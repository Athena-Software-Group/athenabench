"""Model wrappers for OpenAI and HuggingFace models."""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from athena_common.utils import load_api_key
from dotenv import load_dotenv

try:  # Optional dependency used only for OpenAI models
    from openai import OpenAI
except Exception:  # pragma: no cover - library might not be installed during tests
    OpenAI = None  # type: ignore

try:  # Optional dependency for HuggingFace models
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None  # type: ignore

try:  # Optional dependency for Google Gemini models
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None  # type: ignore


@dataclass
class BaseModel:
    name: str

    def generate(self, prompt: str, **_: object) -> str:  # pragma: no cover - interface
        """Return a model response for *prompt*.

        Additional keyword arguments are ignored by default but allow
        specialized models (e.g. the dummy model used for tests) to accept
        extra information such as the ground-truth answer.
        """
        raise NotImplementedError

class OpenAIModel(BaseModel):
    """Wrapper for OpenAI chat and responses APIs."""

    def __init__(self, name: str, api_key: Optional[str] = None):
        super().__init__(name)
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAIModel")
        api_key = api_key or load_api_key("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.0, **_: object) -> str:
        if self.name.startswith("gpt-5"):
            # Support a search-enabled variant aliased as "gpt-5-search":
            # use the underlying gpt-5 model with the web_search tool enabled.
            api_model = "gpt-5" if self.name in {"gpt-5-search", "gpt-5"} else self.name
            kwargs = {}
            if self.name == "gpt-5-search":
                # Best-effort: enable web_search tool if available.
                kwargs["tools"] = [{"type": "web_search"}]
                kwargs["tool_choice"] = "auto"
            resp = self.client.responses.create(
                model=api_model,
                input=prompt,
                **kwargs,
            )
            return (getattr(resp, "output_text", "") or "").strip()
        else:
            resp = self.client.chat.completions.create(
                model=self.name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()


class GeminiModel(BaseModel):
    """Wrapper for Google Gemini models."""

    def __init__(self, name: str, api_key: Optional[str] = None):
        super().__init__(name)
        if genai is None:
            raise ImportError(
                "google-generativeai package is required for GeminiModel"
            )
        api_key = api_key or load_api_key("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name=name)

    def generate(self, prompt: str, temperature: float = 0.0, **_: object) -> str:
        resp = self.client.generate_content(
            prompt, generation_config={"temperature": temperature}
        )
        return (getattr(resp, "text", "") or "").strip()


class HuggingFaceModel(BaseModel):
    """Wrapper for HuggingFace transformer models (Qwen3 + Llama 3/4)."""
    def __init__(
        self, name: str, max_new_tokens: int = 2048, api_key: Optional[str] = None
    ):
        super().__init__(name)
        if pipeline is None:
            raise ImportError("transformers package is required for HuggingFaceModel")

        # --- Auth token resolution ---
        load_dotenv()
        token = api_key or os.getenv("HF_TOKEN") \
                        or os.getenv("HUGGINGFACE_API_KEY") \
                        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        auth_args = {"token": token} if token else {}

        common_load_kwargs = dict(trust_remote_code=True, **auth_args)
        if torch.cuda.is_available():
            model_load_kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16, **common_load_kwargs)
        else:
            model_load_kwargs = dict(**common_load_kwargs)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(name, **common_load_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(
                name, attn_implementation="flash_attention_2", **model_load_kwargs
            )
        except Exception:
            # Fallback without flash-attn
            self.tokenizer = AutoTokenizer.from_pretrained(name, **common_load_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(name, **model_load_kwargs)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,  # only the continuation
        )
        self.max_new_tokens = max_new_tokens

    # ----- helpers -----
    def _format_prompt(self, prompt: str) -> str:
        """Use chat template when available (covers Qwen3 + Llama Instruct)."""
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def _eos_ids(self):
        """Include EOS and EOT if present (Llama often uses <|eot_id|>)."""
        ids = set()
        for attr in ("eos_token_id", "eot_token_id"):
            val = getattr(self.tokenizer, attr, None)
            if val is not None:
                ids.add(val)
        for attr in ("eos_token_id", "eot_token_id"):
            val = getattr(self.model.config, attr, None)
            if val is not None:
                ids.add(val)
        if not ids:
            return None
        return list(ids) if len(ids) > 1 else next(iter(ids))

    def _cap_new_tokens(self, prompt_str: str, requested: int) -> int:
        """Avoid exceeding the model's context window."""
        cfg_ctx = getattr(self.model.config, "max_position_embeddings", None)
        tok_ctx = getattr(self.tokenizer, "model_max_length", None)
        ctx = None
        for v in (cfg_ctx, tok_ctx):
            if isinstance(v, int) and v > 0:
                ctx = v if ctx is None else min(ctx, v)
        if ctx is None:  # last-resort default
            ctx = 4096

        # Tokenize on CPU just to count; pipeline will retokenize on the right device.
        ids = self.tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt")["input_ids"]
        room = max(1, ctx - ids.shape[-1])
        return max(1, min(requested, room))

    def generate(self, prompt: str, temperature: float = 0.0, **_: object) -> str:
        formatted = self._format_prompt(prompt)
        do_sample = temperature > 0.0
        max_new = self._cap_new_tokens(formatted, self.max_new_tokens)
        eos = self._eos_ids()

        gen = self.pipe(
            formatted,
            max_new_tokens=max_new,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            eos_token_id=eos,
            pad_token_id=self.tokenizer.pad_token_id,
        )[0]["generated_text"]
        return gen.strip()

def load_model(cfg: dict) -> BaseModel:
    mtype = cfg.get("type")
    name = cfg.get("name") or cfg.get("model")
    if mtype in {"openai", "chatgpt"}:
        return OpenAIModel(name, api_key=cfg.get("api_key"))
    if mtype in {"hf", "huggingface"}:
        return HuggingFaceModel(name, max_new_tokens=cfg.get("max_new_tokens", 2048), api_key=cfg.get("api_key"))
    if mtype in {"gemini", "google"}:
        return GeminiModel(name, api_key=cfg.get("api_key"))
    if mtype == "dummy":
        return DummyModel(name)
    raise ValueError(f"Unsupported model type: {mtype}")


class DummyModel(BaseModel):
    """Return either the ground truth answer or 'I don't know'.

    This lightweight model is intended for testing the evaluation pipeline. It
    ignores the prompt and, with 50% probability, echoes the provided answer
    (passed as the ``answer`` keyword argument) or the fixed string
    ``"I don't know"``.
    """

    def generate(self, prompt: str, answer: str = "", **_: object) -> str:  # type: ignore[override]
        _ = prompt  # Prompt is unused but kept for interface compatibility
        return answer if random.random() < 0.5 else "I don't know"

