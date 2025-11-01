# Athena Bench

Athena Bench provides cybersecurity benchmarking tasks for evaluating language models on a shared set of threat-intelligence workloads. This guide focuses on running the bundled **mini** benchmark, which contains lightweight subsets of each task for quick iteration.

## Run the Mini Benchmark

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure models and credentials** in `athena_eval/config.yaml`. Each entry specifies a provider (`openai`, `gemini`, `huggingface`, or `dummy`) and model name. API keys can be placed in the environment or a `.env` file that is auto-loaded. Example:
   ```
   OPENAI_API_KEY=""
   GEMINI_API_KEY=""
   HF_TOKEN=""
   ```

3. **Generate predictions on the mini subset**
   ```bash
   python -m athena_eval.run --mini --model gpt-4o --task RCM
   ```
   - Omit `--model` or `--task` to iterate over all configured entries.
   - The `--mini` flag swaps each dataset path for its counterpart in `benchmark-mini/` and writes outputs to `runs-mini/<model>/<task>.jsonl`.
   - Evaluation runs by default; add `--no-evaluate` to skip scoring during generation.

4. **Evaluate previously generated predictions (optional)**
   ```bash
   python -m athena_eval.evaluate --mini --model gpt-4o --task RCM
   ```
   This reads predictions from `runs-mini/` and compares them against the mini datasets in `benchmark-mini/`, emitting task-level metrics and refreshing any `*-scored.jsonl` artifacts.

### Mini Tasks Available

Mini splits are provided for every benchmark task and live in `benchmark-mini/`:
- `athena-cti-mcq-3k.jsonl`
- `athena-cti-ate.jsonl`
- `athena-cti-rcm.jsonl`
- `athena-cti-rms.jsonl`
- `athena-cti-taa.jsonl`
- `athena-cti-vsp.jsonl`

Ensure the task names used with `--task` match the keys in `athena_eval/config.yaml` (e.g.,`MCQ3k`, `ATE`, `RCM`, `RMS`, `TAA`, `VSP`).


## Full Benchmark
For access to the full benchmark dataset, reach out to info@athenasecuritygrp.com.