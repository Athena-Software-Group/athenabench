"""Run models against benchmark tasks and store predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm

from athena_common.utils import load_jsonl, load_yaml

from .answer_extractors import extract_answer
from .models import load_model
from .evaluate import load_alias_dict, load_related_dict, score_record


def existing_ids(path: Path) -> set[int]:
    """Return a set of record identifiers already stored in *path*."""
    ids = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ids.add(obj.get("id"))
                except Exception:
                    continue
    return ids


def run_model_on_task(
    model_cfg: Dict,
    task_name: str,
    dataset_path: str,
    out_dir: Path,
    evaluate: bool = False,
    alias_dict=None,
    related_dict=None,
) -> Dict[str, float] | None:
    """Generate predictions for a dataset and optionally score them.

    Parameters
    ----------
    model_cfg:
        Configuration dictionary for the model.
    task_name:
        Name of the benchmark task.
    dataset_path:
        Path to the dataset file.
    out_dir:
        Directory where prediction files are written.
    evaluate:
        If ``True``, compute running scores and save them alongside
        predictions.
    alias_dict, related_dict:
        Lookup tables used for the TAA task.

    Returns
    -------
    dict | None
        Aggregate metrics if ``evaluate`` is ``True``; otherwise ``None``.
    """

    model_name = model_cfg.get("name") or model_cfg.get("model")
    model = load_model(model_cfg)

    model_dir = out_dir / model_name
    preds_path = model_dir / f"{task_name}.jsonl"
    done = existing_ids(preds_path)

    records = load_jsonl(dataset_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Running aggregates for progress display
    sum_score = 0.0
    sum_correct = 0
    sum_plausible = 0
    sum_combined = 0.0
    count_success = 0

    with preds_path.open("a", encoding="utf-8") as f:
        iterator = enumerate(records)
        iterator = tqdm(iterator, total=len(records), desc=f"{model_name}-{task_name}")
        for idx, row in iterator:
            if idx in done:
                continue
            prompt = row.get("prompt", "")
            answer = row.get("answer", "")
            response = model.generate(prompt, answer=answer)
            pred = extract_answer(task_name, response)
            rec: Dict = {
                "id": idx,
                "prompt": prompt,
                "response": response,
                "prediction": pred,
                "answer": answer,
            }

            if evaluate:
                score, success = score_record(
                    task_name, pred, answer, alias_dict, related_dict
                )
                rec["score"] = score
                rec["success"] = success
                if success:
                    count_success += 1
                    if isinstance(score, dict):
                        sum_correct += score.get("correct", 0)
                        sum_plausible += score.get("plausible", 0)
                        sum_combined += score.get("combined", 0.0)
                        iterator.set_postfix(
                            {
                                "avg_correct": f"{sum_correct / count_success:.3f}",
                                "avg_plausible": f"{sum_plausible / count_success:.3f}",
                                "avg_combined": f"{sum_combined / count_success:.3f}",
                            }
                        )
                    else:
                        sum_score += float(score)
                        iterator.set_postfix(
                            {"avg_score": f"{sum_score / count_success:.3f}"}
                        )
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if not evaluate or count_success == 0:
        return None

    task = task_name.upper()
    if task == "TAA":
        return {
            "correct_accuracy": sum_correct / count_success,
            "plausible_accuracy": sum_plausible / count_success,
            "combined_accuracy": sum_combined / count_success,
        }
    if task == "RMS":
        return {"f1": sum_score / count_success}
    key = "mean_absolute_deviation" if task == "CVSS" else "accuracy"
    return {key: sum_score / count_success}


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point to run models on benchmark tasks."""

    parser = argparse.ArgumentParser(description="Run models on benchmark tasks")
    parser.add_argument("--config", default="athena_eval/config.yaml")
    parser.add_argument("--model", help="model name to run", default=None)
    parser.add_argument("--task", help="task to run", default=None)
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        default=True,
        help="evaluate predictions during generation (default)",
    )
    parser.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        help="skip evaluation step",
    )
    parser.add_argument(
        "--mini",
        dest="mini",
        action="store_true",
        default=False,
        help="run only on mini benchmark subsets; write outputs to runs-mini",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_yaml(args.config)
    pred_dir = Path("runs-mini") if args.mini else Path(cfg.get("predictions_dir", "data/predictions"))

    alias_dict = related_dict = None
    if args.evaluate:
        alias_dict = load_alias_dict("data/processed/taa/aliases.csv")
        related_dict = load_related_dict("data/processed/taa/related_groups.csv")

    model_names = [args.model] if args.model else list(cfg.get("models", {}).keys())
    task_names = [args.task] if args.task else list(cfg.get("tasks", {}).keys())

    for m in model_names:
        model_cfg = cfg["models"][m]
        model_name = model_cfg.get("name") or model_cfg.get("model")
        for t in task_names:
            dataset_path = cfg["tasks"][t]
            # If --mini, point to benchmark-mini/<basename> when available
            if args.mini:
                mini_path = Path("benchmark-mini") / Path(dataset_path).name
                if mini_path.exists():
                    dataset_path = str(mini_path)
                else:
                    print(f"[run] Mini dataset missing for {t} at {mini_path}; using full dataset {dataset_path}")

            # Fallback specific to MCQ3k: if dataset missing, fall back to MCQ (mini if requested, else full)
            if t.upper() == "MCQ3K" and not Path(dataset_path).exists():
                fallback = None
                if args.mini:
                    mcq_mini = Path("benchmark-mini") / Path(cfg["tasks"].get("MCQ", "")).name
                    if mcq_mini.exists():
                        fallback = str(mcq_mini)
                if fallback is None:
                    mcq_full = cfg["tasks"].get("MCQ")
                    if mcq_full and Path(mcq_full).exists():
                        fallback = mcq_full
                if fallback:
                    print(f"[run] {t} dataset not found at {dataset_path}. Falling back to: {fallback}")
                    dataset_path = fallback
            metrics = run_model_on_task(
                model_cfg,
                t,
                dataset_path,
                pred_dir,
                evaluate=args.evaluate,
                alias_dict=alias_dict,
                related_dict=related_dict,
            )
            if metrics is not None:
                print(f"{model_name} {t}: {metrics}")


if __name__ == "__main__":  # pragma: no cover
    main()
