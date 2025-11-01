"""Evaluate model predictions for benchmark tasks."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from cvss import CVSS3
from tqdm import tqdm

from athena_common.utils import load_jsonl, load_yaml
import hashlib
from .answer_extractors import extract_answer

# ---------------------------------------------------------------------------
# TAA evaluation helpers


def load_alias_dict(path: str) -> Dict[str, List[str]]:
    """Load threat actor aliases from a CSV file."""
    alias: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row["ThreatActor"].strip().lower()
            v = row["Alias"].strip().lower()
            alias.setdefault(k, []).append(v)
            alias.setdefault(v, []).append(k)  # bidirectional
    return alias


def load_related_dict(path: str) -> Dict[str, List[str]]:
    """Load related threat actor mappings from a CSV file."""
    related: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row["ThreatActor"].strip().lower()
            v = row["RelatedGroup"].strip().lower()
            related.setdefault(k, []).append(v)
            related.setdefault(v, []).append(k)
    return related


def is_alias_connected(
    actor1: str, actor2: str, alias_dict: Dict[str, List[str]]
) -> bool:
    """Return ``True`` if *actor1* is connected to *actor2* via aliases."""
    visited = set()
    queue = [actor1]
    while queue:
        cur = queue.pop(0)
        if cur == actor2:
            return True
        visited.add(cur)
        for nxt in alias_dict.get(cur, []):
            if nxt not in visited:
                queue.append(nxt)
    return False


def is_related_connected(
    actor1: str,
    actor2: str,
    alias_dict: Dict[str, List[str]],
    related_dict: Dict[str, List[str]],
) -> bool:
    """Return ``True`` if actors are connected via aliases or related groups."""
    visited = set()
    queue = [actor1]
    while queue:
        cur = queue.pop(0)
        if cur == actor2:
            return True
        visited.add(cur)
        neighbours = alias_dict.get(cur, []) + related_dict.get(cur, [])
        for nxt in neighbours:
            if nxt not in visited:
                queue.append(nxt)
    return False


def threat_actor_connection(
    actor1: str,
    actor2: str,
    alias_dict: Dict[str, List[str]],
    related_dict: Dict[str, List[str]],
) -> str:
    """Classify the relationship between two actors.

    Returns ``"C"`` if they are aliases, ``"P"`` if plausibly related and
    ``"I"`` if independent.
    """
    actor1 = actor1.strip().lower()
    actor2 = actor2.strip().lower()
    if is_alias_connected(actor1, actor2, alias_dict):
        return "C"
    if is_related_connected(actor1, actor2, alias_dict, related_dict):
        return "P"
    return "I"


def score_taa(
    pred: str,
    ans: str,
    alias_dict: Dict[str, List[str]],
    related_dict: Dict[str, List[str]],
) -> Tuple[Dict[str, float], bool]:
    """Score a TAA prediction.

    Returns a dictionary with correctness, plausibility and combined metrics
    along with a success boolean.
    """
    res = threat_actor_connection(ans, pred, alias_dict, related_dict)
    score = {
        "correct": 1 if res == "C" else 0,
        "plausible": 1 if res in {"C", "P"} else 0,
        "combined": 1.0 if res == "C" else 0.5 if res == "P" else 0.0,
    }
    return score, True


# ---------------------------------------------------------------------------
# Task scoring


def score_record(task: str, pred: str, ans: str, alias_dict, related_dict):
    """Score a single prediction for *task*.

    Returns a tuple of the score and a boolean indicating whether the score
    is valid (e.g. parsing may fail for CVSS vectors).
    """
    task = task.upper()
    if task == "RCM":
        return (1 if pred.strip().lower() == ans.strip().lower() else 0, True)
    if task == "VSP":
        try:
            p = CVSS3(pred.strip()).scores()[0]
            a = CVSS3(ans.strip()).scores()[0]
            return (abs(p - a), True)
        except Exception:
            return (0.0, False)
    if task == "TAA":
        return score_taa(pred, ans, alias_dict, related_dict)
    if task == "ATE":
        p = pred.strip().split(".")[0].upper()
        a = ans.strip().split(".")[0].upper()
        return (1 if p and p == a else 0, True)
    if task == "RMS":
        p_ids = set(re.findall(r"M\d{4}", pred.upper()))
        a_ids = set(re.findall(r"M\d{4}", ans.upper()))
        tp = len(p_ids & a_ids)
        fp = len(p_ids - a_ids)
        fn = len(a_ids - p_ids)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision == 0 and recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return (f1, True)
    # default MCQ style
    return (1 if pred.strip().lower() == ans.strip().lower() else 0, True)


# ---------------------------------------------------------------------------
# Main evaluation logic


def format_percentage_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Convert accuracy- and f1-style metrics to percentages."""
    formatted: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and (
            "accuracy" in key.lower() or key.lower() == "f1"
        ):
            formatted[key] = value * 100.0
        else:
            formatted[key] = value
    return formatted


def evaluate_records(
    task: str,
    records: List[Dict],
    out_path: Path,
    alias_dict,
    related_dict,
    vsp_denominator: float = 7.7,
) -> Dict[str, float]:
    """Evaluate a list of prediction records and write a scored version."""
    results = []
    sum_score = 0.0
    sum_correct = 0
    sum_plausible = 0
    sum_combined = 0.0
    count_success = 0
    iterator = enumerate(records)
    iterator = tqdm(iterator, total=len(records), desc=str(out_path))
    for _, rec in iterator:
        response = rec.get("response", "")
        pred = extract_answer(task, response)
        ans = rec.get("answer", "")
        score, success = score_record(task, pred, ans, alias_dict, related_dict)
        results.append({**rec, "score": score, "success": success})
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
                iterator.set_postfix({"avg_score": f"{sum_score / count_success:.3f}"})
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    task_up = task.upper()
    if task_up == "TAA":
        corr = [r["score"]["correct"] for r in results if r["success"]]
        plaus = [r["score"]["plausible"] for r in results if r["success"]]
        comb = [r["score"]["combined"] for r in results if r["success"]]
        metrics = {
            "accuracy": sum(corr) / len(corr) if corr else 0.0,
            "plausible_accuracy": sum(plaus) / len(plaus) if plaus else 0.0,
            "combined_accuracy": sum(comb) / len(comb) if comb else 0.0,
        }
        return format_percentage_metrics(metrics)

    scores = [r["score"] for r in results if r["success"]]
    if task_up == "RMS":
        metrics = {"f1": sum(scores) / len(scores) if scores else 0.0}
        return format_percentage_metrics(metrics)

    if task_up == "VSP":
        mad = sum(scores) / len(scores) if scores else 0.0
        denom = vsp_denominator if vsp_denominator else 1.0
        if denom == 0:
            denom = 1.0
        accuracy = 1 - (mad / denom)
        metrics = {"MAD": mad, "accuracy": accuracy}
        return format_percentage_metrics(metrics)

    if task_up == "CVSS":
        metrics = {
            "mean_absolute_deviation": sum(scores) / len(scores) if scores else 0.0
        }
        return metrics

    metrics = {"accuracy": sum(scores) / len(scores) if scores else 0.0}
    return format_percentage_metrics(metrics)


def evaluate_file(
    task: str,
    preds_path: Path,
    out_path: Path,
    alias_dict,
    related_dict,
    vsp_denominator: float = 7.7,
) -> Dict[str, float]:
    """Evaluate a predictions file and write a scored version."""
    records = load_jsonl(str(preds_path))
    return evaluate_records(
        task, records, out_path, alias_dict, related_dict, vsp_denominator
    )


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point to evaluate model predictions."""

    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument("--config", default="athena_eval/config.yaml")
    parser.add_argument("--model", help="model name to eval", default=None)
    parser.add_argument("--task", help="task to eval", default=None)
    parser.add_argument(
        "--mini",
        dest="mini",
        action="store_true",
        default=False,
        help="evaluate only on mini subsets; read/write under runs-mini",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_yaml(args.config)
    pred_dir = Path("runs-mini") if args.mini else Path(cfg.get("predictions_dir", "data/predictions"))

    alias_dict = load_alias_dict("data/processed/taa/aliases.csv")
    related_dict = load_related_dict("data/processed/taa/related_groups.csv")

    model_names = [args.model] if args.model else list(cfg.get("models", {}).keys())
    tasks_cfg = cfg.get("tasks", {})
    default_eval_tasks = cfg.get("default_eval_tasks")
    vsp_denominator = float(cfg.get("vsp_mad_denominator", 7.7))

    def resolve_task_name(name: str) -> str:
        if name in tasks_cfg:
            return name
        for candidate in tasks_cfg.keys():
            if candidate.upper() == name.upper():
                return candidate
        return name

    if args.task:
        resolved = resolve_task_name(args.task)
        task_names = [resolved]
        avg_task_set = {resolved.upper()}
    else:
        raw_task_names = default_eval_tasks or list(tasks_cfg.keys())
        task_names = [resolve_task_name(t) for t in raw_task_names]
        avg_task_set = {t.upper() for t in task_names}

    for m in model_names:
        model_cfg = cfg["models"][m]
        model_name = model_cfg.get("name") or model_cfg.get("model")
        avg_entries: List[Tuple[str, float]] = []
        avg_display: Dict[str, float] = {}

        def emit_metrics(task_key: str, metrics: Dict[str, float]) -> None:
            display_metrics = {
                k: (f"{float(v):.4f}" if isinstance(v, (int, float)) else v)
                for k, v in metrics.items()
            }
            print(f"{model_name} {task_key}: {display_metrics}")
            task_upper = task_key.upper()
            if task_upper in avg_task_set:
                value = metrics.get("accuracy")
                if value is None and task_upper == "RMS":
                    value = metrics.get("f1")
                if value is not None:
                    avg_entries.append((task_key, value))
                    avg_display[task_upper] = value

        for t in task_names:
            model_dir = pred_dir / model_name
            preds_path = model_dir / f"{t}.jsonl"
            out_path = model_dir / f"{t}-scored.jsonl"
            model_dir.mkdir(parents=True, exist_ok=True)
            task_up = t.upper()

            # If --mini, try to evaluate predictions from runs-mini first
            if args.mini:
                if preds_path.exists():
                    metrics = evaluate_file(
                        t, preds_path, out_path, alias_dict, related_dict, vsp_denominator
                    )
                    emit_metrics(t, metrics)
                    continue
                # Build mini dataset path (benchmark-mini/<basename>)
                full_ds = tasks_cfg.get(t)
                mini_ds = Path("benchmark-mini") / Path(full_ds).name if full_ds else None
                if mini_ds is None or not mini_ds.exists():
                    print(f"[eval] Mini dataset missing for {t}; cannot evaluate mini.")
                    continue
                # Attempt to map from full predictions under the regular runs dir
                full_runs_dir = Path(cfg.get("predictions_dir", "runs")) / model_name
                full_preds_path = full_runs_dir / f"{t}.jsonl"
                # For MCQ3k, also consider fallback to MCQ predictions
                if task_up == "MCQ3K" and not full_preds_path.exists():
                    alt = full_runs_dir / "MCQ.jsonl"
                    if alt.exists():
                        full_preds_path = alt
                if not full_preds_path.exists():
                    print(f"[eval] No predictions found for mini {t}. Expected {preds_path} or {full_preds_path}.")
                    continue
                # Load mini and full predictions
                mini_recs = load_jsonl(str(mini_ds))
                # Build mapping from prompt_hash -> answer for mini
                def phash(s: str | None) -> str:
                    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

                mini_map: Dict[str, str] = {}
                mini_order: List[str] = []
                for r in mini_recs:
                    h = r.get("prompt_hash") or phash(r.get("prompt", ""))
                    mini_map[h] = r.get("answer", "")
                    mini_order.append(h)
                full_preds = load_jsonl(str(full_preds_path))
                # Filter and override answers to align with mini
                filtered: List[Dict] = []
                for rec in full_preds:
                    h = rec.get("prompt_hash") or phash(rec.get("prompt", ""))
                    if h in mini_map:
                        new_rec = dict(rec)
                        new_rec["answer"] = mini_map[h]
                        filtered.append(new_rec)
                # Preserve the mini ordering where possible
                order_index = {h: i for i, h in enumerate(mini_order)}
                filtered.sort(
                    key=lambda r: order_index.get(
                        r.get("prompt_hash") or phash(r.get("prompt", "")), 10**9
                    )
                )
                metrics = evaluate_records(
                    t, filtered, out_path, alias_dict, related_dict, vsp_denominator
                )
                emit_metrics(t, metrics)
                continue

            # Non-mini path: preserve MCQ3k fallback by ids if needed
            if task_up == "MCQ3K" and not preds_path.exists():
                mcq_preds = model_dir / "MCQ.jsonl"
                mcq3k_path = Path(cfg["tasks"].get("MCQ3k", ""))
                if mcq_preds.exists() and mcq3k_path.exists():
                    print(f"[eval] {preds_path} not found. Using {mcq_preds} filtered to MCQ3k ids.")
                    subset = load_jsonl(str(mcq3k_path))
                    id_to_answer = {int(r.get("id")): r.get("answer", "") for r in subset}
                    full_preds = load_jsonl(str(mcq_preds))
                    filtered_preds: List[Dict] = []
                    for rec in full_preds:
                        rid = rec.get("id")
                        if rid in id_to_answer:
                            new_rec = dict(rec)
                            new_rec["answer"] = id_to_answer[rid]
                            filtered_preds.append(new_rec)
                    metrics = evaluate_records(
                        t, filtered_preds, out_path, alias_dict, related_dict, vsp_denominator
                    )
                    emit_metrics(t, metrics)
                    continue
                if mcq_preds.exists():
                    print(f"[eval] {preds_path} not found and MCQ3k dataset missing. Evaluating {mcq_preds} as fallback.")
                    metrics = evaluate_file(
                        "MCQ", mcq_preds, out_path, alias_dict, related_dict, vsp_denominator
                    )
                    emit_metrics(t, metrics)
                    continue

            metrics = evaluate_file(
                t, preds_path, out_path, alias_dict, related_dict, vsp_denominator
            )
            emit_metrics(t, metrics)

        if avg_entries:
            avg_value = sum(value for _, value in avg_entries) / len(avg_entries)
            task_list = ", ".join(name for name, _ in avg_entries)
            print(f"{model_name} average accuracy across {task_list}: {avg_value:.4f}")
            task_headers = [
                "MCQ3K(Accuracy)",
                "ATE (Accuracy)",
                "RCM (Accuracy)",
                "RMS (F1-score)",
                "VSP (Acc)",
                "TAA (Accuracy)",
                "Combined",
            ]
            values = [
                avg_display.get("MCQ3K"),
                avg_display.get("ATE"),
                avg_display.get("RCM"),
                avg_display.get("RMS"),
                avg_display.get("VSP"),
                avg_display.get("TAA"),
                avg_value,
            ]
            formatted_values = [
                f"{float(v):.4f}" if isinstance(v, (int, float)) else ""
                for v in values
            ]
            sys.stdout.write("\t".join(task_headers) + "\n")
            sys.stdout.write("\t".join(formatted_values) + "\n")


if __name__ == "__main__":  # pragma: no cover
    main()
