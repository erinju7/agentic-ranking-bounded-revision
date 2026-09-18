"""Consolidate the per-model reranker runs into one comparison table.

Reads every `single_agent_rerank_*` result directory's `{split}_summary_metrics.json`
(the `rerank_mean` block) plus its `run_metadata.json` (model / candidate-text
policy), and prints a tidy model x split x metric table. Also emits a CSV for
direct use in the write-up. Consumes existing outputs only; no API calls.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rq2_core import RESULT_ROOT

METRICS = ["Recall@1", "Recall@5", "Recall@10", "MRR", "mean_true_rank"]


def load_row(directory: Path, split: str) -> dict[str, Any] | None:
    summary_path = directory / f"{split}_summary_metrics.json"
    metadata_path = directory / f"{split}_run_metadata.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    rerank = summary.get("rerank_mean", {})
    if not rerank:
        return None
    row = {
        "condition": directory.name.replace("single_agent_rerank_", ""),
        "policy": metadata.get("candidate_text_policy", ""),
        "model": metadata.get("model", "unknown"),
        "split": split,
        "n_parse_failures": summary.get("n_parse_failures"),
    }
    for metric in METRICS:
        value = rerank.get(metric)
        row[metric] = round(value, 4) if isinstance(value, (int, float)) else None
    # Retriever ceiling reference (repeat-independent), useful alongside.
    ref = summary.get("retriever_baseline_over_topn", {})
    row["retriever_MRR"] = round(ref["MRR"], 4) if "MRR" in ref else None
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 model-comparison table.")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--output-dir", default=str(RESULT_ROOT / "rq2_model_comparison"))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for directory in sorted(RESULT_ROOT.glob("single_agent_rerank_*")):
        if not directory.is_dir() or "mock" in directory.name:
            continue
        for split in args.splits:
            row = load_row(directory, split)
            if row:
                rows.append(row)

    rows.sort(key=lambda r: (r["split"], r["policy"], r["model"]))
    if rows:
        with (out / "rq2_model_comparison.csv").open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    header = f"{'condition':<28}{'model':<22}{'split':<6}" + "".join(
        f"{m:>12}" for m in METRICS
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = "".join(f"{r[m]!s:>12}" for m in METRICS)
        print(f"{r['condition']:<28}{r['model']:<22}{r['split']:<6}{cells}")
    print(f"\nWrote {out/'rq2_model_comparison.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
