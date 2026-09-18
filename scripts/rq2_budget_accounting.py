"""RQ2 budget accounting: turn the per-call usage logs into a cost table.

Every reranker run records `usage_metadata` (prompt/output token counts) on each
query in `{split}_rerank_outputs.jsonl`, and the model string in
`{split}_run_metadata.json`. This script walks the reranker result directories,
sums tokens per (condition, model, split), and prices them with the list prices
in `rq2_core.MODEL_PRICING_USD_PER_MTOK`, replacing the paper's hand-waved
"total API cost was below $1" with an auditable per-condition, per-model table.

It reads only existing outputs; it makes no API calls. Rerun it after any new
model/condition run to refresh the budget.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rq2_core import MODEL_PRICING_USD_PER_MTOK, RESULT_ROOT, usage_cost_usd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def condition_dirs() -> list[Path]:
    return sorted(
        d
        for d in RESULT_ROOT.glob("single_agent_rerank_*")
        if d.is_dir() and "mock" not in d.name
    )


def accumulate(directory: Path, split: str) -> dict[str, Any] | None:
    outputs = directory / f"{split}_rerank_outputs.jsonl"
    metadata_path = directory / f"{split}_run_metadata.json"
    if not outputs.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = metadata.get("model", "unknown")
    if model == "mock":
        return None
    rows = read_jsonl(outputs)
    if not rows:
        return None

    prompt_tokens = output_tokens = 0
    cost = 0.0
    priced_calls = 0
    for row in rows:
        usage = row.get("usage_metadata", {}) or {}
        prompt_tokens += usage.get("prompt_token_count") or 0
        output_tokens += usage.get("candidates_token_count") or 0
        call_cost = usage_cost_usd(model, usage)
        if call_cost is not None:
            cost += call_cost
            priced_calls += 1

    n_calls = len(rows)
    has_price = model in MODEL_PRICING_USD_PER_MTOK
    return {
        "condition": directory.name.replace("single_agent_rerank_", ""),
        "candidate_text_policy": metadata.get("candidate_text_policy", ""),
        "model": model,
        "provider": metadata.get("provider", ""),
        "split": split,
        "n_calls": n_calls,
        "repeats": metadata.get("repeats", 1),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "mean_prompt_tokens_per_call": round(prompt_tokens / n_calls, 1),
        "mean_output_tokens_per_call": round(output_tokens / n_calls, 1),
        "cost_usd": round(cost, 4) if has_price else None,
        "cost_per_call_usd": round(cost / n_calls, 5) if has_price else None,
        "priced_calls": priced_calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 token/cost accounting.")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument(
        "--output-dir",
        default=str(RESULT_ROOT / "rq2_budget_accounting"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for directory in condition_dirs():
        for split in args.splits:
            record = accumulate(directory, split)
            if record is not None:
                records.append(record)

    total_cost = sum(r["cost_usd"] for r in records if r["cost_usd"] is not None)
    total_tokens = sum(r["total_tokens"] for r in records)
    summary = {
        "n_runs": len(records),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 4),
        "price_table_usd_per_mtok": MODEL_PRICING_USD_PER_MTOK,
        "runs": records,
    }

    (output_dir / "rq2_budget.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if records:
        with (output_dir / "rq2_budget.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    print(f"{'condition':<24}{'model':<22}{'split':<6}{'calls':>6}{'tokens':>12}{'cost $':>10}")
    for r in records:
        cost = "n/a" if r["cost_usd"] is None else f"{r['cost_usd']:.4f}"
        print(
            f"{r['condition']:<24}{r['model']:<22}{r['split']:<6}"
            f"{r['n_calls']:>6}{r['total_tokens']:>12}{cost:>10}"
        )
    print(f"\nTotal priced cost: ${total_cost:.4f} over {total_tokens:,} tokens")
    print(f"Wrote {output_dir/'rq2_budget.csv'} and rq2_budget.json")


if __name__ == "__main__":
    main()
