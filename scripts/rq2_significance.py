"""RQ2 statistical significance for the reranker comparisons.

All comparisons are PAIRED: every method is evaluated on the same frozen query
set, so we compare per-query outcomes query-by-query.

For each requested pair (A vs B, same split) we report:
  - Recall@1 for each side and the paired difference, with a McNemar exact test
    (binomial on the discordant hit@1 pairs).
  - MRR for each side and the paired difference, with a paired bootstrap 95% CI
    over queries and a Wilcoxon signed-rank test on per-query reciprocal ranks.
  - mean true rank difference with a paired bootstrap 95% CI.

A difference is "significant" when its bootstrap CI excludes 0 / the test
p < 0.05. This is what makes the null results (context, enrichment) defensible:
a CI that straddles 0 is evidence of no effect, not merely an absent one.

Consumes existing outputs only; no API calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, wilcoxon

from rq2_core import RESULT_ROOT, load_ground_truth

B = 10000
RNG = np.random.default_rng(42)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def per_query(directory: Path, split: str) -> dict[str, dict[str, float]]:
    """{query_id: {rr, hit1, rank}} using the first repeat, top_n+1 for misses."""
    outputs = directory / f"{split}_rerank_outputs.jsonl"
    if not outputs.exists():
        return {}
    meta_path = directory / f"{split}_run_metadata.json"
    top_n = (
        json.loads(meta_path.read_text()).get("top_n", 50)
        if meta_path.exists()
        else 50
    )
    truths = load_ground_truth(split)
    out: dict[str, dict[str, float]] = {}
    for row in read_jsonl(outputs):
        if int(row.get("repeat", 0)) != 0 or not row.get("ranked"):
            continue
        qid = str(row["query_id"])
        truth = set(truths.get(qid, []))
        rank = next(
            (r["rank"] for r in row["ranked"] if r["call_label"] in truth),
            top_n + 1,
        )
        out[qid] = {"rr": 1.0 / rank, "hit1": float(rank == 1), "rank": float(rank)}
    return out


def paired_arrays(a: dict, b: dict) -> tuple[np.ndarray, ...]:
    qids = sorted(set(a) & set(b))
    rr_a = np.array([a[q]["rr"] for q in qids])
    rr_b = np.array([b[q]["rr"] for q in qids])
    h_a = np.array([a[q]["hit1"] for q in qids])
    h_b = np.array([b[q]["hit1"] for q in qids])
    rk_a = np.array([a[q]["rank"] for q in qids])
    rk_b = np.array([b[q]["rank"] for q in qids])
    return rr_a, rr_b, h_a, h_b, rk_a, rk_b


def boot_ci(diff_per_query: np.ndarray) -> tuple[float, float]:
    """Paired bootstrap 95% CI for the mean paired difference."""
    n = len(diff_per_query)
    idx = RNG.integers(0, n, size=(B, n))
    means = diff_per_query[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar_p(h_a: np.ndarray, h_b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar on hit@1: discordant pairs b (A hit, B miss), c (A miss, B hit)."""
    b = int(np.sum((h_a == 1) & (h_b == 0)))
    c = int(np.sum((h_a == 0) & (h_b == 1)))
    n = b + c
    p = 1.0 if n == 0 else binomtest(b, n, 0.5).pvalue
    return b, c, float(p)


def compare(a: dict, b: dict) -> dict[str, Any]:
    rr_a, rr_b, h_a, h_b, rk_a, rk_b = paired_arrays(a, b)
    n = len(rr_a)
    mrr_lo, mrr_hi = boot_ci(rr_a - rr_b)
    rank_lo, rank_hi = boot_ci(rk_a - rk_b)
    bcell, ccell, mcp = mcnemar_p(h_a, h_b)
    # Wilcoxon needs at least one non-zero difference
    try:
        w_p = float(wilcoxon(rr_a, rr_b, zero_method="wilcox").pvalue)
    except ValueError:
        w_p = 1.0
    return {
        "n": n,
        "recall@1_A": round(float(h_a.mean()), 3),
        "recall@1_B": round(float(h_b.mean()), 3),
        "recall@1_diff": round(float(h_a.mean() - h_b.mean()), 3),
        "mcnemar_discordant": [bcell, ccell],
        "mcnemar_p": round(mcp, 4),
        "MRR_A": round(float(rr_a.mean()), 3),
        "MRR_B": round(float(rr_b.mean()), 3),
        "MRR_diff": round(float(rr_a.mean() - rr_b.mean()), 3),
        "MRR_diff_CI95": [round(mrr_lo, 3), round(mrr_hi, 3)],
        "wilcoxon_p": round(w_p, 4),
        "mean_rank_diff": round(float(rk_a.mean() - rk_b.mean()), 2),
        "mean_rank_diff_CI95": [round(rank_lo, 2), round(rank_hi, 2)],
        "MRR_sig": not (mrr_lo <= 0 <= mrr_hi),
        "recall@1_sig": mcp < 0.05,
    }


def run_ci(a: dict) -> dict[str, Any]:
    """Per-run bootstrap 95% CI for MRR and Recall@1."""
    rr = np.array([v["rr"] for v in a.values()])
    h1 = np.array([v["hit1"] for v in a.values()])
    n = len(rr)
    idx = RNG.integers(0, n, size=(B, n))
    mrr = np.percentile(rr[idx].mean(axis=1), [2.5, 97.5])
    r1 = np.percentile(h1[idx].mean(axis=1), [2.5, 97.5])
    return {
        "MRR": round(float(rr.mean()), 3),
        "MRR_CI95": [round(float(mrr[0]), 3), round(float(mrr[1]), 3)],
        "recall@1": round(float(h1.mean()), 3),
        "recall@1_CI95": [round(float(r1[0]), 3), round(float(r1[1]), 3)],
    }


def d(name: str) -> Path:
    return RESULT_ROOT / name


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 paired significance tests.")
    parser.add_argument("--output-dir", default=str(RESULT_ROOT / "rq2_significance"))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    thin = {
        "gemini-2.5-flash-lite": d("single_agent_rerank_baseline"),
        "claude-haiku-4-5": d("single_agent_rerank_baseline_claude-haiku-4-5"),
        "claude-sonnet-5": d("single_agent_rerank_baseline_claude-sonnet-5"),
        "claude-opus-4-8": d("single_agent_rerank_baseline_claude-opus-4-8"),
        "gemini-flash-latest": d("single_agent_rerank_baseline_gemini-flash-latest"),
    }

    # (name, dirA, dirB, split) — paired comparisons that carry the claims.
    comparisons = [
        # Model ladder on test (each step up vs the one below).
        ("haiku vs flash-lite (thin,test)", thin["claude-haiku-4-5"], thin["gemini-2.5-flash-lite"], "test"),
        ("sonnet vs haiku (thin,test)", thin["claude-sonnet-5"], thin["claude-haiku-4-5"], "test"),
        ("opus vs sonnet (thin,test)", thin["claude-opus-4-8"], thin["claude-sonnet-5"], "test"),
        ("flash-latest vs opus (thin,test)", thin["gemini-flash-latest"], thin["claude-opus-4-8"], "test"),
        # Null result: context awareness (Haiku), both splits.
        ("Haiku +historical vs thin (dev)", d("single_agent_rerank_historical_context"), thin["claude-haiku-4-5"], "dev"),
        ("Haiku +historical vs thin (test)", d("single_agent_rerank_historical_context"), thin["claude-haiku-4-5"], "test"),
        # Context awareness (flash-latest) + candidate-text on flash-latest, if present.
        ("flash-latest +historical vs thin (test)", d("single_agent_rerank_historical_context_gemini-flash-latest"), thin["gemini-flash-latest"], "test"),
        ("flash-latest enriched vs thin (test)", d("single_agent_rerank_enriched_gemini-flash-latest"), thin["gemini-flash-latest"], "test"),
        ("flash-latest targeted vs thin (test)", d("single_agent_rerank_enriched_targeted_gemini-flash-latest"), thin["gemini-flash-latest"], "test"),
    ]

    results = {"per_run_ci": {}, "comparisons": {}}
    for model, directory in thin.items():
        pq = per_query(directory, "test")
        if pq:
            results["per_run_ci"][f"{model} (thin,test)"] = run_ci(pq)

    print(f"{'per-run (thin,test)':<28}{'MRR [95% CI]':<26}{'R@1 [95% CI]'}")
    for k, v in results["per_run_ci"].items():
        print(f"{k:<28}{v['MRR']} {v['MRR_CI95']!s:<20}{v['recall@1']} {v['recall@1_CI95']}")

    print(f"\n{'comparison':<42}{'ΔMRR [95% CI]':<26}{'ΔR@1':>7} {'McN p':>7} {'sig'}")
    for name, da, db, split in comparisons:
        a, b = per_query(da, split), per_query(db, split)
        if not a or not b:
            print(f"{name:<42}(skipped: run missing)")
            continue
        r = compare(a, b)
        results["comparisons"][name] = r
        sig = "MRR*" if r["MRR_sig"] else "n.s."
        if r["recall@1_sig"]:
            sig += " R@1*"
        print(f"{name:<42}{r['MRR_diff']:+} {r['MRR_diff_CI95']!s:<19}"
              f"{r['recall@1_diff']:+7} {r['mcnemar_p']:>7} {sig}")

    (out / "rq2_significance.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out/'rq2_significance.json'}")


if __name__ == "__main__":
    main()
