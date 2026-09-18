"""Exp-2 formal comparison: single-pass generalist vs multi-agent, paired over the
held-out test set. Reports Recall@1 (McNemar exact) and MRR (paired bootstrap 95%
CI + Wilcoxon), plus token / latency / cost per arm. Consumes existing arch
outputs only; no API calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

from rq2_core import RESULT_ROOT

B = 10000
RNG = np.random.default_rng(42)
ARCH = RESULT_ROOT / "arch_comparison"


def per_query(arm: str, split: str, model: str) -> dict[str, dict]:
    path = ARCH / f"{arm}_{model}" / f"{split}_arch_outputs.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        true = r["true_call_label"]
        rank = next((c["rank"] for c in r["ranked"] if c["call_label"] == true), len(r["ranked"]) + 1)
        out[str(r["query_id"])] = {"rr": 1.0 / rank, "hit1": int(rank <= 1), "rank": rank}
    return out


def summary(arm: str, split: str, model: str) -> dict:
    return json.loads((ARCH / f"{arm}_{model}" / f"{split}_summary.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="gemini-flash-latest")
    ap.add_argument("--a", default="multiagent", help="arm A (the one whose gain we test)")
    ap.add_argument("--b", default="generalist", help="arm B (baseline)")
    args = ap.parse_args()

    A, Bq = per_query(args.a, args.split, args.model), per_query(args.b, args.split, args.model)
    ids = sorted(set(A) & set(Bq))
    n = len(ids)
    rr_a = np.array([A[i]["rr"] for i in ids]); rr_b = np.array([Bq[i]["rr"] for i in ids])
    h_a = np.array([A[i]["hit1"] for i in ids]); h_b = np.array([Bq[i]["hit1"] for i in ids])

    # MRR paired bootstrap CI on the difference (A - B)
    diff = rr_a - rr_b
    idx = RNG.integers(0, n, size=(B, n))
    boot = diff[idx].mean(axis=1)
    mrr_lo, mrr_hi = np.percentile(boot, [2.5, 97.5])
    try:
        w_p = wilcoxon(rr_a, rr_b).pvalue
    except ValueError:
        w_p = 1.0

    # Recall@1 McNemar exact on discordant pairs
    b = int(((h_a == 1) & (h_b == 0)).sum())
    c = int(((h_a == 0) & (h_b == 1)).sum())
    mcn_p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0

    sa, sb = summary(args.a, args.split, args.model), summary(args.b, args.split, args.model)
    out = {
        "n": n, "arm_A": args.a, "arm_B": args.b,
        "recall@1": {
            "A": round(float(h_a.mean()), 3), "B": round(float(h_b.mean()), 3),
            "diff_A_minus_B": round(float(h_a.mean() - h_b.mean()), 3),
            "discordant_A_better": b, "discordant_B_better": c, "mcnemar_p": round(mcn_p, 4),
        },
        "MRR": {
            "A": round(float(rr_a.mean()), 3), "B": round(float(rr_b.mean()), 3),
            "diff_A_minus_B": round(float(diff.mean()), 3),
            "diff_CI95": [round(float(mrr_lo), 3), round(float(mrr_hi), 3)],
            "wilcoxon_p": round(float(w_p), 4),
        },
        "cost_latency": {
            arm: {
                "cost_usd": s.get("cost_usd"), "total_tokens": s.get("total_tokens"),
                "mean_latency_s_per_query": s.get("mean_latency_s_per_query"),
                "mean_calls_per_query": s.get("mean_calls_per_query"),
            } for arm, s in [(args.a, sa), (args.b, sb)]
        },
    }
    (ARCH / f"{args.split}_arch_significance.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
