"""Diagnostic error analysis for the multi-agent architecture result.

Consumes the instrumented internals dump (scripts/rq2_architecture.py
--log-internals): per query, the theme/action/budget specialist scores, the
coordinator's raw ranking, and the cid->label / true-cid metadata. No API calls.

Separates implementation artifacts from architectural limitations by answering:
  1. Does the coordinator rank by the specialist evidence, or default to input
     (alphabetical-cid) order?  -> rank-correlation of the coordinator order with
     sorted-cid order vs deterministic-fusion order vs retriever order.
  2. Are the 50-point specialist scores missing evidence or neutral evidence?
     -> cross-tab of score==50 against whether that candidate's governed field is
     present in its profile.
  3. Does deterministic fusion (no LLM coordinator) match/beat the coordinator?
  4. What does each specialist contribute alone?
Metrics: Recall@1, MRR, mean true rank of the true call under each ranking.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from rq2_core import RESULT_ROOT
from rq2_architecture import load_profiles, THEME_FIELDS, ACTION_FIELDS, BUDGET_FIELDS

ARCH = RESULT_ROOT / "arch_comparison"
INTERNALS = ARCH / "multiagent_internals" / "test_internals.jsonl"
DIM_FIELDS = {"theme": THEME_FIELDS, "action": ACTION_FIELDS, "budget": BUDGET_FIELDS}


def rank_of(true_cid, ordered):
    return ordered.index(true_cid) + 1 if true_cid in ordered else len(ordered) + 1


def order_by_score(cids, score_of):
    # rank desc by score; missing score -> -inf; ties broken by retriever order (cids order)
    idx = {c: i for i, c in enumerate(cids)}
    return sorted(cids, key=lambda c: (-(score_of(c) if score_of(c) is not None else -1e9), idx[c]))


def spearman(order_a, order_b):
    ra = {c: i for i, c in enumerate(order_a)}
    rb = {c: i for i, c in enumerate(order_b)}
    common = [c for c in order_a if c in rb]
    n = len(common)
    if n < 3:
        return None
    a = [ra[c] for c in common]; b = [rb[c] for c in common]
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else None


def metrics(ranks):
    n = len(ranks)
    return {"n": n,
            "R@1": round(sum(r <= 1 for r in ranks) / n, 3),
            "R@5": round(sum(r <= 5 for r in ranks) / n, 3),
            "MRR": round(sum(1.0 / r for r in ranks) / n, 3),
            "mean_rank": round(st.mean(ranks), 1)}


def main():
    profiles, id_map, label_to_cid = load_profiles()
    rows = [json.loads(l) for l in INTERNALS.read_text().splitlines() if l.strip()]
    print(f"internals rows: {len(rows)}\n")

    ranks = {k: [] for k in ["coordinator", "fusion", "theme", "action", "budget"]}
    echo_exact = 0
    corr_sorted, corr_fusion, corr_retr = [], [], []
    fifty = {d: {"total": 0, "eq50": 0, "eq50_field_present": 0, "eq50_field_missing": 0}
             for d in DIM_FIELDS}

    for r in rows:
        cids = r["cids"]                        # retriever order
        true = r["true_cid"]
        scores = r["scores"]
        coord = [c for c in r["coordinator_ranked_ids"] if c in set(cids)]
        for c in cids:                          # back-fill coordinator to full order
            if c not in coord:
                coord.append(c)

        def sc(dim):
            return lambda c: (scores.get(c) or {}).get(dim)
        def fusion_score(c):
            vals = [v for v in (scores.get(c) or {}).values() if isinstance(v, (int, float))]
            return st.mean(vals) if vals else None

        fusion_order = order_by_score(cids, fusion_score)
        ranks["coordinator"].append(rank_of(true, coord))
        ranks["fusion"].append(rank_of(true, fusion_order))
        for d in DIM_FIELDS:
            ranks[d].append(rank_of(true, order_by_score(cids, sc(d))))

        # coordinator-behaviour correlations
        sorted_order = sorted(cids)
        if coord[:len(sorted_order)] == sorted_order:
            echo_exact += 1
        for store, other in ((corr_sorted, sorted_order), (corr_fusion, fusion_order), (corr_retr, cids)):
            v = spearman(coord, other)
            if v is not None:
                store.append(v)

        # 50-defaults vs missing evidence
        for d, fields in DIM_FIELDS.items():
            for c in cids:
                v = (scores.get(c) or {}).get(d)
                if v is None:
                    continue
                fifty[d]["total"] += 1
                if abs(v - 50.0) < 1e-9:
                    fifty[d]["eq50"] += 1
                    has_field = any(profiles.get(c, {}).get(f) for f in fields)
                    fifty[d]["eq50_field_present" if has_field else "eq50_field_missing"] += 1

    print("=== ranking quality by scheme (true-call rank) ===")
    for k in ["coordinator", "fusion", "theme", "action", "budget"]:
        print(f"  {k:12s} {metrics(ranks[k])}")

    print("\n=== 1. coordinator behaviour ===")
    print(f"  exact echo of alphabetical-cid order: {echo_exact}/{len(rows)}")
    def mean(x): return round(st.mean(x), 3) if x else None
    print(f"  mean Spearman corr(coordinator, sorted-cid order):   {mean(corr_sorted)}")
    print(f"  mean Spearman corr(coordinator, fusion order):        {mean(corr_fusion)}")
    print(f"  mean Spearman corr(coordinator, retriever order):     {mean(corr_retr)}")

    print("\n=== 2. 50-point scores: missing vs neutral evidence ===")
    for d in DIM_FIELDS:
        f = fifty[d]
        if f["total"]:
            print(f"  {d:7s} scored={f['total']:5d}  ==50: {f['eq50']:5d} "
                  f"({f['eq50']/f['total']:.0%})  of which field-present {f['eq50_field_present']}, "
                  f"field-missing {f['eq50_field_missing']}")

    out = ARCH / "arch_error_analysis.json"
    out.write_text(json.dumps({
        "n": len(rows),
        "metrics": {k: metrics(ranks[k]) for k in ranks},
        "coordinator_echo_exact": echo_exact,
        "corr_sorted": mean(corr_sorted), "corr_fusion": mean(corr_fusion),
        "corr_retriever": mean(corr_retr),
        "fifty": fifty,
    }, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
