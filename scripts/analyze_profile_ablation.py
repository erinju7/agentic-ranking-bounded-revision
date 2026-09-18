"""Full vs Compact candidate-profile ablation: per-query analysis + exports.

Reproducible analysis behind the Level-0 profile decision. Compact is kept as the
fixed default (near-equivalent overall ranking at ~76% fewer candidate tokens);
this script documents WHERE the two representations diverge and WHY, and exports
the supporting tables/ids for the dissertation discussion.

Outputs (results/rq2_architecture/level0_profile_ablation/):
  full_vs_compact_per_query.csv   per-query ranks, deltas, strata, flags
  transition_matrix.csv           rank-bucket transitions full -> compact
  query_id_sets.json              top1_gains / recall5_losses / recall10_losses
  temporal_sibling_failures.csv   the sibling-collapse deep-recall losses
  summary.json                    headline metrics + mechanism counts
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "frozen" / "rq2_v1_seed42_description_filtered"
CUR = D / "curated_evidence_subset"
RES = ROOT / "results" / "rq2_architecture"
ABL = RES / "level0_enriched_reranker"
OUT = RES / "level0_profile_ablation"
MODEL = "claude-haiku-4-5"  # the fixed model for the profile ablation


def load_rankings(mode):
    return json.loads((ABL / f"{MODEL}__{mode}" / "curated_rankings.json").read_text())


def load_truths():
    t = {}
    for f in ("dev_queries.jsonl", "test_queries.jsonl"):
        for line in (D / f).open():
            o = json.loads(line)
            t[str(o["query_id"])] = o["true_call_label"]
    return t


def load_top50():
    top = {}
    for f in ("dev_historical_top50_candidates.jsonl", "test_historical_top50_candidates.jsonl"):
        for line in (RES / "deterministic_topn_retriever" / f).open():
            o = json.loads(line)
            top[str(o["query_id"])] = [c["call_label"] for c in o["candidates"]]
    return top


def main():
    FR, CR = load_rankings("full"), load_rankings("compact")
    truth = load_truths()
    top50 = load_top50()
    pool = {c["call_label"]: c for c in json.loads((D / "candidate_pool.json").read_text())["candidates"]}
    profiles = json.loads((CUR / "candidate_profiles.json").read_text())
    id_map = json.loads((CUR / "candidate_id_map.json").read_text())
    lab2id = {v: k for k, v in id_map.items()}
    OUT.mkdir(parents=True, exist_ok=True)

    def rank_of(R, lbl):
        return next((r["rank"] for r in R if r["call_label"] == lbl), 51)

    def family(lbl):
        s = (pool.get(lbl, {}).get("dominant_fundingScheme") or "").upper()
        for p in ("ERC", "MSCA", "RIA", "IA", "CSA", "SME", "BBI", "IMI2", "FCH2", "ERA-NET"):
            if s.startswith(p):
                return p
        return s or "OTHER"

    def nproj(lbl):
        return pool.get(lbl, {}).get("n_projects") or 0

    def size_band(lbl):
        n = nproj(lbl)
        return "small" if n <= 25 else "large" if n >= 150 else "med"

    def compact_sig(lbl):
        p = profiles.get(lab2id.get(lbl, ""), {})
        return (p.get("title"), p.get("action_type"), p.get("call_scale"), p.get("submission_stage"))

    def sig_collisions(q, lbl):
        s = compact_sig(lbl)
        return sum(1 for c in top50.get(q, []) if compact_sig(c) == s)

    def sibling_count(q, lbl):
        stem = re.sub(r"(19|20)\d{2}", "Y", lbl)
        return sum(1 for c in top50.get(q, []) if c != lbl and re.sub(r"(19|20)\d{2}", "Y", c) == stem)

    def bucket(r):
        return "1" if r == 1 else "2-5" if r <= 5 else "6-10" if r <= 10 else ">10"

    rows = []
    for q in FR:
        if q not in CR:
            continue
        t = truth[q]
        rf, rc = rank_of(FR[q], t), rank_of(CR[q], t)
        rows.append({
            "query_id": q, "true_call": t, "family": family(t),
            "size_band": size_band(t), "n_projects": nproj(t),
            "rank_full": rf, "rank_compact": rc, "delta_compact_minus_full": rc - rf,
            "bucket_full": bucket(rf), "bucket_compact": bucket(rc),
            "compact_sig_collisions": sig_collisions(q, t),
            "temporal_siblings_in_top50": sibling_count(q, t),
            "top1_gain": int(rc == 1 and rf > 1),
            "recall5_loss": int(rf <= 5 and rc > 5),
            "recall10_loss": int(rf <= 10 and rc > 10),
            "catastrophic_1_to_tail": int(rf == 1 and rc > 10),
        })

    # per-query CSV
    cols = list(rows[0].keys())
    with (OUT / "full_vs_compact_per_query.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    # transition matrix
    order = ["1", "2-5", "6-10", ">10"]
    tm = collections.Counter((r["bucket_full"], r["bucket_compact"]) for r in rows)
    with (OUT / "transition_matrix.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["full\\compact"] + order)
        for a in order:
            w.writerow([a] + [tm.get((a, b), 0) for b in order])

    # id sets
    id_sets = {
        "top1_gains": [r["query_id"] for r in rows if r["top1_gain"]],
        "recall5_losses": [r["query_id"] for r in rows if r["recall5_loss"]],
        "recall10_losses": [r["query_id"] for r in rows if r["recall10_loss"]],
        "catastrophic_1_to_tail": [r["query_id"] for r in rows if r["catastrophic_1_to_tail"]],
    }
    (OUT / "query_id_sets.json").write_text(json.dumps(id_sets, indent=1))

    # temporal-sibling failures: recall losses whose true call has a sibling in top-50
    sib = [r for r in rows if (r["recall5_loss"] or r["recall10_loss"]) and r["temporal_siblings_in_top50"] > 0]
    with (OUT / "temporal_sibling_failures.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(sib)

    gains = [r for r in rows if r["top1_gain"]]
    loss5 = [r for r in rows if r["recall5_loss"]]
    summary = {
        "model": MODEL, "n_queries": len(rows),
        "unchanged_bucket": sum(1 for r in rows if r["bucket_full"] == r["bucket_compact"]),
        "top1_gains": len(gains), "recall5_losses": len(loss5),
        "recall10_losses": sum(r["recall10_loss"] for r in rows),
        "catastrophic_1_to_tail": sum(r["catastrophic_1_to_tail"] for r in rows),
        "median_nproj_gains": statistics.median([r["n_projects"] for r in gains]),
        "median_nproj_recall5_losses": statistics.median([r["n_projects"] for r in loss5]),
        "recall5_losses_with_temporal_sibling": sum(1 for r in loss5 if r["temporal_siblings_in_top50"] > 0),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nwrote 5 files to {OUT}")


if __name__ == "__main__":
    main()
