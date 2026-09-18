"""Aggregate Study 2 listwise scores across seeds (mean +/- sd) vs James gold.

Reads seed dirs: results/james_validation/listwise/ (seed 42),
listwise/seed_7/, listwise/seed_123/. Reuses score_variant from score_study2_listwise.
Reports mean +/- sd per variant per metric so the tie / B-below-A pattern is not a
single-shuffle artefact.
"""
from __future__ import annotations
import json, os, statistics as st
from pathlib import Path
from score_study2_listwise import score_variant, load_gold, pairwise_kappa

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "james_validation" / "listwise" / os.environ.get("LW_DIR", "claude-haiku-4-5-20251001")
SEEDS = [42] + list(range(43, 52))   # 10 independent replicates (default seed 42 + 43..51)
VARIANTS = ["A", "B", "C", "D_cap1", "D_cap2"]
METRICS = ["nDCG@1", "nDCG@3", "nDCG@6", "Recall@1", "MRR", "per_call_weighted_kappa"]


def seed_dir(s):
    return BASE if s == 42 else BASE / f"seed_{s}"


def main():
    gold = load_gold()
    # per-seed per-variant scores
    per_seed = {}
    for s in SEEDS:
        f = seed_dir(s) / "listwise_results.json"
        if not f.exists():
            print(f"  (missing seed {s}: {f}) — skipping")
            continue
        results = json.loads(f.read_text())
        per_seed[s] = {v: score_variant(results, v, gold) for v in VARIANTS}
    seeds = sorted(per_seed)
    print(f"Study 2 listwise vs James — {len(seeds)} seeds {seeds} (nDCG n=5, rank n=4)\n")

    agg = {v: {} for v in VARIANTS}
    hdr = f"{'variant':9s}" + "".join(f"{m:>16s}" for m in METRICS)
    print(hdr)
    for v in VARIANTS:
        cells = []
        for m in METRICS:
            vals = [per_seed[s][v][m] for s in seeds if per_seed[s][v][m] is not None]
            if not vals:
                agg[v][m] = None; cells.append(f"{'-':>16s}"); continue
            mu = st.mean(vals); sd = st.pstdev(vals) if len(vals) > 1 else 0.0
            agg[v][m] = {"mean": round(mu, 3), "sd": round(sd, 3), "n": len(vals)}
            cells.append(f"{mu:.2f}±{sd:.2f}".rjust(16))
        print(f"{v:9s}" + "".join(cells))

    # Mito FP consistency across seeds
    print("\nMito top-1 false-positive (per seed, per variant):")
    for v in VARIANTS:
        row = []
        for s in seeds:
            m = per_seed[s][v]["mito_top1_fp"]
            row.append(f"{s}:{'FP' if (m and m['false_positive']) else 'ok'}")
        print(f"  {v:9s} " + "  ".join(row))

    out = BASE / "scores_vs_james_multiseed.json"
    out.write_text(json.dumps({"seeds": seeds, "aggregate": agg,
                               "per_seed": per_seed, "pairwise_kappa": pairwise_kappa(gold)}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
