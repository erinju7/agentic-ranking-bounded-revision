"""Aggregate the k temp=0 replicates of the pinned-haiku clean A/B/C/D across the 4 BRIGHT splits.
Reports the DISTRIBUTION of the D-vs-A effect (and B, C) rather than a single p: per-rep pooled
fix/break/net/McNemar-p, plus mean+-sd and how many reps reach significance. This is the correctly-
configured (temp=0), version-locked primary with replicate spread. Reads rep_<k>/clean_abcd.json.
"""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = "claude-haiku-4-5-20251001"
BB = ROOT / "results" / "bright_backbones" / M
SPLITS = ["biology", "earth_science", "psychology", "sustainable_living"]


def mc(f, b):
    n = f + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(f, b) + 1)) / 2 ** n)


def rep_dirs(split):
    d = BB / split
    return sorted([p for p in d.glob("rep_*") if (p / "clean_abcd.json").exists()],
                  key=lambda p: int(p.name.split("_")[1]))


def main():
    # discover reps present in ALL splits
    reps_per_split = {s: {int(p.name.split("_")[1]) for p in rep_dirs(s)} for s in SPLITS}
    common = sorted(set.intersection(*reps_per_split.values())) if all(reps_per_split.values()) else []
    for s in SPLITS:
        print(f"  {s}: reps {sorted(reps_per_split[s])}")
    print(f"complete reps (present in all 4 splits): {common}\n")
    if not common:
        print("no complete reps yet"); return

    pooled = []
    for k in common:
        agg = {"A": 0, "B": 0, "C": 0, "D": 0}; N = 0; fb = {"B": [0, 0], "C": [0, 0], "D": [0, 0]}
        res_models = set()
        for s in SPLITS:
            rep = json.loads((BB / s / f"rep_{k}" / "clean_abcd.json").read_text())
            res_models.add(rep["report"].get("resolved_model"))
            cc = [x for x in rep["per_query"] if all(x["ok"].values())]
            N += len(cc)
            for x in cc:
                for a in "ABCD":
                    agg[a] += 1 if x["top1"][a] else 0
                for a in "BCD":
                    if not x["top1"]["A"] and x["top1"][a]: fb[a][0] += 1
                    if x["top1"]["A"] and not x["top1"][a]: fb[a][1] += 1
        row = {"rep": k, "N": N, "resolved": sorted(res_models),
               "A": agg["A"] / N, "B": agg["B"] / N, "C": agg["C"] / N, "D": agg["D"] / N}
        for a in "BCD":
            fx, bk = fb[a]; row[f"{a}_fix"] = fx; row[f"{a}_break"] = bk
            row[f"{a}_net"] = fx - bk; row[f"{a}_p"] = round(mc(fx, bk), 4)
        pooled.append(row)

    print(f"=== POOLED across 4 splits, per replicate (temp=0, {M}) ===")
    hdr = f"{'rep':>3} {'N':>4} {'A':>6} {'B':>6} {'C':>6} {'D':>6} | {'D_fix':>5} {'D_brk':>5} {'D_net':>5} {'D_p':>7}"
    print(hdr)
    for r in pooled:
        print(f"{r['rep']:>3} {r['N']:>4} {r['A']:>6.3f} {r['B']:>6.3f} {r['C']:>6.3f} {r['D']:>6.3f} | "
              f"{r['D_fix']:>5} {r['D_break']:>5} {r['D_net']:>+5} {r['D_p']:>7}")

    def dist(key):
        xs = [r[key] for r in pooled]
        return f"mean={st.mean(xs):.3f} sd={(st.pstdev(xs) if len(xs)>1 else 0):.3f} min={min(xs):.3f} max={max(xs):.3f}"
    print("\n-- distribution across reps --")
    for a in "ABCD":
        print(f"  {a} R@1: {dist(a)}")
    print(f"  D net (queries): {dist('D_net')}")
    ps = [r["D_p"] for r in pooled]
    print(f"  D McNemar p: values={ps}  #sig(<0.05)={sum(1 for p in ps if p < 0.05)}/{len(ps)}")
    resolved = sorted({m for r in pooled for m in r["resolved"]})
    print(f"\nresolved_model(s): {resolved}")

    (BB / "reps_pooled_summary.json").write_text(json.dumps(
        {"model": M, "temperature": 0, "reps": pooled,
         "resolved_model": resolved}, indent=1))
    print(f"wrote {BB/'reps_pooled_summary.json'}")


if __name__ == "__main__":
    main()
