"""RQ3 headroom analysis rerun on the version-locked, temp=0 primary (claude-haiku-4-5-20251001),
across the 5 replicates. Same pre-registered model as bright_headroom_analysis.py
(flip ~ A_correct * C(split)), but reads the clean_abcd.json complete-case data and reports the
A_correct effect + the A_correct:split interaction as a DISTRIBUTION over the 5 reps -- so the
query-level headroom conclusion is replication-aware, not a single draw. Per-split headroom table
is averaged across reps.
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
import numpy as np
import pandas as pd
from bright_headroom_analysis import per_split_table, logistic, pooled_fisher, SPLITS

ROOT = Path(__file__).resolve().parents[1]
M = "claude-haiku-4-5-20251001"
BB = ROOT / "results" / "bright_backbones" / M
REPS = [1, 2, 3, 4, 5]


def load_rep(var, rep):
    rows = []
    for sp in SPLITS:
        f = BB / sp / f"rep_{rep}" / "clean_abcd.json"
        if not f.exists():
            continue
        for q in json.loads(f.read_text())["per_query"]:
            if not all(q["ok"].values()):          # complete-case only
                continue
            a = int(bool(q["top1"]["A"])); arch = int(bool(q["top1"][var]))
            rows.append({"split": sp, "A_correct": a, "arch": arch, "flip": int(arch != a)})
    return pd.DataFrame(rows)


def main():
    report = {"model": M, "temperature": 0, "reps": REPS, "variants": {}}
    for var in ["D", "B", "C"]:
        per_rep = []
        for rep in REPS:
            df = load_rep(var, rep)
            if df.empty:
                continue
            L = logistic(df, var); F = pooled_fisher(df)
            inter = L["interaction_A_correct_x_split"]
            per_rep.append({"rep": rep, "n": L["n"],
                            "A_correct_coef": L["A_correct_coef"], "A_correct_p": L["A_correct_p"],
                            "interaction_p": inter.get("p"), "interaction_chi2": inter.get("LR_chi2"),
                            "pooled_fisher_p": F["fisher_p"], "fix": F["fix"], "break": F["break"],
                            "fix_rate": F["fix_rate"], "break_rate": F["break_rate"]})
        if not per_rep:
            print(f"{var}: no data"); continue
        # per-split headroom table averaged across reps
        tabs = [per_split_table(load_rep(var, r)) for r in REPS if not load_rep(var, r).empty]
        avg = tabs[0].copy()
        for col in ["A_R@1", "headroom_1-R@1", "fix", "break", "net", "fix_rate", "break_rate"]:
            avg[col] = np.mean([pd.to_numeric(t[col], errors="coerce") for t in tabs], axis=0).round(3)

        def dist(key):
            xs = [r[key] for r in per_rep if r[key] is not None]
            return {"mean": round(st.mean(xs), 4), "sd": round(st.pstdev(xs) if len(xs) > 1 else 0, 4),
                    "min": round(min(xs), 4), "max": round(max(xs), 4), "values": [round(x, 4) for x in xs]}
        acoef = dist("A_correct_coef"); ap = dist("A_correct_p"); ip = dist("interaction_p")
        report["variants"][var] = {"per_rep": per_rep, "A_correct_coef_dist": acoef,
                                   "A_correct_p_dist": ap, "interaction_p_dist": ip,
                                   "per_split_avg": avg.to_dict("records")}
        print(f"\n===== Variant {var} (temp=0, {len(per_rep)} reps) =====")
        print("per-split (avg over reps):")
        print(avg.to_string(index=False))
        print(f"  A_correct coef: mean={acoef['mean']} sd={acoef['sd']} (headroom concentration)")
        print(f"  A_correct p:    {ap['values']}  #sig<0.05={sum(1 for v in ap['values'] if v<0.05)}/{len(ap['values'])}")
        print(f"  interaction A_correct×split p: {ip['values']}  #sig<0.05={sum(1 for v in ip['values'] if v<0.05)}/{len(ip['values'])}")

    out = BB / "headroom_haiku_temp0.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
