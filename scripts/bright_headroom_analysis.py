"""RQ3 headroom analysis (pre-registered 2026-08-28). Pooled query-level primary + 4-point
descriptive supplement. Study 2 is NOT included (independent qualitative point).

PRIMARY (per variant): pooled logistic  flip ~ A_correct * split.
  flip = 1 iff the variant changes top-1 correctness (A-wrong->right = fix; A-right->wrong = break).
  - A_correct main-effect coef + p  -> query-level headroom concentration.
  - A_correct : split interaction (LR test) -> does the concentration vary with task difficulty
    (the direct task-level headroom test). n supports it; the 4-point regression does not.
DESCRIPTIVE: per-split A R@1, fix, break, net, fix_rate, break_rate; monotonicity only, no p.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SPLITS = ["biology", "earth_science", "psychology", "sustainable_living"]
VAR_DIR = {"B": "bright_concept_hardpool", "C": "bright_competing_hypotheses", "D": "bright_ch_anchor"}


def load_variant(var):
    rows = []
    for sp in SPLITS:
        p = RES / VAR_DIR[var] / sp / "per_query.json"
        if not p.exists():
            continue
        for q in json.loads(p.read_text()):
            a = int(bool(q["baseline_top1_gold"])); arch = int(bool(q["arch_top1_gold"]))
            rows.append({"split": sp, "A_correct": a, "arch": arch,
                         "flip": int(arch != a)})
    return pd.DataFrame(rows)


def per_split_table(df):
    out = []
    for sp in SPLITS:
        d = df[df.split == sp]
        if d.empty:
            continue
        aw, ar = d[d.A_correct == 0], d[d.A_correct == 1]
        fix = int((aw.arch == 1).sum()); brk = int((ar.arch == 0).sum())
        out.append({"split": sp, "n": len(d), "A_R@1": round(d.A_correct.mean(), 3),
                    "headroom_1-R@1": round(1 - d.A_correct.mean(), 3),
                    "fix": fix, "break": brk, "net": fix - brk,
                    "fix_rate": round(fix / len(aw), 3) if len(aw) else None,
                    "break_rate": round(brk / len(ar), 3) if len(ar) else None})
    return pd.DataFrame(out)


def logistic(df, var):
    # main-effect model and interaction model; LR test on interaction
    m0 = smf.logit("flip ~ A_correct + C(split)", df).fit(disp=0)
    try:
        m1 = smf.logit("flip ~ A_correct * C(split)", df).fit(disp=0)
        lr = 2 * (m1.llf - m0.llf); dfree = int(m1.df_model - m0.df_model)
        from scipy.stats import chi2
        lr_p = float(chi2.sf(lr, dfree))
        interaction = {"LR_chi2": round(lr, 3), "df": dfree, "p": round(lr_p, 4)}
    except Exception as e:
        interaction = {"error": str(e)[:80]}
    b = m0.params.get("A_correct"); p = m0.pvalues.get("A_correct")
    return {"A_correct_coef": round(float(b), 3), "A_correct_p": round(float(p), 4),
            "interaction_A_correct_x_split": interaction, "n": int(len(df))}


def pooled_fisher(df):
    # 2x2: A_correct(0/1) x flip(0/1); tests concentration of flips in A-wrong
    a = int(((df.A_correct == 0) & (df.flip == 1)).sum())   # fix
    b = int(((df.A_correct == 0) & (df.flip == 0)).sum())
    c = int(((df.A_correct == 1) & (df.flip == 1)).sum())   # break
    d = int(((df.A_correct == 1) & (df.flip == 0)).sum())
    _, p = fisher_exact([[a, b], [c, d]])
    fixr = a / (a + b) if (a + b) else 0; brkr = c / (c + d) if (c + d) else 0
    return {"fix": a, "break": c, "fix_rate": round(fixr, 3), "break_rate": round(brkr, 3),
            "fisher_p": round(float(p), 4)}


def main():
    report = {"splits_present": [], "variants": {}}
    for var in ["D", "B", "C"]:
        df = load_variant(var)
        if df.empty:
            print(f"{var}: no data"); continue
        report["splits_present"] = sorted(df.split.unique().tolist())
        tab = per_split_table(df)
        fr = pd.to_numeric(tab["fix_rate"], errors="coerce")
        hr = pd.to_numeric(tab["headroom_1-R@1"], errors="coerce")
        # monotonicity: does fix_rate rise with headroom? sign of rank correlation (descriptive)
        order = tab.sort_values("headroom_1-R@1")
        mono = "monotone↑" if order["fix_rate"].is_monotonic_increasing else \
               ("monotone↓" if order["fix_rate"].is_monotonic_decreasing else "non-monotone")
        report["variants"][var] = {
            "pooled_logistic": logistic(df, var),
            "pooled_2x2_fisher": pooled_fisher(df),
            "per_split": tab.to_dict("records"),
            "fix_rate_vs_headroom_monotonicity": mono,
            "slope_fix_rate_on_headroom": round(float(np.polyfit(hr, fr, 1)[0]), 3) if fr.notna().all() else None,
        }
        print(f"\n===== Variant {var} (pooled n={len(df)}) =====")
        print(tab.to_string(index=False))
        L = report["variants"][var]["pooled_logistic"]; F = report["variants"][var]["pooled_2x2_fisher"]
        print(f"  pooled Fisher: fix_rate={F['fix_rate']} break_rate={F['break_rate']} p={F['fisher_p']}")
        print(f"  logistic A_correct coef={L['A_correct_coef']} p={L['A_correct_p']} | "
              f"interaction {L['interaction_A_correct_x_split']}")
        print(f"  fix_rate vs headroom: {mono} (slope={report['variants'][var]['slope_fix_rate_on_headroom']})")

    out = RES / "bright_headroom" / "headroom_analysis.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
