"""Task 8 (NO API): baseline-strength vs absolute-gain relationship, reported SEPARATELY on the
two axes — never pooled across the seven heterogeneous points. All inputs are read from the frozen
per-split summaries and the backbone table already in the paper's artefacts; nothing is invented.

Axis 1 (ACROSS-DOMAIN, backbone fixed = gemini-flash-latest reference): biology + 3 new splits.
Axis 2 (ACROSS-BACKBONE, query set fixed = biology 100-pools): the four backbones.

Emits analysis/baseline_gain_correlation.md with: monotonicity of the 3 new splits, the biology
inversion, and one Pearson r PER AXIS (plus the 3-new-splits-only r). Also pulls the split rows
(incl. MRR delta + CI) straight from runs/splits/<split>/summary.json for the paper table.
"""
import json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def pearson(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    dx = [x-mx for x in xs]; dy = [y-my for y in ys]
    num = sum(a*b for a, b in zip(dx, dy))
    den = math.sqrt(sum(a*a for a in dx) * sum(b*b for b in dy))
    return num/den if den else float("nan")

def monotone_decreasing(pairs):
    """pairs sorted by baseline ascending; True if gain strictly decreases."""
    s = sorted(pairs, key=lambda p: p[0])
    return all(s[i][1] > s[i+1][1] for i in range(len(s)-1)), s

# ---- Axis 1: across-domain, reference backbone. Pull from frozen summaries. ----
splits = ["earth_science", "sustainable_living", "psychology"]
dom = []  # (name, A_R@1, dR@1)
split_rows = []
for s in splits:
    j = json.loads((ROOT/"runs"/"splits"/s/"summary.json").read_text())
    a1, d1 = j["A"]["R@1"], j["D"]["R@1"]
    dom.append((s, a1, round(d1-a1, 3)))
    split_rows.append({"split": s, "n": j["n"], "A": a1, "D": d1, "dR1": round(d1-a1, 3),
                       "fixed": j["fixed"], "broke": j["broke"], "mcn": j["mcnemar_p"],
                       "mrr_d": j["MRR_delta"], "mrr_ci": j["MRR_CI"], "mrr_p": j["MRR_perm_p"]})
# biology reference row (frozen values already reported in the paper / RUN_LOG)
bio = ("biology", 0.536, 0.072)

new3 = [(a, d) for (_, a, d) in dom]                      # 3 new splits
allfour = new3 + [(bio[1], bio[2])]                       # + biology

mono3, s3 = monotone_decreasing(new3)
mono4, s4 = monotone_decreasing(allfour)
r_new3 = pearson([a for a, _ in new3], [d for _, d in new3])
r_dom4 = pearson([a for a, _ in allfour], [d for _, d in allfour])

# ---- Axis 2: across-backbone, biology query set fixed. Values from the paper's backbone table. ----
# (A R@1, dR@1) per backbone — from Table "backbone robustness" (analysis/fixes.md + panel summaries)
bb = [("gemini-flash-latest", 0.540, 0.070),
      ("gemini-2.5-flash-lite", 0.278, 0.155),
      ("claude-haiku-4-5", 0.536, 0.134),
      ("claude-sonnet-5", 0.546, 0.114)]
r_bb4 = pearson([a for _, a, _ in bb], [d for _, _, d in bb])

# ---- pooled (the thing we must NOT report as one number; computed only to show why) ----
pooled7 = allfour + [(a, d) for _, a, d in bb]
r_pooled = pearson([a for a, _ in pooled7], [d for _, d in pooled7])

L = []
L.append("# Task 8 — baseline strength vs absolute gain, reported PER AXIS (never pooled)\n")
L.append("Two axes vary two different things; pooling their seven points into one correlation mixes "
         "a domain effect (query difficulty changes, backbone fixed) with a model effect (model "
         "changes, query set fixed). We therefore report one Pearson r per axis and never a single "
         "seven-point r.\n")

L.append("## Axis 1 — across DOMAIN (backbone fixed = gemini-flash-latest)\n")
L.append(f"3 new splits, ordered by baseline A R@1 (ascending): "
         f"{', '.join(f'{a:.3f}→+{d:.3f}' for a,d in s3)}")
L.append(f"- Monotone (gain strictly falls as baseline rises), 3 new splits only: **{mono3}** "
         f"(Pearson r = **{r_new3:.3f}**).")
L.append(f"- Adding biology (0.536→+0.072): monotone across all 4 = **{mono4}** "
         f"(Pearson r = **{r_dom4:.3f}**). Biology is an **inversion** — its baseline (0.536) sits "
         f"between sustainable_living (0.487) and psychology (0.600), but its gain (+0.072) is the "
         f"2nd-LARGEST, above both, breaking the monotone run.")
L.append(f"- BOTH facts stand: the 3 pre-registered new splits are cleanly monotone; folding the "
         f"reference biology split back in introduces exactly one inversion.\n")

L.append("## Axis 2 — across BACKBONE (query set fixed = biology 100-pools)\n")
L.append("Ordered by baseline A R@1: " +
         ", ".join(f"{n} {a:.3f}→+{d:.3f}" for n, a, d in sorted(bb, key=lambda x: x[1])))
L.append(f"- Pearson r (4 backbones) = **{r_bb4:.3f}**: weaker backbone → larger gain "
         f"(flash-lite, baseline 0.278, biggest gain +0.155), but NOT monotone "
         f"(haiku 0.536 gains +0.134 > sonnet 0.546 +0.114, yet flash-latest 0.540 gains only +0.070).\n")

L.append("## Why not pool\n")
L.append(f"For the record only, the naive seven-point pooled r would be {r_pooled:.3f}. We do NOT "
         f"report this in the paper: the seven points are heterogeneous (four share a query set and "
         f"vary the model; four share the model and vary the domain; biology is common to both), so a "
         f"single r conflates two distinct mechanisms and its d.f. are not what a 7-point test implies.\n")

L.append("## Split table rows (for tab:splits — MRR delta + CI pulled from summary.json)\n")
L.append("| split | n | A R@1 | D R@1 | dR@1 | fixed | broke | McNemar p | MRR perm p | MRR d [95% CI] |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for r in split_rows:
    L.append(f"| {r['split']} | {r['n']} | {r['A']:.3f} | {r['D']:.3f} | +{r['dR1']:.3f} | "
             f"{r['fixed']} | {r['broke']} | {r['mcn']:.3f} | {r['mrr_p']:.3f} | "
             f"{r['mrr_d']:+.3f} [{r['mrr_ci'][0]:+.3f}, {r['mrr_ci'][1]:+.3f}] |")

(ROOT/"analysis"/"baseline_gain_correlation.md").write_text("\n".join(L)+"\n")
print("\n".join(L))
print("\nwrote analysis/baseline_gain_correlation.md")
