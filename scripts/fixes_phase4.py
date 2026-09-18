"""Phase 4: (1) recompute break rates with the correct per-system / per-backbone denominators;
(2) post-hoc power for the reference R@1 McNemar. NO API. Frozen per-query outputs only.
break rate = (# queries baseline-correct but system wrong) / (# baseline-correct).
"""
import json, math, statistics as st
from pathlib import Path
from scipy.stats import binom
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"analysis"

def load(p): return {str(x["id"]):x for x in json.load(open(ROOT/p))}
baseA=load("results/bright_hardpool/biology/baseline_per_query.json")
B=load("results/bright_concept_hardpool/biology/per_query.json")
C=load("results/bright_competing_hypotheses/biology/per_query.json")
D=load("results/bright_ch_anchor/biology/per_query.json")

def mcnemar(f,b):
    n=f+b; k=min(f,b); return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n)) if n else 1.0

# ---------- (1) reference-backbone break rates for B/C/D (denominator = baseline-correct) ----------
ids=list(baseA)
base_correct=[q for q in ids if baseA[q]["baseline_top1_gold"]]
den=len(base_correct)   # 52
def breakrate(sysd):
    broke=sum(1 for q in base_correct if not sysd[q]["arch_top1_gold"])
    fixed=sum(1 for q in ids if (not baseA[q]["baseline_top1_gold"]) and sysd[q]["arch_top1_gold"])
    return fixed, broke, broke/den
ref={"A":(0,0,0.0)}  # A is the baseline; break rate vs itself = 0 by construction
for name,sysd in [("B",B),("C",C),("D",D)]:
    ref[name]=breakrate(sysd)

# ---------- per-backbone A/D break rates (denominator differs per backbone) ----------
bb_rows=[]
# reference gemini-flash-latest from D file (has baseline_ + arch_)
bc=[q for q in ids if baseA[q]["baseline_top1_gold"]]
brokeD=sum(1 for q in bc if not D[q]["arch_top1_gold"])
bb_rows.append(("gemini-flash-latest", round(len(bc)/97,3), len(bc), sum(1 for q in ids if (not baseA[q]["baseline_top1_gold"]) and D[q]["arch_top1_gold"]), brokeD, brokeD/len(bc)))
for bb in ["gemini-2.5-flash-lite","claude-haiku-4-5","claude-sonnet-5"]:
    per=load(f"results/bright_backbones/{bb}/per_query.json")
    corr=[q for q in per if per[q]["A_top1_gold"]]
    brk=sum(1 for q in corr if not per[q]["D_top1_gold"])
    fx =sum(1 for q in per if (not per[q]["A_top1_gold"]) and per[q]["D_top1_gold"])
    bb_rows.append((bb, round(len(corr)/len(per),3), len(corr), fx, brk, brk/len(corr) if corr else 0.0))

# ---------- (2) post-hoc power for reference R@1 (10 fixed / 3 broke, 13 discordant) ----------
f_obs, b_obs = 10, 3; n_disc=13; ratio=f_obs/n_disc
# (a) min discordant pairs for p<0.05 at the observed ~10:3 ratio
min_disc=None
for n in range(1,200):
    f=round(n*ratio); b=n-f
    if mcnemar(f,b)<0.05: min_disc=(n,f,b,mcnemar(f,b)); break
# power machinery: n_queries queries, discordant prob psi, P(fixed|discordant)=pi
def power(psi, pi, nq=97, alpha=0.05):
    pw=0.0
    for d in range(0,nq+1):
        pd=binom.pmf(d,nq,psi)
        if pd<1e-15: continue
        rej=0.0
        for f in range(0,d+1):
            if mcnemar(f,d-f)<alpha: rej+=binom.pmf(f,d,pi)
        pw+=pd*rej
    return pw
psi_obs=n_disc/97; pi_obs=f_obs/n_disc
pw_obs=power(psi_obs, pi_obs)
# (b) min detectable pi (-> OR) at n=97, 80% power, holding discordant rate at observed
pi_star=None
p=0.55
while p<=0.999:
    if power(psi_obs,p)>=0.80: pi_star=p; break
    p+=0.005
OR_star = pi_star/(1-pi_star) if pi_star else None

# ---------- write ----------
L=["# Phase 4 — break-rate recompute + post-hoc power (no API)\n",
   "## 1. Break rates (denominator = number of baseline-correct queries)\n",
   "### 1a. Reference backbone, Systems B/C/D vs single-pass baseline A",
   f"Denominator = baseline-correct at reference = **{den}** queries (R@1 {den}/97).\n",
   "| System | fixed | broke | break rate among already-correct |",
   "|---|---|---|---|"]
for name in ["A","B","C","D"]:
    f,b,r=ref[name]
    lbl={"A":"A single-pass (baseline itself)","B":"B concept-guided","C":"C competing-hyp v1","D":"D anchor-and-edit"}[name]
    L.append(f"| {lbl} | {f} | {b} | {'0.0% (baseline vs itself)' if name=='A' else f'{r*100:.1f}%  ({b}/{den})'} |")
L.append("")
L.append("**Correction to the draft.** The draft's **7.1%** for System B is wrong. B broke "
         f"**{ref['B'][1]}** of {den} baseline-correct queries = **{ref['B'][2]*100:.1f}%** (the 7.1% "
         f"appears to use a wrong denominator of 28). Corrected: B = {ref['B'][2]*100:.1f}%, "
         f"C = {ref['C'][2]*100:.1f}%, D = {ref['D'][2]*100:.1f}%.\n")
L.append(f"**Consequence for the abstract.** With the corrected figures, **System B ({ref['B'][2]*100:.1f}%) "
         f"has a LOWER break rate than System D ({ref['D'][2]*100:.1f}%).** So the claim that "
         f"Anchor-and-Edit reduces damage to already-correct queries must be **scoped to the comparison "
         f"against regeneration (System C, {ref['C'][2]*100:.1f}%)**, not against all systems. The clean "
         f"statement: D breaks far fewer already-correct queries than C (regeneration), but not fewer "
         f"than B (concept-guided) — B is even more conservative, it just also fixes fewer.\n")
L.append("### 1b. Per-backbone A vs D break rate (denominator = that backbone's baseline-correct count)")
L.append("| Backbone | A R@1 | baseline-correct (denom) | fixed | broke | D break rate |")
L.append("|---|---|---|---|---|---|")
for bb,ar1,denk,fx,brk,rate in bb_rows:
    L.append(f"| {bb} | {ar1} | {denk} | {fx} | {brk} | {rate*100:.1f}%  ({brk}/{denk}) |")
L.append("")
L.append("**§7 conflict check.** §7 claims `claude-sonnet-5` has zero collateral breaks (11 fixed / 0 "
         f"broke) and that no-harm is best on the most capable model. Per-backbone recompute: "
         f"**sonnet broke = {bb_rows[3][4]} → break rate {bb_rows[3][5]*100:.1f}% — the §7 zero-breaks "
         f"claim is CONFIRMED.** Across the panel the break rate is "
         + ", ".join(f"{r[0].split('-')[0]}…={r[5]*100:.1f}%" for r in bb_rows) +
         ". Sonnet is indeed the lowest (0%), consistent with §7, but the relationship is **not "
         "strictly monotonic in capability** (haiku 7.7% exceeds the reference gemini-flash-latest "
         "5.8%), so phrase §7 as 'sonnet incurred zero breaks' rather than 'harm decreases "
         "monotonically with capability'.\n")

L.append("## 2. Post-hoc power for the reference R@1 McNemar (10 fixed / 3 broke, 13 discordant)\n")
L.append(f"- Observed exact two-sided McNemar p = **0.0923** (not < 0.05).")
L.append(f"- **Minimum discordant pairs for p<0.05 at the observed ~10:3 ratio:** n = **{min_disc[0]}** "
         f"(i.e. {min_disc[1]} fixed / {min_disc[2]} broke → p = {min_disc[3]:.4f}). The reference had 13, "
         f"so it was **{min_disc[0]-13} discordant pairs short** of significance at this ratio.")
L.append(f"- **Power of the reference test** (n=97, discordant rate {psi_obs:.3f}, fixed-share "
         f"{pi_obs:.3f}) = **{pw_obs*100:.1f}%** — i.e. the single-backbone R@1 test was underpowered "
         f"(well below 80%). This is the quantitative replacement for the word \"directional\".")
if pi_star:
    L.append(f"- **Minimum detectable effect at n=97 with 80% power**, holding the discordant rate at the "
             f"observed {psi_obs:.3f}: fixed-share $\\pi \\ge$ **{pi_star:.3f}** of discordant pairs, i.e. a "
             f"matched **odds ratio $\\ge$ {OR_star:.1f}** (vs the observed OR {pi_obs/(1-pi_obs):.2f}). At the "
             f"observed discordant rate, 97 queries can only reliably detect a substantially larger "
             f"per-query effect than the one present — which is why the pooled 4-backbone panel "
             f"(70 discordant) is the appropriate basis for the significance claim.")
(OUT/"fixes.md").write_text("\n".join(L))

# LaTeX
lat=["% Corrected break rates (reference) + per-backbone",
     "\\begin{tabular}{lccc}\n\\toprule\nSystem & fixed & broke & break rate (/%d) \\\\\n\\midrule"%den]
for name in ["B","C","D"]:
    f,b,r=ref[name]; lat.append(f"{name} & {f} & {b} & {r*100:.1f}\\%% \\\\")
lat.append("\\bottomrule\n\\end{tabular}")
(OUT/"fixes_tables.tex").write_text("\n".join(lat).replace("\\%%","\\%"))

print("Reference break rates (denom=%d): "%den, {k:(ref[k][1],round(ref[k][2]*100,1)) for k in ['B','C','D']})
print("Per-backbone A/D break rate:")
for bb,ar1,denk,fx,brk,rate in bb_rows: print(f"  {bb:<24} denom={denk:>2} fixed={fx:>2} broke={brk:>2} rate={rate*100:.1f}%")
print(f"min discordant for p<0.05 at 10:3 ratio = {min_disc}")
print(f"power of reference test = {pw_obs*100:.1f}% | min detectable OR at n=97,80% power = {OR_star:.2f} (pi={pi_star:.3f})")
print("wrote analysis/fixes.md + fixes_tables.tex")
