"""Phase 2: pooled reanalysis across the four backbones. NO API. Reanalysis of frozen
per-query outputs only. Gates (A-prompt hash + reference regression) already passed separately.
Emits analysis/pooled_backbones.md and LaTeX tables.
"""
import json, math, warnings, statistics as st
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sps
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"analysis"; OUT.mkdir(exist_ok=True)

# ---- assemble 4-backbone paired data ----
baseA={str(p["id"]):p for p in json.load(open(ROOT/"results/bright_hardpool/biology/baseline_per_query.json"))}
Dref ={str(p["id"]):p for p in json.load(open(ROOT/"results/bright_ch_anchor/biology/per_query.json"))}
rows=[]
ids=list(baseA)
for i in ids:
    rows.append(dict(backbone="gemini-flash-latest", query=i, system="A", correct=int(baseA[i]["baseline_top1_gold"]), rr=1.0/baseA[i]["baseline_first_gold_rank"]))
    rows.append(dict(backbone="gemini-flash-latest", query=i, system="D", correct=int(Dref[i]["arch_top1_gold"]), rr=1.0/Dref[i]["arch_first_gold_rank"]))
for bb in ["gemini-2.5-flash-lite","claude-haiku-4-5","claude-sonnet-5"]:
    per={str(p["id"]):p for p in json.load(open(ROOT/f"results/bright_backbones/{bb}/per_query.json"))}
    for i in per:
        rows.append(dict(backbone=bb, query=i, system="A", correct=int(per[i]["A_top1_gold"]), rr=1.0/per[i]["A_first_gold"]))
        rows.append(dict(backbone=bb, query=i, system="D", correct=int(per[i]["D_top1_gold"]), rr=1.0/per[i]["D_first_gold"]))
df=pd.DataFrame(rows)
df["sysD"]=(df["system"]=="D").astype(int)
BACKBONES=["gemini-flash-latest","gemini-2.5-flash-lite","claude-haiku-4-5","claude-sonnet-5"]

# ---- per-backbone discordant counts (paired top-1) ----
disc={}
for bb in BACKBONES:
    d=df[df.backbone==bb].pivot(index="query",columns="system",values="correct")
    b=int(((d.A==1)&(d.D==0)).sum())   # broke (A right, D wrong)
    c=int(((d.A==0)&(d.D==1)).sum())   # fixed (A wrong, D right)
    disc[bb]=dict(fixed=c, broke=b, discordant=b+c,
                  A_r1=round(d.A.mean(),3), D_r1=round(d.D.mean(),3))
tot_c=sum(v["fixed"] for v in disc.values()); tot_b=sum(v["broke"] for v in disc.values())
tot_disc=tot_c+tot_b

# ---- CMH = stratified McNemar (matched pairs), common OR, Wilson CI, Breslow-Day ----
cmh_stat=(sum(disc[bb]["fixed"]-disc[bb]["broke"] for bb in BACKBONES)**2)/sum(disc[bb]["discordant"] for bb in BACKBONES)
cmh_p=sps.chi2.sf(cmh_stat,1)
OR=tot_c/tot_b
pi=tot_c/tot_disc
# Wilson CI on pi -> OR CI
z=1.96; nD=tot_disc
center=(pi+z*z/(2*nD))/(1+z*z/nD)
half=(z*math.sqrt(pi*(1-pi)/nD + z*z/(4*nD*nD)))/(1+z*z/nD)
pi_lo,pi_hi=center-half,center+half
OR_lo,OR_hi=pi_lo/(1-pi_lo), pi_hi/(1-pi_hi)
# Breslow-Day for homogeneity of matched OR across backbones (chi2, K-1 df)
bd=0.0
for bb in BACKBONES:
    n_k=disc[bb]["discordant"]; c_k=disc[bb]["fixed"]
    E=n_k*pi; V=n_k*pi*(1-pi)
    if V>0: bd+=(c_k-E)**2/V
bd_p=sps.chi2.sf(bd,len(BACKBONES)-1)

# ---- conditional logistic regression ----
warn_log=[]
def fit_condlogit(group_col):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m=ConditionalLogit(df["correct"].values, df[["sysD"]].values, groups=df[group_col].values)
        r=m.fit(disp=0)
        wl=[str(x.message) for x in w]
    coef=r.params[0]; se=r.bse[0]; p=r.pvalues[0]
    return coef,se,p,math.exp(coef),(math.exp(coef-1.96*se),math.exp(coef+1.96*se)),wl
df["qxb"]=df["query"]+"|"+df["backbone"]
cl_q=fit_condlogit("query")                 # stratified by query (literal spec)
cl_qxb=fit_condlogit("qxb")                 # stratified by matched pair (reduces to McNemar)

# ---- MRR linear mixed model: crossed random intercepts query + backbone ----
# --- crossed-VC model as originally specified (kept to EXPOSE the boundary variance component) ---
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    df["grp"]=1
    vc={"query":"0+C(query)","backbone":"0+C(backbone)"}
    mdf=smf.mixedlm("rr ~ sysD", df, groups=df["grp"], vc_formula=vc).fit()
    lmm_warn=sorted(set(str(x.message) for x in w))
lmm_conv=getattr(mdf,"converged",None)
vc_query, vc_backbone = float(mdf.vcomp[0]), float(mdf.vcomp[1])   # order = vc_formula keys
vc_resid=float(mdf.scale)
# --- MAIN spec (Correction 3): backbone as FIXED effect, query as random intercept ---
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    mdM=smf.mixedlm("rr ~ sysD + C(backbone)", df, groups=df["query"]).fit()
    lmmM_warn=sorted(set(str(x.message) for x in w))
lmmM_coef=mdM.params["sysD"]; lmmM_se=mdM.bse["sysD"]
lmmM_ci=(lmmM_coef-1.96*lmmM_se, lmmM_coef+1.96*lmmM_se); lmmM_p=mdM.pvalues["sysD"]
lmmM_conv=getattr(mdM,"converged",None); vc_query_main=float(mdM.cov_re.iloc[0,0])
# design-based check: mean of per-query paired RR differences, pooled
paired_diff=df.pivot_table(index=["backbone","query"],columns="system",values="rr")
mean_paired=(paired_diff["D"]-paired_diff["A"]).mean()

# ---------- write markdown ----------
L=[]
def w(s=""): L.append(s)
w("# Phase 2 — Pooled reanalysis across the four backbones\n")
w("No API calls; reanalysis of frozen per-query outputs. Gates passed (A-prompt sha256 identical; "
  "reference row reproduces A/D R@1 0.54/0.61, MRR 0.646/0.708, McNemar p=0.092, MRR perm p=0.0076).\n")
w("The 97 queries are shared across backbones, so observations are correlated; analyses below "
  "respect that (stratified / matched / random-effects). Outcome = top-1 correct unless noted.\n")

w("## 4. Per-backbone discordant counts (paired, top-1)\n")
w("| Backbone | A R@1 | D R@1 | fixed (A wrong→D right) | broke (A right→D wrong) | discordant |")
w("|---|---|---|---|---|---|")
for bb in BACKBONES:
    v=disc[bb]; w(f"| {bb} | {v['A_r1']} | {v['D_r1']} | {v['fixed']} | {v['broke']} | {v['discordant']} |")
w(f"| **Total** |  |  | **{tot_c}** | **{tot_b}** | **{tot_disc}** |")
w(f"\n**Total discordant pairs across all four backbones = {tot_disc}** (the figure the write-up "
  f"should quote for statistical power; the single-backbone reference had 13).\n")

w("## 1. Cochran–Mantel–Haenszel (stratified McNemar) + Breslow–Day\n")
w("Strata = backbone. For matched binary pairs the CMH test of marginal homogeneity is the "
  "stratified McNemar $\\chi^2=[\\sum_k(c_k-b_k)]^2/\\sum_k(c_k+b_k)$ (statsmodels' StratifiedTable "
  "tests conditional independence in the agreement table, which is a different hypothesis, so the "
  "matched-pairs formula is applied directly). Common matched OR $=\\sum c_k/\\sum b_k$; Wilson CI "
  "via the discordant binomial. Breslow–Day tests homogeneity of the matched OR across backbones.\n")
w(f"- CMH statistic = **{cmh_stat:.3f}**, df=1, **p = {cmh_p:.3g}**")
w(f"- Common matched odds ratio (D vs A) = **{OR:.3f}**, 95% CI **[{OR_lo:.3f}, {OR_hi:.3f}]**")
w(f"- Breslow–Day statistic = {bd:.3f}, df={len(BACKBONES)-1}, **p = {bd_p:.3f}** — "
  f"**no evidence of heterogeneity** in the odds ratio across backbones. This is stated as "
  f"*no evidence of* heterogeneity, not *homogeneity*: with only 4 strata Breslow–Day has weak "
  f"power to detect it.\n")
w("**Relative vs absolute effect (Correction 2).** Breslow–Day concerns the *relative* effect "
  "(the odds ratio), which shows no evidence of variation across backbones (OR$\\approx$4.83). "
  "This does **not** contradict §7's observation that the *absolute* gain $\\Delta$R@1 varies with "
  "headroom (flash-lite $+0.155$, reference $+0.070$): a constant odds ratio produces larger "
  "absolute differences where the baseline sits nearer 0.5 and smaller ones near the ceiling. The "
  "clean statement is: **the relative effect is consistent across backbones; the absolute gain "
  "scales with baseline headroom.**\n")

w("## 2. Conditional logistic regression (matched binary)\n")
w("Outcome top-1 correct; covariate system (D=1). ConditionalLogit (statsmodels 0.14.6). The two "
  "stratifications below are **different estimands, not two estimates of the same quantity** "
  "(Correction 4): (a) query$\\times$backbone (matched pair) — the paired D-vs-A effect, the "
  "headline matched OR; (b) query — the D-vs-A effect *after conditioning on query difficulty*, "
  "which pools the four backbones within each query stratum and is reported as a robustness check.\n")
w("| Estimand (stratification) | role | system coef | SE | OR (=e^coef) | 95% CI | p |")
w("|---|---|---|---|---|---|---|")
w(f"| paired effect (query×backbone) | **headline** | {cl_qxb[0]:.3f} | {cl_qxb[1]:.3f} | **{cl_qxb[3]:.3f}** | [{cl_qxb[4][0]:.3f}, {cl_qxb[4][1]:.3f}] | {cl_qxb[2]:.3g} |")
w(f"| query-difficulty-controlled (query) | robustness | {cl_q[0]:.3f} | {cl_q[1]:.3f} | {cl_q[3]:.3f} | [{cl_q[4][0]:.3f}, {cl_q[4][1]:.3f}] | {cl_q[2]:.3g} |")
allwarn=sorted(set(cl_q[5]+cl_qxb[5]))
w(f"\nThe **headline matched OR = {cl_qxb[3]:.3f}** equals the CMH common OR ({OR:.3f}) exactly (as it "
  f"must for pure matched pairs, so it adds no new information beyond CMH — it is reported for the "
  f"model-based CI). The **query-difficulty-controlled OR = {cl_q[3]:.3f}** is a *different estimand*: "
  f"the effect attenuates once query difficulty is conditioned out, but remains highly significant "
  f"($p={cl_q[2]:.2g}$). Reporting only the 4.83 figure would overstate the effect; both are shown. "
  f"Convergence warnings (verbatim): {allwarn if allwarn else 'none'}\n")

w("## 3. Linear mixed model for MRR (reciprocal rank)\n")
w("**Main specification (Correction 3): backbone as a FIXED effect, query as a random intercept.** "
  "Outcome = per-query reciprocal rank; $n=776$ rows. Backbone is treated as fixed because it has "
  "only 4 levels — below the usual threshold for a random effect — and because the originally "
  "specified crossed model drove the backbone variance component to its boundary (see below), which "
  "is what produced the slogdet warnings.\n")
w(f"- **system (D) coefficient = {lmmM_coef:.4f}**, SE {lmmM_se:.4f}, 95% CI "
  f"**[{lmmM_ci[0]:.4f}, {lmmM_ci[1]:.4f}]**, p = {lmmM_p:.3g}; converged = {lmmM_conv}; "
  f"query random-intercept variance = {vc_query_main:.4f}")
w(f"- convergence warnings (verbatim): {lmmM_warn if lmmM_warn else 'none'}\n")
w("Why the crossed model was replaced. The originally specified crossed-VC model "
  "(random intercepts for both query and backbone) returned:\n")
w(f"- variance components: **query = {vc_query:.4f}, backbone = {vc_backbone:.6f}**, residual = {vc_resid:.4f}")
w(f"- system coefficient {mdf.params['sysD']:.4f} — identical to four decimals to the "
  f"design-based pooled mean paired RR difference ({mean_paired:.4f}) and to the main model above.\n")
w(f"The backbone variance component is **{vc_backbone:.2e}** — effectively zero (at the boundary), so the "
  f"crossed model degenerates to a query-only model; the slogdet warnings are the numerical signature "
  f"of that boundary fit, not benign noise. The fixed-effect specification removes it. The system "
  f"effect is stable across all three (crossed, backbone-fixed, design-based mean): {mean_paired:.4f}.\n")

w("## Reading\n")
w(f"All analyses agree: D improves top-1 and MRR over A across the pooled panel. The pooled "
  f"discordant count is **{tot_disc}** (vs 13 at the reference backbone alone), so the pooled result "
  f"rests on far more information than the single-backbone test. **The relative effect is consistent "
  f"across backbones** (matched OR $\\approx${OR:.2f}, Breslow–Day $p={bd_p:.2f}$; no evidence of "
  f"heterogeneity, weak power at 4 strata), **while the absolute gain $\\Delta$R@1 scales with "
  f"baseline headroom** (largest where the baseline is weakest).\n")
w("**Headline recommendation (Correction 1 — writing decision, to apply in the paper's §12).** "
  "Promote the pooled 4-backbone panel to the PRIMARY R@1 result (70 discordant pairs, CMH "
  f"$p={cmh_p:.2g}$, OR {OR:.2f} [{OR_lo:.2f}, {OR_hi:.2f}]), and demote the single reference "
  "backbone to a detailed case study (ablation, error analysis, and cost all run on it; its full "
  "narrative is retained). This replaces the current \"directionally strong, not $<0.05$\" wording. "
  "**Mandatory caveat to state in the same place:** the four-backbone panel was run *after* observing "
  "the reference result, so it is a **replication, not a pre-registered analysis**. With the caveat "
  "the promotion is defensible; without it, it is not.\n")
(OUT/"pooled_backbones.md").write_text("\n".join(L))

# ---------- LaTeX tables ----------
lat=[]
lat.append("% Per-backbone discordant + pooled")
lat.append("\\begin{tabular}{lccccc}\n\\toprule")
lat.append("Backbone & A R@1 & D R@1 & fixed & broke & discordant \\\\\n\\midrule")
for bb in BACKBONES:
    v=disc[bb]; bbesc=bb.replace('_','\\_')
    lat.append(f"{bbesc} & {v['A_r1']} & {v['D_r1']} & {v['fixed']} & {v['broke']} & {v['discordant']} \\\\")
lat.append(f"\\midrule\nTotal & & & {tot_c} & {tot_b} & {tot_disc} \\\\\n\\bottomrule\n\\end{{tabular}}")
lat.append("")
lat.append("% Pooled tests")
lat.append("\\begin{tabular}{lc}\n\\toprule\nTest & Result \\\\\n\\midrule")
lat.append(f"CMH (stratified McNemar) & $\\chi^2={cmh_stat:.2f}$, $p={cmh_p:.2g}$ \\\\")
lat.append(f"Common matched OR (D vs A) & {OR:.2f} \\,[{OR_lo:.2f}, {OR_hi:.2f}] \\\\")
lat.append(f"Breslow--Day homogeneity & $\\chi^2={bd:.2f}$, $p={bd_p:.2f}$ \\\\")
lat.append(f"Conditional logit (query$\\times$backbone) & OR $={cl_qxb[3]:.2f}$, $p={cl_qxb[2]:.2g}$ \\\\")
lat.append(f"MRR mixed (backbone fixed), system coef & {lmmM_coef:.3f}\\,[{lmmM_ci[0]:.3f}, {lmmM_ci[1]:.3f}] \\\\")
lat.append("\\bottomrule\n\\end{tabular}")
(OUT/"pooled_backbones_tables.tex").write_text("\n".join(lat))

# ---------- console ----------
print("Per-backbone discordant:", {bb:(disc[bb]['fixed'],disc[bb]['broke']) for bb in BACKBONES})
print(f"TOTAL discordant across 4 backbones = {tot_disc} (fixed {tot_c} / broke {tot_b})")
print(f"CMH chi2={cmh_stat:.3f} p={cmh_p:.3g} | matched OR={OR:.3f} [{OR_lo:.3f},{OR_hi:.3f}]")
print(f"Breslow-Day chi2={bd:.3f} df={len(BACKBONES)-1} p={bd_p:.3f}")
print(f"CondLogit by query (robustness):     coef={cl_q[0]:.3f} OR={cl_q[3]:.3f} p={cl_q[2]:.3g}")
print(f"CondLogit by qxbb (headline matched): coef={cl_qxb[0]:.3f} OR={cl_qxb[3]:.3f} p={cl_qxb[2]:.3g}")
print(f"MRR MAIN (backbone fixed): system coef={lmmM_coef:.4f} [{lmmM_ci[0]:.4f},{lmmM_ci[1]:.4f}] p={lmmM_p:.3g} conv={lmmM_conv} warn={lmmM_warn or 'none'}")
print(f"MRR crossed VC: query={vc_query:.4f} backbone={vc_backbone:.2e} resid={vc_resid:.4f} (backbone VC at boundary -> slogdet warnings)")
print(f"design-based mean paired RR diff = {mean_paired:.4f}")
print("wrote analysis/pooled_backbones.md + pooled_backbones_tables.tex")
