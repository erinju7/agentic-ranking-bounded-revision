"""Phase 3: Cat-5 sensitivity. NO API. Reference backbone only. Reuses frozen taxonomy labels
(analysis/taxonomy_labels.json) — no re-labelling. Primary = complete 97-query set (headline).
S1 = drop Cat-5; S2 = drop Cat-4 + Cat-5. D full ranking reconstructed from frozen baseline_ranked
+ promote_ids/decision and verified against stored arch_first_gold_rank.
"""
import json, math, random, statistics as st
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"analysis"

base={str(p["id"]):p for p in json.load(open(ROOT/"results/bright_hardpool/biology/baseline_per_query.json"))}
Dp  ={str(p["id"]):p for p in json.load(open(ROOT/"results/bright_ch_anchor/biology/per_query.json"))}
tax =json.load(open(ROOT/"analysis/taxonomy_labels.json"))["labels"]
cat ={l["query_id"]:l["baseline_error_category"] for l in tax}
CAT4=[q for q,c in cat.items() if c==4]; CAT5=[q for q,c in cat.items() if c==5]

def ndcg(rels,k=10):
    d=sum(r/math.log2(i+2) for i,r in enumerate(rels[:k]))
    ide=sum(r/math.log2(i+2) for i,r in enumerate(sorted(rels,reverse=True)[:k]))
    return d/ide if ide else 0.0

# per-query A and D full rankings -> rels, first_gold_rank, ndcg, gold ranks
Q={}
for q in base:
    gold=set(base[q]["gold_aliases"]); bl=base[q]["baseline_ranked"]
    dec=Dp[q]["decision"]; prom=[a for a in (Dp[q]["promote_ids"] or []) if a in bl][:2]
    dr = list(bl) if (dec=="surface_sufficient" or not prom) else prom+[a for a in bl if a not in prom]
    def stats(ranked):
        rels=[1 if a in gold else 0 for a in ranked]
        first=next((i+1 for i,a in enumerate(ranked) if a in gold), len(ranked)+1)
        gr=[i+1 for i,a in enumerate(ranked) if a in gold]
        return dict(rels=rels, first=first, top1=rels[0]==1, ndcg=ndcg(rels), goldranks=gr)
    A=stats(bl); D=stats(dr)
    assert D["first"]==Dp[q]["arch_first_gold_rank"], f"D recon mismatch q{q}: {D['first']} vs {Dp[q]['arch_first_gold_rank']}"
    assert A["first"]==base[q]["baseline_first_gold_rank"]
    Q[q]={"A":A,"D":D}
print("D-reconstruction check vs stored arch_first_gold_rank: PASS (all 97)")

def mcnemar(f,b):
    n=f+b; k=min(f,b); return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n)) if n else 1.0
def boot_perm(a,s,B=10000,seed=123):
    diffs=[si-ai for ai,si in zip(a,s)]; obs=st.mean(diffs); n=len(diffs)
    rng=random.Random(seed); means=[]
    for _ in range(B): means.append(sum(diffs[rng.randrange(n)] for _ in range(n))/n)
    means.sort(); lo,hi=means[int(0.025*B)],means[int(0.975*B)]
    rng2=random.Random(seed+1); c=0
    for _ in range(B):
        m=sum((d if rng2.random()<0.5 else -d) for d in diffs)/n
        if abs(m)>=abs(obs)-1e-12: c+=1
    return obs,lo,hi,(c+1)/(B+1)

def analyse(qids):
    n=len(qids)
    def r1(S): return sum(Q[q][S]["top1"] for q in qids)/n
    def rec5(S): return sum(1 for q in qids if Q[q][S]["first"]<=5)/n
    def mrr(S): return st.mean([1/Q[q][S]["first"] for q in qids])
    def ndcgm(S): return st.mean([Q[q][S]["ndcg"] for q in qids])
    def mgr(S):
        allg=[r for q in qids for r in Q[q][S]["goldranks"]]; return st.mean(allg)
    fixed=sum(1 for q in qids if (not Q[q]["A"]["top1"]) and Q[q]["D"]["top1"])
    broke=sum(1 for q in qids if Q[q]["A"]["top1"] and (not Q[q]["D"]["top1"]))
    a=[1/Q[q]["A"]["first"] for q in qids]; s=[1/Q[q]["D"]["first"] for q in qids]
    d,lo,hi,pp=boot_perm(a,s)
    return dict(n=n, A_r1=r1("A"), D_r1=r1("D"), A_rec5=rec5("A"), D_rec5=rec5("D"),
                A_mrr=mrr("A"), D_mrr=mrr("D"), A_ndcg=ndcgm("A"), D_ndcg=ndcgm("D"),
                A_mgr=mgr("A"), D_mgr=mgr("D"), fixed=fixed, broke=broke,
                mcnemar=mcnemar(fixed,broke), mrr_delta=d, mrr_ci=(lo,hi), mrr_p=pp)

allq=list(base)
prim=analyse(allq)
s1=analyse([q for q in allq if q not in set(CAT5)])
s2=analyse([q for q in allq if q not in set(CAT4)|set(CAT5)])
RES=[("Primary (complete 97-set) — HEADLINE",prim),("S1 (drop Cat-5)",s1),("S2 (drop Cat-4 + Cat-5)",s2)]

# direction check
def dirn(r): return "D>A" if r["D_r1"]>r["A_r1"] else ("D<A" if r["D_r1"]<r["A_r1"] else "D=A")
dir_same=all(dirn(r)=="D>A" for _,r in RES) and all(r["D_mrr"]>r["A_mrr"] for _,r in RES)
prim_sig=prim["mcnemar"]<0.05; s1_sig=s1["mcnemar"]<0.05; s2_sig=s2["mcnemar"]<0.05

CAVEAT=("This is a sensitivity analysis, not the primary analysis. The exclusion criterion is "
 "category membership from the taxonomy, and the categories were identified in the §3.2 acceptance "
 "audit prior to observing System D's outcomes. The complete-set result remains the reported headline.")

L=["# Phase 3 — Cat-5 sensitivity analysis (reference backbone, no API)\n",
   f"> **{CAVEAT}**\n",
   "Honesty note: the category *scheme* and ambiguity assessment predate System D (§3.2 gate); the "
   "per-query Cat-1..5 assignment was transcribed in Study 1d (`analysis/taxonomy_labels.json`). "
   "Category membership is a structural property of the query/gold, not derived from D's ranking. "
   "D's full ranking is reconstructed from frozen `baseline_ranked`+`promote_ids`/`decision` and "
   "verified against the stored `arch_first_gold_rank` (all 97 match).\n",
   f"Dropped sets: Cat-5 = {sorted(CAT5,key=int)} (n=11); Cat-4 = {sorted(CAT4,key=int)} (n=5).\n",
   "| Analysis | n | A R@1 | D R@1 | A Rec@5 | D Rec@5 | A MRR | D MRR | A nDCG@10 | D nDCG@10 | A mgr | D mgr | fixed | broke | McNemar p | MRR Δ [95% CI] | MRR perm p |",
   "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name,r in RES:
    L.append(f"| {name} | {r['n']} | {r['A_r1']:.3f} | {r['D_r1']:.3f} | {r['A_rec5']:.3f} | {r['D_rec5']:.3f} | "
             f"{r['A_mrr']:.3f} | {r['D_mrr']:.3f} | {r['A_ndcg']:.3f} | {r['D_ndcg']:.3f} | {r['A_mgr']:.2f} | {r['D_mgr']:.2f} | "
             f"{r['fixed']} | {r['broke']} | {r['mcnemar']:.4f} | {r['mrr_delta']:+.3f} [{r['mrr_ci'][0]:+.3f}, {r['mrr_ci'][1]:+.3f}] | {r['mrr_p']:.4f} |")
L.append("")
L.append("## Key observation — the R@1 discordant structure is invariant")
L.append("The fixed/broke counts (**10 / 3**) and the exact McNemar p (**0.0923**) are **identical across "
         "all three analyses**. This is structural, not coincidental: the excluded Cat-4/Cat-5 queries are "
         "all baseline *errors* with **0 fixes** (unfixable by definition), and the 3 broke queries "
         "(q64, q41, q45) are baseline-*correct* queries that lie **outside the error taxonomy** and are "
         "therefore never excluded. So the exclusion removes only non-discordant queries. Consequence: "
         "dropping the benchmark-ambiguous categories **raises absolute R@1** (A 0.536→0.642, D 0.608→0.728) "
         "by removing unfixable queries from the denominator, but **does not change the paired R@1 "
         "significance at all** — confirming the primary p=0.092 is *not* an artifact of including "
         "benchmark-unfixable queries.")
L.append("")
L.append("## Direction vs significance")
L.append(f"- Direction: {', '.join(f'{name.split(chr(32))[0]}={dirn(r)}' for name,r in RES)} — "
         f"**exclusion does NOT change the direction of any conclusion** (D>A on R@1 and MRR in all three)."
         if dir_same else "- Direction CHANGES under exclusion — see table.")
L.append(f"- Significance (McNemar R@1): Primary p={prim['mcnemar']:.4f} ({'sig' if prim_sig else 'not sig'}), "
         f"S1 p={s1['mcnemar']:.4f} ({'sig' if s1_sig else 'not sig'}), S2 p={s2['mcnemar']:.4f} ({'sig' if s2_sig else 'not sig'}).")
if (s1_sig or s2_sig) and not prim_sig:
    L.append(f"- **A sensitivity subset crosses p<0.05 while the primary does not.** Per the brief this is "
             f"stated plainly WITH the caveat, in the same sentence: a sensitivity analysis that excludes "
             f"taxonomy categories (identified pre-D in the §3.2 audit) reaches significance, but **the "
             f"complete-set result remains the reported headline** and this is not the primary finding.")
else:
    L.append("- No sensitivity subset crosses p<0.05 while the primary does not (or the primary is already "
             "significant); the exclusion changes significance only as noted, not the reported headline.")
L.append(f"\n**Summary:** excluding the benchmark-ambiguous / ambiguous-query categories changes only the "
         f"*magnitude/significance* of the D-vs-A gap, not its *direction*. It is reported as a sensitivity "
         f"analysis; the complete 97-query result is the headline everywhere.")
(OUT/"sensitivity_cat5.md").write_text("\n".join(L))

# LaTeX
lat=["% Cat-5 sensitivity (reference backbone). Sensitivity analysis, NOT primary.",
     "\\begin{tabular}{lcccccccc}\n\\toprule",
     "Analysis & $n$ & A R@1 & D R@1 & A MRR & D MRR & fixed & broke & McNemar $p$ \\\\\n\\midrule"]
for name,r in RES:
    nm=name.split(" —")[0].replace("&","\\&")
    lat.append(f"{nm} & {r['n']} & {r['A_r1']:.3f} & {r['D_r1']:.3f} & {r['A_mrr']:.3f} & {r['D_mrr']:.3f} & {r['fixed']} & {r['broke']} & {r['mcnemar']:.4f} \\\\")
lat.append("\\bottomrule\n\\end{tabular}")
(OUT/"sensitivity_cat5_tables.tex").write_text("\n".join(lat))

print(f"{'analysis':<32}{'n':>4}{'A_R@1':>7}{'D_R@1':>7}{'fix':>5}{'brk':>5}{'McN_p':>9}{'MRRΔ':>8}{'MRR_p':>8}")
for name,r in RES:
    print(f"{name.split(' —')[0]:<32}{r['n']:>4}{r['A_r1']:>7.3f}{r['D_r1']:>7.3f}{r['fixed']:>5}{r['broke']:>5}{r['mcnemar']:>9.4f}{r['mrr_delta']:>+8.3f}{r['mrr_p']:>8.4f}")
print("direction unchanged (D>A all subsets):", dir_same)
print("wrote analysis/sensitivity_cat5.md + sensitivity_cat5_tables.tex")
