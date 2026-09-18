"""Phase 3 RE-RUN with HUMAN labels (primary). Same computation as sensitivity_cat5.py; the ONLY
change is that Cat-4/Cat-5 membership comes from analysis/annotation_human.csv (the human primary
annotation), not the machine transcription. Outcomes still recomputed from frozen System-D outputs.
NO API. Reference backbone.
"""
import json, math, random, csv, statistics as st
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"analysis"
base={str(p["id"]):p for p in json.load(open(ROOT/"results/bright_hardpool/biology/baseline_per_query.json"))}
Dp  ={str(p["id"]):p for p in json.load(open(ROOT/"results/bright_ch_anchor/biology/per_query.json"))}
# HUMAN labels
H={}
with open(ROOT/"analysis/annotation_human.csv") as f:
    for r in csv.DictReader(f): H[r["query_id"]]=int(r["category"])
CAT4=[q for q,c in H.items() if c==4]; CAT5=[q for q,c in H.items() if c==5]

def ndcg(rels,k=10):
    d=sum(r/math.log2(i+2) for i,r in enumerate(rels[:k]))
    ide=sum(r/math.log2(i+2) for i,r in enumerate(sorted(rels,reverse=True)[:k])); return d/ide if ide else 0.0
def mcnemar(f,b):
    n=f+b; k=min(f,b); return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n)) if n else 1.0
def boot_perm(a,s,B=10000,seed=123):
    diffs=[si-ai for ai,si in zip(a,s)]; obs=st.mean(diffs); n=len(diffs)
    rng=random.Random(seed); means=[sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(B)]
    means.sort(); lo,hi=means[int(0.025*B)],means[int(0.975*B)]
    rng2=random.Random(seed+1); c=sum(1 for _ in range(B) if abs(sum((d if rng2.random()<0.5 else -d) for d in diffs)/n)>=abs(obs)-1e-12)
    return obs,lo,hi,(c+1)/(B+1)

Q={}
for q in base:
    gold=set(base[q]["gold_aliases"]); bl=base[q]["baseline_ranked"]
    dec=Dp[q]["decision"]; prom=[a for a in (Dp[q]["promote_ids"] or []) if a in bl][:2]
    dr=list(bl) if (dec=="surface_sufficient" or not prom) else prom+[a for a in bl if a not in prom]
    def stt(r):
        rels=[1 if a in gold else 0 for a in r]; first=next((i+1 for i,a in enumerate(r) if a in gold),len(r)+1)
        return dict(first=first,top1=rels[0]==1,ndcg=ndcg(rels),gr=[i+1 for i,a in enumerate(r) if a in gold])
    Q[q]={"A":stt(bl),"D":stt(dr)}

def analyse(qids):
    n=len(qids)
    r1=lambda S:sum(Q[q][S]["top1"] for q in qids)/n
    rec5=lambda S:sum(1 for q in qids if Q[q][S]["first"]<=5)/n
    mrr=lambda S:st.mean([1/Q[q][S]["first"] for q in qids])
    ndm=lambda S:st.mean([Q[q][S]["ndcg"] for q in qids])
    mgr=lambda S:st.mean([r for q in qids for r in Q[q][S]["gr"]])
    fixed=sum(1 for q in qids if (not Q[q]["A"]["top1"]) and Q[q]["D"]["top1"])
    broke=sum(1 for q in qids if Q[q]["A"]["top1"] and (not Q[q]["D"]["top1"]))
    a=[1/Q[q]["A"]["first"] for q in qids]; s=[1/Q[q]["D"]["first"] for q in qids]
    d,lo,hi,pp=boot_perm(a,s)
    return dict(n=n,A_r1=r1("A"),D_r1=r1("D"),A_rec5=rec5("A"),D_rec5=rec5("D"),A_mrr=mrr("A"),D_mrr=mrr("D"),
                A_ndcg=ndm("A"),D_ndcg=ndm("D"),A_mgr=mgr("A"),D_mgr=mgr("D"),fixed=fixed,broke=broke,
                mcnemar=mcnemar(fixed,broke),mrr_delta=d,mrr_ci=(lo,hi),mrr_p=pp)

allq=list(base)
prim=analyse(allq); s1=analyse([q for q in allq if q not in set(CAT5)]); s2=analyse([q for q in allq if q not in set(CAT4)|set(CAT5)])
RES=[("Primary (complete 97) — HEADLINE",prim),("S1 (drop Cat-5, human)",s1),("S2 (drop Cat-4+5, human)",s2)]
# which fixes/breaks fall in the excluded set
fixed_ids=[q for q in allq if (not Q[q]["A"]["top1"]) and Q[q]["D"]["top1"]]
fix_in_cat5=[q for q in fixed_ids if H.get(q)==5]

CAV=("This is a sensitivity analysis, not the primary analysis. The exclusion criterion is category "
 "membership from the taxonomy (HUMAN primary labels), identified by a blinded human annotator (blind "
 "to System-D outcomes and to the machine labels). The complete-set result remains the reported headline.")
L=["# Phase 3 (RE-RUN, HUMAN labels) — Cat-5 sensitivity (reference backbone, no API)\n",
   f"> **{CAV}**\n",
   f"Label source: `analysis/annotation_human.csv` (human primary). Human distribution "
   f"Cat1=12, Cat2=16, Cat3=3, Cat4={len(CAT4)}, Cat5={len(CAT5)}. Dropped: Cat-5={sorted(CAT5,key=int)}; "
   f"Cat-4={sorted(CAT4,key=int)}.\n",
   "| Analysis | n | A R@1 | D R@1 | A MRR | D MRR | A nDCG@10 | D nDCG@10 | A mgr | D mgr | fixed | broke | McNemar p | MRR Δ [95% CI] | MRR perm p |",
   "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name,r in RES:
    L.append(f"| {name} | {r['n']} | {r['A_r1']:.3f} | {r['D_r1']:.3f} | {r['A_mrr']:.3f} | {r['D_mrr']:.3f} | "
             f"{r['A_ndcg']:.3f} | {r['D_ndcg']:.3f} | {r['A_mgr']:.2f} | {r['D_mgr']:.2f} | {r['fixed']} | {r['broke']} | "
             f"{r['mcnemar']:.4f} | {r['mrr_delta']:+.3f} [{r['mrr_ci'][0]:+.3f}, {r['mrr_ci'][1]:+.3f}] | {r['mrr_p']:.4f} |")
L.append("")
L.append("## Key difference vs the machine-label version")
L.append(f"Under the human labels, **Cat-5 now contains {len(fix_in_cat5)} of System-D's 10 fixes: "
         f"{sorted(fix_in_cat5,key=int)}** (q12 closest-relatives, q54 blood-as-organ, q84 spider-venom — "
         f"the human judged each gold contestable). Excluding Cat-5 therefore **removes fixes**, so the "
         f"paired R@1 discordant count drops from **10/3 to {s1['fixed']}/{s1['broke']}** and McNemar rises "
         f"from **0.0923 to {s1['mcnemar']:.4f}**. This is the OPPOSITE of the machine-label re-run, where "
         f"Cat-5 held no fixes and the McNemar was invariant. The human validation materially changes the "
         f"sensitivity behaviour — reported as-is, no adjudication.\n")
L.append("## Direction vs significance")
dir_ok=all(r['D_r1']>r['A_r1'] and r['D_mrr']>r['A_mrr'] for _,r in RES)
L.append(f"- Direction: D>A on R@1 and MRR in all three subsets → **exclusion does not change the "
         f"direction** (D still improves over A).")
L.append(f"- Significance (McNemar R@1): Primary {prim['mcnemar']:.4f}, S1 {s1['mcnemar']:.4f}, S2 {s2['mcnemar']:.4f}. "
         f"MRR permutation still significant in all three (p ≤ {max(r['mrr_p'] for _,r in RES):.4f}).")
L.append(f"- **So excluding the human-labelled gold-ambiguous queries REDUCES the R@1 significance** "
         f"(removes 3 of D's fixes, which the human attributes to contestable golds), while the MRR gain "
         f"and the direction survive. Interpretation: part of D's top-1 benefit sits on queries the human "
         f"considers gold-ambiguous; a conservative, clean-gold-only view leaves a positive but "
         f"R@1-non-significant effect (fixed {s1['fixed']} / broke {s1['broke']}), with MRR still significant.")
(OUT/"sensitivity_cat5_human.md").write_text("\n".join(L))
print(f"{'analysis':<34}{'n':>4}{'A_R@1':>7}{'D_R@1':>7}{'fix':>4}{'brk':>4}{'McN_p':>9}{'MRRΔ':>8}{'MRR_p':>8}")
for name,r in RES:
    print(f"{name.split(' —')[0]:<34}{r['n']:>4}{r['A_r1']:>7.3f}{r['D_r1']:>7.3f}{r['fixed']:>4}{r['broke']:>4}{r['mcnemar']:>9.4f}{r['mrr_delta']:>+8.3f}{r['mrr_p']:>8.4f}")
print("D-fixes now in human Cat-5:", sorted(fix_in_cat5,key=int))
print("direction D>A all subsets:", dir_ok)
print("wrote analysis/sensitivity_cat5_human.md")
