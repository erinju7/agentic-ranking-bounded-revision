"""Corrected analysis of the k=5 ablation (reads per-query rep files; no re-run).
Handles the three statistical fixes:
  (1) repeated measures: reps are the SAME queries -> NO pooled 485-case McNemar.
      Aggregate inference = CLUSTER BOOTSTRAP over queries (resample query ids, keep all
      of a query's reps together); plus per-rep McNemar; plus cross-rep SD (reproducibility).
  (2) multiple testing: full-D vs {no_anchor,no_concept,no_surface} is a family of 3 ->
      HOLM correction within each rep; report raw AND Holm-corrected p. Ordering by point est.
  (3) per-comparison matched samples: each pair uses queries where BOTH those arms are ok.
Output: results/bright_ablation/<MODEL>/analysis.json + console table.
"""
import json, math, random, statistics as st
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; M="claude-haiku-4-5-20251001"
BASE=ROOT/"results"/"bright_ablation"/M
ABL=["no_anchor","no_concept","no_surface"]

def mcnemar(f,b):
    n=f+b; return 1.0 if n==0 else min(1.0,2*sum(math.comb(n,i) for i in range(min(f,b)+1))/2**n)
def holm(ps):
    order=sorted(range(len(ps)),key=lambda i:ps[i]); out=[0]*len(ps); m=len(ps)
    for rank,i in enumerate(order): out[i]=min(1.0,(m-rank)*ps[i])
    for a in range(1,len(order)): out[order[a]]=max(out[order[a]],out[order[a-1]])
    return out

reps=[]
for k in range(1,6):
    f=BASE/f"rep_{k}"/"per_query.json"
    if f.exists(): reps.append(json.load(open(f)))
K=len(reps); print(f"loaded {K} reps")

def ok(rec,arm):  # arm produced a usable (non-fallback) output; use top1 presence as proxy
    return arm in rec and "top1_gold" in rec[arm]

# per-rep McNemar (per-comparison matched) + Holm within rep
perrep={a:[] for a in ABL}
for rep in reps:
    ps=[]
    for a in ABL:
        m=[x for x in rep if ok(x,"full") and ok(x,a)]
        f=sum(1 for x in m if x["full"]["top1_gold"] and not x[a]["top1_gold"])
        b=sum(1 for x in m if not x["full"]["top1_gold"] and x[a]["top1_gold"])
        ps.append((a,f,b,len(m),mcnemar(f,b)))
    hp=holm([p[4] for p in ps])
    for (a,f,b,n,raw),hc in zip(ps,hp):
        perrep[a].append({"fix_full>abl":f,"abl>full":b,"n":n,"raw_p":round(raw,4),"holm_p":round(hc,4)})

# cluster bootstrap over queries (keep all reps of a query) for Hit@1 Δ and MRR Δ
def qkeyed(metric, arm):
    # returns {qid: [per-rep paired diff full-arm]} using matched (both ok) rep-instances
    d={}
    for rep in reps:
        for x in rep:
            if ok(x,"full") and ok(x,arm):
                v=(x["full"]["top1_gold"]-x[arm]["top1_gold"]) if metric=="hit" else (x["full"]["mrr"]-x[arm]["mrr"])
                d.setdefault(x["id"],[]).append(v)
    return d
def clusterboot(d,B=10000,seed=0):
    qs=list(d); rng=random.Random(seed)
    def stat(sample):
        vals=[v for q in sample for v in d[q]]; return sum(vals)/len(vals)
    obs=stat(qs); bs=sorted(stat([qs[rng.randrange(len(qs))] for _ in qs]) for _ in range(B))
    return round(obs,4), round(bs[int(.025*B)],4), round(bs[int(.975*B)],4)

out={"model":M,"reps":K,"per_rep":perrep,"aggregate":{}}
print("\n=== full-D vs each ablation (per-comparison matched; Holm within rep; cluster bootstrap over queries) ===")
for a in ABL:
    sig_raw=sum(1 for r in perrep[a] if r["raw_p"]<0.05); sig_holm=sum(1 for r in perrep[a] if r["holm_p"]<0.05)
    hit=clusterboot(qkeyed("hit",a)); mrr=clusterboot(qkeyed("mrr",a))
    sd=st.pstdev([ (r["fix_full>abl"]-r["abl>full"]) for r in perrep[a] ]) if K>1 else 0
    out["aggregate"][a]={"per_rep_raw_p":[r["raw_p"] for r in perrep[a]],
                          "per_rep_holm_p":[r["holm_p"] for r in perrep[a]],
                          "sig_raw_k":f"{sig_raw}/{K}","sig_holm_k":f"{sig_holm}/{K}",
                          "Hit@1_delta_cluster95CI":hit,"MRR_delta_cluster95CI":mrr}
    print(f"  full vs {a}: raw sig {sig_raw}/{K}, Holm sig {sig_holm}/{K}")
    print(f"     Hit@1 Δ (full−abl) cluster-boot 95% CI: {hit[0]:+.3f} [{hit[1]:+.3f},{hit[2]:+.3f}]")
    print(f"     MRR   Δ cluster-boot 95% CI:           {mrr[0]:+.3f} [{mrr[1]:+.3f},{mrr[2]:+.3f}]")
# point-estimate component ordering (descriptive, NOT a significance claim)
order=sorted(ABL,key=lambda a:out["aggregate"][a]["Hit@1_delta_cluster95CI"][0],reverse=True)
print("\npoint-estimate Hit@1 drop from full (largest first = most important component):")
for a in order: print(f"  {a}: full−abl Hit@1 Δ = {out['aggregate'][a]['Hit@1_delta_cluster95CI'][0]:+.3f}")
# arm Hit@1 means
print("\narm Hit@1 (mean over reps):")
for arm in ["A","full","no_anchor","no_concept","no_surface"]:
    vals=[st.mean(x[arm]["top1_gold"] for x in rep if ok(x,arm)) for rep in reps]
    print(f"  {arm:11} {st.mean(vals):.3f}")
    out.setdefault("arm_hit1",{})[arm]=round(st.mean(vals),4)
json.dump(out, open(BASE/"analysis.json","w"), indent=1)
print(f"\nwrote {BASE/'analysis.json'}")
