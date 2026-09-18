"""Step 6b (API): per pre-registered split — closed-book acceptance gate, then System A + System D
at the reference backbone. Per-call response cache keyed by (split, system, agent, qid, prompt-hash)
so re-runs are free/resumable; token+cost logged per call; global $10 HARD-ABORT. Gates all splits
first; if any shows substantial memorization it STOPS before running A/D (report for a decision).
"""
import json, math, time, random, hashlib, re, statistics as st
from pathlib import Path
from rq2_core import make_client, extract_json_object, usage_cost_usd
from bright_backbone import p_rerank_plain, p_a1, p_a2, p_a3_anchor, rank_from_ids

ROOT=Path(__file__).resolve().parents[1]
SPLITS=["earth_science","sustainable_living","psychology"]
MODEL="gemini-flash-latest"; COST_CAP=10.0
GATE_MEM_THRESHOLD=0.50   # fraction reproducing reference answer (MuSiQue-50% reference point)
ledger={"cost":0.0,"calls":0}

def norm(s): return re.sub(r"[^a-z0-9 ]"," ",(s or "").lower())
def f1(pred,gold):
    p,g=norm(pred).split(),norm(gold).split()
    if not p or not g: return 0.0
    ps,gs=set(p),set(g); inter=len(ps&gs)
    if not inter: return 0.0
    pr,rc=inter/len(ps),inter/len(gs); return 2*pr*rc/(pr+rc)
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
    return round(obs,4),round(lo,4),round(hi,4),round((c+1)/(B+1),4)

def p_closed_book(q):
    return (f"""Answer this question as fully and specifically as you can from your own knowledge, """
            f"""with the key facts/entities. Return JSON only.\nSchema: {{"answer": "<answer>"}}\n\nQuestion: {q[:1500]}""")

client=make_client(MODEL)
def call(split, system, agent, qid, prompt):
    cache=ROOT/"runs"/"splits"/split/"cache"; cache.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256(prompt.encode()).hexdigest()[:8]
    cf=cache/f"{system}_{agent}_{qid}_{h}.json"
    if cf.exists(): d=json.loads(cf.read_text())
    else:
        t=time.perf_counter(); raw=client.generate(prompt); dt=time.perf_counter()-t
        u=client.last_usage_metadata or {}
        d={"raw":raw,"in":u.get("prompt_token_count") or 0,"out":u.get("candidates_token_count") or 0,
           "cost":usage_cost_usd(MODEL,u) or 0.0,"lat":dt}
        cf.write_text(json.dumps(d)); ledger["cost"]+=d["cost"]; ledger["calls"]+=1
        if ledger["cost"]>COST_CAP: raise RuntimeError(f"COST HARD-ABORT: ${ledger['cost']:.2f} > ${COST_CAP}")
    try: obj=extract_json_object(d["raw"])
    except Exception: obj={}
    return obj,d

# ---------- gate all splits first ----------
gates={}
for split in SPLITS:
    pools=json.loads((ROOT/"runs"/"splits"/split/"pools.json").read_text())
    gate_ids=json.loads((ROOT/"runs"/"splits"/split/"gate_sample.json").read_text())
    f1s=[]
    for qid in gate_ids:
        obj,_=call(split,"gate","closedbook",qid,p_closed_book(pools[qid]["query"]))
        ans=obj.get("answer","") if isinstance(obj,dict) else ""
        f1s.append(f1(ans, pools[qid].get("gold_answer") or ""))
    frac=sum(1 for x in f1s if x>=0.4)/len(f1s); meanf1=st.mean(f1s)
    passed = frac < GATE_MEM_THRESHOLD
    gates[split]={"n":len(f1s),"mean_f1":round(meanf1,3),"frac_ge_0.4":round(frac,3),"passed":passed}
    (ROOT/"runs"/"splits"/split/"gate.json").write_text(json.dumps(gates[split],indent=1))
    print(f"GATE {split:<20} mean_f1={meanf1:.3f} frac>=0.4={frac:.3f} -> {'PASS' if passed else 'FLAG (substantial memorization)'}")

if not all(g["passed"] for g in gates.values()):
    print("\nGATE FAILURE — stopping before A/D per brief. Flagged:",
          [s for s,g in gates.items() if not g['passed']])
    print(f"cost so far ${ledger['cost']:.4f}, calls {ledger['calls']}")
    raise SystemExit(0)
print(f"\nAll gates PASS. cost so far ${ledger['cost']:.4f}. Running A+D...\n")

# ---------- A + D per split ----------
results={}
for split in SPLITS:
    pools=json.loads((ROOT/"runs"/"splits"/split/"pools.json").read_text())
    perA=[]; perD=[]; decisions={}
    for qid,pl in pools.items():
        q=pl["query"]; a2t=pl["alias_to_text"]; goldset=set(pl["gold_aliases"]); order=list(a2t)
        body=[{"id":a,"text":a2t[a]} for a in order]
        # A
        objA,_=call(split,"A","rerank",qid,p_rerank_plain(q,body))
        rA=rank_from_ids(objA.get("ranked_ids",[]),order)
        relsA=[1 if a in goldset else 0 for a in rA]; firstA=next((i+1 for i,a in enumerate(rA) if a in goldset),len(rA)+1)
        # D
        h1,_=call(split,"D","a1",qid,p_a1(q)); h2,_=call(split,"D","a2",qid,p_a2(q))
        co,_=call(split,"D","a3",qid,p_a3_anchor(q,h1,h2,body))
        dec=(co.get("decision") or "surface_sufficient"); decisions[dec]=decisions.get(dec,0)+1
        prom=[str(x.get("id") if isinstance(x,dict) else x) for x in (co.get("promote_ids") or [])]
        prom=[a for a in prom if a in a2t][:2]
        rD=list(rA) if (dec=="surface_sufficient" or not prom) else prom+[a for a in rA if a not in prom]
        relsD=[1 if a in goldset else 0 for a in rD]; firstD=next((i+1 for i,a in enumerate(rD) if a in goldset),len(rD)+1)
        perA.append({"id":qid,"first":firstA,"top1":relsA[0]==1,"ndcg":ndcg(relsA),"gr":[i+1 for i,a in enumerate(rA) if a in goldset]})
        perD.append({"id":qid,"first":firstD,"top1":relsD[0]==1,"ndcg":ndcg(relsD),"gr":[i+1 for i,a in enumerate(rD) if a in goldset],"decision":dec})
    (ROOT/"runs"/"splits"/split/"A_per_query.json").write_text(json.dumps(perA,indent=1))
    (ROOT/"runs"/"splits"/split/"D_per_query.json").write_text(json.dumps(perD,indent=1))
    n=len(perA)
    Aidx={x["id"]:x for x in perA}; Didx={x["id"]:x for x in perD}
    def r1(idx): return sum(idx[q]["top1"] for q in idx)/n
    def rec5(idx): return sum(1 for q in idx if idx[q]["first"]<=5)/n
    def mrr(idx): return st.mean([1/idx[q]["first"] for q in idx])
    def ndm(idx): return st.mean([idx[q]["ndcg"] for q in idx])
    def mgr(idx): return st.mean([r for q in idx for r in idx[q]["gr"]])
    fixed=sum(1 for q in Aidx if (not Aidx[q]["top1"]) and Didx[q]["top1"])
    broke=sum(1 for q in Aidx if Aidx[q]["top1"] and (not Didx[q]["top1"]))
    corr=sum(1 for q in Aidx if Aidx[q]["top1"])
    a=[1/Aidx[q]["first"] for q in Aidx]; s=[1/Didx[q]["first"] for q in Aidx]
    dlt,lo,hi,pp=boot_perm(a,s)
    summ={"split":split,"n":n,"gate":gates[split],
          "A":{"R@1":round(r1(Aidx),3),"Recall@5":round(rec5(Aidx),3),"MRR":round(mrr(Aidx),3),"nDCG@10":round(ndm(Aidx),3),"mean_gold_rank":round(mgr(Aidx),2)},
          "D":{"R@1":round(r1(Didx),3),"Recall@5":round(rec5(Didx),3),"MRR":round(mrr(Didx),3),"nDCG@10":round(ndm(Didx),3),"mean_gold_rank":round(mgr(Didx),2)},
          "fixed":fixed,"broke":broke,"break_rate_among_correct":round(broke/corr,3) if corr else 0.0,
          "mcnemar_p":round(mcnemar(fixed,broke),4),"MRR_delta":dlt,"MRR_CI":[lo,hi],"MRR_perm_p":pp,
          "decisions":decisions}
    (ROOT/"runs"/"splits"/split/"summary.json").write_text(json.dumps(summ,indent=1))
    results[split]=summ
    print(f"{split:<20} A R@1={summ['A']['R@1']} D R@1={summ['D']['R@1']} fix={fixed} brk={broke} McN={summ['mcnemar_p']} MRRp={pp}")

(ROOT/"runs"/"splits"/"ledger.json").write_text(json.dumps(ledger,indent=1))
print(f"\nTOTAL spend this run = ${ledger['cost']:.4f} over {ledger['calls']} live calls (cached calls free).")
print("wrote per-split pools/gate/A/D/summary under runs/splits/<split>/")
