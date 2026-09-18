"""Study 1 component ablation on the PINNED primary backbone (claude-haiku-4-5-20251001,
BRIGHT biology, pool 100, temp 0), pre-registered re-run.

Arms on MATCHED queries with a shared anchor A:
  A          : single-pass baseline (the anchor)
  full       : A1 + A2 + anchor-coordinator (promote onto A)   [= System D]
  no_surface : A2 only + anchor
  no_concept : A1 only + anchor
  no_anchor  : A1 + A2 + full-rerank coordinator (regenerate; no anchor)

Persists per query: FULL rankings (ids) for every arm, raw model responses, decision/
rationale/promote_ids, first-gold rank, top1. Persists per arm: cost/latency/tokens/calls.
Direct comparisons (pre-registered, reported regardless of significance): exact McNemar on
Hit@1 for full-D vs each of {no_anchor, no_concept, no_surface}, plus MRR delta with a
query-level bootstrap 95% CI. k reps (default 5). Output: results/bright_ablation/<MODEL>/rep_<k>/.
"""
from __future__ import annotations
import os, sys, json, math, time, random, argparse, statistics as st
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bright_backbone import POOLS, p_rerank_plain, client_for, usage_cost_usd, ROOT
from rq2_core import extract_json_object
for _e in (ROOT/".env", ROOT.parent/".env"):
    if _e.exists():
        for _l in _e.read_text().splitlines():
            _l=_l.strip()
            if _l and not _l.startswith("#") and "=" in _l:
                _k,_v=_l.split("=",1); os.environ.setdefault(_k.strip(),_v.strip().strip('"').strip("'"))

MODEL="claude-haiku-4-5-20251001"; PRICE=(1.00,5.00)  # $/Mtok in,out
ARMS=["A","full","no_surface","no_concept","no_anchor"]
ABL=["no_anchor","no_concept","no_surface"]   # compared directly to full


def ndcg(r,k):
    dcg=sum(x/math.log2(i+2) for i,x in enumerate(r[:k]))
    ide=sum(x/math.log2(i+2) for i,x in enumerate(sorted(r,reverse=True)[:k]))
    return dcg/ide if ide else 0.0
def mcnemar(f,b):
    n=f+b; return 1.0 if n==0 else min(1.0,2*sum(math.comb(n,i) for i in range(min(f,b)+1))/2**n)
def boot_ci(diffs,B=10000,seed=0):
    n=len(diffs); rng=random.Random(seed); bs=sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(B))
    return round(st.mean(diffs),4), round(bs[int(.025*B)],4), round(bs[int(.975*B)],4)

def p_a1(q): return f"""You are the SURFACE/LEXICAL hypothesis agent. Describe what the question
LITERALLY asks -- the surface reading, terms an answer doc would contain, answer type,
and any hard constraints. Do NOT abstract to deeper principles. Return JSON only.
Schema: {{"surface_intent":"<one sentence>","key_terms":["..."],"expected_answer_type":"<...>","explicit_constraints":["..."]}}
Question: {q[:1500]}"""
def p_a2(q): return f"""You are the LATENT CONCEPT hypothesis agent. Identify the underlying concept(s)
that actually answer the question -- mechanism, principle, compound, or law -- NOT the
surface topic. Alternatives are first-class. Return JSON only.
Schema: {{"primary_concept":"<...>","reasoning":"<1-2 sentences>","concept_terms":["..."],"alternatives":[{{"concept":"<...>","confidence":"high|medium|low"}}]}}
Question: {q[:1500]}"""
def p_anchor(q,hyp,c): return f"""You are the COORDINATOR in an ANCHOR-AND-EDIT retrieval system. A single-pass
baseline has ALREADY produced a good ranking of the candidates (you do not see it). Your
job is NOT to re-rank everything. Your job is to decide whether the baseline's surface
reading suffices, and if not, to name the FEW documents that the correct latent reading
promotes to the top.

{hyp}

Decide:
- decision="surface_sufficient": the answer is the document a literal/lexical reading
  would pick; the baseline is trusted UNCHANGED. promote_ids MUST be [].
- decision="concept_decisive": the answer is a conceptually-distant document that only the
  latent concept (or one of its alternatives) explains. Put in promote_ids the 1-2
  candidate ids that best embody that concept, best first.
- decision="mixed": ambiguous; promote_ids may hold 1-2 ids you are confident belong at top.
Reason from the EVIDENCE in the candidate texts. Do NOT list more than 2 promote_ids. Return JSON only.
Schema: {{"decision":"surface_sufficient|concept_decisive|mixed","promote_ids":["<id>"],"rationale":"<short, evidence-based>"}}
Question: {q[:1500]}

Documents:
{json.dumps(c, ensure_ascii=False)}"""
def p_full(q,h1,h2,c): return f"""You are the COORDINATOR: a reasoning arbiter, NOT a rule engine and NOT a
score-averager. You are given the question, TWO competing hypotheses about what it is
really asking, and the actual candidate documents. Decide -- using the EVIDENCE in the
documents -- which reading the answer supports, and produce the final ranking.

Two competing hypotheses (neither is privileged):
  SURFACE/LEXICAL: {json.dumps(h1, ensure_ascii=False)}
  LATENT CONCEPT : {json.dumps(h2, ensure_ascii=False)}

Rules:
- If the documents show the surface reading already identifies the right answer, rank by it.
- If the answer is a conceptually-distant document that only the latent concept explains, rank that to the top.
- Do NOT average. Reason document-by-document for the top of the ranking. Rank ALL {len(c)} ids.
Return JSON only.
Schema: {{"ranked_ids": ["<all ids, most->least relevant>"]}}
Question: {q[:1500]}

Documents:
{json.dumps(c, ensure_ascii=False)}"""

def rank_from_ids(ids, order):
    out=[]
    for x in ids:
        v=str(x.get("id") if isinstance(x,dict) else x)
        if v in order and v not in out: out.append(v)
    for a in order:
        if a not in out: out.append(a)
    return out


def run_rep(cl, rep, outdir):
    acc={a:{"cost":0.0,"lat":0.0,"calls":0,"tin":0,"tout":0} for a in ARMS}
    per=[]
    def call(prompt, key):
        t=time.perf_counter(); raw=cl.generate(prompt); dt=time.perf_counter()-t
        u=cl.last_usage_metadata or {}; a=acc[key]
        a["cost"]+=(u.get("prompt_token_count",0)*PRICE[0]+u.get("candidates_token_count",0)*PRICE[1])/1e6
        a["lat"]+=dt; a["calls"]+=1; a["tin"]+=u.get("prompt_token_count",0); a["tout"]+=u.get("candidates_token_count",0)
        try: o=extract_json_object(raw)
        except Exception: o={}
        return o, raw
    for i,(qid,pl) in enumerate(POOLS.items(),1):
        q=pl["query"]; a2t=pl["alias_to_text"]; goldset=set(pl["gold_aliases"]); order=list(a2t)
        body=[{"id":a,"text":a2t[a]} for a in order]
        oaA,rawA=call(p_rerank_plain(q,body),"A"); A=rank_from_ids(oaA.get("ranked_ids",[]),order)
        o1,raw1=call(p_a1(q),"full"); o2,raw2=call(p_a2(q),"full")
        h1,h2=o1,o2
        def anchor(hyp,key):
            co,raw=call(p_anchor(q,hyp,body),key)
            dec=co.get("decision") or "surface_sufficient"
            prom=[str(x.get("id") if isinstance(x,dict) else x) for x in (co.get("promote_ids") or [])]
            prom=[a for a in prom if a in a2t][:2]
            rk=list(A) if (dec=="surface_sufficient" or not prom) else prom+[a for a in A if a not in prom]
            return rk,dec,prom,(co.get("rationale") or "")[:500],raw
        both=(f"Hypothesis about what the question is really asking:\n  SURFACE/LEXICAL: {json.dumps(h1,ensure_ascii=False)}\n  LATENT CONCEPT : {json.dumps(h2,ensure_ascii=False)}")
        surf=f"Hypothesis about what the question is really asking:\n  SURFACE/LEXICAL: {json.dumps(h1,ensure_ascii=False)}"
        conc=f"Hypothesis about what the question is really asking:\n  LATENT CONCEPT : {json.dumps(h2,ensure_ascii=False)}"
        rk={}
        rk["A"]=A
        fr=anchor(both,"full"); rk["full"]=fr[0]
        nsr=anchor(conc,"no_surface"); rk["no_surface"]=nsr[0]
        ncr=anchor(surf,"no_concept"); rk["no_concept"]=ncr[0]
        co,rawNA=call(p_full(q,h1,h2,body),"no_anchor"); rk["no_anchor"]=rank_from_ids(co.get("ranked_ids",[]),order)
        rec={"id":qid,"gold":list(goldset)}
        for a in ARMS:
            r=rk[a]; rels=[1 if x in goldset else 0 for x in r]
            first=next((j+1 for j,x in enumerate(r) if x in goldset),len(r)+1)
            rec[a]={"ranked":r[:20],"top1":r[0],"top1_gold":rels[0]==1,"first_gold":first,
                    "mrr":1.0/first,"ndcg10":ndcg(rels,10)}
        rec["full_decision"]=fr[1]; rec["full_promote"]=fr[2]; rec["full_rationale"]=fr[3]
        rec["raw"]={"A":rawA[:1500],"h1":raw1[:800],"h2":raw2[:800],"full":fr[4][:1500],"no_anchor":rawNA[:1500]}
        per.append(rec)
        print(f"[rep{rep} {i}/{len(POOLS)}] {qid}",flush=True)
    # metrics + direct comparisons
    def summ(a):
        return {m:round(st.mean(x[a][k] for x in per),4) for m,k in [("Hit@1","top1_gold"),("MRR","mrr"),("nDCG@10","ndcg10")]}
    comps={}
    for a in ABL:
        f=sum(1 for x in per if x["full"]["top1_gold"] and not x[a]["top1_gold"])   # full right, abl wrong
        b=sum(1 for x in per if not x["full"]["top1_gold"] and x[a]["top1_gold"])   # abl right, full wrong
        mrrd=[x["full"]["mrr"]-x[a]["mrr"] for x in per]
        d,lo,hi=boot_ci(mrrd)
        comps[f"full_vs_{a}"]={"full>abl":f,"abl>full":b,"mcnemar_p":round(mcnemar(f,b),4),
                               "MRR_delta":d,"MRR_CI":[lo,hi]}
    out={"model":MODEL,"resolved_model":getattr(cl,"last_model",None),"temperature":0,"pool":100,
         "domain":"biology","n":len(per),"rep":rep,"run_utc":datetime.now(timezone.utc).isoformat(),
         "metrics":{a:summ(a) for a in ARMS},"cost":{a:{**acc[a],"cost":round(acc[a]["cost"],4)} for a in ARMS},
         "full_vs_ablation":comps}
    outdir.mkdir(parents=True,exist_ok=True)
    json.dump(out, open(outdir/"ablation.json","w"), indent=1)
    json.dump(per, open(outdir/"per_query.json","w"), indent=1, ensure_ascii=False)
    print(f"  rep{rep}: "+" ".join(f"{a}={out['metrics'][a]['Hit@1']}" for a in ARMS))
    for a in ABL:
        c=comps[f"full_vs_{a}"]; print(f"    full vs {a}: {c['full>abl']}/{c['abl>full']} p={c['mcnemar_p']} ΔMRR={c['MRR_delta']} CI{c['MRR_CI']}")
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--reps",type=int,default=5); args=ap.parse_args()
    cl=client_for(MODEL)
    base=ROOT/"results"/"bright_ablation"/MODEL
    reps=[]
    for k in range(1,args.reps+1):
        od=base/f"rep_{k}"
        if (od/"ablation.json").exists():
            reps.append(json.load(open(od/"ablation.json"))); print(f"rep{k} exists, skip"); continue
        reps.append(run_rep(cl,k,od))
    # aggregate direct comparisons across reps
    print(f"\n=== ABLATION AGG ({len(reps)} reps) — full-D vs each ablation ===")
    for a in ABL:
        sig=sum(1 for r in reps if r["full_vs_ablation"][f"full_vs_{a}"]["mcnemar_p"]<0.05)
        dd=st.mean(r["full_vs_ablation"][f"full_vs_{a}"]["MRR_delta"] for r in reps)
        print(f"  full vs {a}: McNemar sig {sig}/{len(reps)}; mean ΔMRR {dd:+.4f}")
    for a in ARMS:
        print(f"  {a}: Hit@1 {st.mean(r['metrics'][a]['Hit@1'] for r in reps):.3f}")


if __name__=="__main__":
    main()
