"""Backbone robustness for the BRIGHT anchor-and-edit finding. For each model given on the
command line, runs that model's OWN single-pass baseline A and anchor-and-edit D over the
IDENTICAL frozen 100-doc pools, and reports the paired D-vs-A comparison. Reuses the verbatim
frozen prompts. Writes to results/bright_backbones/<model>/ -- the frozen gemini-flash-latest
record is NOT touched. Usage: python bright_backbone.py <model1> <model2> ...
"""
from __future__ import annotations
import sys, json, math, time, random, statistics as st
from pathlib import Path
from rq2_core import make_client, extract_json_object, usage_cost_usd
import os

class RawClaude:
    """Schema-free Claude caller (bypasses rq2_core AnthropicClient's forced ranked_call_labels
    schema) so arbitrary JSON prompts work. Same generate()/last_usage_metadata surface."""
    def __init__(self, model):
        import anthropic
        # explicit per-request timeout + built-in retries so a dead socket (e.g. laptop slept
        # mid-request) fails fast and our retry loop can recover instead of blocking forever.
        self._c = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"),
                                      timeout=180.0, max_retries=2)
        self.model_name = model; self.last_usage_metadata = {}; self.last_model = None; self.temperature = 0
    def generate(self, prompt):
        import time as _t
        kw = {"model": self.model_name, "max_tokens": 8192,
              "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if not self.model_name.startswith("claude-haiku"):
            kw["thinking"] = {"type": "disabled"}
        last = None
        for attempt in range(6):
            try:
                r = self._c.messages.create(**kw)
                self.last_model = getattr(r, "model", None)
                u = getattr(r, "usage", None)
                if u is not None:
                    self.last_usage_metadata = {"prompt_token_count": getattr(u,"input_tokens",0),
                                                "candidates_token_count": getattr(u,"output_tokens",0)}
                return "".join(b.text for b in getattr(r,"content",[]) if getattr(b,"type",None)=="text")
            except Exception as e:
                msg = str(e).lower()
                if "temperature" in msg and "temperature" in kw:   # model rejects temperature -> drop it, retry now
                    kw.pop("temperature"); self.temperature = None; continue
                last = e; _t.sleep(5*(attempt+1))
        raise last

def client_for(model):
    return RawClaude(model) if model.startswith("claude") else make_client(model)

ROOT = Path(__file__).resolve().parents[1]
POOLS = json.loads((ROOT/"data"/"bright_hardpool"/"biology"/"pools.json").read_text())
MODELS = sys.argv[1:] or ["gemini-2.5-flash-lite"]


def dcg(r): return sum(x/math.log2(i+2) for i,x in enumerate(r))
def ndcg(r,k):
    d=dcg(r[:k]); ide=dcg(sorted(r,reverse=True)[:k]); return d/ide if ide else 0.0
def mcnemar(f,b):
    n=f+b
    if n==0: return 1.0
    k=min(f,b); return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))
def perm_mrr(a,s,B=5000,seed=123):
    diffs=[si-ai for ai,si in zip(a,s)]; obs=st.mean(diffs); n=len(diffs); rng=random.Random(seed); c=0
    for _ in range(B):
        m=sum((d if rng.random()<0.5 else -d) for d in diffs)/n
        if abs(m)>=abs(obs)-1e-12: c+=1
    return round(obs,4),(c+1)/(B+1)


def p_rerank_plain(q, body):
    return f"""You are given a question and {len(body)} candidate documents (opaque
ids). Rank ALL ids from most to least relevant for answering the question. Return
JSON only.

Schema: {{"ranked_ids": ["<id>", ...]}}

Question: {q[:1500]}

Documents:
{json.dumps(body, ensure_ascii=False)}
"""
def p_a1(q):
    return f"""You are the SURFACE/LEXICAL hypothesis agent. Describe what the question
LITERALLY asks -- the surface reading, terms an answer doc would contain, answer type,
and any hard constraints. Do NOT abstract to deeper principles. Return JSON only.
Schema: {{"surface_intent":"<one sentence>","key_terms":["..."],"expected_answer_type":"<...>","explicit_constraints":["..."]}}
Question: {q[:1500]}"""
def p_a2(q):
    return f"""You are the LATENT CONCEPT hypothesis agent. Identify the underlying concept(s)
that actually answer the question -- mechanism, principle, compound, or law -- NOT the
surface topic. Alternatives are first-class. Return JSON only.
Schema: {{"primary_concept":"<...>","reasoning":"<1-2 sentences>","concept_terms":["..."],"alternatives":[{{"concept":"<...>","confidence":"high|medium|low"}}]}}
Question: {q[:1500]}"""
def p_a3_anchor(q,h1,h2,c):
    return f"""You are the COORDINATOR in an ANCHOR-AND-EDIT retrieval system. A single-pass
baseline has ALREADY produced a good ranking of the candidates (you do not see it). Your
job is NOT to re-rank everything. Your job is to decide whether the baseline's surface
reading suffices, and if not, to name the FEW documents that the correct latent reading
promotes to the top.

Two competing hypotheses (neither privileged):
  SURFACE/LEXICAL: {json.dumps(h1, ensure_ascii=False)}
  LATENT CONCEPT : {json.dumps(h2, ensure_ascii=False)}

Decide:
- decision="surface_sufficient": the answer is the document a literal/lexical reading
  would pick; the baseline is trusted UNCHANGED. Use this whenever the surface reading is
  adequate -- do not disturb an already-good ranking. promote_ids MUST be [].
- decision="concept_decisive": the answer is a conceptually-distant document that only the
  latent concept (or one of its alternatives) explains. Put in promote_ids the 1-2
  candidate ids that best embody that concept, best first.
- decision="mixed": ambiguous; promote_ids may hold 1-2 ids you are confident belong at top.

Reason from the EVIDENCE in the candidate texts. Do NOT list more than 2 promote_ids.
Return JSON only.
Schema: {{"decision":"surface_sufficient|concept_decisive|mixed","promote_ids":["<id>"],"rationale":"<short, evidence-based>"}}

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


def run_model(MODEL):
    OUT=ROOT/"results"/"bright_backbones"/MODEL.replace("/","_"); OUT.mkdir(parents=True,exist_ok=True)
    client=client_for(MODEL)
    def call(prompt):
        t=time.perf_counter(); raw=client.generate(prompt); dt=time.perf_counter()-t
        u=client.last_usage_metadata or {}
        m={"lat":dt,"in":u.get("prompt_token_count") or 0,"out":u.get("candidates_token_count") or 0,"cost":usage_cost_usd(MODEL,u) or 0.0}
        try: o=extract_json_object(raw)
        except Exception: o={}
        return o,m
    cost={"A":0.0,"D":0.0}; lat={"A":0.0,"D":0.0}; calls={"A":0,"D":0}
    a_r1=a_rec5=0; a_mrr=[]; a_ndcg=[]; a_first={}
    d_r1=d_rec5=0; d_mrr=[]; d_ndcg=[]
    fixed=[]; broke=[]; decisions={}; per=[]
    for qid,pl in POOLS.items():
        q=pl["query"]; a2t=pl["alias_to_text"]; goldset=set(pl["gold_aliases"]); order=list(a2t)
        body=[{"id":a,"text":a2t[a]} for a in order]
        # A baseline
        o,m=call(p_rerank_plain(q,body)); cost["A"]+=m["cost"];lat["A"]+=m["lat"];calls["A"]+=1
        a_ranked=rank_from_ids(o.get("ranked_ids",[]),order)
        a_rel=[1 if x in goldset else 0 for x in a_ranked]
        af=next((i+1 for i,x in enumerate(a_ranked) if x in goldset),len(a_ranked)+1)
        a_r1+=a_rel[0]==1; a_rec5+=any(a_rel[:5]); a_mrr.append(1/af); a_ndcg.append(ndcg(a_rel,10)); a_first[qid]=af
        a_top1_gold=a_rel[0]==1
        # D anchor-edit
        h1,m1=call(p_a1(q)); h2,m2=call(p_a2(q)); co,m3=call(p_a3_anchor(q,h1,h2,body))
        for m in (m1,m2,m3): cost["D"]+=m["cost"];lat["D"]+=m["lat"];calls["D"]+=1
        dec=(co.get("decision") or "surface_sufficient")
        promote=[str(x.get("id") if isinstance(x,dict) else x) for x in (co.get("promote_ids") or [])]
        promote=[a for a in promote if a in a2t][:2]
        if dec=="surface_sufficient" or not promote:
            d_ranked=list(a_ranked)
        else:
            d_ranked=promote+[a for a in a_ranked if a not in promote]
        decisions[dec]=decisions.get(dec,0)+1
        d_rel=[1 if x in goldset else 0 for x in d_ranked]
        df=next((i+1 for i,x in enumerate(d_ranked) if x in goldset),len(d_ranked)+1)
        d_r1+=d_rel[0]==1; d_rec5+=any(d_rel[:5]); d_mrr.append(1/df); d_ndcg.append(ndcg(d_rel,10))
        d_top1_gold=d_rel[0]==1
        if (not a_top1_gold) and d_top1_gold: fixed.append(qid)
        if a_top1_gold and (not d_top1_gold): broke.append(qid)
        per.append({"id":qid,"decision":dec,"promote":promote,"A_first_gold":af,"D_first_gold":df,
                    "A_top1_gold":a_top1_gold,"D_top1_gold":d_top1_gold})
    n=len(POOLS)
    p_mc=mcnemar(len(fixed),len(broke))
    mrr_delta,mrr_p=perm_mrr(a_mrr,d_mrr)
    summ={"model":MODEL,"n":n,"pool":100,
        "A":{"R@1":round(a_r1/n,3),"Recall@5":round(a_rec5/n,3),"MRR":round(st.mean(a_mrr),3),"nDCG@10":round(st.mean(a_ndcg),3)},
        "D":{"R@1":round(d_r1/n,3),"Recall@5":round(d_rec5/n,3),"MRR":round(st.mean(d_mrr),3),"nDCG@10":round(st.mean(d_ndcg),3)},
        "fixed":len(fixed),"broke":len(broke),"mcnemar_p_R@1":round(p_mc,4),
        "MRR_delta":mrr_delta,"MRR_perm_p":round(mrr_p,4),
        "decisions":decisions,"break_rate_baseline_correct":round(len(broke)/max(1,a_r1),3),
        "cost_usd":{"A":round(cost["A"],4),"D":round(cost["D"],4),"total":round(cost["A"]+cost["D"],4)},
        "latency_s":{"A":round(lat["A"],1),"D":round(lat["D"],1)},"calls":calls}
    (OUT/"summary.json").write_text(json.dumps(summ,indent=1))
    (OUT/"per_query.json").write_text(json.dumps(per,indent=1))
    print(f"### {MODEL}  (n={n})")
    print(f"   A: R@1={summ['A']['R@1']} MRR={summ['A']['MRR']} nDCG@10={summ['A']['nDCG@10']}")
    print(f"   D: R@1={summ['D']['R@1']} MRR={summ['D']['MRR']} nDCG@10={summ['D']['nDCG@10']}")
    print(f"   fixed={len(fixed)} broke={len(broke)} McNemar_p={p_mc:.4f} | MRR Δ={mrr_delta:+.4f} perm_p={mrr_p:.4f}")
    print(f"   decisions={decisions} | cost=${summ['cost_usd']['total']} | calls={calls}")
    return summ


def main():
    alls=[]
    for M in MODELS:
        print(f"\n===== running backbone: {M} =====")
        try: alls.append(run_model(M))
        except Exception as e:
            print(f"   FAILED {M}: {e}")
    (ROOT/"results"/"bright_backbones"/"panel_summary.json").write_text(json.dumps(alls,indent=1))
    print("\n===== PANEL SUMMARY (D vs A per backbone) =====")
    print(f"{'model':<26}{'A R@1':>7}{'D R@1':>7}{'A MRR':>8}{'D MRR':>8}{'fix':>5}{'brk':>5}{'McN_p':>8}{'MRRp':>8}")
    # include frozen reference
    ref={"model":"gemini-flash-latest (frozen ref)","A":{"R@1":0.54,"MRR":0.646},"D":{"R@1":0.61,"MRR":0.708},"fixed":10,"broke":3,"mcnemar_p_R@1":0.092,"MRR_perm_p":0.0076}
    for s in [ref]+alls:
        print(f"{s['model']:<26}{s['A']['R@1']:>7}{s['D']['R@1']:>7}{s['A']['MRR']:>8}{s['D']['MRR']:>8}{s['fixed']:>5}{s['broke']:>5}{s['mcnemar_p_R@1']:>8}{s['MRR_perm_p']:>8}")


if __name__ == "__main__":
    main()
