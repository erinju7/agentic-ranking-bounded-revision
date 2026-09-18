"""FINALIZE (no optimisation): (1) evaluate System B (Concept-guided reranking) on the
SAME frozen 100-doc pools as A/C/D so the comparison uses identical inputs; (2) measure
real per-query latency and cost (real token usage) for A,B,C,D on a shared 12-query probe.
All prompts copied verbatim from the frozen system scripts. Nothing is tuned.
"""
from __future__ import annotations
import json, math, time, statistics as st
from pathlib import Path
from rq2_core import make_client, extract_json_object, usage_cost_usd

ROOT = Path(__file__).resolve().parents[1]
POOLS = json.loads((ROOT/"data"/"bright_hardpool"/"biology"/"pools.json").read_text())
BASE = {str(p["id"]): p for p in json.loads((ROOT/"results"/"bright_hardpool"/"biology"/"baseline_per_query.json").read_text())}
OUTB = ROOT/"results"/"bright_concept_hardpool"/"biology"; OUTB.mkdir(parents=True, exist_ok=True)
OUTM = ROOT/"results"/"bright_finalize"; OUTM.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-flash-latest"


def dcg(r): return sum(x/math.log2(i+2) for i,x in enumerate(r))
def ndcg(r,k):
    d=dcg(r[:k]); ide=dcg(sorted(r,reverse=True)[:k]); return d/ide if ide else 0.0
def mcnemar(f,b):
    n=f+b
    if n==0: return 1.0
    k=min(f,b); return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))
def rank_from_ids(ids, order):
    out=[]
    for x in ids:
        v=str(x.get("id") if isinstance(x,dict) else x)
        if v in order and v not in out: out.append(v)
    for a in order:
        if a not in out: out.append(a)
    return out

# ---------- verbatim prompts ----------
def p_rerank_plain(q, body):
    return f"""You are given a question and {len(body)} candidate documents (opaque
ids). Rank ALL ids from most to least relevant for answering the question. Return
JSON only.

Schema: {{"ranked_ids": ["<id>", ...]}}

Question: {q[:1500]}

Documents:
{json.dumps(body, ensure_ascii=False)}
"""
def p_concept_B(q):
    return f"""You are a scientific concept-abstraction agent. Read the question and
identify the LATENT scientific concept(s) that actually answer it -- the underlying
mechanism, principle, compound, or law -- NOT merely the surface topic the question
mentions. Give concise reasoning. If genuinely uncertain, give one lower-confidence
alternative concept. Return JSON only.

Schema: {{"concept": "<the core latent concept(s)>",
          "reasoning": "<1-2 sentences: why this concept, not the surface topic>",
          "alternative": "<optional alternative concept, or empty string>"}}

Question: {q[:1500]}
"""
def p_rerank_B(q, body, ca):
    cb=f"""\nLatent concept hypothesis (from a concept-abstraction step; treat as a
strong hint about what the question is really about, but rank by the documents):
  concept: {ca.get('concept','')}
  reasoning: {ca.get('reasoning','')}
  alternative: {ca.get('alternative','')}\n"""
    return f"""You are given a question and {len(body)} candidate documents (opaque
ids). Rank ALL ids from most to least relevant for answering the question. Return
JSON only.

Schema: {{"ranked_ids": ["<id>", ...]}}
{cb}
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
def p_a3_full(q,h1,h2,c):
    return f"""You are the COORDINATOR: a reasoning arbiter, NOT a rule engine and NOT a
score-averager. You are given the question, TWO competing hypotheses about what it is
really asking, and the actual candidate documents. Decide -- using the EVIDENCE in the
documents -- which reading the answer supports, and produce the final ranking.

Two competing hypotheses (neither is privileged):
  SURFACE/LEXICAL: {json.dumps(h1, ensure_ascii=False)}
  LATENT CONCEPT : {json.dumps(h2, ensure_ascii=False)}

Rules:
- If the documents show the surface reading already identifies the right answer, set
  decision="surface_sufficient" and rank by the surface reading -- do NOT let the latent
  concept perturb an answer that is already correct.
- If the answer is a conceptually-distant document that only the latent concept (or one
  of its alternatives) explains, set decision="concept_decisive".
- Otherwise decision="mixed". Treat the concept's alternatives as live competitors.
- Do NOT average. Reason document-by-document for the top of the ranking.
- Rank ALL {len(c)} ids.

Return JSON only.
Schema: {{"decision": "surface_sufficient | concept_decisive | mixed",
          "rationale": "<why, referencing evidence in the candidates>",
          "per_candidate": [{{"id": "<id>", "satisfies": "surface|concept|alternative|both|neither", "note": "<short>"}}],
          "ranked_ids": ["<all ids, most->least relevant>"]}}
(Provide per_candidate for your TOP 10 only; ranked_ids must include ALL ids.)

Question: {q[:1500]}

Documents:
{json.dumps(c, ensure_ascii=False)}"""
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


def call(client, prompt):
    t=time.perf_counter()
    raw=client.generate(prompt)
    dt=time.perf_counter()-t
    cost=usage_cost_usd(MODEL, client.last_usage_metadata) or 0.0
    try: obj=extract_json_object(raw)
    except Exception: obj={}
    return obj, raw, dt, cost


def main():
    client=make_client(MODEL)
    # ---------- Part 1: System B on frozen 100-pool (accuracy) ----------
    r1=rec5=0; mrrs=[]; ndcgs=[]; goldranks=[]; b_r1=0; fixed=[]; broke=[]; perB=[]
    for qid,pl in POOLS.items():
        q=pl["query"]; a2t=pl["alias_to_text"]; goldset=set(pl["gold_aliases"]); order=list(a2t)
        ca,_,_,_=call(client, p_concept_B(q))
        body=[{"id":a,"text":a2t[a]} for a in order]
        obj,_,_,_=call(client, p_rerank_B(q,body,ca))
        ranked=rank_from_ids(obj.get("ranked_ids",[]), order)
        rels=[1 if a in goldset else 0 for a in ranked]
        first=next((i+1 for i,a in enumerate(ranked) if a in goldset), len(ranked)+1)
        r1+=rels[0]==1; rec5+=any(rels[:5]); mrrs.append(1/first)
        ndcgs.append(ndcg(rels,10)); goldranks+=[i+1 for i,a in enumerate(ranked) if a in goldset]
        bt=BASE[qid]["baseline_top1_gold"]; ct=rels[0]==1; b_r1+=bt
        if (not bt) and ct: fixed.append(qid)
        if bt and (not ct): broke.append(qid)
        perB.append({"id":qid,"concept":ca.get("concept",""),"baseline_top1_gold":bt,"arch_top1_gold":ct,
                     "baseline_first_gold_rank":BASE[qid]["baseline_first_gold_rank"],"arch_first_gold_rank":first})
    n=len(POOLS)
    summB={"system":"B_concept_guided_hardpool","n":n,"pool":100,
           "R@1":r1/n,"Recall@5":rec5/n,"MRR":st.mean(mrrs),"nDCG@10":st.mean(ndcgs),"mean_gold_rank":st.mean(goldranks),
           "n_fixed":len(fixed),"n_broke":len(broke),"fixed":fixed,"broke":broke,"mcnemar_p":mcnemar(len(fixed),len(broke))}
    (OUTB/"per_query.json").write_text(json.dumps(perB,indent=1,ensure_ascii=False))
    (OUTB/"summary.json").write_text(json.dumps(summB,indent=1))
    print("=== System B on frozen 100-pool ===")
    print(f"  R@1={r1/n:.2f} Recall@5={rec5/n:.2f} MRR={st.mean(mrrs):.3f} nDCG@10={st.mean(ndcgs):.3f} mean_gold_rank={st.mean(goldranks):.2f} fixed={len(fixed)} broke={len(broke)}")

    # ---------- Part 2: latency + real cost probe (shared 12-query subset) ----------
    probe_ids=list(POOLS)[:12]
    lat={s:[] for s in "ABCD"}; cost={s:[] for s in "ABCD"}
    for qid in probe_ids:
        pl=POOLS[qid]; q=pl["query"]; a2t=pl["alias_to_text"]; order=list(a2t)
        body=[{"id":a,"text":a2t[a]} for a in order]
        # A: 1 call
        _,_,dt,c=call(client,p_rerank_plain(q,body)); lat["A"].append(dt); cost["A"].append(c)
        # B: 2 calls
        ca,_,dt1,c1=call(client,p_concept_B(q)); _,_,dt2,c2=call(client,p_rerank_B(q,body,ca))
        lat["B"].append(dt1+dt2); cost["B"].append(c1+c2)
        # C: 3 calls (A1,A2,A3-full)
        h1,_,d1,x1=call(client,p_a1(q)); h2,_,d2,x2=call(client,p_a2(q)); _,_,d3,x3=call(client,p_a3_full(q,h1,h2,body))
        lat["C"].append(d1+d2+d3); cost["C"].append(x1+x2+x3)
        # D: 3 calls (A1,A2,A3-anchor)
        h1,_,d1,x1=call(client,p_a1(q)); h2,_,d2,x2=call(client,p_a2(q)); _,_,d3,x3=call(client,p_a3_anchor(q,h1,h2,body))
        lat["D"].append(d1+d2+d3); cost["D"].append(x1+x2+x3)
    meas={s:{"calls_per_query":{"A":1,"B":2,"C":3,"D":3}[s],
             "latency_s_per_query":round(st.mean(lat[s]),2),
             "cost_usd_per_query":round(st.mean(cost[s]),5)} for s in "ABCD"}
    (OUTM/"cost_latency.json").write_text(json.dumps({"probe_n":len(probe_ids),"measured":meas},indent=1))
    print("\n=== latency + real cost (12-query probe) ===")
    for s in "ABCD":
        print(f"  {s}: calls/q={meas[s]['calls_per_query']}  latency/q={meas[s]['latency_s_per_query']}s  cost/q=${meas[s]['cost_usd_per_query']}")
    print(f"\nwrote {OUTB/'summary.json'} and {OUTM/'cost_latency.json'}")


if __name__ == "__main__":
    main()
