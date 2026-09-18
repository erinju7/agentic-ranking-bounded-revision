"""Study 1c: controlled ablation of the final Anchor-and-Edit system (System D) on BRIGHT
biology (frozen 100-pool, n=97, gemini-flash-latest, temp 0). Every variant uses System D's
EXACT Surface (A1) and Concept (A2) prompts; only the named component is removed:
  A (full)      : reused frozen from results/bright_ch_anchor (NOT rerun).
  B (no surface): A2 + anchor-coordinator receiving ONLY the concept hypothesis; anchor kept.
  C (no concept): A1 + anchor-coordinator receiving ONLY the surface hypothesis; anchor kept.
  D (no anchor) : A1 + A2 + full-rerank coordinator (regenerates ranking; NO baseline anchor).
                  NOT the frozen CH-v1: v1's A1/A2 wording differs, which would confound the
                  ablation; here A1/A2 are D's exact prompts so only the anchor is removed.
Reference for paired stats = the frozen single-pass baseline. No prompt/param/pool/retriever
changes. Computes R@1/Recall@5/MRR/nDCG@10/mean-gold-rank, fixed/broke, McNemar, MRR bootstrap
CI + permutation, latency/tokens/cost. Frozen A/D-full cost from the cost-profiling replication.
"""
from __future__ import annotations
import json, math, time, random, statistics as st
from pathlib import Path
from rq2_core import make_client, extract_json_object, usage_cost_usd

ROOT = Path(__file__).resolve().parents[1]
POOLS = json.loads((ROOT/"data"/"bright_hardpool"/"biology"/"pools.json").read_text())
BASE = {str(p["id"]): p for p in json.loads((ROOT/"results"/"bright_hardpool"/"biology"/"baseline_per_query.json").read_text())}
A_FULL = {str(p["id"]): p for p in json.loads((ROOT/"results"/"bright_ch_anchor"/"biology"/"per_query.json").read_text())}
A_SUMM = json.loads((ROOT/"results"/"bright_ch_anchor"/"biology"/"summary.json").read_text())["architecture"]
COSTP = json.loads((ROOT/"results"/"bright_finalize"/"cost_profile.json").read_text())["systems"]
OUT = ROOT/"results"/"bright_ablation"; OUT.mkdir(parents=True, exist_ok=True)
MODEL="gemini-flash-latest"


def dcg(r): return sum(x/math.log2(i+2) for i,x in enumerate(r))
def ndcg(r,k):
    d=dcg(r[:k]); ide=dcg(sorted(r,reverse=True)[:k]); return d/ide if ide else 0.0
def mcnemar(f,b):
    n=f+b
    if n==0: return 1.0
    k=min(f,b); return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))
def boot_perm(a,s,B=10000,seed=123):
    diffs=[si-ai for ai,si in zip(a,s)]; obs=st.mean(diffs); n=len(diffs); rng=random.Random(seed)
    means=[]
    for _ in range(B):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n))/n)
    means.sort(); lo=means[int(0.025*B)]; hi=means[int(0.975*B)]
    rng2=random.Random(seed+1); c=0
    for _ in range(B):
        m=sum((d if rng2.random()<0.5 else -d) for d in diffs)/n
        if abs(m)>=abs(obs)-1e-12: c+=1
    return round(obs,4),round(lo,4),round(hi,4),round((c+1)/(B+1),4)


# ---- verbatim final-D prompts ----
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

def p_a3_anchor(q, hyp_block, c):
    # verbatim final anchor coordinator; only the hypotheses block varies (component removal)
    return f"""You are the COORDINATOR in an ANCHOR-AND-EDIT retrieval system. A single-pass
baseline has ALREADY produced a good ranking of the candidates (you do not see it). Your
job is NOT to re-rank everything. Your job is to decide whether the baseline's surface
reading suffices, and if not, to name the FEW documents that the correct latent reading
promotes to the top.

{hyp_block}

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

def p_a3_fullrerank(q,h1,h2,c):
    # no-anchor coordinator: regenerates the full ranking (verbatim CH-v1 coordinator)
    return f"""You are the COORDINATOR: a reasoning arbiter, NOT a rule engine and NOT a
score-averager. You are given the question, TWO competing hypotheses about what it is
really asking, and the actual candidate documents. Decide -- using the EVIDENCE in the
documents -- which reading the answer supports, and produce the final ranking.

Two competing hypotheses (neither is privileged):
  SURFACE/LEXICAL: {json.dumps(h1, ensure_ascii=False)}
  LATENT CONCEPT : {json.dumps(h2, ensure_ascii=False)}

Rules:
- If the documents show the surface reading already identifies the right answer, rank by it.
- If the answer is a conceptually-distant document that only the latent concept (or one of
  its alternatives) explains, rank that to the top.
- Do NOT average. Reason document-by-document for the top of the ranking.
- Rank ALL {len(c)} ids.

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


def run_variant(client, kind):
    cost=lat=0.0; calls=0; tin=tout=0
    per=[]; r1=rec5=0; mrrs=[]; ndcgs=[]; goldranks=[]; decisions={}
    def call(prompt):
        nonlocal cost,lat,calls,tin,tout
        t=time.perf_counter(); raw=client.generate(prompt); dt=time.perf_counter()-t
        u=client.last_usage_metadata or {}
        cost+=usage_cost_usd(MODEL,u) or 0.0; lat+=dt; calls+=1
        tin+=u.get("prompt_token_count") or 0; tout+=u.get("candidates_token_count") or 0
        try: return extract_json_object(raw)
        except Exception: return {}
    for qid,pl in POOLS.items():
        q=pl["query"]; a2t=pl["alias_to_text"]; goldset=set(pl["gold_aliases"]); order=list(a2t)
        body=[{"id":a,"text":a2t[a]} for a in order]
        base_ranked=BASE[qid]["baseline_ranked"]
        if kind=="no_anchor":
            h1=call(p_a1(q)); h2=call(p_a2(q)); co=call(p_a3_fullrerank(q,h1,h2,body))
            ranked=rank_from_ids(co.get("ranked_ids",[]),order); dec="regenerate"
        else:
            if kind=="no_surface":
                h2=call(p_a2(q))
                hyp=f"Hypothesis about what the question is really asking:\n  LATENT CONCEPT : {json.dumps(h2, ensure_ascii=False)}"
            else:  # no_concept
                h1=call(p_a1(q))
                hyp=f"Hypothesis about what the question is really asking:\n  SURFACE/LEXICAL: {json.dumps(h1, ensure_ascii=False)}"
            co=call(p_a3_anchor(q,hyp,body))
            dec=(co.get("decision") or "surface_sufficient")
            promote=[str(x.get("id") if isinstance(x,dict) else x) for x in (co.get("promote_ids") or [])]
            promote=[a for a in promote if a in a2t][:2]
            ranked=list(base_ranked) if (dec=="surface_sufficient" or not promote) else promote+[a for a in base_ranked if a not in promote]
        decisions[dec]=decisions.get(dec,0)+1
        rels=[1 if x in goldset else 0 for x in ranked]
        first=next((i+1 for i,x in enumerate(ranked) if x in goldset),len(ranked)+1)
        r1+=rels[0]==1; rec5+=any(rels[:5]); mrrs.append(1/first); ndcgs.append(ndcg(rels,10))
        goldranks+=[i+1 for i,x in enumerate(ranked) if x in goldset]
        per.append({"id":qid,"first_gold_rank":first,"top1_gold":rels[0]==1})
    n=len(POOLS)
    return {"per":per,"decisions":decisions,
            "metrics":{"R@1":round(r1/n,3),"Recall@5":round(rec5/n,3),"MRR":round(st.mean(mrrs),3),
                       "nDCG@10":round(st.mean(ndcgs),3),"mean_gold_rank":round(st.mean(goldranks),2)},
            "cost":{"total_usd":round(cost,4),"per_query":round(cost/n,6),"latency_s":round(lat,1),
                    "latency_per_query":round(lat/n,2),"tokens_in":tin,"tokens_out":tout,"calls":calls}}


def paired(variant_per, kind_cost, name):
    ids=list(BASE)
    a=[1.0/BASE[i]["baseline_first_gold_rank"] for i in ids]
    vper={x["id"]:x for x in variant_per}
    s=[1.0/vper[i]["first_gold_rank"] for i in ids]
    fixed=sum(1 for i in ids if (not BASE[i]["baseline_top1_gold"]) and vper[i]["top1_gold"])
    broke=sum(1 for i in ids if BASE[i]["baseline_top1_gold"] and (not vper[i]["top1_gold"]))
    d,lo,hi,pp=boot_perm(a,s)
    return {"fixed":fixed,"broke":broke,"mcnemar_p":round(mcnemar(fixed,broke),4),
            "MRR_delta":d,"MRR_CI":[lo,hi],"MRR_perm_p":pp}


def main():
    client=make_client(MODEL)
    # Variant A (full) reused frozen
    a_ids=list(BASE)
    aA=[1.0/BASE[i]["baseline_first_gold_rank"] for i in a_ids]
    sA=[1.0/A_FULL[i]["arch_first_gold_rank"] for i in a_ids]
    fixedA=sum(1 for i in a_ids if (not BASE[i]["baseline_top1_gold"]) and A_FULL[i]["arch_top1_gold"])
    brokeA=sum(1 for i in a_ids if BASE[i]["baseline_top1_gold"] and (not A_FULL[i]["arch_top1_gold"]))
    dA,loA,hiA,ppA=boot_perm(aA,sA)
    variants={"A_full":{"metrics":A_SUMM,"stats":{"fixed":fixedA,"broke":brokeA,"mcnemar_p":round(mcnemar(fixedA,brokeA),4),
              "MRR_delta":dA,"MRR_CI":[loA,hiA],"MRR_perm_p":ppA},
              "cost":{"per_query":COSTP["D"]["cost_usd_per_query"],"latency_per_query":COSTP["D"]["latency_s_per_query"],
                      "tokens_in":COSTP["D"]["input_tokens_per_query"],"tokens_out":COSTP["D"]["output_tokens_per_query"],"calls":COSTP["D"]["calls_per_query"]},
              "reused":"frozen results/bright_ch_anchor"}}
    for key,kind in [("B_no_surface","no_surface"),("C_no_concept","no_concept"),("D_no_anchor","no_anchor")]:
        print(f"running variant {key} ...")
        r=run_variant(client,kind)
        stats=paired(r["per"],r["cost"],key)
        (OUT/f"{key}_per_query.json").write_text(json.dumps(r["per"],indent=1))
        variants[key]={"metrics":r["metrics"],"stats":stats,"cost":r["cost"],"decisions":r["decisions"]}
    (OUT/"ablation_summary.json").write_text(json.dumps({"reference":"frozen single-pass baseline (R@1 0.54)","baseline":{"R@1":0.536,"MRR":0.646,"nDCG@10":0.646,"mean_gold_rank":11.01},"variants":variants},indent=1))

    order=[("A_full","A · Full Anchor-and-Edit"),("B_no_surface","B · No Surface"),("C_no_concept","C · No Concept"),("D_no_anchor","D · No Anchor")]
    print("\n"+"="*118)
    print("STUDY 1c ABLATION (n=97, frozen 100-pool, gemini-flash-latest t=0). Paired vs frozen single-pass baseline (R@1 0.536, MRR 0.646).")
    print("="*118)
    print(f"{'Variant':<28}{'R@1':>6}{'Rec@5':>7}{'MRR':>7}{'nDCG10':>8}{'gRank':>7}{'fix':>5}{'brk':>5}{'McN_p':>8}{'MRRΔ':>8}{'MRR_CI':>18}{'MRRp':>8}{'call':>5}{'$/q':>9}")
    for k,label in order:
        v=variants[k]; m=v["metrics"]; s=v["stats"]; c=v["cost"]
        ci=f"[{s['MRR_CI'][0]:+.3f},{s['MRR_CI'][1]:+.3f}]"
        print(f"{label:<28}{m['R@1']:>6}{m['Recall@5']:>7}{m['MRR']:>7}{m['nDCG@10']:>8}{m['mean_gold_rank']:>7}{s['fixed']:>5}{s['broke']:>5}{s['mcnemar_p']:>8}{s['MRR_delta']:>+8.3f}{ci:>18}{s['MRR_perm_p']:>8}{c['calls']:>5}{c['per_query']:>9.5f}")
    print(f"\nwrote {OUT/'ablation_summary.json'}")


if __name__ == "__main__":
    main()
