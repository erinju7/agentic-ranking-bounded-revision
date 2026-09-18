"""Competing-Hypotheses architecture, ANCHOR-AND-EDIT variant (corrected coordinator).
Fixes the mechanism failure found in bright_competing_hypotheses.py: there the coordinator
regenerated all 100 ranks from scratch, so its 'surface_sufficient' decision was decorative
(23/53 surface_sufficient rankings were perturbed; 5/7 breaks occurred under it).

Here the decision is ENFORCED mechanically:
  - decision == surface_sufficient  -> return the frozen baseline ranking UNCHANGED (protect).
  - else                            -> take promote_ids (>=1 docs the winning reading picks)
                                       and move them to the FRONT of the baseline ranking;
                                       everything else keeps baseline order (surgical edit).
A1/A2 unchanged. Same frozen 100-doc pools, strictly paired. No averaging/gates/debate/loops.
"""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path
from rq2_core import make_client, extract_json_object

ROOT = Path(__file__).resolve().parents[1]
POOLS = ROOT / "data" / "bright_hardpool" / "biology" / "pools.json"
BASE = ROOT / "results" / "bright_hardpool" / "biology" / "baseline_per_query.json"
OUT = ROOT / "results" / "bright_ch_anchor" / "biology"; OUT.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-flash-latest"


def dcg(rels): return sum(r/math.log2(i+2) for i, r in enumerate(rels))
def ndcg(rels, k):
    d = dcg(rels[:k]); ideal = dcg(sorted(rels, reverse=True)[:k]); return d/ideal if ideal else 0.0
def mcnemar_exact(fixed, broke):
    n = fixed + broke
    if n == 0: return 1.0
    k = min(fixed, broke)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k+1)) / (2**n))


def a1_surface(client, q):
    p = f"""You are the SURFACE/LEXICAL hypothesis agent. Describe what the question
LITERALLY asks -- the surface reading, terms an answer doc would contain, answer type,
and any hard constraints. Do NOT abstract to deeper principles. Return JSON only.
Schema: {{"surface_intent":"<one sentence>","key_terms":["..."],"expected_answer_type":"<...>","explicit_constraints":["..."]}}
Question: {q[:1500]}"""
    try: return extract_json_object(client.generate(p))
    except Exception: return {"surface_intent":"","key_terms":[],"expected_answer_type":"","explicit_constraints":[]}


def a2_concept(client, q):
    p = f"""You are the LATENT CONCEPT hypothesis agent. Identify the underlying concept(s)
that actually answer the question -- mechanism, principle, compound, or law -- NOT the
surface topic. Alternatives are first-class. Return JSON only.
Schema: {{"primary_concept":"<...>","reasoning":"<1-2 sentences>","concept_terms":["..."],"alternatives":[{{"concept":"<...>","confidence":"high|medium|low"}}]}}
Question: {q[:1500]}"""
    try: return extract_json_object(client.generate(p))
    except Exception: return {"primary_concept":"","reasoning":"","concept_terms":[],"alternatives":[]}


def a3_anchor(client, q, h1, h2, cands):
    p = f"""You are the COORDINATOR in an ANCHOR-AND-EDIT retrieval system. A single-pass
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
{json.dumps(cands, ensure_ascii=False)}"""
    try: return extract_json_object(client.generate(p))
    except Exception: return {"decision":"surface_sufficient","promote_ids":[],"rationale":""}


def main():
    pools = json.loads(POOLS.read_text())
    base = {str(p["id"]): p for p in json.loads(BASE.read_text())}
    client = make_client(MODEL)
    r1 = rec5 = 0; mrrs=[]; ndcgs=[]; goldranks=[]; b_r1=0
    fixed, broke, per = [], [], []
    decisions = {}
    for qid, pl in pools.items():
        q = pl["query"]; alias_to_text = pl["alias_to_text"]; goldset = set(pl["gold_aliases"])
        alias_order = list(alias_to_text)
        baseline_ranked = base[qid]["baseline_ranked"]
        h1 = a1_surface(client, q); h2 = a2_concept(client, q)
        cands = [{"id": a, "text": alias_to_text[a]} for a in alias_order]
        co = a3_anchor(client, q, h1, h2, cands)
        dec = (co.get("decision") or "surface_sufficient")
        promote = [str(x.get("id") if isinstance(x, dict) else x) for x in (co.get("promote_ids") or [])]
        promote = [a for a in promote if a in alias_to_text][:2]
        # ENFORCE anchor-and-edit
        if dec == "surface_sufficient" or not promote:
            ranked = list(baseline_ranked)
        else:
            ranked = promote + [a for a in baseline_ranked if a not in promote]
        decisions[dec] = decisions.get(dec, 0) + 1
        rels = [1 if a in goldset else 0 for a in ranked]
        first = next((i+1 for i,a in enumerate(ranked) if a in goldset), len(ranked)+1)
        r1 += rels[0]==1; rec5 += any(rels[:5]); mrrs.append(1/first)
        ndcgs.append(ndcg(rels,10)); goldranks += [i+1 for i,a in enumerate(ranked) if a in goldset]
        bt = base[qid]["baseline_top1_gold"]; ct = rels[0]==1; b_r1 += bt
        if (not bt) and ct: fixed.append(qid)
        if bt and (not ct): broke.append(qid)
        per.append({"id":qid,"query":q[:250],"decision":dec,"promote_ids":promote,
                    "primary_concept":h2.get("primary_concept",""),"baseline_top1_gold":bt,"arch_top1_gold":ct,
                    "baseline_first_gold_rank":base[qid]["baseline_first_gold_rank"],"arch_first_gold_rank":first,
                    "rationale":(co.get("rationale","") or "")[:250]})
    (OUT/"per_query.json").write_text(json.dumps(per, indent=1, ensure_ascii=False))
    n=len(pools)
    bcorr=[p for p in per if p["baseline_top1_gold"]]
    breaks_by_dec={}
    for p in per:
        if p["baseline_top1_gold"] and not p["arch_top1_gold"]:
            breaks_by_dec[p["decision"]]=breaks_by_dec.get(p["decision"],0)+1
    p_mc=mcnemar_exact(len(fixed),len(broke))
    summ={"n":n,"pool":100,"variant":"anchor_and_edit",
          "baseline":{"R@1":b_r1/n},
          "architecture":{"R@1":r1/n,"Recall@5":rec5/n,"MRR":st.mean(mrrs),"nDCG@10":st.mean(ndcgs),"mean_gold_rank":st.mean(goldranks)},
          "fixed":fixed,"broke":broke,"n_fixed":len(fixed),"n_broke":len(broke),"mcnemar_p":p_mc,
          "decisions":decisions,"break_rate_baseline_correct":len(broke)/len(bcorr) if bcorr else 0,
          "breaks_by_decision":breaks_by_dec}
    (OUT/"summary.json").write_text(json.dumps(summ, indent=1))
    print(f"=== Competing-Hypotheses ANCHOR-AND-EDIT (n={n}, POOL=100, {MODEL}) ===")
    print(f"  BASELINE       R@1={b_r1/n:.2f}")
    print(f"  ARCHITECTURE   R@1={r1/n:.2f}  Recall@5={rec5/n:.2f}  MRR={st.mean(mrrs):.3f}  nDCG@10={st.mean(ndcgs):.3f}  mean_gold_rank={st.mean(goldranks):.2f}")
    print(f"  R@1: {b_r1}/{n} -> {r1}/{n}   fixed={len(fixed)} broke={len(broke)}   McNemar p={p_mc:.4f}")
    print(f"  decisions: {decisions}")
    print(f"  break rate among baseline-correct: {summ['break_rate_baseline_correct']:.3f} (v1 full-rerank=0.135, universal Concept->Rerank=0.071)")
    print(f"  breaks by decision: {breaks_by_dec}  (surface_sufficient breaks should now be 0 by construction)")
    print(f"  fixed={fixed}\n  broke={broke}")


if __name__ == "__main__":
    main()
