"""Competing-Hypotheses + Coordinator architecture (implements competing_hypotheses_design.md).
Two symmetric hypothesis agents (A1 surface/lexical, A2 latent concept) run per query;
a reasoning coordinator (A3) arbitrates over the ACTUAL candidate documents and may
decide the surface reading suffices (leaving the ranking unperturbed). Reranks the
byte-identical frozen 100-doc pools from the hard-pool baseline; strictly paired.
No averaging, no gates, no debate, no reflection, no re-retrieval, no pool changes.
Adaptation for POOL=100: coordinator ranks all ids but annotates only its top-10
(per_candidate for 100 docs is impractical); noted in the report.
"""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path
from rq2_core import make_client, extract_json_object

ROOT = Path(__file__).resolve().parents[1]
POOLS = ROOT / "data" / "bright_hardpool" / "biology" / "pools.json"
BASE = ROOT / "results" / "bright_hardpool" / "biology" / "baseline_per_query.json"
OUT = ROOT / "results" / "bright_competing_hypotheses" / "biology"; OUT.mkdir(parents=True, exist_ok=True)
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
    p = f"""You are the SURFACE/LEXICAL hypothesis agent. Read the question and describe
what it LITERALLY asks -- the surface reading, the terms an answer document would
contain, the answer type, and any hard constraints stated. Do NOT abstract to deeper
principles; that is another agent's job. Return JSON only.

Schema: {{"surface_intent": "<one sentence>",
          "key_terms": ["<surface terms/entities the answer doc likely contains>"],
          "expected_answer_type": "<e.g. a chemical | a mechanism | an organism | a method>",
          "explicit_constraints": ["<hard constraints stated in the question>"]}}

Question: {q[:1500]}"""
    try: return extract_json_object(client.generate(p))
    except Exception: return {"surface_intent": "", "key_terms": [], "expected_answer_type": "", "explicit_constraints": []}


def a2_concept(client, q):
    p = f"""You are the LATENT CONCEPT hypothesis agent. Identify the underlying
scientific concept(s) that actually answer the question -- the mechanism, principle,
compound, or law -- NOT the surface topic. Alternatives are first-class: if more than
one latent concept is plausible, list them. Return JSON only.

Schema: {{"primary_concept": "<core latent concept>",
          "reasoning": "<1-2 sentences: why this, not the surface topic>",
          "concept_terms": ["<terms the concept-bearing doc likely contains>"],
          "alternatives": [{{"concept": "<competing latent concept>", "confidence": "high|medium|low"}}]}}

Question: {q[:1500]}"""
    try: return extract_json_object(client.generate(p))
    except Exception: return {"primary_concept": "", "reasoning": "", "concept_terms": [], "alternatives": []}


def a3_coordinator(client, q, h_surface, h_concept, candidates):
    p = f"""You are the COORDINATOR: a reasoning arbiter, NOT a rule engine and NOT a
score-averager. You are given the question, TWO competing hypotheses about what it is
really asking, and the actual candidate documents. Decide -- using the EVIDENCE in the
documents -- which reading the answer supports, and produce the final ranking.

Two competing hypotheses (neither is privileged):
  SURFACE/LEXICAL: {json.dumps(h_surface, ensure_ascii=False)}
  LATENT CONCEPT : {json.dumps(h_concept, ensure_ascii=False)}

Rules:
- If the documents show the surface reading already identifies the right answer, set
  decision="surface_sufficient" and rank by the surface reading -- do NOT let the latent
  concept perturb an answer that is already correct.
- If the answer is a conceptually-distant document that only the latent concept (or one
  of its alternatives) explains, set decision="concept_decisive".
- Otherwise decision="mixed". Treat the concept's alternatives as live competitors.
- Do NOT average. Reason document-by-document for the top of the ranking.
- Rank ALL {len(candidates)} ids.

Return JSON only.
Schema: {{"decision": "surface_sufficient | concept_decisive | mixed",
          "rationale": "<why, referencing evidence in the candidates>",
          "per_candidate": [{{"id": "<id>", "satisfies": "surface|concept|alternative|both|neither", "note": "<short>"}}],
          "ranked_ids": ["<all ids, most->least relevant>"]}}
(Provide per_candidate for your TOP 10 only; ranked_ids must include ALL ids.)

Question: {q[:1500]}

Documents:
{json.dumps(candidates, ensure_ascii=False)}"""
    try: return extract_json_object(client.generate(p))
    except Exception: return {"decision": "", "rationale": "", "per_candidate": [], "ranked_ids": []}


def rank_from_ids(ids, alias_order):
    ranked = []
    for x in ids:
        v = str(x.get("id") if isinstance(x, dict) else x)
        if v in alias_order and v not in ranked: ranked.append(v)
    for a in alias_order:
        if a not in ranked: ranked.append(a)
    return ranked


def main():
    pools = json.loads(POOLS.read_text())
    base = {str(p["id"]): p for p in json.loads(BASE.read_text())}
    client = make_client(MODEL)
    r1 = rec5 = 0; mrrs = []; ndcgs = []; goldranks = []
    fixed, broke, per = [], [], []
    decisions = {}
    b_r1 = 0
    for qid, pl in pools.items():
        q = pl["query"]; alias_to_text = pl["alias_to_text"]; goldset = set(pl["gold_aliases"])
        alias_order = list(alias_to_text)
        h1 = a1_surface(client, q); h2 = a2_concept(client, q)
        cands = [{"id": a, "text": alias_to_text[a]} for a in alias_order]
        co = a3_coordinator(client, q, h1, h2, cands)
        ranked = rank_from_ids(co.get("ranked_ids", []), alias_order)
        rels = [1 if a in goldset else 0 for a in ranked]
        first = next((i+1 for i, a in enumerate(ranked) if a in goldset), len(ranked)+1)
        r1 += rels[0] == 1; rec5 += any(rels[:5]); mrrs.append(1/first)
        ndcgs.append(ndcg(rels, 10)); goldranks += [i+1 for i, a in enumerate(ranked) if a in goldset]
        dec = co.get("decision", "") or "unparsed"; decisions[dec] = decisions.get(dec, 0) + 1
        bt = base[qid]["baseline_top1_gold"]; ct = rels[0] == 1; b_r1 += bt
        if (not bt) and ct: fixed.append(qid)
        if bt and (not ct): broke.append(qid)
        per.append({"id": qid, "query": q[:250], "decision": dec,
                    "primary_concept": h2.get("primary_concept", ""),
                    "alternatives": h2.get("alternatives", []),
                    "surface_intent": h1.get("surface_intent", ""),
                    "baseline_top1_gold": bt, "arch_top1_gold": ct,
                    "baseline_first_gold_rank": base[qid]["baseline_first_gold_rank"],
                    "arch_first_gold_rank": first, "rationale": (co.get("rationale", "") or "")[:300]})
    (OUT/"per_query.json").write_text(json.dumps(per, indent=1, ensure_ascii=False))
    n = len(pools)
    # mechanism: break rate among baseline-correct, by decision
    bcorr = [p for p in per if p["baseline_top1_gold"]]
    protect = {"surface_sufficient": 0, "other": 0}
    for p in bcorr:
        if not p["arch_top1_gold"]:
            protect["surface_sufficient" if p["decision"] == "surface_sufficient" else "other"] += 1
    p_mc = mcnemar_exact(len(fixed), len(broke))
    summ = {"n": n, "pool": 100,
            "baseline": {"R@1": b_r1/n},
            "architecture": {"R@1": r1/n, "Recall@5": rec5/n, "MRR": st.mean(mrrs),
                             "nDCG@10": st.mean(ndcgs), "mean_gold_rank": st.mean(goldranks)},
            "fixed": fixed, "broke": broke, "n_fixed": len(fixed), "n_broke": len(broke),
            "mcnemar_p": p_mc, "decisions": decisions,
            "break_rate_baseline_correct": len(broke)/len(bcorr) if bcorr else 0,
            "breaks_by_decision": protect}
    (OUT/"summary.json").write_text(json.dumps(summ, indent=1))
    print(f"=== Competing-Hypotheses + Coordinator (n={n}, POOL=100, {MODEL}) ===")
    print(f"  BASELINE (hard pool)  R@1={b_r1/n:.2f}")
    print(f"  ARCHITECTURE          R@1={r1/n:.2f}  Recall@5={rec5/n:.2f}  MRR={st.mean(mrrs):.3f}  nDCG@10={st.mean(ndcgs):.3f}  mean_gold_rank={st.mean(goldranks):.2f}")
    print(f"  R@1: {b_r1}/{n} -> {r1}/{n}   fixed={len(fixed)} broke={len(broke)}   McNemar p={p_mc:.4f}")
    print(f"  decisions: {decisions}")
    print(f"  break rate among baseline-correct: {summ['break_rate_baseline_correct']:.3f} (was 0.071 for universal Concept->Rerank)")
    print(f"  breaks by decision: {protect}")
    print(f"  fixed={fixed}\n  broke={broke}")


if __name__ == "__main__":
    main()
