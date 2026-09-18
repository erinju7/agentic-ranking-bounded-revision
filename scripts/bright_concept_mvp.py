"""BRIGHT Concept-Agent PoC (biology). Pipeline: Question -> Concept Agent -> LLM Reranker.
Tests ONLY whether explicit concept abstraction improves reranking. No debate, no
coordinator, no reflection, no retrieval loops, no extra specialists. The retriever,
candidate pool, sample, and baseline are UNCHANGED and re-derived identically (same
seeds) from bright_mvp.py so the comparison is strictly paired. Baseline numbers are
loaded from the frozen data/bright_mvp/mvp_per_query.json (never re-run here).
"""
from __future__ import annotations
import json, re, math, random, urllib.request, statistics as st
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rq2_core import make_client, extract_json_object

DIR = Path(__file__).resolve().parents[1] / "data" / "bright_mvp"
OUT = Path(__file__).resolve().parents[1] / "results" / "bright_concept_mvp"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-flash-latest"
DOM, N, POOL, MAX_GOLD, DOC_CAP = "biology", 30, 20, 6, 600
BASE = "https://huggingface.co/datasets/xlangai/BRIGHT/resolve/refs%2Fconvert%2Fparquet"


def get(url, path): return pd.read_parquet(path)  # parquet already cached by bright_mvp.py
def dcg(rels): return sum(r/math.log2(i+2) for i, r in enumerate(rels))
def ndcg(rels, k):
    d = dcg(rels[:k]); ideal = dcg(sorted(rels, reverse=True)[:k]); return d/ideal if ideal else 0.0


# ---- reranker: IDENTICAL to baseline, optionally with a concept hypothesis block ----
def rerank(client, q, body, concept_block=""):
    prompt = f"""You are given a question and {len(body)} candidate documents (opaque
ids). Rank ALL ids from most to least relevant for answering the question. Return
JSON only.

Schema: {{"ranked_ids": ["<id>", ...]}}
{concept_block}
Question: {q[:1500]}

Documents:
{json.dumps(body, ensure_ascii=False)}
"""
    try:
        ids = extract_json_object(client.generate(prompt)).get("ranked_ids", [])
    except Exception:
        ids = []
    return ids


def concept_agent(client, q):
    prompt = f"""You are a scientific concept-abstraction agent. Read the question and
identify the LATENT scientific concept(s) that actually answer it -- the underlying
mechanism, principle, compound, or law -- NOT merely the surface topic the question
mentions. Give concise reasoning. If genuinely uncertain, give one lower-confidence
alternative concept. Return JSON only.

Schema: {{"concept": "<the core latent concept(s)>",
          "reasoning": "<1-2 sentences: why this concept, not the surface topic>",
          "alternative": "<optional alternative concept, or empty string>"}}

Question: {q[:1500]}
"""
    try:
        return extract_json_object(client.generate(prompt))
    except Exception:
        return {"concept": "", "reasoning": "", "alternative": ""}


def rank_from_ids(ids, alias):
    ranked = []
    for x in ids:
        v = str(x.get("id") if isinstance(x, dict) else x)
        if v in alias and v not in ranked: ranked.append(v)
    for a in alias:
        if a not in ranked: ranked.append(a)
    return ranked


def main():
    docs_df = get(None, DIR/"documents.parquet"); ex_df = get(None, DIR/"examples.parquet")
    id2doc = dict(zip(docs_df["id"].astype(str), docs_df["content"].astype(str)))
    corpus_ids = list(id2doc); corpus_txt = [id2doc[i] for i in corpus_ids]
    ex = ex_df.to_dict("records")
    # ---- reproduce frozen sample + pools EXACTLY (seeds 42 then 7) ----
    elig = [e for e in ex if 1 <= len([g for g in list(e["gold_ids"]) if g in id2doc]) <= MAX_GOLD]
    elig.sort(key=lambda e: str(e["id"])); random.Random(42).shuffle(elig)
    sample = elig[:N]
    vec = TfidfVectorizer(stop_words="english", max_features=50000); X = vec.fit_transform(corpus_txt)
    rng = random.Random(7)

    frozen = {str(p["id"]): p for p in json.loads((DIR/"mvp_per_query.json").read_text())}
    client = make_client(MODEL)
    c = {"r1": 0, "rec5": 0, "mrr": [], "ndcg": []}
    b = {"r1": 0, "rec5": 0, "mrr": []}   # baseline from frozen (per-query recoverable metrics)
    became_correct, became_worse, per = [], [], []
    for e in sample:
        q = e["query"]
        gold = [g for g in list(e["gold_ids"]) if g in id2doc][:MAX_GOLD]
        excl = set(list(e.get("excluded_ids") or [])) | set(gold)
        sims = linear_kernel(vec.transform([q]), X).ravel(); order = sims.argsort()[::-1]
        distract = []
        for row in order:
            cid = corpus_ids[row]
            if cid not in excl: distract.append(cid)
            if len(gold) + len(distract) >= POOL: break
        pool = gold + distract; rng.shuffle(pool)               # SAME rng progression as baseline
        alias = {f"D{i:02d}": cid for i, cid in enumerate(pool)}
        goldset = {a for a, cid in alias.items() if cid in gold}
        body = [{"id": a, "text": id2doc[cid][:DOC_CAP]} for a, cid in alias.items()]
        # ---- concept agent -> concept-conditioned rerank ----
        ca = concept_agent(client, q)
        cb = f"""\nLatent concept hypothesis (from a concept-abstraction step; treat as a
strong hint about what the question is really about, but rank by the documents):
  concept: {ca.get('concept','')}
  reasoning: {ca.get('reasoning','')}
  alternative: {ca.get('alternative','')}\n"""
        ranked = rank_from_ids(rerank(client, q, body, cb), alias)
        rels = [1 if a in goldset else 0 for a in ranked]
        first = next((i+1 for i, a in enumerate(ranked) if a in goldset), len(ranked)+1)
        c["r1"] += rels[0] == 1; c["rec5"] += any(rels[:5]); c["mrr"].append(1/first); c["ndcg"].append(ndcg(rels, 10))
        # ---- frozen baseline (paired; pool identical) ----
        fz = frozen[str(e["id"])]; b_first = fz["first_gold_rank"]; b_top1 = fz["top1_is_gold"]
        b["r1"] += b_top1; b["rec5"] += b_first <= 5; b["mrr"].append(1/b_first)
        c_top1 = rels[0] == 1
        if (not b_top1) and c_top1: became_correct.append(e["id"])
        if b_top1 and (not c_top1): became_worse.append(e["id"])
        per.append({"id": e["id"], "query": q[:300],
                    "concept": ca.get("concept", ""), "reasoning": ca.get("reasoning", ""),
                    "alternative": ca.get("alternative", ""),
                    "baseline_top1_gold": bool(b_top1), "concept_top1_gold": bool(c_top1),
                    "baseline_first_gold_rank": b_first, "concept_first_gold_rank": first,
                    "concept_top1_doc": id2doc[alias[ranked[0]]][:300],
                    "gold_docs": [id2doc[g][:200] for g in gold]})
    (OUT/"per_query.json").write_text(json.dumps(per, indent=1, ensure_ascii=False))
    n = len(sample)
    def line(name, d, nd=None):
        s = f"  {name:<26} R@1={d['r1']/n:.2f}  Recall@5={d['rec5']/n:.2f}  MRR={st.mean(d['mrr']):.3f}"
        if nd is not None: s += f"  nDCG@10={nd:.3f}"
        return s
    print(f"=== BRIGHT Concept-Agent PoC vs frozen baseline (n={n}, pool={POOL}, {MODEL}) ===")
    print(line("BASELINE (frozen)", b, 0.821))
    print(line("CONCEPT->RERANK", c, st.mean(c["ndcg"])))
    print(f"\n  R@1: {b['r1']}/{n} -> {c['r1']}/{n}   (delta {c['r1']-b['r1']:+d})")
    print(f"\n1. previously WRONG -> now CORRECT ({len(became_correct)}): {became_correct}")
    print(f"2. previously CORRECT -> now WORSE ({len(became_worse)}): {became_worse}")
    (OUT/"summary.json").write_text(json.dumps({
        "n": n, "baseline": {"R@1": b["r1"]/n, "Recall@5": b["rec5"]/n, "MRR": st.mean(b["mrr"]), "nDCG@10": 0.821},
        "concept": {"R@1": c["r1"]/n, "Recall@5": c["rec5"]/n, "MRR": st.mean(c["mrr"]), "nDCG@10": st.mean(c["ndcg"])},
        "became_correct": became_correct, "became_worse": became_worse}, indent=1))
    print(f"\nwrote {OUT/'summary.json'} and {OUT/'per_query.json'}")


if __name__ == "__main__":
    main()
