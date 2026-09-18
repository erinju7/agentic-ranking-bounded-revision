"""BRIGHT Concept-Agent PoC on a SECOND domain (default earth_science), for
significance. Runs BOTH arms (baseline rerank + concept->rerank) in one strictly
paired run, using the IDENTICAL prompts/seeds/params/pool construction as the biology
PoC (bright_concept_mvp.py). Nothing about the biology setup is changed. Reports paired
metrics, fixed/broken, per-domain + pooled McNemar. Usage: python bright_domain_eval.py [domain]
"""
from __future__ import annotations
import sys, json, re, math, random, urllib.request, statistics as st
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rq2_core import make_client, extract_json_object

DOM = sys.argv[1] if len(sys.argv) > 1 else "earth_science"
N_ARG = int(sys.argv[2]) if len(sys.argv) > 2 else None
ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "bright_mvp" / DOM
DIR.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "results" / "bright_concept_mvp" / DOM
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-flash-latest"
N, POOL, MAX_GOLD, DOC_CAP = (N_ARG or 30), 20, 6, 600
BASE = "https://huggingface.co/datasets/xlangai/BRIGHT/resolve/refs%2Fconvert%2Fparquet"
BIO_FROZEN = ROOT / "results" / "bright_concept_mvp" / "summary.json"  # biology paired result


def get(url, path):
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return pd.read_parquet(path)


def dcg(rels): return sum(r/math.log2(i+2) for i, r in enumerate(rels))
def ndcg(rels, k):
    d = dcg(rels[:k]); ideal = dcg(sorted(rels, reverse=True)[:k]); return d/ideal if ideal else 0.0


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
        return extract_json_object(client.generate(prompt)).get("ranked_ids", [])
    except Exception:
        return []


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


def mcnemar_exact(fixed, broke):
    n = fixed + broke
    if n == 0: return 1.0
    k = min(fixed, broke)
    p = 2 * sum(math.comb(n, i) for i in range(k+1)) / (2**n)
    return min(1.0, p)


def main():
    docs_df = get(f"{BASE}/documents/{DOM}/0000.parquet", DIR/"documents.parquet")
    ex_df = get(f"{BASE}/examples/{DOM}/0000.parquet", DIR/"examples.parquet")
    id2doc = dict(zip(docs_df["id"].astype(str), docs_df["content"].astype(str)))
    corpus_ids = list(id2doc); corpus_txt = [id2doc[i] for i in corpus_ids]
    ex = ex_df.to_dict("records")
    elig = [e for e in ex if 1 <= len([g for g in list(e["gold_ids"]) if g in id2doc]) <= MAX_GOLD]
    elig.sort(key=lambda e: str(e["id"])); random.Random(42).shuffle(elig)
    sample = elig[:N]
    print(f"BRIGHT {DOM}: corpus={len(corpus_ids)} docs, examples={len(ex)}, eligible={len(elig)}, sampled={len(sample)}")
    vec = TfidfVectorizer(stop_words="english", max_features=50000); X = vec.fit_transform(corpus_txt)
    rng = random.Random(7)
    client = make_client(MODEL)
    b = {"r1": 0, "rec5": 0, "mrr": [], "ndcg": []}
    c = {"r1": 0, "rec5": 0, "mrr": [], "ndcg": []}
    fixed, broke, per = [], [], []
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
        pool = gold + distract; rng.shuffle(pool)
        alias = {f"D{i:02d}": cid for i, cid in enumerate(pool)}
        goldset = {a for a, cid in alias.items() if cid in gold}
        body = [{"id": a, "text": id2doc[cid][:DOC_CAP]} for a, cid in alias.items()]
        # baseline arm (concept_block empty == biology baseline prompt, verbatim)
        b_ranked = rank_from_ids(rerank(client, q, body, ""), alias)
        b_rels = [1 if a in goldset else 0 for a in b_ranked]
        b_first = next((i+1 for i, a in enumerate(b_ranked) if a in goldset), len(b_ranked)+1)
        b["r1"] += b_rels[0] == 1; b["rec5"] += any(b_rels[:5]); b["mrr"].append(1/b_first); b["ndcg"].append(ndcg(b_rels, 10))
        # concept arm
        ca = concept_agent(client, q)
        cb = f"""\nLatent concept hypothesis (from a concept-abstraction step; treat as a
strong hint about what the question is really about, but rank by the documents):
  concept: {ca.get('concept','')}
  reasoning: {ca.get('reasoning','')}
  alternative: {ca.get('alternative','')}\n"""
        c_ranked = rank_from_ids(rerank(client, q, body, cb), alias)
        c_rels = [1 if a in goldset else 0 for a in c_ranked]
        c_first = next((i+1 for i, a in enumerate(c_ranked) if a in goldset), len(c_ranked)+1)
        c["r1"] += c_rels[0] == 1; c["rec5"] += any(c_rels[:5]); c["mrr"].append(1/c_first); c["ndcg"].append(ndcg(c_rels, 10))
        bt, ct = b_rels[0] == 1, c_rels[0] == 1
        if (not bt) and ct: fixed.append(e["id"])
        if bt and (not ct): broke.append(e["id"])
        per.append({"id": e["id"], "query": q[:300], "concept": ca.get("concept", ""),
                    "reasoning": ca.get("reasoning", ""), "alternative": ca.get("alternative", ""),
                    "baseline_top1_gold": bt, "concept_top1_gold": ct,
                    "baseline_first_gold_rank": b_first, "concept_first_gold_rank": c_first,
                    "concept_top1_doc": id2doc[alias[c_ranked[0]]][:300],
                    "gold_docs": [id2doc[g][:200] for g in gold]})
    (OUT/"per_query.json").write_text(json.dumps(per, indent=1, ensure_ascii=False))
    n = len(sample)
    p_dom = mcnemar_exact(len(fixed), len(broke))
    print(f"\n=== {DOM} paired (n={n}, pool={POOL}, {MODEL}) ===")
    print(f"  BASELINE       R@1={b['r1']/n:.2f}  Recall@5={b['rec5']/n:.2f}  MRR={st.mean(b['mrr']):.3f}  nDCG@10={st.mean(b['ndcg']):.3f}")
    print(f"  CONCEPT->RERANK R@1={c['r1']/n:.2f}  Recall@5={c['rec5']/n:.2f}  MRR={st.mean(c['mrr']):.3f}  nDCG@10={st.mean(c['ndcg']):.3f}")
    print(f"  R@1: {b['r1']}/{n} -> {c['r1']}/{n}   fixed={fixed}  broke={broke}")
    print(f"  McNemar exact (this domain): p={p_dom:.4f}")
    p_pool = p_dom
    if DOM != "biology" and BIO_FROZEN.exists():
        # pooled with the frozen biology paired result
        bio = json.loads(BIO_FROZEN.read_text())
        bf, bb = len(bio["became_correct"]), len(bio["became_worse"])
        tf, tb = bf + len(fixed), bb + len(broke)
        p_pool = mcnemar_exact(tf, tb)
        print(f"\n=== POOLED biology+{DOM} ===")
        print(f"  biology: fixed={bf} broke={bb} (R@1 {bio['baseline']['R@1']:.2f}->{bio['concept']['R@1']:.2f})")
        print(f"  {DOM}: fixed={len(fixed)} broke={len(broke)}")
        print(f"  pooled discordant: fixed={tf} broke={tb}  ->  McNemar exact p={p_pool:.4f}")
    (OUT/"summary.json").write_text(json.dumps({
        "domain": DOM, "n": n,
        "baseline": {"R@1": b["r1"]/n, "Recall@5": b["rec5"]/n, "MRR": st.mean(b["mrr"]), "nDCG@10": st.mean(b["ndcg"])},
        "concept": {"R@1": c["r1"]/n, "Recall@5": c["rec5"]/n, "MRR": st.mean(c["mrr"]), "nDCG@10": st.mean(c["ndcg"])},
        "fixed": fixed, "broke": broke, "n_fixed": len(fixed), "n_broke": len(broke),
        "mcnemar_p_domain": p_dom, "mcnemar_p_pooled": p_pool}, indent=1))
    print(f"\nwrote {OUT/'summary.json'} and {OUT/'per_query.json'}")


if __name__ == "__main__":
    main()
