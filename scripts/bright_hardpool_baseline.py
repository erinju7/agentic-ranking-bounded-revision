"""Harder-pool baseline (BRIGHT biology, n=97). Same single-pass generalist reranker
and same seeds as before, but POOL=100 (gold + top-99 TF-IDF hard negatives) to remove
the 'pool too easy' ceiling. Freezes the pools (alias->text, gold aliases, shuffled
order) AND the full baseline ranking to disk, so the competing-hypotheses architecture
can later rerank byte-identical inputs. Does NOT touch the 20-pool results.
"""
from __future__ import annotations
import os, sys, json, math, random, statistics as st
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rq2_core import make_client, extract_json_object

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "biology"
# biology parquet lives in bright_mvp/ (flat names); the additional splits in
# bright_splits/<domain>/ with *_0000 names. Everything else is IDENTICAL across domains.
if DOMAIN == "biology":
    SRC, DOCF, EXF = ROOT / "data" / "bright_mvp", "documents.parquet", "examples.parquet"
else:
    SRC, DOCF, EXF = ROOT / "data" / "bright_splits" / DOMAIN, "documents_0000.parquet", "examples_0000.parquet"
POOLDIR = ROOT / "data" / "bright_hardpool" / DOMAIN; POOLDIR.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "results" / "bright_hardpool" / DOMAIN; OUT.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-flash-latest"
N, POOL, MAX_GOLD, DOC_CAP = 97, 100, 6, 600


def load_env():
    for env in (ROOT / ".env", ROOT.parent / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def dcg(rels): return sum(r/math.log2(i+2) for i, r in enumerate(rels))
def ndcg(rels, k):
    d = dcg(rels[:k]); ideal = dcg(sorted(rels, reverse=True)[:k]); return d/ideal if ideal else 0.0


def rerank(client, q, body):
    prompt = f"""You are given a question and {len(body)} candidate documents (opaque
ids). Rank ALL ids from most to least relevant for answering the question. Return
JSON only.

Schema: {{"ranked_ids": ["<id>", ...]}}

Question: {q[:1500]}

Documents:
{json.dumps(body, ensure_ascii=False)}
"""
    try:
        return extract_json_object(client.generate(prompt)).get("ranked_ids", [])
    except Exception:
        return []


def rank_from_ids(ids, alias):
    ranked = []
    for x in ids:
        v = str(x.get("id") if isinstance(x, dict) else x)
        if v in alias and v not in ranked: ranked.append(v)
    for a in alias:
        if a not in ranked: ranked.append(a)
    return ranked


def main():
    load_env()
    docs_df = pd.read_parquet(SRC/DOCF); ex_df = pd.read_parquet(SRC/EXF)
    id2doc = dict(zip(docs_df["id"].astype(str), docs_df["content"].astype(str)))
    corpus_ids = list(id2doc); corpus_txt = [id2doc[i] for i in corpus_ids]
    ex = ex_df.to_dict("records")
    elig = [e for e in ex if 1 <= len([g for g in list(e["gold_ids"]) if g in id2doc]) <= MAX_GOLD]
    elig.sort(key=lambda e: str(e["id"])); random.Random(42).shuffle(elig)
    sample = elig[:N]
    print(f"BRIGHT {DOMAIN} hard-pool: corpus={len(corpus_ids)}, eligible={len(elig)}, sampled={len(sample)}, POOL={POOL}")
    vec = TfidfVectorizer(stop_words="english", max_features=50000); X = vec.fit_transform(corpus_txt)
    rng = random.Random(7)
    client = make_client(MODEL)
    r1 = rec5 = 0; mrrs = []; ndcgs = []; goldranks = []
    pools = {}; per = []
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
        gold_aliases = [a for a, cid in alias.items() if cid in gold]
        alias_to_text = {a: id2doc[cid][:DOC_CAP] for a, cid in alias.items()}
        # freeze pool for the architecture (opaque, self-contained)
        pools[str(e["id"])] = {"query": q, "gold_aliases": gold_aliases, "alias_to_text": alias_to_text}
        # single-pass baseline
        body = [{"id": a, "text": alias_to_text[a]} for a in alias]
        ranked = rank_from_ids(rerank(client, q, body), alias)
        goldset = set(gold_aliases)
        rels = [1 if a in goldset else 0 for a in ranked]
        first = next((i+1 for i, a in enumerate(ranked) if a in goldset), len(ranked)+1)
        r1 += rels[0] == 1; rec5 += any(rels[:5]); mrrs.append(1/first)
        ndcgs.append(ndcg(rels, 10)); goldranks += [i+1 for i, a in enumerate(ranked) if a in goldset]
        per.append({"id": e["id"], "query": q[:300], "baseline_ranked": ranked,
                    "gold_aliases": gold_aliases, "baseline_top1": ranked[0],
                    "baseline_top1_gold": ranked[0] in goldset, "baseline_first_gold_rank": first})
    (POOLDIR/"pools.json").write_text(json.dumps(pools, ensure_ascii=False))
    (OUT/"baseline_per_query.json").write_text(json.dumps(per, indent=1, ensure_ascii=False))
    n = len(sample)
    summ = {"n": n, "pool": POOL, "R@1": r1/n, "Recall@5": rec5/n, "MRR": st.mean(mrrs),
            "nDCG@10": st.mean(ndcgs), "mean_gold_rank": st.mean(goldranks)}
    (OUT/"baseline_summary.json").write_text(json.dumps(summ, indent=1))
    print(f"=== hard-pool baseline (n={n}, POOL={POOL}, {MODEL}) ===")
    print(f"  R@1={summ['R@1']:.2f}  Recall@5={summ['Recall@5']:.2f}  MRR={summ['MRR']:.3f}  nDCG@10={summ['nDCG@10']:.3f}  mean_gold_rank={summ['mean_gold_rank']:.2f}")
    print(f"  (compare 20-pool baseline: R@1=0.72 Recall@5=0.92 MRR=0.805 nDCG@10=0.823)")
    print(f"  froze {len(pools)} pools -> {POOLDIR/'pools.json'}")


if __name__ == "__main__":
    main()
