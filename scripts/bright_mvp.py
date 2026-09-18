"""BRIGHT dataset-gating MVP (biology domain). Reranking headroom + closed-book
memorization probe. No multi-agent. Deterministic 30-query sample; candidate pool =
all gold docs + TF-IDF hard distractors from the same domain (pool frozen for reuse).
Reads official BRIGHT (xlangai/BRIGHT) parquet from HuggingFace.
"""
from __future__ import annotations
import json, re, math, random, urllib.request, statistics as st
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rq2_core import make_client, extract_json_object

DIR = Path(__file__).resolve().parents[1] / "data" / "bright_mvp"
DIR.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-flash-latest"
DOM = "biology"
N = 30
POOL = 20
MAX_GOLD = 6
DOC_CAP = 600
BASE = "https://huggingface.co/datasets/xlangai/BRIGHT/resolve/refs%2Fconvert%2Fparquet"


def get(url, path):
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return pd.read_parquet(path)


def norm(s): return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g: return 0.0
    ps, gs = set(p), set(g); inter = len(ps & gs)
    if inter == 0: return 0.0
    prec, rec = inter/len(ps), inter/len(gs)
    return 2*prec*rec/(prec+rec)


def dcg(rels): return sum(r/math.log2(i+2) for i, r in enumerate(rels))
def ndcg(rels, k):
    d = dcg(rels[:k]); ideal = dcg(sorted(rels, reverse=True)[:k]); return d/ideal if ideal else 0.0


def main():
    docs_df = get(f"{BASE}/documents/{DOM}/0000.parquet", DIR/"documents.parquet")
    ex_df = get(f"{BASE}/examples/{DOM}/0000.parquet", DIR/"examples.parquet")
    id2doc = dict(zip(docs_df["id"].astype(str), docs_df["content"].astype(str)))
    corpus_ids = list(id2doc); corpus_txt = [id2doc[i] for i in corpus_ids]

    ex = ex_df.to_dict("records")
    # deterministic sample: manageable gold count, then seeded shuffle
    elig = [e for e in ex if 1 <= len([g for g in list(e["gold_ids"]) if g in id2doc]) <= MAX_GOLD]
    elig.sort(key=lambda e: str(e["id"]))
    random.Random(42).shuffle(elig)
    sample = elig[:N]
    print(f"BRIGHT {DOM}: corpus={len(corpus_ids)} docs, examples={len(ex)}, sampled={len(sample)}")

    # TF-IDF over corpus for hard distractors
    vec = TfidfVectorizer(stop_words="english", max_features=50000)
    X = vec.fit_transform(corpus_txt)
    id2row = {i: r for r, i in enumerate(corpus_ids)}
    rng = random.Random(7)

    client = make_client(MODEL)
    r1 = 0; rec5 = 0; mrrs = []; ndcgs = []; goldranks = []
    cb_f1s = []; per = []
    for e in sample:
        q = e["query"]
        gold = [g for g in list(e["gold_ids"]) if g in id2doc][:MAX_GOLD]
        excl = set(list(e.get("excluded_ids") or [])) | set(gold)
        qv = vec.transform([q]); sims = linear_kernel(qv, X).ravel()
        order = sims.argsort()[::-1]
        distract = []
        for row in order:
            cid = corpus_ids[row]
            if cid not in excl:
                distract.append(cid)
            if len(gold) + len(distract) >= POOL:
                break
        pool = gold + distract
        rng.shuffle(pool)
        alias = {f"D{i:02d}": cid for i, cid in enumerate(pool)}
        goldset = {a for a, cid in alias.items() if cid in gold}
        body = [{"id": a, "text": id2doc[cid][:DOC_CAP]} for a, cid in alias.items()]
        prompt = f"""You are given a question and {len(body)} candidate documents (opaque
ids). Rank ALL ids from most to least relevant for answering the question. Return
JSON only.

Schema: {{"ranked_ids": ["<id>", ...]}}

Question: {q[:1500]}

Documents:
{json.dumps(body, ensure_ascii=False)}
"""
        try:
            ids = extract_json_object(client.generate(prompt)).get("ranked_ids", [])
        except Exception:
            ids = []
        ranked = []
        for x in ids:
            v = str(x.get("id") if isinstance(x, dict) else x)
            if v in alias and v not in ranked: ranked.append(v)
        for a in alias:
            if a not in ranked: ranked.append(a)
        rels = [1 if a in goldset else 0 for a in ranked]
        first = next((i+1 for i, a in enumerate(ranked) if a in goldset), len(ranked)+1)
        r1 += rels[0] == 1; rec5 += any(rels[:5]); mrrs.append(1/first)
        ndcgs.append(ndcg(rels, 10)); goldranks += [i+1 for i, a in enumerate(ranked) if a in goldset]
        # closed-book
        try:
            cbraw = client.generate(f"""Answer this question as fully and specifically as you
can from your own knowledge, with the key facts/entities. Return JSON only.
Schema: {{"answer": "<answer>"}}

Question: {q[:1500]}""")
            cbans = extract_json_object(cbraw).get("answer", "") if cbraw.strip().startswith("{") else cbraw
        except Exception:
            cbans = ""
        cbf1 = f1(cbans, e.get("gold_answer") or "")
        cb_f1s.append(cbf1)
        per.append({"id": e["id"], "query": q[:400], "top1_alias": ranked[0],
                    "top1_doc": id2doc[alias[ranked[0]]][:500], "top1_is_gold": ranked[0] in goldset,
                    "gold_docs": [id2doc[g][:500] for g in gold], "gold_ids": gold,
                    "first_gold_rank": first, "closed_book_f1": round(cbf1, 3), "closed_book_ans": cbans[:400]})
    (DIR/"mvp_per_query.json").write_text(json.dumps(per, indent=1, ensure_ascii=False))
    n = len(sample)
    # closed-book vs ranking correlation
    hi = [p for p in per if p["closed_book_f1"] >= 0.4]; lo = [p for p in per if p["closed_book_f1"] < 0.4]
    def mrank(g): return round(st.mean([p["first_gold_rank"] for p in g]), 2) if g else None
    print(f"\n=== 1. RANKING headroom (n={n}, pool={POOL}) ===")
    print(f"  R@1={r1/n:.2f}  Recall@5={rec5/n:.2f}  MRR={st.mean(mrrs):.3f}  nDCG@10={st.mean(ndcgs):.3f}  mean_gold_rank={st.mean(goldranks):.2f}")
    print(f"\n=== 2. CLOSED-BOOK memorization probe ===")
    print(f"  mean closed-book answer-F1 vs gold_answer: {st.mean(cb_f1s):.3f}")
    print(f"  queries with F1>=0.4 (substantial memory): {len(hi)}/{n} = {len(hi)/n:.2f}")
    print(f"  first-gold-rank | closed-book HIGH(F1>=.4): {mrank(hi)}   LOW: {mrank(lo)}  (lower=easier ranking)")
    print(f"  -> does closed-book success predict easier ranking? compare the two means above")
    print(f"\nwrote {DIR/'mvp_per_query.json'}")


if __name__ == "__main__":
    main()
