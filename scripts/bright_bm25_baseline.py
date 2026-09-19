"""Deterministic Okapi BM25 reranker baseline on the BRIGHT Biology pools (same
100-candidate pools used by A/B/C/D). Reports Hit@1 and MRR. BM25 is CPU-only and
needs no API. Provides a conventional lexical reference for the LLM designs."""
import json, re, math
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
def bm25_rank(query, docs, k1=1.5, b=0.75):
    # docs: list of (alias, text)
    toks = [tok(t) for _, t in docs]; N = len(docs)
    avgdl = sum(len(d) for d in toks) / max(N, 1)
    df = Counter()
    for d in toks:
        for w in set(d): df[w] += 1
    idf = {w: math.log((N - n + 0.5) / (n + 0.5) + 1) for w, n in df.items()}
    scores = []
    for (alias, _), d in zip(docs, toks):
        tf = Counter(d); dl = len(d); s = 0.0
        for w in set(tok(query)):
            if w in tf:
                s += idf.get(w, 0) * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * dl / avgdl))
        scores.append((s, alias))
    scores.sort(key=lambda x: (-x[0]))
    return [a for _, a in scores]
def run(domain):
    P = json.loads((ROOT / "data" / "bright_hardpool" / domain / "pools.json").read_text())
    hit1 = 0; rr = 0.0; n = 0
    for qid, pl in P.items():
        gold = set(pl["gold_aliases"]); docs = list(pl["alias_to_text"].items())
        ranked = bm25_rank(pl["query"], docs)
        n += 1
        if ranked[0] in gold: hit1 += 1
        rank = next((i for i, a in enumerate(ranked, 1) if a in gold), None)
        rr += (1 / rank) if rank else 0
    print(f"{domain:20s} n={n}  BM25 Hit@1={hit1/n:.3f}  MRR={rr/n:.3f}")
if __name__ == "__main__":
    for d in ["biology", "earth_science", "psychology", "sustainable_living"]:
        try: run(d)
        except Exception as e: print(d, "skip:", e)
