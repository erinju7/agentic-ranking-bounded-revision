"""ADDENDUM (NO API): build 100-doc pools for the queries the MAX_GOLD=6 eligibility cap EXCLUDED
(>6 in-corpus golds), for the three new splits. Construction is byte-for-byte the frozen rule from
build_split_pools.py (POOL=100, insert up to MAX_GOLD=6 golds, TF-IDF hard-negative distractors,
600-char truncation, opaque D00.. aliases) EXCEPT it selects the excluded set and uses an independent
RNG so the frozen pools.json (retained set) is never touched. Writes pools_addendum.json per split.
Purpose: robustness re-run to test whether the eligibility filter created the BM25 inversion.
"""
import json, random
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["earth_science", "sustainable_living", "psychology"]
POOL, MAX_GOLD, DOC_CAP = 100, 6, 600

summary = {}
for split in SPLITS:
    docs = pd.read_parquet(ROOT/"data"/"bright_splits"/split/"documents_0000.parquet")
    ex = pd.read_parquet(ROOT/"data"/"bright_splits"/split/"examples_0000.parquet")
    id2doc = dict(zip(docs["id"].astype(str), docs["content"].astype(str)))
    corpus_ids = list(id2doc); corpus_txt = [id2doc[i] for i in corpus_ids]
    exl = ex.to_dict("records")
    # EXCLUDED set: >MAX_GOLD in-corpus golds (the mirror of build_split_pools.py's eligibility)
    excl_ex = [e for e in exl if len([g for g in list(e["gold_ids"]) if g in id2doc]) > MAX_GOLD]
    excl_ex.sort(key=lambda e: str(e["id"]))          # deterministic order
    random.Random(42).shuffle(excl_ex)                # same seed family, independent list
    vec = TfidfVectorizer(stop_words="english", max_features=50000)
    X = vec.fit_transform(corpus_txt)                 # identical fit (same corpus) -> identical distractors
    rng = random.Random(7)                            # independent pool-shuffle RNG (retained pools untouched)
    pools = {}
    for e in excl_ex:
        q = e["query"]
        gold = [g for g in list(e["gold_ids"]) if g in id2doc][:MAX_GOLD]   # cap at 6, identical rule
        excl = set(list(e.get("excluded_ids") or [])) | set(gold)
        sims = linear_kernel(vec.transform([q]), X).ravel(); order = sims.argsort()[::-1]
        distract = []
        for row in order:
            cid = corpus_ids[row]
            if cid not in excl: distract.append(cid)
            if len(gold) + len(distract) >= POOL: break
        pool = gold + distract; rng.shuffle(pool)
        alias = {f"D{i:02d}": cid for i, cid in enumerate(pool)}
        pools[str(e["id"])] = {"query": q,
                               "gold_aliases": [a for a, cid in alias.items() if cid in gold],
                               "alias_to_text": {a: id2doc[cid][:DOC_CAP] for a, cid in alias.items()},
                               "gold_answer": e.get("gold_answer", ""),
                               "n_golds_in_corpus": len([g for g in list(e["gold_ids"]) if g in id2doc])}
    (ROOT/"runs"/"splits"/split/"pools_addendum.json").write_text(json.dumps(pools, ensure_ascii=False))
    N = len(pools)
    est = N * (0.0043 + 0.0006 + 0.0006 + 0.0043)     # A rerank + D a1 + a2 + a3
    summary[split] = {"excluded_N": N, "proj_cost_usd": round(est, 3)}
    print(f"{split:18} excluded_N={N}  in-pool golds capped at 6  proj_cost=${est:.2f}")

tot = sum(v["proj_cost_usd"] for v in summary.values())
(ROOT/"runs"/"splits"/"addendum_build_summary.json").write_text(json.dumps(summary, indent=1))
print(f"\nPROJECTED TOTAL (A+D, reference backbone) = ${tot:.2f}  (hard cap $10 in the runner)")
