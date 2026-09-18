"""Fair, information-parity ranking baseline for RQ2.

The historical-profile TF-IDF retriever is used only to *generate* the frozen
top-N candidate set (its job is recall). It must not double as the ranking
baseline, because it sees historical project text that the reranker does not ---
an unfair information advantage.

This script provides the fair baseline: it re-ranks each query's frozen top-N
using a catalogue-only lexical (TF-IDF) score built from exactly the fields the
reranker sees --- query title/abstract/keywords vs.\ candidate
call_label + funding_call_descriptions --- with no historical text and no
fundingScheme. It therefore isolates "LLM reasoning vs.\ lexical matching on the
same information".
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from rq2_core import (
    FROZEN_DIR,
    RESULT_ROOT,
    load_candidate_pool,
    load_frozen_split,
    load_ground_truth,
    read_jsonl,
)

RETRIEVER_DIR = RESULT_ROOT / "deterministic_topn_retriever"
OUT_DIR = RESULT_ROOT / "catalogue_lexical_baseline"


def compact(value: Any) -> str:
    return " ".join(str(value or "").split())


def query_doc(q: dict[str, Any]) -> str:
    return compact(" ".join([q.get("project_title", ""), q.get("query_text", ""),
                             q.get("project_keywords", "")]))


def candidate_doc(c: dict[str, Any]) -> str:
    return compact(" ".join([c.get("call_label", ""),
                             " ".join(c.get("funding_call_descriptions", []))]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--retriever-file", default=None,
                   help="Frozen top-N JSONL; defaults to {split}_historical_top50_candidates.jsonl")
    p.add_argument("--allow-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to touch test split without --allow-test.")

    retriever_file = Path(args.retriever_file) if args.retriever_file else (
        RETRIEVER_DIR / f"{args.split}_historical_top50_candidates.jsonl")
    if not retriever_file.exists():
        raise SystemExit(f"Retriever file not found: {retriever_file}")

    queries = load_frozen_split(args.split, FROZEN_DIR)
    truths = load_ground_truth(args.split, FROZEN_DIR)
    candidates = load_candidate_pool(FROZEN_DIR)
    labels = [c["call_label"] for c in candidates]
    idx = {L: i for i, L in enumerate(labels)}

    # Global catalogue-only TF-IDF (same fields the reranker sees).
    vec = TfidfVectorizer(lowercase=True, stop_words="english", min_df=1,
                          ngram_range=(1, 2), sublinear_tf=True)
    cand_matrix = vec.fit_transform([candidate_doc(c) for c in candidates])
    qvecs = {str(q["query_id"]): vec.transform([query_doc(q)]) for q in queries}

    topn = {str(r["query_id"]): sorted(r["candidates"], key=lambda c: c["rank"])
            for r in read_jsonl(retriever_file)}

    true_ranks = []
    for q in queries:
        qid = str(q["query_id"])
        if qid not in topn:
            continue
        truth = set(truths[qid])
        cand_ids = [idx[c["call_label"]] for c in topn[qid] if c["call_label"] in idx]
        sims = linear_kernel(qvecs[qid], cand_matrix[cand_ids])[0]
        order = [cand_ids[j] for j in np.argsort(-sims)]
        rank = next((r for r, ci in enumerate(order, 1) if labels[ci] in truth),
                    len(order) + 1)
        true_ranks.append(rank)

    r = np.array(true_ranks)
    n = len(r)
    summary = {
        "baseline": "catalogue_only_lexical_tfidf",
        "note": "information-parity with reranker; no historical text, no fundingScheme",
        "split": args.split,
        "retriever_file": retriever_file.name,
        "n_queries": n,
        "Recall@1": float((r <= 1).mean()),
        "Recall@5": float((r <= 5).mean()),
        "Recall@10": float((r <= 10).mean()),
        "MRR": float((1.0 / r).mean()),
        "NDCG@5": float(np.mean([(1 / math.log2(x + 1)) if x <= 5 else 0.0 for x in r])),
        "mean_true_rank": float(r.mean()),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
