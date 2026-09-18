from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from rq2_core import FROZEN_DIR, RESULT_ROOT, load_candidate_pool, load_frozen_split, load_ground_truth


OUT_DIR = RESULT_ROOT / "deterministic_topn_retriever"
SOURCE_PATH = FROZEN_DIR.parents[1] / "cleaned" / "project_objectives_with_call_descriptions.csv"
RETRIEVER_VERSION = "rq2_tfidf_topn_v4_global_fit_loo"
DEFAULT_TOP_N = 50


def compact_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def query_document(query: dict[str, Any]) -> str:
    return compact_text(
        " ".join(
            [
                query.get("project_title", ""),
                query.get("query_text", ""),
                query.get("project_keywords", ""),
                query.get("fundingScheme", ""),
            ]
        )
    )


def candidate_document(candidate: dict[str, Any]) -> str:
    return compact_text(
        " ".join(
            [
                candidate.get("call_label", ""),
                candidate.get("dominant_fundingScheme", ""),
                " ".join(candidate.get("funding_call_descriptions", [])),
            ]
        )
    )


def load_historical_groups(candidate_labels: list[str]) -> dict[str, pd.DataFrame]:
    source = pd.read_csv(SOURCE_PATH, low_memory=False, encoding_errors="replace")
    source["id"] = pd.to_numeric(source["id"], errors="coerce")
    source = source.dropna(subset=["id"]).copy()
    source["id"] = source["id"].astype("int64")
    source = source[source["subCall"].astype(str).isin(candidate_labels)].copy()
    return {
        str(label): group.copy()
        for label, group in source.groupby("subCall", sort=False)
    }


def historical_text(group: pd.DataFrame, exclude_project_id: int | None = None) -> str:
    if exclude_project_id is not None:
        group = group[group["id"] != exclude_project_id]
    if group.empty:
        return ""
    parts = []
    for column in ["project_title", "project_objective", "project_keywords"]:
        if column in group.columns:
            parts.append(" ".join(group[column].fillna("").astype(str).tolist()))
    return compact_text(" ".join(parts))


def candidate_document_with_history(
    candidate: dict[str, Any],
    historical_groups: dict[str, pd.DataFrame],
    *,
    exclude_project_id: int | None = None,
) -> str:
    call_label = candidate["call_label"]
    group = historical_groups.get(call_label)
    history = historical_text(group, exclude_project_id) if group is not None else ""
    return compact_text(
        " ".join(
            [
                candidate_document(candidate),
                history,
            ]
        )
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic TF-IDF top-N candidate sets for RQ2.",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument(
        "--candidate-text",
        choices=["catalogue", "historical"],
        default="catalogue",
        help="Use fixed catalogue fields only, or add leave-one-out historical project text.",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required to build/evaluate the sealed test split.",
    )
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to touch test split without --allow-test.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = load_frozen_split(args.split, FROZEN_DIR)
    candidates = load_candidate_pool(FROZEN_DIR)
    truths = load_ground_truth(args.split, FROZEN_DIR)
    candidate_labels = [candidate["call_label"] for candidate in candidates]
    historical_groups = (
        load_historical_groups(candidate_labels)
        if args.candidate_text == "historical"
        else {}
    )

    query_docs = [query_document(query) for query in queries]
    candidate_docs = [candidate_document(candidate) for candidate in candidates]

    if args.candidate_text == "catalogue":
        fit_corpus = candidate_docs
    else:
        # Global fit corpus uses the unrestricted historical text (no per-query
        # exclusion) so the vocabulary/idf are fixed once, exactly like the
        # catalogue baseline. Leave-one-out is applied afterwards, per query,
        # by re-vectorizing only the true candidate with the already-fitted
        # vectorizer (transform only, never re-fit).
        fit_corpus = [
            candidate_document_with_history(candidate, historical_groups, exclude_project_id=None)
            for candidate in candidates
        ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=1,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(fit_corpus)
    query_matrix = vectorizer.transform(query_docs)
    scores = linear_kernel(query_matrix, matrix)

    top_n = min(args.top_n, len(candidates))
    topn_rows = []
    true_ranks = []
    for query_idx, query in enumerate(queries):
        query_id = str(query["query_id"])
        truth = set(truths[query_id])
        query_scores = scores[query_idx].copy()
        if args.candidate_text == "historical":
            query_project_id = int(query["project_id"])
            query_vector = query_matrix[query_idx]
            for candidate_idx, candidate in enumerate(candidates):
                if candidate["call_label"] not in truth:
                    continue
                loo_doc = candidate_document_with_history(
                    candidate,
                    historical_groups,
                    exclude_project_id=query_project_id,
                )
                loo_vector = vectorizer.transform([loo_doc])
                query_scores[candidate_idx] = linear_kernel(query_vector, loo_vector)[0, 0]
        order = np.argsort(-query_scores)
        true_rank = next(
            (
                rank
                for rank, candidate_idx in enumerate(order, start=1)
                if candidate_labels[candidate_idx] in truth
            ),
            len(candidates) + 1,
        )
        true_ranks.append(true_rank)
        topn_rows.append(
            {
                "query_id": query_id,
                "top_n": top_n,
                "true_call_label": list(truths[query_id]),
                "true_rank_in_full_candidate_pool": true_rank,
                "true_in_top_n": true_rank <= top_n,
                "candidates": [
                    {
                        "rank": rank,
                        "call_label": candidate_labels[candidate_idx],
                        "retriever_score": float(query_scores[candidate_idx]),
                    }
                    for rank, candidate_idx in enumerate(order[:top_n], start=1)
                ],
            }
        )

    summary = {
        "retriever_version": RETRIEVER_VERSION,
        "split": args.split,
        "candidate_text": args.candidate_text,
        "leave_one_out_true_call": args.candidate_text == "historical",
        "vectorizer_fit": (
            "global_candidate_corpus_with_unrestricted_history;"
            " true_call_label_revectorized_via_transform_only_after_per_query_leave_one_out"
            if args.candidate_text == "historical"
            else "global_candidate_corpus_only"
        ),
        "top_n": top_n,
        "n_queries": len(queries),
        "n_candidates": len(candidates),
        "Recall@50": float(np.mean(np.array(true_ranks) <= 50)),
        f"Recall@{top_n}": float(np.mean(np.array(true_ranks) <= top_n)),
        "Recall@100": float(np.mean(np.array(true_ranks) <= 100)),
        "MRR": float(np.mean(1.0 / np.array(true_ranks))),
        "mean_true_rank": float(np.mean(true_ranks)),
        "median_true_rank": float(np.median(true_ranks)),
        "max_true_rank": int(np.max(true_ranks)),
        "query_document_fields": [
            "project_title",
            "query_text",
            "project_keywords",
            "fundingScheme",
        ],
        "candidate_document_fields": [
            "call_label",
            "dominant_fundingScheme",
            "funding_call_descriptions",
            *(
                [
                    "historical_project_title",
                    "historical_project_objective",
                    "historical_project_keywords",
                ]
                if args.candidate_text == "historical"
                else []
            ),
        ],
        "outputs": {
            "topn_candidates": f"{args.split}_{args.candidate_text}_top{top_n}_candidates.jsonl",
            "summary": f"{args.split}_{args.candidate_text}_top{top_n}_summary.json",
        },
    }

    write_jsonl(
        output_dir / f"{args.split}_{args.candidate_text}_top{top_n}_candidates.jsonl",
        topn_rows,
    )
    (output_dir / f"{args.split}_{args.candidate_text}_top{top_n}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
