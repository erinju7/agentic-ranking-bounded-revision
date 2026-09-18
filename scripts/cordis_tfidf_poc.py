from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


DATA_PATH = Path(
    "CORDIS - EU research projects under Horizon 2020 (2014-2020)"
    "/Publications Office/9-cordis-h2020projects-csv/project.csv"
)
LABEL_COLUMN = "subCall"
QUERY_COLUMN = "objective"
SAMPLE_SIZE = 200
MIN_PROJECTS_PER_CALL = 5
RANDOM_SEED = 42


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def shorten(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def main() -> None:
    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        quotechar='"',
        on_bad_lines="skip",
        low_memory=False,
    )
    df[LABEL_COLUMN] = clean_text(df[LABEL_COLUMN])
    df[QUERY_COLUMN] = clean_text(df[QUERY_COLUMN])
    df["title"] = clean_text(df["title"])

    usable = df[
        df[LABEL_COLUMN].ne("") & df[QUERY_COLUMN].ne("")
    ].copy()
    call_sizes = usable[LABEL_COLUMN].value_counts()
    eligible_calls = call_sizes[
        call_sizes >= MIN_PROJECTS_PER_CALL
    ].index
    eligible = usable[usable[LABEL_COLUMN].isin(eligible_calls)].copy()
    eligible["_source_index"] = eligible.index

    if len(eligible) < SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(eligible)} eligible projects; cannot sample {SAMPLE_SIZE}."
        )

    queries = eligible.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_SEED,
        replace=False,
    ).reset_index(drop=True)

    call_labels = sorted(eligible_calls)
    call_to_index = {label: index for index, label in enumerate(call_labels)}
    call_docs = (
        eligible.groupby(LABEL_COLUMN)[QUERY_COLUMN]
        .agg(" ".join)
        .reindex(call_labels)
    )

    # Fit once for a fast proof of concept. Every call is represented by its
    # concatenated objectives; the true call is replaced by a leave-one-out
    # document for each query before ranking.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=2,
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(
        call_docs.tolist() + queries[QUERY_COLUMN].tolist()
    )
    call_matrix = matrix[: len(call_labels)]
    query_matrix = matrix[len(call_labels) :]
    similarities = linear_kernel(query_matrix, call_matrix)

    for query_index, row in queries.iterrows():
        true_call = row[LABEL_COLUMN]
        true_call_index = call_to_index[true_call]
        other_objectives = eligible.loc[
            (eligible[LABEL_COLUMN] == true_call)
            & (eligible["_source_index"] != row["_source_index"]),
            QUERY_COLUMN,
        ]
        leave_one_out_doc = " ".join(other_objectives)
        leave_one_out_vector = vectorizer.transform([leave_one_out_doc])
        similarities[query_index, true_call_index] = linear_kernel(
            query_matrix[query_index],
            leave_one_out_vector,
        )[0, 0]

    rankings = np.argsort(-similarities, axis=1)
    true_indices = np.array(
        [call_to_index[label] for label in queries[LABEL_COLUMN]]
    )
    true_ranks = np.empty(SAMPLE_SIZE, dtype=int)
    for index in range(SAMPLE_SIZE):
        true_ranks[index] = (
            np.flatnonzero(rankings[index] == true_indices[index])[0] + 1
        )

    recall_at_1 = np.mean(true_ranks <= 1)
    recall_at_5 = np.mean(true_ranks <= 5)
    mrr = np.mean(1.0 / true_ranks)

    print("CORDIS TF-IDF retrieval proof of concept")
    print(f"Parsed projects: {len(df):,}")
    print(f"Usable projects: {len(usable):,}")
    print(
        f"Eligible calls (>={MIN_PROJECTS_PER_CALL} usable projects): "
        f"{len(call_labels):,}"
    )
    print(f"Projects in eligible calls: {len(eligible):,}")
    print(f"Sampled queries: {len(queries):,} (seed={RANDOM_SEED})")
    print(f"TF-IDF features: {len(vectorizer.get_feature_names_out()):,}")
    print()
    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"Recall@5: {recall_at_5:.4f}")
    print(f"MRR:      {mrr:.4f}")

    example_indices = [
        0,
        int(np.argmin(np.abs(true_ranks - np.median(true_ranks)))),
        int(np.argmax(true_ranks)),
    ]
    print("\nExamples")
    for example_number, query_index in enumerate(example_indices, start=1):
        row = queries.iloc[query_index]
        true_call = row[LABEL_COLUMN]
        print(f"\nExample {example_number}")
        print(f"Project: {row['id']} | {row['title']}")
        print(f"Objective: {shorten(row[QUERY_COLUMN])}")
        print(f"True call: {true_call}")
        print(f"True-call rank: {true_ranks[query_index]}")
        print("Top 5 calls:")
        for rank, call_index in enumerate(rankings[query_index, :5], start=1):
            label = call_labels[call_index]
            marker = " <-- TRUE" if label == true_call else ""
            print(
                f"  {rank}. {label} "
                f"(cosine={similarities[query_index, call_index]:.4f})"
                f"{marker}"
            )


if __name__ == "__main__":
    main()
