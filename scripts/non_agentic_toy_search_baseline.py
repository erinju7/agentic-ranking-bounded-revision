from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned" / "project_objectives_with_call_descriptions.csv"
RESULT_DIR = ROOT / "results" / "non_agentic_toy_search"
REPORT_PATH = ROOT / "reports" / "non_agentic_toy_search_baseline.md"

DEFAULT_SAMPLE_SIZE = 200
DEFAULT_RANDOM_SEED = 42
DEFAULT_MIN_PROJECTS_PER_CALL = 5
TFIDF_MAX_FEATURES = 50_000
EPS_BUDGET_EUR = 1.0
EPS_DURATION_MONTHS = 1.0
INVALID_TEXT_MARKERS = {"", "false", "true", "nan", "none", "null", "na", "n/a", "<na>"}

CONFIGS = {
    "call_description_only": ["call_description"],
    "historical_objectives_only": ["historical_objectives"],
    "keywords_only": ["keywords"],
    "budget_only": ["budget"],
    "duration_only": ["duration"],
    "description_plus_objectives": ["call_description", "historical_objectives"],
    "text_only": ["call_description", "historical_objectives", "keywords"],
    "text_plus_budget": [
        "call_description",
        "historical_objectives",
        "keywords",
        "budget",
    ],
    "text_plus_duration": [
        "call_description",
        "historical_objectives",
        "keywords",
        "duration",
    ],
    "text_plus_budget_duration": [
        "call_description",
        "historical_objectives",
        "keywords",
        "budget",
        "duration",
    ],
}


@dataclass(frozen=True)
class RunConfig:
    sample_size: int
    random_seed: int
    min_projects_per_call: int


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def is_valid_text(series: pd.Series) -> pd.Series:
    return ~clean_text(series).str.lower().isin(INVALID_TEXT_MARKERS)


def shorten(text: str, limit: int = 180) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def rank_metrics(true_ranks: np.ndarray) -> dict[str, float]:
    return {
        "Recall@1": float(np.mean(true_ranks <= 1)),
        "Recall@5": float(np.mean(true_ranks <= 5)),
        "Recall@10": float(np.mean(true_ranks <= 10)),
        "MRR": float(np.mean(1.0 / true_ranks)),
        "n_queries": int(len(true_ranks)),
    }


def build_query_text(df: pd.DataFrame) -> pd.Series:
    return clean_text(
        df["project_title"]
        + " "
        + df["project_objective"]
        + " "
        + df["project_keywords"]
    )


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, encoding_errors="replace")
    required = {
        "id",
        "project_title",
        "project_objective",
        "project_keywords",
        "funding_call_id",
        "funding_call_description",
        "subCall",
        "masterCall",
        "fundingScheme",
        "totalCost_eur",
        "ecMaxContribution_eur",
        "duration_months",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in required - {"id"}:
        if column in {"totalCost_eur", "ecMaxContribution_eur", "duration_months"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = clean_text(df[column])
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype("int64")
    df = df[
        is_valid_text(df["project_objective"])
        & is_valid_text(df["project_title"])
        & is_valid_text(df["project_keywords"])
        & is_valid_text(df["funding_call_id"])
        & is_valid_text(df["funding_call_description"])
    ].copy()
    df["query_text"] = build_query_text(df)
    return df


def first_nonempty(values: pd.Series) -> str:
    clean = values.dropna().astype(str).str.strip()
    clean = clean[clean.ne("")]
    return "" if clean.empty else clean.iloc[0]


def iqr(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def robust_closeness(value: float, median: float, spread: float, epsilon: float) -> float:
    if pd.isna(value) or pd.isna(median):
        return 0.0
    denom = max(float(spread) if not pd.isna(spread) else 0.0, epsilon)
    distance = abs(float(value) - float(median)) / denom
    return float(1.0 / (1.0 + distance))


def robust_closeness_vector(
    value: float,
    medians: np.ndarray,
    spreads: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    if pd.isna(value):
        return np.zeros_like(medians, dtype=float)
    safe_spreads = np.maximum(np.nan_to_num(spreads, nan=0.0), epsilon)
    distances = np.abs(float(value) - medians) / safe_spreads
    scores = 1.0 / (1.0 + distances)
    scores[~np.isfinite(scores)] = 0.0
    scores[np.isnan(medians)] = 0.0
    return scores.astype(float)


def build_call_index(df: pd.DataFrame, min_projects_per_call: int) -> pd.DataFrame:
    call_sizes = df["funding_call_id"].value_counts()
    candidate_calls = sorted(call_sizes[call_sizes >= min_projects_per_call].index)
    eligible = df[df["funding_call_id"].isin(candidate_calls)].copy()

    grouped = eligible.groupby("funding_call_id", sort=True)
    index = grouped.agg(
        n_projects=("id", "size"),
        funding_call_description=("funding_call_description", first_nonempty),
        historical_objectives=("query_text", " ".join),
        historical_keywords=("project_keywords", " ".join),
        dominant_subCall=("subCall", lambda x: x.mode().iloc[0]),
        dominant_masterCall=("masterCall", lambda x: x.mode().iloc[0]),
        median_totalCost_eur=("totalCost_eur", "median"),
        iqr_totalCost_eur=("totalCost_eur", iqr),
        median_ecMaxContribution_eur=("ecMaxContribution_eur", "median"),
        iqr_ecMaxContribution_eur=("ecMaxContribution_eur", iqr),
        median_duration_months=("duration_months", "median"),
        iqr_duration_months=("duration_months", iqr),
    ).reset_index()
    index = index.rename(columns={"funding_call_id": "call_id"})
    return index


def fit_tfidf_scores(
    candidate_docs: list[str],
    query_docs: list[str],
    min_df: int = 1,
) -> tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=min_df,
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(candidate_docs + query_docs)
    candidate_matrix = matrix[: len(candidate_docs)]
    query_matrix = matrix[len(candidate_docs) :]
    return vectorizer, linear_kernel(query_matrix, candidate_matrix)


def leave_one_out_text(
    grouped: dict[str, pd.DataFrame],
    call_id: str,
    query_id: int,
    column: str,
) -> str:
    group = grouped[call_id]
    return " ".join(group.loc[group["id"] != query_id, column].dropna().astype(str))


def build_dimension_scores(
    eligible: pd.DataFrame,
    call_index: pd.DataFrame,
    sample: pd.DataFrame,
) -> dict[str, np.ndarray]:
    candidate_calls = call_index["call_id"].tolist()
    call_to_index = {label: index for index, label in enumerate(candidate_calls)}
    grouped = {
        label: group.copy()
        for label, group in eligible.groupby("funding_call_id", sort=False)
    }

    query_texts = sample["query_text"].tolist()
    query_keywords = sample["project_keywords"].tolist()

    description_vectorizer, call_description_scores = fit_tfidf_scores(
        call_index["funding_call_description"].tolist(),
        query_texts,
        min_df=1,
    )
    objectives_vectorizer, historical_objective_scores = fit_tfidf_scores(
        call_index["historical_objectives"].tolist(),
        query_texts,
        min_df=2,
    )
    keywords_vectorizer, keyword_scores = fit_tfidf_scores(
        call_index["historical_keywords"].tolist(),
        query_keywords,
        min_df=1,
    )

    total_cost_medians = call_index["median_totalCost_eur"].to_numpy(dtype=float)
    total_cost_iqrs = call_index["iqr_totalCost_eur"].to_numpy(dtype=float)
    ec_contribution_medians = call_index["median_ecMaxContribution_eur"].to_numpy(
        dtype=float
    )
    ec_contribution_iqrs = call_index["iqr_ecMaxContribution_eur"].to_numpy(dtype=float)
    duration_medians = call_index["median_duration_months"].to_numpy(dtype=float)
    duration_iqrs = call_index["iqr_duration_months"].to_numpy(dtype=float)

    budget_scores = np.zeros((len(sample), len(candidate_calls)), dtype=float)
    duration_scores = np.zeros_like(budget_scores)

    for query_idx, query in sample.reset_index(drop=True).iterrows():
        true_call = str(query["funding_call_id"])
        true_call_idx = call_to_index[true_call]
        query_id = int(query["id"])

        true_objectives_text = leave_one_out_text(
            grouped,
            true_call,
            query_id,
            "query_text",
        )
        true_objectives_vector = objectives_vectorizer.transform([true_objectives_text])
        historical_objective_scores[query_idx, true_call_idx] = linear_kernel(
            objectives_vectorizer.transform([query["query_text"]]),
            true_objectives_vector,
        )[0, 0]

        true_keywords_text = leave_one_out_text(
            grouped,
            true_call,
            query_id,
            "project_keywords",
        )
        true_keywords_vector = keywords_vectorizer.transform([true_keywords_text])
        keyword_scores[query_idx, true_call_idx] = linear_kernel(
            keywords_vectorizer.transform([query["project_keywords"]]),
            true_keywords_vector,
        )[0, 0]

        total_cost_scores = robust_closeness_vector(
            query["totalCost_eur"],
            total_cost_medians,
            total_cost_iqrs,
            EPS_BUDGET_EUR,
        )
        ec_contribution_scores = robust_closeness_vector(
            query["ecMaxContribution_eur"],
            ec_contribution_medians,
            ec_contribution_iqrs,
            EPS_BUDGET_EUR,
        )
        budget_scores[query_idx] = (total_cost_scores + ec_contribution_scores) / 2.0
        duration_scores[query_idx] = robust_closeness_vector(
            query["duration_months"],
            duration_medians,
            duration_iqrs,
            EPS_DURATION_MONTHS,
        )

        true_group = grouped[true_call]
        true_group_leave_out = true_group[true_group["id"] != query_id]
        if len(true_group_leave_out) == 0:
            budget_scores[query_idx, true_call_idx] = 0.0
            duration_scores[query_idx, true_call_idx] = 0.0
        else:
            true_total_cost_score = robust_closeness(
                query["totalCost_eur"],
                true_group_leave_out["totalCost_eur"].median(),
                iqr(true_group_leave_out["totalCost_eur"]),
                EPS_BUDGET_EUR,
            )
            true_ec_contribution_score = robust_closeness(
                query["ecMaxContribution_eur"],
                true_group_leave_out["ecMaxContribution_eur"].median(),
                iqr(true_group_leave_out["ecMaxContribution_eur"]),
                EPS_BUDGET_EUR,
            )
            budget_scores[query_idx, true_call_idx] = float(
                np.mean([true_total_cost_score, true_ec_contribution_score])
            )
            duration_scores[query_idx, true_call_idx] = robust_closeness(
                query["duration_months"],
                true_group_leave_out["duration_months"].median(),
                iqr(true_group_leave_out["duration_months"]),
                EPS_DURATION_MONTHS,
            )

    return {
        "call_description": call_description_scores,
        "historical_objectives": historical_objective_scores,
        "keywords": keyword_scores,
        "budget": budget_scores,
        "duration": duration_scores,
        "_feature_counts": {
            "call_description_tfidf_features": len(
                description_vectorizer.get_feature_names_out()
            ),
            "historical_objectives_tfidf_features": len(
                objectives_vectorizer.get_feature_names_out()
            ),
            "keywords_tfidf_features": len(keywords_vectorizer.get_feature_names_out()),
        },
    }


def evaluate(
    df: pd.DataFrame,
    run_config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    call_index = build_call_index(df, run_config.min_projects_per_call)
    candidate_calls = call_index["call_id"].tolist()
    call_to_index = {label: index for index, label in enumerate(candidate_calls)}
    eligible = df[df["funding_call_id"].isin(candidate_calls)].copy()

    sample_size = min(run_config.sample_size, len(eligible))
    sample = eligible.sample(
        n=sample_size,
        random_state=run_config.random_seed,
        replace=False,
    ).reset_index(drop=True)

    dimension_scores = build_dimension_scores(eligible, call_index, sample)
    feature_counts = dimension_scores.pop("_feature_counts")
    true_indices = sample["funding_call_id"].astype(str).map(call_to_index).to_numpy()

    metric_rows = []
    per_query_rows = []
    ranking_rows = []

    for config_name, dimensions in CONFIGS.items():
        combined = np.zeros(
            (len(sample), len(candidate_calls)),
            dtype=float,
        )
        for dimension in dimensions:
            combined += dimension_scores[dimension]
        combined = combined / len(dimensions)

        rankings = np.argsort(-combined, axis=1)
        true_ranks = np.empty(len(sample), dtype=int)
        for query_idx in range(len(sample)):
            true_ranks[query_idx] = (
                np.flatnonzero(rankings[query_idx] == true_indices[query_idx])[0] + 1
            )

        metrics = rank_metrics(true_ranks)
        metric_rows.append(
            {
                "config": config_name,
                "features": "+".join(dimensions),
                **metrics,
            }
        )

        for query_idx, query in sample.reset_index(drop=True).iterrows():
            per_query_rows.append(
                {
                    "config": config_name,
                    "query_id": int(query["id"]),
                    "true_call": query["funding_call_id"],
                    "totalCost_eur": query["totalCost_eur"],
                    "ecMaxContribution_eur": query["ecMaxContribution_eur"],
                    "duration_months": query["duration_months"],
                    "true_rank": int(true_ranks[query_idx]),
                    "true_score": float(combined[query_idx, true_indices[query_idx]]),
                    "call_description_true_score": float(
                        dimension_scores["call_description"][
                            query_idx,
                            true_indices[query_idx],
                        ]
                    ),
                    "historical_objectives_true_score": float(
                        dimension_scores["historical_objectives"][
                            query_idx,
                            true_indices[query_idx],
                        ]
                    ),
                    "keywords_true_score": float(
                        dimension_scores["keywords"][query_idx, true_indices[query_idx]]
                    ),
                    "budget_true_score": float(
                        dimension_scores["budget"][query_idx, true_indices[query_idx]]
                    ),
                    "duration_true_score": float(
                        dimension_scores["duration"][query_idx, true_indices[query_idx]]
                    ),
                }
            )

            for rank, call_idx in enumerate(rankings[query_idx, :10], start=1):
                call_row = call_index.iloc[call_idx]
                ranking_rows.append(
                    {
                        "config": config_name,
                        "query_id": int(query["id"]),
                        "rank": rank,
                        "call_id": call_row["call_id"],
                        "call_description": call_row["funding_call_description"],
                        "score": float(combined[query_idx, call_idx]),
                        "is_true_call": call_row["call_id"]
                        == str(query["funding_call_id"]),
                    }
                )

    metadata = {
        "source_data": str(DATA_PATH.relative_to(ROOT)),
        "candidate_calls": len(candidate_calls),
        "eligible_projects": len(eligible),
        "sample_size": len(sample),
        "random_seed": run_config.random_seed,
        "min_projects_per_call": run_config.min_projects_per_call,
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "configs": CONFIGS,
        **feature_counts,
    }

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(per_query_rows),
        pd.DataFrame(ranking_rows),
        call_index,
        metadata,
    )


def search_calls(
    df: pd.DataFrame,
    query: str,
    query_keywords: str,
    total_cost_eur: float | None,
    ec_contribution_eur: float | None,
    duration_months: float | None,
    min_projects_per_call: int,
    top_k: int,
) -> pd.DataFrame:
    call_index = build_call_index(df, min_projects_per_call)

    query_text = " ".join([query, query_keywords]).strip()
    keyword_text = query_keywords.strip() or query_text

    _, description_scores = fit_tfidf_scores(
        call_index["funding_call_description"].tolist(),
        [query_text],
        min_df=1,
    )
    _, objective_scores = fit_tfidf_scores(
        call_index["historical_objectives"].tolist(),
        [query_text],
        min_df=2,
    )
    _, keyword_scores = fit_tfidf_scores(
        call_index["historical_keywords"].tolist(),
        [keyword_text],
        min_df=1,
    )

    score_parts = [
        description_scores[0],
        objective_scores[0],
        keyword_scores[0],
    ]

    budget_scores = np.zeros(len(call_index), dtype=float)
    if total_cost_eur is not None or ec_contribution_eur is not None:
        budget_parts = []
        if total_cost_eur is not None:
            budget_parts.append(
                robust_closeness_vector(
                    total_cost_eur,
                    call_index["median_totalCost_eur"].to_numpy(dtype=float),
                    call_index["iqr_totalCost_eur"].to_numpy(dtype=float),
                    EPS_BUDGET_EUR,
                )
            )
        if ec_contribution_eur is not None:
            budget_parts.append(
                robust_closeness_vector(
                    ec_contribution_eur,
                    call_index["median_ecMaxContribution_eur"].to_numpy(dtype=float),
                    call_index["iqr_ecMaxContribution_eur"].to_numpy(dtype=float),
                    EPS_BUDGET_EUR,
                )
            )
        budget_scores = np.mean(budget_parts, axis=0)
        score_parts.append(budget_scores)

    duration_scores = np.zeros(len(call_index), dtype=float)
    if duration_months is not None:
        duration_scores = robust_closeness_vector(
            duration_months,
            call_index["median_duration_months"].to_numpy(dtype=float),
            call_index["iqr_duration_months"].to_numpy(dtype=float),
            EPS_DURATION_MONTHS,
        )
        score_parts.append(duration_scores)

    combined = np.mean(score_parts, axis=0)
    rankings = np.argsort(-combined)[:top_k]

    rows = []
    for rank, call_idx in enumerate(rankings, start=1):
        call = call_index.iloc[call_idx]
        rows.append(
            {
                "rank": rank,
                "call_id": call["call_id"],
                "call_description": call["funding_call_description"],
                "score": float(combined[call_idx]),
                "call_description_score": float(description_scores[0, call_idx]),
                "historical_objectives_score": float(objective_scores[0, call_idx]),
                "keywords_score": float(keyword_scores[0, call_idx]),
                "budget_score": float(budget_scores[call_idx]),
                "duration_score": float(duration_scores[call_idx]),
                "n_projects": int(call["n_projects"]),
                "median_totalCost_eur": call["median_totalCost_eur"],
                "median_ecMaxContribution_eur": call["median_ecMaxContribution_eur"],
                "median_duration_months": call["median_duration_months"],
            }
        )
    return pd.DataFrame(rows)


def print_search_results(results: pd.DataFrame) -> None:
    print("Toy funding-call search results")
    print()
    for row in results.itertuples(index=False):
        print(f"{row.rank}. {row.call_id} | score={row.score:.4f}")
        print(f"   {shorten(row.call_description, 140)}")
        print(
            "   components: "
            f"description={row.call_description_score:.4f}, "
            f"objectives={row.historical_objectives_score:.4f}, "
            f"keywords={row.keywords_score:.4f}, "
            f"budget={row.budget_score:.4f}, "
            f"duration={row.duration_score:.4f}; "
            f"n_projects={row.n_projects}, "
            f"median_ec={row.median_ecMaxContribution_eur:.0f}, "
            f"median_duration={row.median_duration_months:.1f} months"
        )
    print()


def write_report(metrics_df: pd.DataFrame, metadata: dict[str, object]) -> None:
    table_rows = []
    for _, row in metrics_df.iterrows():
        table_rows.append(
            "| {config} | {features} | {r1:.3f} | {r5:.3f} | {r10:.3f} | {mrr:.3f} |".format(
                config=row["config"],
                features=row["features"],
                r1=row["Recall@1"],
                r5=row["Recall@5"],
                r10=row["Recall@10"],
                mrr=row["MRR"],
            )
        )

    best_row = metrics_df.sort_values(["MRR", "Recall@10"], ascending=False).iloc[0]
    report = f"""# Non-Agentic Toy Search Baseline

## Purpose

This is a transparent, non-agentic retrieval baseline built directly from:

```text
{metadata['source_data']}
```

The task is:

```text
query = project title + objective + keywords + budget/duration metadata
ground truth = funding_call_id
candidate item = funding_call_id
```

Each candidate call is represented as a small search-engine document. No LLM,
agent, prompt, chain-of-thought, or external knowledge is used.

## Dataset Snapshot

| Item | Value |
|---|---:|
| Candidate funding calls | {metadata['candidate_calls']:,} |
| Eligible projects | {metadata['eligible_projects']:,} |
| Evaluation queries | {metadata['sample_size']:,} |
| Minimum projects per call | {metadata['min_projects_per_call']} |
| Random seed | {metadata['random_seed']} |

## Feature / Dimension Rationale

| Feature group | Columns used | Why it is included |
|---|---|---|
| Call description semantics | `funding_call_description` | Tests whether the official topic title alone is enough to recover the call. |
| Historical objective profile | `project_title`, `project_objective`, `project_keywords` grouped by `funding_call_id` | Tests whether funded projects reveal the empirical topic scope better than the short call title. |
| Project keywords | `project_keywords` grouped by `funding_call_id` | Tests whether author-provided keywords add compact topical signal. |
| Budget fit | `totalCost_eur`, `ecMaxContribution_eur` | Tests whether the project has a similar financial scale to historical projects under the call. |
| Duration fit | `duration_months` | Tests whether the project has a similar time horizon to historical projects under the call. |

The historical objective, keyword, budget, and duration features apply a
leave-one-out control for the query's own true call, so the query project is not
matched against itself.

`fundingScheme`, `subCall`, and `masterCall` are deliberately excluded from the
baseline features because they can act as near-label proxies for
`funding_call_id`.

## Results

All feature scores are in `[0, 1]`; multi-feature configurations use an
unweighted mean.

| Configuration | Features | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

Best configuration by MRR:

```text
{best_row['config']} ({best_row['features']})
```

## Interpretation

This baseline supports feature selection by showing which observable fields
improve ranking without any agentic reasoning. If a feature helps here, it is a
good candidate dimension for the richer matcher. If it fails here, it should
either be dropped, transformed, or reserved for qualitative explanation rather
than ranking.

## Toy Search CLI

Use the same index for an ad hoc non-agentic search:

```bash
python main_experiment/scripts/non_agentic_toy_search_baseline.py \\
  --query "machine learning for clinical decision support in cancer diagnosis" \\
  --query-keywords "AI, healthcare, cancer, diagnostics" \\
  --ec-contribution-eur 3000000 \\
  --duration-months 36 \\
  --top-k 5 \\
  --skip-eval
```

## Outputs

```text
results/non_agentic_toy_search/summary_metrics.csv
results/non_agentic_toy_search/per_query_results.csv
results/non_agentic_toy_search/top10_rankings.csv
results/non_agentic_toy_search/call_index.csv
results/non_agentic_toy_search/run_metadata.json
```
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_baseline(args: argparse.Namespace) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(Path(args.data_path))

    if args.query:
        search_results = search_calls(
            df=df,
            query=args.query,
            query_keywords=args.query_keywords,
            total_cost_eur=args.total_cost_eur,
            ec_contribution_eur=args.ec_contribution_eur,
            duration_months=args.duration_months,
            min_projects_per_call=args.min_projects_per_call,
            top_k=args.top_k,
        )
        print_search_results(search_results)
        if args.skip_eval:
            return

    run_config = RunConfig(
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        min_projects_per_call=args.min_projects_per_call,
    )
    metrics_df, per_query_df, top10_df, call_index, metadata = evaluate(df, run_config)

    metrics_df.to_csv(RESULT_DIR / "summary_metrics.csv", index=False)
    per_query_df.to_csv(RESULT_DIR / "per_query_results.csv", index=False)
    top10_df.to_csv(RESULT_DIR / "top10_rankings.csv", index=False)
    call_index.to_csv(RESULT_DIR / "call_index.csv", index=False)
    (RESULT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(metrics_df, metadata)

    print(metrics_df.to_string(index=False))
    print()
    print(f"Wrote {RESULT_DIR.relative_to(ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and evaluate a non-agentic toy search baseline.",
    )
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--min-projects-per-call",
        type=int,
        default=DEFAULT_MIN_PROJECTS_PER_CALL,
    )
    parser.add_argument(
        "--query",
        default="",
        help="Optional free-text proposal query to rank funding calls.",
    )
    parser.add_argument(
        "--query-keywords",
        default="",
        help="Optional keyword string for the query mode.",
    )
    parser.add_argument(
        "--total-cost-eur",
        type=float,
        default=None,
        help="Optional total project cost in EUR for query mode.",
    )
    parser.add_argument(
        "--ec-contribution-eur",
        type=float,
        default=None,
        help="Optional requested EC contribution in EUR for query mode.",
    )
    parser.add_argument(
        "--duration-months",
        type=float,
        default=None,
        help="Optional planned project duration in months for query mode.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only run query mode; do not regenerate evaluation outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_baseline(parse_args())
