from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "cleaned" / "cordis_projects_clean.csv"
PROFILE_PATH = ROOT / "data" / "cleaned" / "call_profiles_subcall.csv"
SAMPLE_PATH = ROOT / "data" / "samples" / "eval_sample_200_seed42.csv"
SMALL_CALL_FLAGS_PATH = (
    ROOT / "data" / "samples" / "eval_sample_200_seed42_small_call_flags.csv"
)
RESULT_DIR = ROOT / "results" / "rq1_dimension_ablation"

MIN_PROJECTS_PER_CALL = 5
TFIDF_MAX_FEATURES = 50_000
EPS_EC_CONTRIBUTION = 1.0
EPS_DURATION_MONTHS = 1.0

CONFIGS = {
    "thematic_only": ["thematic"],
    "technical_only": ["technical"],
    "budget_duration_only": ["budget_duration"],
    "thematic_plus_technical": ["thematic", "technical"],
    "thematic_plus_budget_duration": ["thematic", "budget_duration"],
    "all_dimensions": ["thematic", "technical", "budget_duration"],
}


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


def rank_metrics(true_ranks: np.ndarray) -> dict[str, float]:
    return {
        "Recall@1": float(np.mean(true_ranks <= 1)),
        "Recall@5": float(np.mean(true_ranks <= 5)),
        "Recall@10": float(np.mean(true_ranks <= 10)),
        "MRR": float(np.mean(1.0 / true_ranks)),
        "n_queries": int(len(true_ranks)),
    }


def prepare_leave_one_out_text(
    grouped_objectives: dict[str, pd.DataFrame],
    call_label: str,
    query_id: object,
) -> str:
    group = grouped_objectives[call_label]
    leave_out = group.loc[group["id"] != query_id, "objective"]
    return " ".join(leave_out.dropna().astype(str))


def leave_one_out_profile(
    grouped: dict[str, pd.DataFrame],
    call_label: str,
    query_id: object,
) -> pd.DataFrame:
    group = grouped[call_label]
    return group[group["id"] != query_id]


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(CLEAN_PATH, low_memory=False)
    profiles = pd.read_csv(PROFILE_PATH)
    sample = pd.read_csv(SAMPLE_PATH, low_memory=False)

    candidate_calls = sorted(profiles["call_label"].astype(str).tolist())
    call_to_index = {label: index for index, label in enumerate(candidate_calls)}
    eligible = clean[clean["call_label"].isin(candidate_calls)].copy()
    eligible["objective"] = eligible["objective"].fillna("").astype(str)

    grouped = {
        label: group.copy()
        for label, group in eligible.groupby("call_label", sort=False)
    }
    call_sizes = np.array([len(grouped[label]) for label in candidate_calls], dtype=float)
    scheme_counts_by_call = {
        label: grouped[label]["fundingScheme"].value_counts().to_dict()
        for label in candidate_calls
    }
    full_ec_medians = np.array(
        [grouped[label]["ecMaxContribution_eur"].median() for label in candidate_calls],
        dtype=float,
    )
    full_ec_iqrs = np.array(
        [iqr(grouped[label]["ecMaxContribution_eur"]) for label in candidate_calls],
        dtype=float,
    )
    full_duration_medians = np.array(
        [grouped[label]["duration_months"].median() for label in candidate_calls],
        dtype=float,
    )
    full_duration_iqrs = np.array(
        [iqr(grouped[label]["duration_months"]) for label in candidate_calls],
        dtype=float,
    )

    full_call_docs = [
        " ".join(grouped[label]["objective"].dropna().astype(str))
        for label in candidate_calls
    ]

    query_texts = sample["objective"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=2,
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(full_call_docs + query_texts)
    full_call_matrix = matrix[: len(candidate_calls)]
    query_matrix = matrix[len(candidate_calls) :]
    thematic_scores = linear_kernel(query_matrix, full_call_matrix)

    technical_scores = np.zeros((len(sample), len(candidate_calls)), dtype=float)
    budget_duration_scores = np.zeros_like(technical_scores)

    for query_idx, query in sample.reset_index(drop=True).iterrows():
        true_call = str(query["call_label"])
        true_call_idx = call_to_index[true_call]

        # Thematic leave-one-out: replace only the true call representation.
        true_call_text = prepare_leave_one_out_text(grouped, true_call, query["id"])
        true_call_vector = vectorizer.transform([true_call_text])
        thematic_scores[query_idx, true_call_idx] = linear_kernel(
            query_matrix[query_idx],
            true_call_vector,
        )[0, 0]

        query_scheme = query["fundingScheme"]
        query_ec = query["ecMaxContribution_eur"]
        query_duration = query["duration_months"]

        technical_scores[query_idx] = np.array(
            [
                scheme_counts_by_call[label].get(query_scheme, 0) / call_sizes[idx]
                for idx, label in enumerate(candidate_calls)
            ],
            dtype=float,
        )
        ec_scores = robust_closeness_vector(
            query_ec,
            full_ec_medians,
            full_ec_iqrs,
            EPS_EC_CONTRIBUTION,
        )
        duration_scores = robust_closeness_vector(
            query_duration,
            full_duration_medians,
            full_duration_iqrs,
            EPS_DURATION_MONTHS,
        )
        budget_duration_scores[query_idx] = (ec_scores + duration_scores) / 2.0

        # Replace the true call with a leave-one-out profile for all non-text
        # dimensions.
        true_group = leave_one_out_profile(grouped, true_call, query["id"])
        if len(true_group) == 0:
            technical_scores[query_idx, true_call_idx] = 0.0
            budget_duration_scores[query_idx, true_call_idx] = 0.0
        else:
            technical_scores[query_idx, true_call_idx] = float(
                (true_group["fundingScheme"] == query_scheme).mean()
            )
            true_ec_score = robust_closeness(
                query_ec,
                true_group["ecMaxContribution_eur"].median(),
                iqr(true_group["ecMaxContribution_eur"]),
                EPS_EC_CONTRIBUTION,
            )
            true_duration_score = robust_closeness(
                query_duration,
                true_group["duration_months"].median(),
                iqr(true_group["duration_months"]),
                EPS_DURATION_MONTHS,
            )
            budget_duration_scores[query_idx, true_call_idx] = float(
                np.mean([true_ec_score, true_duration_score])
            )

    dimension_matrices = {
        "thematic": thematic_scores,
        "technical": technical_scores,
        "budget_duration": budget_duration_scores,
    }
    true_indices = sample["call_label"].astype(str).map(call_to_index).to_numpy()

    metric_rows = []
    per_query_rows = []
    ranking_rows = []

    for config_name, dimensions in CONFIGS.items():
        combined = np.zeros_like(thematic_scores)
        for dimension in dimensions:
            combined += dimension_matrices[dimension]

        rankings = np.argsort(-combined, axis=1)
        true_ranks = np.empty(len(sample), dtype=int)
        for query_idx in range(len(sample)):
            true_ranks[query_idx] = (
                np.flatnonzero(rankings[query_idx] == true_indices[query_idx])[0]
                + 1
            )

        metrics = rank_metrics(true_ranks)
        metric_rows.append(
            {
                "config": config_name,
                "dimensions": "+".join(dimensions),
                **metrics,
            }
        )

        for query_idx, query in sample.reset_index(drop=True).iterrows():
            per_query_rows.append(
                {
                    "config": config_name,
                    "query_id": query["id"],
                    "true_call": query["call_label"],
                    "fundingScheme": query["fundingScheme"],
                    "true_rank": int(true_ranks[query_idx]),
                    "true_score": float(combined[query_idx, true_indices[query_idx]]),
                    "thematic_true_score": float(
                        thematic_scores[query_idx, true_indices[query_idx]]
                    ),
                    "technical_true_score": float(
                        technical_scores[query_idx, true_indices[query_idx]]
                    ),
                    "budget_duration_true_score": float(
                        budget_duration_scores[query_idx, true_indices[query_idx]]
                    ),
                }
            )

        for query_idx, query in sample.reset_index(drop=True).iterrows():
            for rank, call_idx in enumerate(rankings[query_idx, :10], start=1):
                ranking_rows.append(
                    {
                        "config": config_name,
                        "query_id": query["id"],
                        "rank": rank,
                        "call_label": candidate_calls[call_idx],
                        "score": float(combined[query_idx, call_idx]),
                        "is_true_call": candidate_calls[call_idx]
                        == str(query["call_label"]),
                    }
                )

    metrics_df = pd.DataFrame(metric_rows)
    per_query_df = pd.DataFrame(per_query_rows)
    top10_df = pd.DataFrame(ranking_rows)

    metrics_df.to_csv(RESULT_DIR / "ablation_summary_metrics.csv", index=False)
    per_query_df.to_csv(RESULT_DIR / "ablation_per_query_results.csv", index=False)
    top10_df.to_csv(RESULT_DIR / "ablation_top10_rankings.csv", index=False)

    sensitivity_rows = []
    if SMALL_CALL_FLAGS_PATH.exists():
        small_flags = pd.read_csv(SMALL_CALL_FLAGS_PATH)
        small_ids = set(small_flags["id"])
    else:
        small_ids = set()

    all_dimensions = metrics_df.loc[
        metrics_df["config"] == "all_dimensions"
    ].iloc[0].to_dict()
    sensitivity_rows.append(
        {
            "condition": "include_small_call_queries",
            "excluded_queries": 0,
            "n_queries": int(all_dimensions["n_queries"]),
            "Recall@1": all_dimensions["Recall@1"],
            "Recall@5": all_dimensions["Recall@5"],
            "Recall@10": all_dimensions["Recall@10"],
            "MRR": all_dimensions["MRR"],
        }
    )

    all_per_query = per_query_df[per_query_df["config"] == "all_dimensions"].copy()
    filtered = all_per_query[~all_per_query["query_id"].isin(small_ids)]
    sensitivity_metrics = rank_metrics(filtered["true_rank"].to_numpy())
    sensitivity_rows.append(
        {
            "condition": "exclude_small_call_queries",
            "excluded_queries": int(len(all_per_query) - len(filtered)),
            **sensitivity_metrics,
        }
    )
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(
        RESULT_DIR / "all_dimensions_small_call_sensitivity.csv",
        index=False,
    )

    metadata = {
        "sample_path": str(SAMPLE_PATH.relative_to(ROOT)),
        "clean_path": str(CLEAN_PATH.relative_to(ROOT)),
        "candidate_calls": len(candidate_calls),
        "sample_size": len(sample),
        "min_projects_per_call": MIN_PROJECTS_PER_CALL,
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "epsilon_ec_contribution": EPS_EC_CONTRIBUTION,
        "epsilon_duration_months": EPS_DURATION_MONTHS,
        "configs": CONFIGS,
        "outputs": {
            "summary_metrics": "results/rq1_dimension_ablation/ablation_summary_metrics.csv",
            "per_query_results": "results/rq1_dimension_ablation/ablation_per_query_results.csv",
            "top10_rankings": "results/rq1_dimension_ablation/ablation_top10_rankings.csv",
            "small_call_sensitivity": "results/rq1_dimension_ablation/all_dimensions_small_call_sensitivity.csv",
        },
    }
    (RESULT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(metrics_df.to_string(index=False))
    print()
    print("Small-call sensitivity:")
    print(sensitivity_df.to_string(index=False))


if __name__ == "__main__":
    main()
