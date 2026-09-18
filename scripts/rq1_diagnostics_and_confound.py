from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "cleaned" / "cordis_projects_clean.csv"
PROFILE_PATH = ROOT / "data" / "cleaned" / "call_profiles_subcall.csv"
SAMPLE_PATH = ROOT / "data" / "samples" / "eval_sample_200_seed42.csv"
PER_QUERY_PATH = (
    ROOT
    / "results"
    / "rq1_dimension_ablation"
    / "ablation_per_query_results.csv"
)
OUT_DIR = ROOT / "results" / "rq1_dimension_ablation" / "diagnostics"

EPS_EC_CONTRIBUTION = 1.0
EPS_DURATION_MONTHS = 1.0
PERMUTATION_RUNS = 20
PERMUTATION_SEED = 20260621


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


def scheme_family(value: object) -> str:
    text = "" if pd.isna(value) else str(value).upper().strip()
    if text.startswith("MSCA"):
        return "MSCA"
    if text.startswith("ERC"):
        return "ERC"
    if text.startswith("SME"):
        return "SME"
    if text == "RIA":
        return "RIA"
    if text == "IA":
        return "IA"
    if text == "CSA":
        return "CSA"
    return "OTHER"


def rank_metrics(true_ranks: np.ndarray) -> dict[str, float]:
    return {
        "Recall@1": float(np.mean(true_ranks <= 1)),
        "Recall@5": float(np.mean(true_ranks <= 5)),
        "Recall@10": float(np.mean(true_ranks <= 10)),
        "MRR": float(np.mean(1.0 / true_ranks)),
        "n_queries": int(len(true_ranks)),
    }


def rank_from_scores(
    scores: np.ndarray,
    sample: pd.DataFrame,
    candidate_calls: list[str],
    call_to_index: dict[str, int],
) -> np.ndarray:
    rankings = np.argsort(-scores, axis=1)
    true_indices = sample["call_label"].astype(str).map(call_to_index).to_numpy()
    true_ranks = np.empty(len(sample), dtype=int)
    for query_idx in range(len(sample)):
        true_ranks[query_idx] = (
            np.flatnonzero(rankings[query_idx] == true_indices[query_idx])[0] + 1
        )
    return true_ranks


def technical_score_matrix(
    eligible: pd.DataFrame,
    sample: pd.DataFrame,
    candidate_calls: list[str],
    scheme_column: str,
) -> np.ndarray:
    call_sizes = (
        eligible.groupby("call_label")
        .size()
        .reindex(candidate_calls, fill_value=0)
        .to_numpy(dtype=float)
    )
    scheme_call_counts = (
        eligible.groupby([scheme_column, "call_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=candidate_calls, fill_value=0)
    )
    score_matrix = np.zeros((len(sample), len(candidate_calls)), dtype=float)

    for query_idx, query in sample.reset_index(drop=True).iterrows():
        query_scheme = query[scheme_column]
        true_call = str(query["call_label"])
        true_idx = candidate_calls.index(true_call)
        if query_scheme in scheme_call_counts.index:
            numerators = scheme_call_counts.loc[query_scheme].to_numpy(dtype=float)
        else:
            numerators = np.zeros(len(candidate_calls), dtype=float)
        denominators = call_sizes.copy()

        # Leave the query project out of its true call. Because the numerator
        # counts projects with the same scheme as the query, the query itself
        # contributes one to the true-call numerator before leave-one-out.
        numerators = numerators.copy()
        numerators[true_idx] = max(0.0, numerators[true_idx] - 1.0)
        denominators[true_idx] = max(0.0, denominators[true_idx] - 1.0)
        score_matrix[query_idx] = np.divide(
            numerators,
            denominators,
            out=np.zeros_like(numerators, dtype=float),
            where=denominators > 0,
        )
    return score_matrix


def budget_duration_diagnostics(
    eligible: pd.DataFrame,
    sample: pd.DataFrame,
    candidate_calls: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    grouped = {
        label: group.copy()
        for label, group in eligible.groupby("call_label", sort=False)
    }
    call_to_index = {label: index for index, label in enumerate(candidate_calls)}
    full_sizes = np.array([len(grouped[label]) for label in candidate_calls], dtype=int)
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
    rows = []
    true_rows = []

    for query_idx, query in sample.reset_index(drop=True).iterrows():
        true_call = str(query["call_label"])
        true_idx = call_to_index[true_call]
        sizes = full_sizes.copy()
        ec_medians = full_ec_medians.copy()
        ec_iqrs = full_ec_iqrs.copy()
        duration_medians = full_duration_medians.copy()
        duration_iqrs = full_duration_iqrs.copy()

        true_group = grouped[true_call]
        true_group = true_group[true_group["id"] != query["id"]]
        sizes[true_idx] = len(true_group)
        ec_medians[true_idx] = true_group["ecMaxContribution_eur"].median()
        ec_iqrs[true_idx] = iqr(true_group["ecMaxContribution_eur"])
        duration_medians[true_idx] = true_group["duration_months"].median()
        duration_iqrs[true_idx] = iqr(true_group["duration_months"])

        ec_scores = robust_closeness_vector(
            query["ecMaxContribution_eur"],
            ec_medians,
            ec_iqrs,
            EPS_EC_CONTRIBUTION,
        )
        duration_scores = robust_closeness_vector(
            query["duration_months"],
            duration_medians,
            duration_iqrs,
            EPS_DURATION_MONTHS,
        )
        budget_scores = (ec_scores + duration_scores) / 2.0

        for call_idx, call_label in enumerate(candidate_calls):
            is_true_call = call_idx == true_idx
            score = float(budget_scores[call_idx])
            row = {
                "query_id": query["id"],
                "candidate_call": call_label,
                "is_true_call": is_true_call,
                "candidate_members_after_leaveout": int(sizes[call_idx]),
                "ec_median": ec_medians[call_idx],
                "ec_iqr": ec_iqrs[call_idx],
                "duration_median": duration_medians[call_idx],
                "duration_iqr": duration_iqrs[call_idx],
                "ec_score": ec_scores[call_idx],
                "duration_score": duration_scores[call_idx],
                "budget_duration_score": score,
                "ec_iqr_zero_or_missing": pd.isna(ec_iqrs[call_idx])
                or ec_iqrs[call_idx] == 0,
                "duration_iqr_zero_or_missing": pd.isna(duration_iqrs[call_idx])
                or duration_iqrs[call_idx] == 0,
            }
            rows.append(row)
            if is_true_call:
                true_rows.append(row)

    all_scores = pd.DataFrame(rows)
    true_scores = pd.DataFrame(true_rows)
    summary = {
        "n_query_candidate_scores": int(len(all_scores)),
        "score_min": float(all_scores["budget_duration_score"].min()),
        "score_max": float(all_scores["budget_duration_score"].max()),
        "score_nan_count": int(all_scores["budget_duration_score"].isna().sum()),
        "score_lt_0_count": int((all_scores["budget_duration_score"] < 0).sum()),
        "score_gt_1_count": int((all_scores["budget_duration_score"] > 1).sum()),
        "ec_score_nan_count": int(all_scores["ec_score"].isna().sum()),
        "duration_score_nan_count": int(all_scores["duration_score"].isna().sum()),
        "true_call_score_min": float(true_scores["budget_duration_score"].min()),
        "true_call_score_max": float(true_scores["budget_duration_score"].max()),
        "true_call_zero_or_missing_ec_iqr": int(
            true_scores["ec_iqr_zero_or_missing"].sum()
        ),
        "true_call_zero_or_missing_duration_iqr": int(
            true_scores["duration_iqr_zero_or_missing"].sum()
        ),
        "small_true_call_profiles_le_5_members": int(
            (true_scores["candidate_members_after_leaveout"] <= 5).sum()
        ),
    }
    return all_scores, true_scores, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(CLEAN_PATH, low_memory=False)
    profiles = pd.read_csv(PROFILE_PATH)
    sample = pd.read_csv(SAMPLE_PATH, low_memory=False)
    per_query = pd.read_csv(PER_QUERY_PATH)

    candidate_calls = sorted(profiles["call_label"].astype(str).tolist())
    call_to_index = {label: index for index, label in enumerate(candidate_calls)}
    eligible = clean[clean["call_label"].isin(candidate_calls)].copy()

    # 1. Check why all_dimensions can underperform thematic_plus_technical.
    ranks = per_query.pivot(
        index="query_id",
        columns="config",
        values="true_rank",
    )
    ranks["all_minus_thematic_plus_technical"] = (
        ranks["all_dimensions"] - ranks["thematic_plus_technical"]
    )
    rank_shift_summary = {
        "improved_when_adding_budget_to_thematic_technical": int(
            (ranks["all_minus_thematic_plus_technical"] < 0).sum()
        ),
        "unchanged_when_adding_budget_to_thematic_technical": int(
            (ranks["all_minus_thematic_plus_technical"] == 0).sum()
        ),
        "worsened_when_adding_budget_to_thematic_technical": int(
            (ranks["all_minus_thematic_plus_technical"] > 0).sum()
        ),
        "mean_rank_shift_all_minus_thematic_technical": float(
            ranks["all_minus_thematic_plus_technical"].mean()
        ),
        "median_rank_shift_all_minus_thematic_technical": float(
            ranks["all_minus_thematic_plus_technical"].median()
        ),
    }
    ranks.reset_index().to_csv(OUT_DIR / "all_vs_thematic_technical_rank_shifts.csv", index=False)

    # 2. Budget-duration score range and leave-one-out profile diagnostics.
    all_budget_scores, true_budget_scores, budget_summary = budget_duration_diagnostics(
        eligible,
        sample,
        candidate_calls,
    )
    all_budget_scores.to_csv(OUT_DIR / "budget_duration_all_scores_diagnostics.csv", index=False)
    true_budget_scores.to_csv(OUT_DIR / "budget_duration_true_call_diagnostics.csv", index=False)

    # 3. Technical confound controls.
    eligible = eligible.copy()
    sample = sample.copy()
    eligible["scheme_family"] = eligible["fundingScheme"].map(scheme_family)
    sample["scheme_family"] = sample["fundingScheme"].map(scheme_family)

    original_scores = technical_score_matrix(
        eligible,
        sample,
        candidate_calls,
        "fundingScheme",
    )
    family_scores = technical_score_matrix(
        eligible,
        sample,
        candidate_calls,
        "scheme_family",
    )

    original_metrics = rank_metrics(
        rank_from_scores(original_scores, sample, candidate_calls, call_to_index)
    )
    family_metrics = rank_metrics(
        rank_from_scores(family_scores, sample, candidate_calls, call_to_index)
    )

    confound_rows = [
        {
            "condition": "original_exact_fundingScheme",
            "run": 0,
            **original_metrics,
        },
        {
            "condition": "broad_scheme_family",
            "run": 0,
            **family_metrics,
        },
    ]

    rng = np.random.default_rng(PERMUTATION_SEED)
    for run in range(1, PERMUTATION_RUNS + 1):
        permuted = eligible[["id", "fundingScheme"]].copy()
        permuted["fundingScheme_permuted"] = rng.permutation(
            permuted["fundingScheme"].to_numpy()
        )
        eligible_perm = eligible.merge(
            permuted[["id", "fundingScheme_permuted"]],
            on="id",
            how="left",
            validate="one_to_one",
        )
        sample_perm = sample.merge(
            permuted[["id", "fundingScheme_permuted"]],
            on="id",
            how="left",
            validate="one_to_one",
        )
        perm_scores = technical_score_matrix(
            eligible_perm,
            sample_perm,
            candidate_calls,
            "fundingScheme_permuted",
        )
        perm_metrics = rank_metrics(
            rank_from_scores(perm_scores, sample_perm, candidate_calls, call_to_index)
        )
        confound_rows.append(
            {
                "condition": "permuted_fundingScheme",
                "run": run,
                **perm_metrics,
            }
        )

    confound_df = pd.DataFrame(confound_rows)
    confound_df.to_csv(OUT_DIR / "technical_confound_controls.csv", index=False)

    perm_summary = (
        confound_df[confound_df["condition"] == "permuted_fundingScheme"][
            ["Recall@1", "Recall@5", "Recall@10", "MRR"]
        ]
        .agg(["mean", "std", "min", "max"])
        .T
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    perm_summary.to_csv(
        OUT_DIR / "technical_permutation_summary.csv",
        index=False,
    )

    summary = {
        "rank_shift_summary": rank_shift_summary,
        "budget_duration_summary": budget_summary,
        "technical_confound": {
            "original_exact_fundingScheme": original_metrics,
            "broad_scheme_family": family_metrics,
            "permutation_runs": PERMUTATION_RUNS,
            "permutation_seed": PERMUTATION_SEED,
            "permutation_summary": perm_summary.to_dict(orient="records"),
        },
    }
    (OUT_DIR / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
