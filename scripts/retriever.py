from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned" / "project_objectives_with_call_descriptions.csv"
ARTIFACT_DIR = ROOT / "retriever_artifacts"

DEFAULT_MIN_PROJECTS_PER_CALL = 5
TFIDF_MAX_FEATURES = 50_000
EPS_BUDGET_EUR = 1.0
EPS_DURATION_MONTHS = 1.0
TEXT_WEIGHT = 0.70
BUDGET_WEIGHT = 0.15
DURATION_WEIGHT = 0.15
INVALID_TEXT_MARKERS = {"", "false", "true", "nan", "none", "null", "na", "n/a", "<na>"}

TEXT_FEATURES = [
    "call_description_score",
    "historical_objectives_score",
    "keywords_score",
]


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def is_valid_text(series: pd.Series) -> pd.Series:
    return ~clean_text(series).str.lower().isin(INVALID_TEXT_MARKERS)


def shorten(text: str, limit: int = 500) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def iqr(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def robust_closeness_vector(
    value: float | None,
    medians: np.ndarray,
    spreads: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    if value is None or pd.isna(value):
        return np.zeros_like(medians, dtype=float)
    safe_spreads = np.maximum(np.nan_to_num(spreads, nan=0.0), epsilon)
    distances = np.abs(float(value) - medians) / safe_spreads
    scores = 1.0 / (1.0 + distances)
    scores[~np.isfinite(scores)] = 0.0
    scores[np.isnan(medians)] = 0.0
    return scores.astype(float)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
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
    df["query_text"] = clean_text(
        df["project_title"]
        + " "
        + df["project_objective"]
        + " "
        + df["project_keywords"]
    )
    return df


def first_nonempty(values: pd.Series) -> str:
    clean = values.dropna().astype(str).str.strip()
    clean = clean[clean.ne("")]
    return "" if clean.empty else clean.iloc[0]


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
        median_totalCost_eur=("totalCost_eur", "median"),
        iqr_totalCost_eur=("totalCost_eur", iqr),
        median_ecMaxContribution_eur=("ecMaxContribution_eur", "median"),
        iqr_ecMaxContribution_eur=("ecMaxContribution_eur", iqr),
        median_duration_months=("duration_months", "median"),
        iqr_duration_months=("duration_months", iqr),
    ).reset_index()
    index = index.rename(columns={"funding_call_id": "funding_call_id"})
    index["historical_profile_summary"] = index["historical_objectives"].map(shorten)
    return index


def fit_vectorizer(docs: pd.Series, min_df: int) -> tuple[TfidfVectorizer, sparse.spmatrix]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=min_df,
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(docs.fillna("").astype(str).tolist())
    return vectorizer, matrix


def build_index(
    data_path: Path = DATA_PATH,
    artifact_dir: Path = ARTIFACT_DIR,
    min_projects_per_call: int = DEFAULT_MIN_PROJECTS_PER_CALL,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data_path)
    call_index = build_call_index(df, min_projects_per_call)

    vectorizers: dict[str, TfidfVectorizer] = {}
    matrices: dict[str, sparse.spmatrix] = {}
    vectorizers["call_description"], matrices["call_description"] = fit_vectorizer(
        call_index["funding_call_description"],
        min_df=1,
    )
    vectorizers["historical_objectives"], matrices["historical_objectives"] = (
        fit_vectorizer(
            call_index["historical_objectives"],
            min_df=2,
        )
    )
    vectorizers["keywords"], matrices["keywords"] = fit_vectorizer(
        call_index["historical_keywords"],
        min_df=1,
    )

    call_index.to_parquet(artifact_dir / "funding_calls.parquet", index=False)
    with (artifact_dir / "tfidf_vectorizers.pkl").open("wb") as f:
        pickle.dump(vectorizers, f)
    for name, matrix in matrices.items():
        sparse.save_npz(artifact_dir / f"{name}_matrix.npz", matrix)

    metadata = {
        "source_data": str(data_path),
        "candidate_calls": int(len(call_index)),
        "eligible_projects": int(call_index["n_projects"].sum()),
        "min_projects_per_call": int(min_projects_per_call),
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "retriever_weights": {
            "text": TEXT_WEIGHT,
            "budget": BUDGET_WEIGHT,
            "duration": DURATION_WEIGHT,
        },
        "features": [
            "call_description",
            "historical_objectives",
            "keywords",
            "budget",
            "duration",
        ],
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


class FundingCallRetriever:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR) -> None:
        self.artifact_dir = artifact_dir
        self.calls_df = pd.read_parquet(artifact_dir / "funding_calls.parquet")
        with (artifact_dir / "tfidf_vectorizers.pkl").open("rb") as f:
            self.vectorizers: dict[str, TfidfVectorizer] = pickle.load(f)
        self.matrices = {
            "call_description": sparse.load_npz(artifact_dir / "call_description_matrix.npz"),
            "historical_objectives": sparse.load_npz(
                artifact_dir / "historical_objectives_matrix.npz"
            ),
            "keywords": sparse.load_npz(artifact_dir / "keywords_matrix.npz"),
        }

    @classmethod
    def load_default(cls) -> "FundingCallRetriever":
        required = [
            ARTIFACT_DIR / "funding_calls.parquet",
            ARTIFACT_DIR / "tfidf_vectorizers.pkl",
            ARTIFACT_DIR / "call_description_matrix.npz",
            ARTIFACT_DIR / "historical_objectives_matrix.npz",
            ARTIFACT_DIR / "keywords_matrix.npz",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Retriever artifacts are missing. Run "
                "`python main_experiment/scripts/build_retriever_index.py` first. "
                f"Missing: {missing}"
            )
        return cls(ARTIFACT_DIR)

    def get_top_k_calls(
        self,
        project_objective: str,
        project_title: str = "",
        project_keywords: list[str] | str | None = None,
        total_cost_eur: float | None = None,
        ec_contribution_eur: float | None = None,
        duration_months: float | None = None,
        k: int = 50,
    ) -> list[dict[str, Any]]:
        if isinstance(project_keywords, list):
            keyword_text = " ".join(project_keywords)
        else:
            keyword_text = project_keywords or ""
        query_text = " ".join([project_title, project_objective, keyword_text]).strip()

        call_description_scores = linear_kernel(
            self.vectorizers["call_description"].transform([query_text]),
            self.matrices["call_description"],
        )[0]
        objective_scores = linear_kernel(
            self.vectorizers["historical_objectives"].transform([query_text]),
            self.matrices["historical_objectives"],
        )[0]
        keyword_scores = linear_kernel(
            self.vectorizers["keywords"].transform([keyword_text or query_text]),
            self.matrices["keywords"],
        )[0]

        text_scores = np.mean(
            [
                call_description_scores,
                objective_scores,
                keyword_scores,
            ],
            axis=0,
        )

        budget_scores = np.zeros(len(self.calls_df), dtype=float)
        budget_weight = 0.0
        if total_cost_eur is not None or ec_contribution_eur is not None:
            budget_parts = []
            if total_cost_eur is not None:
                budget_parts.append(
                    robust_closeness_vector(
                        total_cost_eur,
                        self.calls_df["median_totalCost_eur"].to_numpy(dtype=float),
                        self.calls_df["iqr_totalCost_eur"].to_numpy(dtype=float),
                        EPS_BUDGET_EUR,
                    )
                )
            if ec_contribution_eur is not None:
                budget_parts.append(
                    robust_closeness_vector(
                        ec_contribution_eur,
                        self.calls_df["median_ecMaxContribution_eur"].to_numpy(dtype=float),
                        self.calls_df["iqr_ecMaxContribution_eur"].to_numpy(dtype=float),
                        EPS_BUDGET_EUR,
                    )
            )
            budget_scores = np.mean(budget_parts, axis=0)
            budget_weight = BUDGET_WEIGHT

        duration_scores = np.zeros(len(self.calls_df), dtype=float)
        duration_weight = 0.0
        if duration_months is not None:
            duration_scores = robust_closeness_vector(
                duration_months,
                self.calls_df["median_duration_months"].to_numpy(dtype=float),
                self.calls_df["iqr_duration_months"].to_numpy(dtype=float),
                EPS_DURATION_MONTHS,
            )
            duration_weight = DURATION_WEIGHT

        text_weight = 1.0 - budget_weight - duration_weight
        if text_weight <= 0:
            text_weight = TEXT_WEIGHT
        combined = (
            text_weight * text_scores
            + budget_weight * budget_scores
            + duration_weight * duration_scores
        )
        top_k = min(max(int(k), 1), len(self.calls_df))
        rankings = np.argsort(-combined)[:top_k]

        results = []
        for rank, call_idx in enumerate(rankings, start=1):
            row = self.calls_df.iloc[call_idx]
            results.append(
                {
                    "rank": rank,
                    "funding_call_id": row["funding_call_id"],
                    "funding_call_description": row["funding_call_description"],
                    "historical_profile_summary": row["historical_profile_summary"],
                    "score": float(combined[call_idx]),
                    "call_description_score": float(call_description_scores[call_idx]),
                    "historical_objectives_score": float(objective_scores[call_idx]),
                    "keywords_score": float(keyword_scores[call_idx]),
                    "budget_score": float(budget_scores[call_idx]),
                    "duration_score": float(duration_scores[call_idx]),
                    "n_projects": int(row["n_projects"]),
                    "median_totalCost_eur": _none_if_nan(row["median_totalCost_eur"]),
                    "median_ecMaxContribution_eur": _none_if_nan(
                        row["median_ecMaxContribution_eur"]
                    ),
                    "median_duration_months": _none_if_nan(
                        row["median_duration_months"]
                    ),
                }
            )
        return results


def _none_if_nan(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


_DEFAULT_RETRIEVER: FundingCallRetriever | None = None


def get_default_retriever() -> FundingCallRetriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = FundingCallRetriever.load_default()
    return _DEFAULT_RETRIEVER


def get_top_k_calls(
    project_objective: str,
    project_title: str = "",
    project_keywords: list[str] | str | None = None,
    total_cost_eur: float | None = None,
    ec_contribution_eur: float | None = None,
    duration_months: float | None = None,
    k: int = 50,
) -> list[dict[str, Any]]:
    return get_default_retriever().get_top_k_calls(
        project_objective=project_objective,
        project_title=project_title,
        project_keywords=project_keywords,
        total_cost_eur=total_cost_eur,
        ec_contribution_eur=ec_contribution_eur,
        duration_months=duration_months,
        k=k,
    )
