from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "cleaned" / "project_objectives_with_call_descriptions.csv"
OUTPUT_DIR = ROOT / "data" / "frozen" / "rq2_v1_seed42_description_filtered"

VERSION = "rq2_v1_seed42_description_filtered"
LABEL_FIELD = "subCall"
QUERY_FIELD = "project_objective"
DESCRIPTION_FIELD = "funding_call_description"
MIN_PROJECTS_PER_CALL = 5
TEST_SIZE = 200
TEST_SEED = 42
DEV_SIZE = 200
DEV_SEED = 43
INVALID_TEXT_MARKERS = {"", "false", "true", "nan", "none", "null", "na", "n/a", "<na>"}

TEXT_COLUMNS = [
    "project_title",
    "project_objective",
    "project_keywords",
    "funding_call_id",
    "funding_call_description",
    "subCall",
    "masterCall",
    "fundingScheme",
]
NUMERIC_COLUMNS = ["totalCost_eur", "ecMaxContribution_eur", "duration_months"]


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def is_valid_text(series: pd.Series) -> pd.Series:
    return ~clean_text(series).str.lower().isin(INVALID_TEXT_MARKERS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_mode(series: pd.Series) -> str:
    clean = clean_text(series)
    clean = clean[clean.ne("")]
    if clean.empty:
        return ""
    return clean.value_counts().sort_index().idxmax()


def unique_sorted(values: pd.Series) -> list[str]:
    clean = clean_text(values)
    return sorted(value for value in clean.unique().tolist() if value)


def json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def row_to_query(row: pd.Series) -> dict[str, Any]:
    return {
        "query_id": str(int(row["id"])),
        "project_id": int(row["id"]),
        "project_title": row["project_title"],
        "query_text": row["project_objective"],
        "project_keywords": row["project_keywords"],
        "funding_call_id": row["funding_call_id"],
        "funding_call_description": row["funding_call_description"],
        "true_call_label": row["subCall"],
        "masterCall": row["masterCall"],
        "fundingScheme": row["fundingScheme"],
        "totalCost_eur": json_safe(row["totalCost_eur"]),
        "ecMaxContribution_eur": json_safe(row["ecMaxContribution_eur"]),
        "duration_months": json_safe(row["duration_months"]),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_source() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_PATH, low_memory=False, encoding_errors="replace")
    required = {"id", *TEXT_COLUMNS, *NUMERIC_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype("int64")
    for column in TEXT_COLUMNS:
        df[column] = clean_text(df[column])
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    valid = (
        is_valid_text(df[QUERY_FIELD])
        & is_valid_text(df["project_title"])
        & is_valid_text(df["project_keywords"])
        & is_valid_text(df["funding_call_id"])
        & is_valid_text(df[DESCRIPTION_FIELD])
        & is_valid_text(df[LABEL_FIELD])
    )
    return df[valid].drop_duplicates(subset=["id"], keep="first").copy()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = load_source()
    call_sizes = source[LABEL_FIELD].value_counts()
    candidate_labels = sorted(call_sizes[call_sizes >= MIN_PROJECTS_PER_CALL].index)
    eligible = source[source[LABEL_FIELD].isin(candidate_labels)].copy()

    if len(eligible) < TEST_SIZE + DEV_SIZE:
        raise ValueError(
            f"Only {len(eligible)} eligible rows; cannot sample "
            f"{TEST_SIZE} test and {DEV_SIZE} dev rows."
        )

    test = eligible.sample(n=TEST_SIZE, random_state=TEST_SEED).sort_values("id")
    dev_pool = eligible[~eligible["id"].isin(test["id"])]
    dev = dev_pool.sample(n=DEV_SIZE, random_state=DEV_SEED).sort_values("id")

    grouped = eligible.groupby(LABEL_FIELD, sort=True)
    candidates = []
    for call_label, group in grouped:
        candidates.append(
            {
                "call_label": call_label,
                "n_projects": int(len(group)),
                "funding_call_ids": unique_sorted(group["funding_call_id"]),
                "funding_call_descriptions": unique_sorted(group[DESCRIPTION_FIELD]),
                "dominant_masterCall": first_mode(group["masterCall"]),
                "dominant_fundingScheme": first_mode(group["fundingScheme"]),
            }
        )

    test_rows = [row_to_query(row) for _, row in test.iterrows()]
    dev_rows = [row_to_query(row) for _, row in dev.iterrows()]
    ground_truth = {
        "version": VERSION,
        "label_field": LABEL_FIELD,
        "query_split": "test",
        "truth_format": "query_id -> list[correct_subCall]",
        "truths": {row["query_id"]: [row["true_call_label"]] for row in test_rows},
    }
    candidate_pool = {
        "version": VERSION,
        "candidate_pool_type": "all eligible subCall labels",
        "eligibility": {
            "source_file": str(SOURCE_PATH.relative_to(ROOT)),
            "query_field": QUERY_FIELD,
            "label_field": LABEL_FIELD,
            "required_valid_description_field": DESCRIPTION_FIELD,
            "invalid_text_markers": sorted(INVALID_TEXT_MARKERS),
            "minimum_projects_per_call": MIN_PROJECTS_PER_CALL,
        },
        "n_candidates": len(candidates),
        "candidates": candidates,
    }

    outputs = {
        "test_queries": OUTPUT_DIR / "test_queries.jsonl",
        "dev_queries": OUTPUT_DIR / "dev_queries.jsonl",
        "candidate_pool": OUTPUT_DIR / "candidate_pool.json",
        "ground_truth": OUTPUT_DIR / "ground_truth.json",
    }
    write_jsonl(outputs["test_queries"], test_rows)
    write_jsonl(outputs["dev_queries"], dev_rows)
    outputs["candidate_pool"].write_text(
        json.dumps(candidate_pool, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["ground_truth"].write_text(
        json.dumps(ground_truth, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_path = OUTPUT_DIR / "split_manifest.json"
    manifest = {
        "version": VERSION,
        "created_by": "main_experiment/scripts/build_frozen_eval_splits.py",
        "source_file": str(SOURCE_PATH.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE_PATH),
        "test_size": len(test_rows),
        "test_seed": TEST_SEED,
        "dev_size": len(dev_rows),
        "dev_seed": DEV_SEED,
        "candidate_pool_type": "all eligible subCall labels",
        "candidate_count": len(candidates),
        "eligible_project_count": int(len(eligible)),
        "valid_source_project_count": int(len(source)),
        "minimum_projects_per_call": MIN_PROJECTS_PER_CALL,
        "label_field": LABEL_FIELD,
        "query_field": QUERY_FIELD,
        "required_valid_description_field": DESCRIPTION_FIELD,
        "outputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in outputs.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote frozen split to {OUTPUT_DIR.relative_to(ROOT)}")
    print(f"Valid source projects: {len(source):,}")
    print(f"Eligible projects: {len(eligible):,}")
    print(f"Candidate subCalls: {len(candidates):,}")
    print(f"Test queries: {len(test_rows):,} (seed {TEST_SEED})")
    print(f"Dev queries: {len(dev_rows):,} (seed {DEV_SEED})")


if __name__ == "__main__":
    main()
