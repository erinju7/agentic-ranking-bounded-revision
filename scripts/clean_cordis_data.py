from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment_config.json"

CORE_COLUMNS = [
    "id",
    "acronym",
    "status",
    "title",
    "startDate",
    "endDate",
    "totalCost",
    "ecMaxContribution",
    "topics",
    "frameworkProgramme",
    "masterCall",
    "subCall",
    "fundingScheme",
    "objective",
    "keywords",
]

TEXT_COLUMNS = [
    "id",
    "acronym",
    "status",
    "title",
    "topics",
    "frameworkProgramme",
    "masterCall",
    "subCall",
    "fundingScheme",
    "objective",
    "keywords",
]

MISSING_MARKERS = {"", "nan", "none", "null", "na", "n/a", "<na>"}


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalise_text(value: Any) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in MISSING_MARKERS:
        return pd.NA
    return text


def parse_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in MISSING_MARKERS:
        return np.nan
    text = text.replace("\u00a0", "").replace(" ", "")
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")
    return pd.to_numeric(text, errors="coerce")


def iqr(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def dominant_value(series: pd.Series) -> str | pd.NA:
    counts = series.dropna().astype(str).str.strip()
    counts = counts[counts.ne("")]
    if counts.empty:
        return pd.NA
    return counts.value_counts().index[0]


def write_report(
    report_path: Path,
    *,
    raw_rows: int,
    raw_columns: list[str],
    missing_expected: list[str],
    exact_duplicate_rows: int,
    duplicate_id_rows: int,
    clean: pd.DataFrame,
    usable: pd.DataFrame,
    eligible: pd.DataFrame,
    call_profiles: pd.DataFrame,
    sample: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    label = config["retrieval_task"]["label_field"]
    query = config["retrieval_task"]["query_field"]
    min_projects = config["retrieval_task"]["minimum_projects_per_call"]
    sample_size = config["sampling"]["sample_size"]
    seed = config["sampling"]["random_seed"]

    missing_counts = clean[CORE_COLUMNS].isna().sum().sort_values(ascending=False)
    call_sizes = usable["call_label"].value_counts()
    top_calls = call_sizes.head(15)

    lines = [
        "# CORDIS Data Cleaning Report",
        "",
        "## Source",
        "",
        f"- Raw project CSV: `{config['dataset']['raw_project_csv']}`",
        f"- Separator: `{config['dataset']['csv_sep']}`",
        f"- Quote character: `{config['dataset']['csv_quotechar']}`",
        f"- Bad-line handling: `{config['dataset']['on_bad_lines']}`",
        "",
        "## Row Counts",
        "",
        f"- Raw parsed rows: {raw_rows:,}",
        f"- Raw parsed columns: {len(raw_columns):,}",
        f"- Exact duplicate rows removed: {exact_duplicate_rows:,}",
        f"- Duplicate project-id rows removed: {duplicate_id_rows:,}",
        f"- Cleaned project rows: {len(clean):,}",
        f"- Usable rows with non-empty `{query}` and `{label}`: {len(usable):,}",
        f"- Eligible calls with >= {min_projects} usable projects: {len(call_profiles):,}",
        f"- Projects in eligible calls: {len(eligible):,}",
        f"- Evaluation sample size: {len(sample):,} requested / {sample_size:,} configured",
        f"- Evaluation sample seed: {seed}",
        "",
        "## Expected Columns",
        "",
        "- Missing expected columns: "
        + (", ".join(f"`{column}`" for column in missing_expected) if missing_expected else "none"),
        "",
        "## Derived Columns",
        "",
        "- `totalCost_eur`",
        "- `ecMaxContribution_eur`",
        "- `startDate_parsed`",
        "- `endDate_parsed`",
        "- `duration_days`",
        "- `duration_months`",
        "- `ec_contribution_ratio`",
        "- `call_label` = `subCall`",
        "- `objective_word_count`",
        "",
        "## Derived Field Quality",
        "",
        f"- Missing `totalCost_eur`: {int(clean['totalCost_eur'].isna().sum()):,}",
        f"- Missing `ecMaxContribution_eur`: {int(clean['ecMaxContribution_eur'].isna().sum()):,}",
        f"- Missing `startDate_parsed`: {int(clean['startDate_parsed'].isna().sum()):,}",
        f"- Missing `endDate_parsed`: {int(clean['endDate_parsed'].isna().sum()):,}",
        f"- Missing `duration_months`: {int(clean['duration_months'].isna().sum()):,}",
        f"- Missing `ec_contribution_ratio`: {int(clean['ec_contribution_ratio'].isna().sum()):,}",
        "",
        "## Missing Values In Core Columns",
        "",
        "| Column | Missing rows |",
        "|---|---:|",
    ]
    lines.extend(f"| `{column}` | {count:,} |" for column, count in missing_counts.items())

    lines.extend(
        [
            "",
            "## Call-Size Distribution",
            "",
            f"- Unique usable `{label}` calls: {call_sizes.size:,}",
            f"- Min projects per call: {int(call_sizes.min()) if not call_sizes.empty else 0:,}",
            f"- Median projects per call: {call_sizes.median():.2f}",
            f"- Mean projects per call: {call_sizes.mean():.2f}",
            f"- Max projects per call: {int(call_sizes.max()) if not call_sizes.empty else 0:,}",
            f"- Calls with >= 2 projects: {int((call_sizes >= 2).sum()):,}",
            f"- Calls with >= 5 projects: {int((call_sizes >= 5).sum()):,}",
            f"- Calls with >= 10 projects: {int((call_sizes >= 10).sum()):,}",
            "",
            "## Top Calls",
            "",
            "| subCall | Projects |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{call}` | {count:,} |" for call, count in top_calls.items())

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `data/cleaned/cordis_projects_clean.csv`",
            "- `data/cleaned/call_profiles_subcall.csv`",
            f"- `{config['sampling']['sample_output']}`",
            "- `reports/data_cleaning_summary.json`",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = load_config()
    raw_path = (ROOT / config["dataset"]["raw_project_csv"]).resolve()

    clean_dir = ROOT / "data" / "cleaned"
    sample_dir = ROOT / "data" / "samples"
    report_dir = ROOT / "reports"
    clean_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(
        raw_path,
        sep=config["dataset"]["csv_sep"],
        quotechar=config["dataset"]["csv_quotechar"],
        on_bad_lines=config["dataset"]["on_bad_lines"],
        low_memory=False,
    )

    missing_expected = [column for column in CORE_COLUMNS if column not in raw.columns]
    if missing_expected:
        raise ValueError(f"Missing expected columns: {missing_expected}")

    exact_duplicate_rows = int(raw.duplicated().sum())
    raw = raw.drop_duplicates().copy()

    duplicate_id_rows = int(raw.duplicated(subset=["id"], keep="first").sum())
    raw = raw.drop_duplicates(subset=["id"], keep="first").copy()

    clean = raw[CORE_COLUMNS].copy()
    for column in TEXT_COLUMNS:
        clean[column] = clean[column].map(normalise_text)

    clean["totalCost_eur"] = clean["totalCost"].map(parse_number)
    clean["ecMaxContribution_eur"] = clean["ecMaxContribution"].map(parse_number)
    clean["startDate_parsed"] = pd.to_datetime(clean["startDate"], errors="coerce")
    clean["endDate_parsed"] = pd.to_datetime(clean["endDate"], errors="coerce")
    clean["duration_days"] = (
        clean["endDate_parsed"] - clean["startDate_parsed"]
    ).dt.days
    clean.loc[clean["duration_days"] < 0, "duration_days"] = np.nan
    clean["duration_months"] = clean["duration_days"] / 30.4375
    clean["ec_contribution_ratio"] = (
        clean["ecMaxContribution_eur"] / clean["totalCost_eur"]
    )
    clean.loc[
        ~np.isfinite(clean["ec_contribution_ratio"]),
        "ec_contribution_ratio",
    ] = np.nan
    clean["call_label"] = clean["subCall"]
    clean["objective_word_count"] = (
        clean["objective"].fillna("").astype(str).str.split().str.len()
    )

    query_column = config["retrieval_task"]["query_field"]
    label_column = config["retrieval_task"]["label_field"]
    min_projects = config["retrieval_task"]["minimum_projects_per_call"]
    sample_size = config["sampling"]["sample_size"]
    seed = config["sampling"]["random_seed"]

    usable = clean[
        clean[query_column].notna()
        & clean[label_column].notna()
        & clean[query_column].astype(str).str.strip().ne("")
        & clean[label_column].astype(str).str.strip().ne("")
    ].copy()

    call_sizes = usable["call_label"].value_counts()
    eligible_call_labels = call_sizes[call_sizes >= min_projects].index
    eligible = usable[usable["call_label"].isin(eligible_call_labels)].copy()

    profile_rows = []
    for call_label, group in eligible.groupby("call_label", sort=True):
        profile_rows.append(
            {
                "call_label": call_label,
                "n_projects": len(group),
                "concatenated_objectives": " ".join(group["objective"].dropna()),
                "median_totalCost_eur": group["totalCost_eur"].median(),
                "iqr_totalCost_eur": iqr(group["totalCost_eur"]),
                "median_ecMaxContribution_eur": group["ecMaxContribution_eur"].median(),
                "iqr_ecMaxContribution_eur": iqr(group["ecMaxContribution_eur"]),
                "median_duration_months": group["duration_months"].median(),
                "iqr_duration_months": iqr(group["duration_months"]),
                "dominant_fundingScheme": dominant_value(group["fundingScheme"]),
                "dominant_masterCall": dominant_value(group["masterCall"]),
                "dominant_topics": dominant_value(group["topics"]),
            }
        )
    call_profiles = pd.DataFrame(profile_rows).sort_values(
        ["n_projects", "call_label"],
        ascending=[False, True],
    )

    if len(eligible) < sample_size:
        raise ValueError(
            f"Only {len(eligible)} eligible rows; cannot sample {sample_size}."
        )
    sample = eligible.sample(n=sample_size, random_state=seed).sort_values("id")

    clean_output = clean_dir / "cordis_projects_clean.csv"
    profile_output = clean_dir / "call_profiles_subcall.csv"
    sample_output = ROOT / config["sampling"]["sample_output"]
    report_output = report_dir / "data_cleaning_report.md"
    summary_output = report_dir / "data_cleaning_summary.json"

    clean.to_csv(clean_output, index=False)
    call_profiles.to_csv(profile_output, index=False)
    sample.to_csv(sample_output, index=False)

    summary = {
        "raw_rows": int(len(raw) + exact_duplicate_rows + duplicate_id_rows),
        "cleaned_rows": int(len(clean)),
        "usable_rows": int(len(usable)),
        "eligible_calls": int(len(call_profiles)),
        "eligible_projects": int(len(eligible)),
        "sample_size": int(len(sample)),
        "sample_seed": int(seed),
        "label_field": "subCall",
        "query_field": query_column,
        "min_projects_per_call": int(min_projects),
        "derived_missing": {
            "totalCost_eur": int(clean["totalCost_eur"].isna().sum()),
            "ecMaxContribution_eur": int(clean["ecMaxContribution_eur"].isna().sum()),
            "startDate_parsed": int(clean["startDate_parsed"].isna().sum()),
            "endDate_parsed": int(clean["endDate_parsed"].isna().sum()),
            "duration_months": int(clean["duration_months"].isna().sum()),
            "ec_contribution_ratio": int(clean["ec_contribution_ratio"].isna().sum()),
        },
        "outputs": {
            "cleaned_projects": str(clean_output.relative_to(ROOT)),
            "call_profiles": str(profile_output.relative_to(ROOT)),
            "evaluation_sample": str(sample_output.relative_to(ROOT)),
            "cleaning_report": str(report_output.relative_to(ROOT)),
        },
    }
    summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_report(
        report_output,
        raw_rows=int(len(raw) + exact_duplicate_rows + duplicate_id_rows),
        raw_columns=list(pd.read_csv(raw_path, sep=config["dataset"]["csv_sep"], quotechar=config["dataset"]["csv_quotechar"], on_bad_lines=config["dataset"]["on_bad_lines"], low_memory=False, nrows=0).columns),
        missing_expected=missing_expected,
        exact_duplicate_rows=exact_duplicate_rows,
        duplicate_id_rows=duplicate_id_rows,
        clean=clean,
        usable=usable,
        eligible=eligible,
        call_profiles=call_profiles,
        sample=sample,
        config=config,
    )

    print(f"Wrote {clean_output.relative_to(ROOT)} ({len(clean):,} rows)")
    print(f"Wrote {profile_output.relative_to(ROOT)} ({len(call_profiles):,} calls)")
    print(f"Wrote {sample_output.relative_to(ROOT)} ({len(sample):,} rows)")
    print(f"Wrote {report_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
