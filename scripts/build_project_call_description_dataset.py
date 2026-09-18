from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = (
    ROOT.parent
    / "CORDIS - EU research projects under Horizon 2020 (2014-2020)"
    / "Publications Office"
    / "9-cordis-h2020projects-csv"
)
CLEAN_PROJECTS = ROOT / "data" / "cleaned" / "cordis_projects_clean.csv"
OUTPUT = ROOT / "data" / "cleaned" / "project_objectives_with_call_descriptions.csv"
REPORT = ROOT / "reports" / "project_call_description_dataset_summary.md"
INVALID_TEXT_MARKERS = {"", "false", "true", "nan", "none", "null", "na", "n/a", "<na>"}


def clean_string(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def is_valid_text(series: pd.Series) -> pd.Series:
    return ~clean_string(series).str.lower().isin(INVALID_TEXT_MARKERS)


def main() -> None:
    projects = pd.read_csv(CLEAN_PROJECTS, low_memory=False, encoding_errors="replace")
    topics = pd.read_csv(
        RAW_DIR / "topics.csv",
        sep=";",
        quotechar='"',
        on_bad_lines="skip",
        low_memory=False,
        encoding_errors="replace",
    )

    projects = projects[
        [
            "id",
            "title",
            "objective",
            "keywords",
            "subCall",
            "masterCall",
            "fundingScheme",
            "totalCost_eur",
            "ecMaxContribution_eur",
            "startDate",
            "endDate",
            "duration_months",
        ]
    ].copy()
    projects["id"] = pd.to_numeric(projects["id"], errors="coerce")
    projects["project_title"] = clean_string(projects["title"])
    projects["project_objective"] = clean_string(projects["objective"])
    projects["project_keywords"] = clean_string(projects["keywords"])

    topics = topics.rename(
        columns={
            "projectID": "id",
            "topic": "funding_call_id",
            "title": "funding_call_description",
        }
    )
    topics["id"] = pd.to_numeric(topics["id"], errors="coerce")
    topics["funding_call_id"] = clean_string(topics["funding_call_id"])
    topics["funding_call_description"] = clean_string(topics["funding_call_description"])

    merged = projects.merge(
        topics[["id", "funding_call_id", "funding_call_description"]],
        on="id",
        how="left",
        validate="one_to_one",
    )

    before_drop = len(merged)
    merged = merged[
        is_valid_text(merged["project_objective"])
        & is_valid_text(merged["project_title"])
        & is_valid_text(merged["project_keywords"])
        & is_valid_text(merged["funding_call_id"])
        & is_valid_text(merged["funding_call_description"])
    ].copy()

    output = merged[
        [
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
            "startDate",
            "endDate",
            "duration_months",
        ]
    ].sort_values("id")

    output.to_csv(OUTPUT, index=False)

    missing_description_rows = before_drop - len(output)
    report = f"""# Project Objective + Funding Call Description Dataset

Output file:

```text
{OUTPUT.relative_to(ROOT)}
```

## Definition

This file joins cleaned CORDIS project rows with `topics.csv`.

| Output column | Source |
|---|---|
| `id` | project id |
| `project_title` | cleaned project `title` |
| `project_objective` | cleaned project `objective` |
| `project_keywords` | cleaned project `keywords` |
| `funding_call_id` | `topics.csv` field `topic` |
| `funding_call_description` | `topics.csv` field `title` |
| `subCall` | original project `subCall` |
| `masterCall` | original project `masterCall` |
| `fundingScheme` | original project `fundingScheme` |
| `totalCost_eur` | cleaned project `totalCost` parsed as EUR |
| `ecMaxContribution_eur` | cleaned project `ecMaxContribution` parsed as EUR |
| `startDate` | original project `startDate` |
| `endDate` | original project `endDate` |
| `duration_months` | cleaned project duration derived from `startDate` and `endDate` |

Note: `funding_call_description` is the short CORDIS topic title, not the full official call scope text.

## Counts

- Cleaned project rows before join: {len(projects):,}
- Rows after join before dropping missing text/description: {before_drop:,}
- Rows kept: {len(output):,}
- Rows dropped due to missing project text, missing/placeholder project keywords, or funding-call description: {missing_description_rows:,}
- Unique funding call ids kept: {output['funding_call_id'].nunique():,}
- Unique funding call descriptions kept: {output['funding_call_description'].nunique():,}
- Rows with `totalCost_eur`: {output['totalCost_eur'].notna().sum():,}
- Rows with `ecMaxContribution_eur`: {output['ecMaxContribution_eur'].notna().sum():,}
- Rows with `duration_months`: {output['duration_months'].notna().sum():,}
"""
    REPORT.write_text(report, encoding="utf-8")

    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(output):,} rows)")
    print(f"Wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
