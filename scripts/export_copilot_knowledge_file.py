from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "retriever_artifacts" / "funding_calls.parquet"
OUTPUT_DIR = ROOT / "copilot_studio_upload"
OUTPUT_CSV = OUTPUT_DIR / "funding_call_knowledge.csv"
OUTPUT_XLSX = OUTPUT_DIR / "funding_call_knowledge.xlsx"
REPORT = ROOT / "reports" / "copilot_studio_knowledge_file.md"

PROFILE_LIMIT = 1_200


def compact_text(value: object, limit: int = PROFILE_LIMIT) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calls = pd.read_parquet(SOURCE)

    knowledge = pd.DataFrame(
        {
            "funding_call_id": calls["funding_call_id"],
            "description": calls["funding_call_description"].map(compact_text),
            "historical_profile_summary": calls["historical_profile_summary"].map(
                compact_text
            ),
            "median_budget_eur": calls["median_ecMaxContribution_eur"],
            "median_total_cost_eur": calls["median_totalCost_eur"],
            "median_duration_months": calls["median_duration_months"],
            "n_historical_projects": calls["n_projects"],
        }
    )

    forbidden = {"fundingScheme", "subCall", "masterCall"}
    leaked = forbidden.intersection(knowledge.columns)
    if leaked:
        raise ValueError(f"Forbidden leakage-prone columns included: {sorted(leaked)}")

    knowledge = knowledge.sort_values("funding_call_id").reset_index(drop=True)
    knowledge.to_csv(OUTPUT_CSV, index=False)
    knowledge.to_excel(OUTPUT_XLSX, index=False)

    report = f"""# Copilot Studio Knowledge File

Generated upload files:

```text
{OUTPUT_CSV.relative_to(ROOT)}
{OUTPUT_XLSX.relative_to(ROOT)}
```

Each row is one candidate `funding_call_id`.

## Included Columns

| Column | Meaning |
|---|---|
| `funding_call_id` | Candidate call identifier to rank and return. |
| `description` | CORDIS call/topic description. |
| `historical_profile_summary` | Compact summary of historical funded project text under the call. |
| `median_budget_eur` | Median EC contribution among historical projects under the call. |
| `median_total_cost_eur` | Median total project cost among historical projects under the call. |
| `median_duration_months` | Median project duration among historical projects under the call. |
| `n_historical_projects` | Number of historical projects used to build the profile. |

## Deliberately Excluded

These fields are not included because they can act as near-label proxies:

```text
fundingScheme
subCall
masterCall
```

## Counts

```text
rows: {len(knowledge):,}
unique funding_call_id: {knowledge['funding_call_id'].nunique():,}
profile character limit: {PROFILE_LIMIT}
```
"""
    REPORT.write_text(report, encoding="utf-8")

    print(f"Wrote {OUTPUT_CSV.relative_to(ROOT)} ({len(knowledge):,} rows)")
    print(f"Wrote {OUTPUT_XLSX.relative_to(ROOT)} ({len(knowledge):,} rows)")
    print(f"Wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
