from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "samples" / "eval_sample_200_seed42.csv"
PROFILE_PATH = ROOT / "data" / "cleaned" / "call_profiles_subcall.csv"
REPORT_PATH = ROOT / "reports" / "eval_sample_200_diagnostics.md"
SUMMARY_PATH = ROOT / "reports" / "eval_sample_200_diagnostics.json"
IDS_PATH = ROOT / "data" / "samples" / "eval_sample_200_seed42_ids.csv"
FLAGGED_PATH = ROOT / "data" / "samples" / "eval_sample_200_seed42_small_call_flags.csv"


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


def main() -> None:
    sample = pd.read_csv(SAMPLE_PATH, low_memory=False)
    profiles = pd.read_csv(PROFILE_PATH)

    sample = sample.copy()
    sample["scheme_family"] = sample["fundingScheme"].map(scheme_family)
    sample = sample.merge(
        profiles[["call_label", "n_projects"]],
        on="call_label",
        how="left",
        validate="many_to_one",
    )
    sample["leave_one_out_members"] = sample["n_projects"] - 1
    sample["small_call_leaveout_risk"] = sample["n_projects"].between(5, 6)

    family_counts = (
        sample["scheme_family"]
        .value_counts()
        .rename_axis("scheme_family")
        .reset_index(name="n_queries")
    )
    family_counts["share"] = family_counts["n_queries"] / len(sample)

    six_families = ["MSCA", "ERC", "SME", "RIA", "IA", "CSA"]
    family_lookup = family_counts.set_index("scheme_family")["n_queries"].to_dict()
    six_family_counts = {family: int(family_lookup.get(family, 0)) for family in six_families}
    low_count_families = {
        family: count for family, count in six_family_counts.items() if count < 30
    }

    risky = sample[sample["small_call_leaveout_risk"]].copy()
    ids = sample[["id", "call_label", "fundingScheme", "scheme_family"]].copy()
    ids.to_csv(IDS_PATH, index=False)
    risky[
        [
            "id",
            "title",
            "call_label",
            "fundingScheme",
            "scheme_family",
            "n_projects",
            "leave_one_out_members",
        ]
    ].to_csv(FLAGGED_PATH, index=False)

    summary = {
        "sample_size": int(len(sample)),
        "unique_query_ids": int(sample["id"].nunique()),
        "six_family_counts": six_family_counts,
        "low_count_families_lt_30": low_count_families,
        "other_count": int(family_lookup.get("OTHER", 0)),
        "small_call_queries_n_projects_5_or_6": int(len(risky)),
        "small_call_unique_labels": int(risky["call_label"].nunique()),
        "min_call_size_in_sample": int(sample["n_projects"].min()),
        "median_call_size_in_sample": float(sample["n_projects"].median()),
        "max_call_size_in_sample": int(sample["n_projects"].max()),
        "query_ids_file": str(IDS_PATH.relative_to(ROOT)),
        "small_call_flags_file": str(FLAGGED_PATH.relative_to(ROOT)),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Evaluation Sample Diagnostics",
        "",
        "Sample file:",
        "",
        f"```text\n{SAMPLE_PATH.relative_to(ROOT)}\n```",
        "",
        "## Fixed Query IDs",
        "",
        f"- Sample size: {len(sample):,}",
        f"- Unique query IDs: {sample['id'].nunique():,}",
        f"- Saved ID list: `{IDS_PATH.relative_to(ROOT)}`",
        "",
        "## FundingScheme Family Distribution",
        "",
        "| Family | Queries | Share |",
        "|---|---:|---:|",
    ]
    family_order = six_families + [family for family in family_counts["scheme_family"] if family not in six_families]
    seen = set()
    for family in family_order:
        if family in seen:
            continue
        seen.add(family)
        count = int(family_lookup.get(family, 0))
        lines.append(f"| `{family}` | {count:,} | {count / len(sample):.1%} |")

    lines.extend(
        [
            "",
            "Families below 30 queries:",
            "",
        ]
    )
    if low_count_families:
        for family, count in low_count_families.items():
            lines.append(f"- `{family}`: {count}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Small subCall Leave-One-Out Risk",
            "",
            "A query is flagged if its `subCall` contains only 5 or 6 usable projects. After leaving the query out, the true-call profile has only 4 or 5 projects.",
            "",
            f"- Flagged queries: {len(risky):,}",
            f"- Unique flagged subCalls: {risky['call_label'].nunique():,}",
            f"- Saved flagged list: `{FLAGGED_PATH.relative_to(ROOT)}`",
            "",
            "Call-size distribution among sampled queries:",
            "",
            f"- Min call size: {int(sample['n_projects'].min()):,}",
            f"- Median call size: {sample['n_projects'].median():.1f}",
            f"- Max call size: {int(sample['n_projects'].max()):,}",
            "",
            "## Flagged Queries",
            "",
            "| id | call_label | scheme | n_projects | leave-one-out members |",
            "|---|---|---|---:|---:|",
        ]
    )
    for _, row in risky.sort_values(["n_projects", "call_label", "id"]).iterrows():
        lines.append(
            f"| `{row['id']}` | `{row['call_label']}` | `{row['fundingScheme']}` | "
            f"{int(row['n_projects'])} | {int(row['leave_one_out_members'])} |"
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)}")
    print(f"Wrote {IDS_PATH.relative_to(ROOT)}")
    print(f"Wrote {FLAGGED_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
