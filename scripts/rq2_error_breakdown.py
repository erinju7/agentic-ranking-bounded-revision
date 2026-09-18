"""RQ2 error-analysis breakdown.

Deepens the paper's Error Analysis from a single call-size figure into a set of
reproducible cross-tabs computed directly from the ranked outputs:

  1. Call-size strata  - true-call rank vs. call size (n_projects), using the
     paper's >=150-project "large" threshold, to quantify the call-size bias.
  2. ERC-sibling stratum - ERC-* calls (label prefix) vs. the rest.
  3. Top-1 confusion decomposition - correct / temporal-sibling (right family,
     wrong year) / same-masterCall / unrelated, so the "most errors are not
     near-miss year confusions" claim is measured, not asserted.
  4. Promotion bias - median size of the call the model puts at rank 1 vs. the
     median size of the true call.

Consumes only existing `{split}_rerank_outputs.jsonl` files; no API calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

from rq2_core import RESULT_ROOT, load_candidate_pool, load_ground_truth

YEAR = re.compile(r"(?:19|20)\d{2}")
LARGE_CALL_MIN_PROJECTS = 150  # paper's definition of a large, broadly-scoped call
SMALL_CALL_MAX_PROJECTS = 25


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def family_root(label: str) -> str:
    """Call label with year tokens removed, so temporal siblings collapse."""
    return YEAR.sub("", label)


def pool_metadata() -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for candidate in load_candidate_pool():
        n_projects = candidate.get("n_projects")
        try:
            n_projects = int(n_projects)
        except (TypeError, ValueError):
            n_projects = None
        meta[candidate["call_label"]] = {
            "n_projects": n_projects,
            "master_call": candidate.get("dominant_masterCall", ""),
        }
    return meta


def size_bin(n_projects: int | None) -> str:
    if n_projects is None:
        return "unknown"
    if n_projects >= LARGE_CALL_MIN_PROJECTS:
        return "large (>=150)"
    if n_projects <= SMALL_CALL_MAX_PROJECTS:
        return "small (<=25)"
    return "medium (26-149)"


def stratum_metrics(true_ranks: list[int]) -> dict[str, Any]:
    if not true_ranks:
        return {"n": 0}
    n = len(true_ranks)
    return {
        "n": n,
        "recall@1": round(sum(r <= 1 for r in true_ranks) / n, 3),
        "recall@5": round(sum(r <= 5 for r in true_ranks) / n, 3),
        "mean_true_rank": round(sum(true_ranks) / n, 2),
    }


def ranked_by_query(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """One ranked list per query; keep the first repeat when repeats were run."""
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row.get("repeat", 0)) != 0:
            continue
        if row.get("ranked"):
            result.setdefault(str(row["query_id"]), row["ranked"])
    return result


def breakdown(directory: Path, split: str, meta: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    outputs = directory / f"{split}_rerank_outputs.jsonl"
    metadata_path = directory / f"{split}_run_metadata.json"
    if not outputs.exists():
        return None
    run_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    ranked = ranked_by_query(read_jsonl(outputs))
    truths = load_ground_truth(split)
    top_n = run_metadata.get("top_n", 50)

    size_strata: dict[str, list[int]] = {}
    erc_strata: dict[str, list[int]] = {"ERC siblings": [], "non-ERC": []}
    confusion = {"correct": 0, "temporal_sibling": 0, "same_masterCall": 0, "unrelated": 0}
    top1_sizes: list[int] = []
    true_sizes: list[int] = []

    for query_id, ranked_list in ranked.items():
        truth = truths.get(str(query_id))
        if not truth:
            continue
        true_label = truth[0]
        truth_set = set(truth)
        true_rank = next(
            (row["rank"] for row in ranked_list if row["call_label"] in truth_set),
            top_n + 1,
        )
        top1_label = ranked_list[0]["call_label"] if ranked_list else ""

        true_meta = meta.get(true_label, {})
        size_strata.setdefault(size_bin(true_meta.get("n_projects")), []).append(true_rank)
        erc_key = "ERC siblings" if true_label.startswith("ERC") else "non-ERC"
        erc_strata[erc_key].append(true_rank)

        if true_meta.get("n_projects") is not None:
            true_sizes.append(true_meta["n_projects"])
        top1_size = meta.get(top1_label, {}).get("n_projects")
        if top1_size is not None:
            top1_sizes.append(top1_size)

        if top1_label in truth_set:
            confusion["correct"] += 1
        elif family_root(top1_label) == family_root(true_label):
            confusion["temporal_sibling"] += 1
        elif (
            meta.get(top1_label, {}).get("master_call")
            and meta.get(top1_label, {}).get("master_call")
            == true_meta.get("master_call")
        ):
            confusion["same_masterCall"] += 1
        else:
            confusion["unrelated"] += 1

    n_queries = sum(len(v) for v in size_strata.values())
    n_errors = n_queries - confusion["correct"]
    return {
        "condition": directory.name.replace("single_agent_rerank_", ""),
        "model": run_metadata.get("model", "unknown"),
        "split": split,
        "n_queries": n_queries,
        "call_size_strata": {k: stratum_metrics(v) for k, v in sorted(size_strata.items())},
        "erc_sibling_strata": {k: stratum_metrics(v) for k, v in erc_strata.items()},
        "top1_confusion": confusion,
        "top1_error_share": (
            {k: round(confusion[k] / n_errors, 3) for k in confusion if k != "correct"}
            if n_errors
            else {}
        ),
        "promotion_bias": {
            "median_projects_of_top1_call": (
                int(statistics.median(top1_sizes)) if top1_sizes else None
            ),
            "median_projects_of_true_call": (
                int(statistics.median(true_sizes)) if true_sizes else None
            ),
        },
    }


def condition_dirs() -> list[Path]:
    return sorted(
        d
        for d in RESULT_ROOT.glob("single_agent_rerank_*")
        if d.is_dir() and "mock" not in d.name
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 error-analysis breakdown.")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--output-dir", default=str(RESULT_ROOT / "rq2_error_breakdown"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = pool_metadata()
    reports: list[dict[str, Any]] = []
    for directory in condition_dirs():
        for split in args.splits:
            report = breakdown(directory, split, meta)
            if report is not None:
                reports.append(report)

    (output_dir / "rq2_error_breakdown.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Flat CSV of the call-size strata for direct table use.
    csv_rows = []
    for report in reports:
        for stratum, metrics in report["call_size_strata"].items():
            csv_rows.append(
                {
                    "condition": report["condition"],
                    "model": report["model"],
                    "split": report["split"],
                    "stratum": stratum,
                    **metrics,
                }
            )
    if csv_rows:
        with (output_dir / "rq2_callsize_strata.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    for report in reports:
        print(f"\n=== {report['condition']} / {report['model']} / {report['split']} "
              f"(N={report['n_queries']}) ===")
        for stratum, m in report["call_size_strata"].items():
            if m["n"]:
                print(f"  {stratum:<16} n={m['n']:<4} R@1={m['recall@1']:<5} "
                      f"mean_rank={m['mean_true_rank']}")
        c = report["top1_confusion"]
        print(f"  top-1: correct={c['correct']} temporal={c['temporal_sibling']} "
              f"sameMaster={c['same_masterCall']} unrelated={c['unrelated']}")
        pb = report["promotion_bias"]
        print(f"  promotion bias: rank-1 median size={pb['median_projects_of_top1_call']} "
              f"vs true median size={pb['median_projects_of_true_call']}")
    print(f"\nWrote {output_dir/'rq2_error_breakdown.json'} and rq2_callsize_strata.csv")


if __name__ == "__main__":
    main()
