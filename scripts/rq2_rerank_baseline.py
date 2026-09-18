"""RQ2 single-agent reranker over a fixed deterministic retriever top-N set.

Protocol
--------
Stage 1 (deterministic, offline): the leave-one-out historical-profile TF-IDF
retriever produces a frozen top-N candidate set per query
(`rq2_build_deterministic_topn.py`). This runner consumes that file; it never
re-runs retrieval.

Stage 2 (this script): a single-node LangGraph agent reranks only the retriever's
top-N candidates. The candidate text exposed to the model is intentionally
information-floor: `call_label` + `funding_call_descriptions` only. fundingScheme
and masterCall are excluded as near-label leakage (enforced in rq2_core by
BLOCKED_MODEL_INPUT_FIELDS).

Because rerank is confined to the retriever's top-N, retriever recall is a hard
ceiling: if the true subCall is not in the top-N, no rerank can recover it. The
summary reports both the rerank metrics and the retriever baseline over the same
top-N set, plus a head-to-head rank comparison so the rerank contribution is
isolated from retrieval.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langgraph.graph import StateGraph

from rq2_core import (
    DEFAULT_GEMINI_MODEL,
    EVALUATOR_VERSION,
    FROZEN_DIR,
    GEMINI_TEMPERATURE,
    PARSER_VERSION,
    PROMPT_VERSION,
    RESULT_ROOT,
    RQ2State,
    build_rank_prompt,
    load_candidate_pool,
    load_frozen_split,
    load_ground_truth,
    make_client,
    provider_of,
    rank_metrics,
    rank_with_llm,
    raw_output_parse_ok,
    read_jsonl,
)


DEFAULT_OUTPUT_DIR = RESULT_ROOT / "single_agent_rerank_baseline"
ENRICHED_OUTPUT_DIR = RESULT_ROOT / "single_agent_rerank_enriched"
ENRICHED_TARGETED_OUTPUT_DIR = RESULT_ROOT / "single_agent_rerank_enriched_targeted"
HISTORICAL_CONTEXT_OUTPUT_DIR = RESULT_ROOT / "single_agent_rerank_historical_context"
MOCK_OUTPUT_DIR = RESULT_ROOT / "single_agent_rerank_baseline_mock"
DEFAULT_RETRIEVER_FILE = (
    RESULT_ROOT
    / "deterministic_topn_retriever"
    / "dev_historical_top50_candidates.jsonl"
)


def _relative_retriever_path(path: Path) -> str:
    """Repo-relative path for provenance; robust to relative/absolute inputs."""
    try:
        return str(path.resolve().relative_to(RESULT_ROOT.parents[1]))
    except ValueError:
        return str(path)


class MockRankClient:
    """Deterministic plumbing check; not an experimental model."""

    def generate(self, prompt: str) -> str:
        payload = json.loads(prompt.split("Input:", 1)[1].strip())
        labels = [candidate["call_label"] for candidate in payload["candidates"]]
        return json.dumps(
            {
                "ranked_call_labels": labels,
                "scores": {label: 100 - index for index, label in enumerate(labels)},
            },
            ensure_ascii=False,
        )


def build_graph(client: Any, *, anonymize: bool = False):
    def rank_node(state: RQ2State) -> RQ2State:
        ranked, raw_output = rank_with_llm(
            client,
            state["query"],
            state["candidates"],
            anonymize=anonymize,
        )
        return {**state, "ranked": ranked, "raw_output": raw_output}

    graph = StateGraph(RQ2State)
    graph.add_node("rank_node", rank_node)
    graph.set_entry_point("rank_node")
    graph.set_finish_point("rank_node")
    return graph.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank a fixed deterministic retriever top-N set for RQ2.",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--retriever-file",
        default=str(DEFAULT_RETRIEVER_FILE),
        help="Frozen top-N candidate JSONL from rq2_build_deterministic_topn.py.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Optionally truncate the retriever set further (must be <= file top_n).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Rerank each query this many times to measure stability/variance.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of LLM calls to run before checkpointing outputs.",
    )
    parser.add_argument("--sleep-between-calls", type=float, default=0.0)
    parser.add_argument("--sleep-between-batches", type=float, default=0.0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing (query_id, repeat) outputs in the output directory.",
    )
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--mock", action="store_true", help="Deterministic mock output.")
    parser.add_argument(
        "--dry-run-prompt",
        action="store_true",
        help="Write the first assembled prompt and exit without calling the graph.",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required to rerank the sealed test split.",
    )
    parser.add_argument(
        "--anonymize-candidates",
        action="store_true",
        help=(
            "Memorization control: replace each candidate's real subCall label "
            "with an opaque per-query alias and scrub years / call-codes from the "
            "descriptions, so a memorizing backend cannot recover the answer from "
            "weights. Aliases are mapped back to real labels for scoring."
        ),
    )
    parser.add_argument(
        "--candidate-descriptions",
        default=None,
        help=(
            "Lever 2 enrichment: JSON map {subCall: [official scope text]} from "
            "fetch_topic_descriptions.py. Overrides the thin catalogue title while "
            "leaving retrieval and the frozen pool untouched (clean ablation)."
        ),
    )
    parser.add_argument(
        "--desc-char-budget",
        type=int,
        default=900,
        help="Per-candidate char cap for enriched scope text (prompt-size guard).",
    )
    parser.add_argument(
        "--enrich-thin-only",
        action="store_true",
        help=(
            "Targeted enrichment: only replace the catalogue title with official "
            "scope when the title is degenerate (<= --thin-threshold words). Crisp "
            "discriminative titles are kept as-is."
        ),
    )
    parser.add_argument(
        "--thin-threshold",
        type=int,
        default=5,
        help="Catalogue word count at/below which a title counts as degenerate.",
    )
    parser.add_argument(
        "--historical-context",
        action="store_true",
        help=(
            "Context-awareness condition: append the leave-one-out historical "
            "project profile (title + objective + keywords of the other projects "
            "funded under each call) to the candidate text the reranker sees. "
            "This is the same signal the Stage-1 retriever uses; it tests whether "
            "the LLM exploits richer context. Mutually exclusive with "
            "--candidate-descriptions."
        ),
    )
    parser.add_argument(
        "--historical-char-budget",
        type=int,
        default=1200,
        help="Per-candidate char cap for injected historical profile text.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_retriever_sets(
    path: Path,
    top_n: int | None,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Return {query_id: retriever record} and the effective top_n."""
    records: dict[str, dict[str, Any]] = {}
    file_top_n = None
    for row in read_jsonl(path):
        file_top_n = row.get("top_n", file_top_n)
        query_id = str(row["query_id"])
        candidates = sorted(row["candidates"], key=lambda c: c["rank"])
        if top_n is not None:
            candidates = candidates[:top_n]
        records[query_id] = {
            "candidates": candidates,
            "true_call_label": row.get("true_call_label", []),
            "true_in_top_n": row.get("true_in_top_n"),
            "true_rank_in_full_candidate_pool": row.get(
                "true_rank_in_full_candidate_pool"
            ),
        }
    effective = top_n if top_n is not None else (file_top_n or 0)
    if top_n is not None and file_top_n is not None and top_n > file_top_n:
        raise SystemExit(
            f"--top-n {top_n} exceeds retriever file top_n {file_top_n}."
        )
    return records, effective


def retriever_ranked_list(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"rank": index, "call_label": candidate["call_label"]}
        for index, candidate in enumerate(record["candidates"], start=1)
    ]


def true_rank_in_list(ranked: list[dict[str, Any]], truth: set[str], top_n: int) -> int:
    """Rank of the true label within a ranked list; top_n+1 if absent (retriever miss)."""
    return next(
        (row["rank"] for row in ranked if row["call_label"] in truth),
        top_n + 1,
    )


def build_per_query_comparison(
    *,
    query_ids: list[str],
    retriever_records: dict[str, dict[str, Any]],
    rerank_ranks: dict[str, list[int]],
    truths: dict[str, list[str]],
    top_n: int,
) -> list[dict[str, Any]]:
    rows = []
    for query_id in query_ids:
        record = retriever_records[query_id]
        truth = set(truths[query_id])
        retr_rank = true_rank_in_list(retriever_ranked_list(record), truth, top_n)
        ranks = rerank_ranks.get(query_id, [])
        rerank_mean = statistics.mean(ranks) if ranks else float(top_n + 1)
        rows.append(
            {
                "query_id": query_id,
                "true_call_label": ";".join(record.get("true_call_label", [])),
                "true_in_top_n": bool(record.get("true_in_top_n")),
                "retriever_rank_in_topn": retr_rank,
                "retriever_rank_full_pool": record.get(
                    "true_rank_in_full_candidate_pool"
                ),
                "rerank_rank_mean": round(rerank_mean, 4),
                "rerank_rank_min": min(ranks) if ranks else None,
                "rerank_rank_max": max(ranks) if ranks else None,
                "rerank_repeats": len(ranks),
                # positive => rerank moved the true label up vs the retriever
                "delta_rank_vs_retriever": round(retr_rank - rerank_mean, 4),
            }
        )
    return rows


def summarize(
    *,
    ranked_by_repeat: dict[int, dict[str, list[dict[str, Any]]]],
    retriever_records: dict[str, dict[str, Any]],
    truths: dict[str, list[str]],
    top_n: int,
    query_ids: list[str],
) -> dict[str, Any]:
    # retriever baseline over the same top-N set (repeat-independent)
    retriever_ranked = {
        query_id: retriever_ranked_list(retriever_records[query_id])
        for query_id in query_ids
    }
    retriever_metrics = rank_metrics(retriever_ranked, truths)

    # rerank metrics per repeat, then aggregate mean/std across repeats
    per_repeat_metrics = [
        rank_metrics(ranked_by_repeat[repeat], truths)
        for repeat in sorted(ranked_by_repeat)
        if ranked_by_repeat[repeat]
    ]
    metric_keys = [
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR",
        "NDCG@5",
        "mean_true_rank",
    ]
    rerank_mean = {}
    rerank_std = {}
    for key in metric_keys:
        values = [m[key] for m in per_repeat_metrics if key in m]
        if values:
            rerank_mean[key] = statistics.mean(values)
            rerank_std[key] = statistics.pstdev(values) if len(values) > 1 else 0.0

    # head-to-head using repeat-averaged per-query rerank rank
    rerank_ranks: dict[str, list[int]] = {qid: [] for qid in query_ids}
    for repeat_map in ranked_by_repeat.values():
        for query_id, ranked in repeat_map.items():
            truth = set(truths[query_id])
            rerank_ranks[query_id].append(true_rank_in_list(ranked, truth, top_n))

    improved = worsened = tied = 0
    for query_id in query_ids:
        record = retriever_records[query_id]
        truth = set(truths[query_id])
        retr_rank = true_rank_in_list(retriever_ranked_list(record), truth, top_n)
        ranks = rerank_ranks[query_id]
        if not ranks:
            continue
        rerank_mean_rank = statistics.mean(ranks)
        if rerank_mean_rank < retr_rank:
            improved += 1
        elif rerank_mean_rank > retr_rank:
            worsened += 1
        else:
            tied += 1

    true_in_top_n = sum(
        1 for qid in query_ids if retriever_records[qid].get("true_in_top_n")
    )
    return {
        "n_queries": len(query_ids),
        "top_n": top_n,
        "repeats": len(per_repeat_metrics),
        "retriever_ceiling_true_in_top_n": true_in_top_n / len(query_ids)
        if query_ids
        else 0.0,
        "retriever_baseline_over_topn": retriever_metrics,
        "rerank_mean": rerank_mean,
        "rerank_std_across_repeats": rerank_std,
        "head_to_head_vs_retriever": {
            "improved": improved,
            "worsened": worsened,
            "tied": tied,
        },
    }


def write_checkpoint(
    *,
    output_dir: Path,
    split: str,
    raw_rows: list[dict[str, Any]],
    ranked_by_repeat: dict[int, dict[str, list[dict[str, Any]]]],
    retriever_records: dict[str, dict[str, Any]],
    truths: dict[str, list[str]],
    top_n: int,
    query_ids: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    summary = summarize(
        ranked_by_repeat=ranked_by_repeat,
        retriever_records=retriever_records,
        truths=truths,
        top_n=top_n,
        query_ids=query_ids,
    )
    # Salvaged (truncated/malformed) model outputs: parse_ok is absent on rows
    # written before this field existed, so only count explicit False.
    summary["n_parse_failures"] = sum(
        1 for row in raw_rows if row.get("parse_ok") is False
    )
    rerank_ranks: dict[str, list[int]] = {qid: [] for qid in query_ids}
    for repeat_map in ranked_by_repeat.values():
        for query_id, ranked in repeat_map.items():
            truth = set(truths[query_id])
            rerank_ranks[query_id].append(true_rank_in_list(ranked, truth, top_n))
    per_query = build_per_query_comparison(
        query_ids=query_ids,
        retriever_records=retriever_records,
        rerank_ranks=rerank_ranks,
        truths=truths,
        top_n=top_n,
    )
    write_jsonl(output_dir / f"{split}_rerank_outputs.jsonl", raw_rows)
    write_csv(output_dir / f"{split}_per_query_comparison.csv", per_query)
    (output_dir / f"{split}_summary_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"{split}_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to rerank test split without --allow-test.")

    if args.historical_context and args.candidate_descriptions:
        raise SystemExit(
            "--historical-context and --candidate-descriptions are mutually "
            "exclusive; each defines a different candidate-text policy."
        )

    enriched_map: dict[str, list[str]] | None = None
    if args.candidate_descriptions:
        enriched_map = json.loads(
            Path(args.candidate_descriptions).read_text(encoding="utf-8")
        )

    # Context-awareness condition: build the per-call leave-one-out historical
    # profile the Stage-1 retriever uses, so it can be appended to candidate text.
    historical_groups = None
    if args.historical_context:
        from rq2_build_deterministic_topn import (
            historical_text,
            load_historical_groups,
        )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.historical_context:
        output_dir = HISTORICAL_CONTEXT_OUTPUT_DIR
    elif enriched_map is not None and args.enrich_thin_only:
        output_dir = ENRICHED_TARGETED_OUTPUT_DIR
    elif enriched_map is not None:
        output_dir = ENRICHED_OUTPUT_DIR
    else:
        output_dir = DEFAULT_OUTPUT_DIR
    if args.mock and output_dir in (
        DEFAULT_OUTPUT_DIR,
        ENRICHED_OUTPUT_DIR,
        ENRICHED_TARGETED_OUTPUT_DIR,
        HISTORICAL_CONTEXT_OUTPUT_DIR,
    ):
        output_dir = MOCK_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    retriever_file = Path(args.retriever_file)
    if not retriever_file.exists():
        raise SystemExit(f"Retriever file not found: {retriever_file}")
    retriever_records, top_n = load_retriever_sets(retriever_file, args.top_n)

    queries = load_frozen_split(args.split, FROZEN_DIR)
    truths = load_ground_truth(args.split, FROZEN_DIR)
    # keep only queries that exist in the retriever file, preserve split order
    queries = [q for q in queries if str(q["query_id"]) in retriever_records]
    if args.limit is not None:
        queries = queries[: args.limit]
    query_ids = [str(q["query_id"]) for q in queries]

    pool_by_label = {c["call_label"]: c for c in load_candidate_pool(FROZEN_DIR)}
    project_id_by_query = {
        str(q["query_id"]): q.get("project_id") for q in queries
    }

    if args.historical_context:
        historical_groups = load_historical_groups(list(pool_by_label))

    def enriched_descriptions(label: str, base: dict[str, Any]) -> list[str]:
        """Official scope text for a subCall, budget-capped; fall back to catalogue."""
        catalogue = base.get("funding_call_descriptions", [])
        scope = (enriched_map or {}).get(label) or []
        if not scope:
            return catalogue
        if args.enrich_thin_only:
            # Targeted: keep crisp discriminative titles; only rescue degenerate ones.
            if len(" ".join(catalogue).split()) > args.thin_threshold:
                return catalogue
        joined = " | ".join(scope)
        return [joined[: args.desc_char_budget]]

    def historical_descriptions(
        label: str, base: dict[str, Any], exclude_project_id: int | None
    ) -> list[str]:
        """Catalogue text plus the leave-one-out historical profile for this call."""
        catalogue = base.get("funding_call_descriptions", [])
        group = historical_groups.get(label) if historical_groups is not None else None
        profile = historical_text(group, exclude_project_id) if group is not None else ""
        if not profile:
            return catalogue
        capped = profile[: args.historical_char_budget]
        return list(catalogue) + [f"Historical funded-project profile: {capped}"]

    def candidate_dicts(query_id: str) -> list[dict[str, Any]]:
        exclude_project_id = None
        if args.historical_context:
            raw_project_id = project_id_by_query.get(query_id)
            if raw_project_id is not None:
                try:
                    exclude_project_id = int(raw_project_id)
                except (TypeError, ValueError):
                    exclude_project_id = None
        out = []
        for candidate in retriever_records[query_id]["candidates"]:
            label = candidate["call_label"]
            base = pool_by_label.get(label)
            if base is None:
                continue
            if enriched_map is not None:
                base = {
                    **base,
                    "funding_call_descriptions": enriched_descriptions(label, base),
                }
            elif args.historical_context:
                base = {
                    **base,
                    "funding_call_descriptions": historical_descriptions(
                        label, base, exclude_project_id
                    ),
                }
            out.append(base)
        return out

    if args.dry_run_prompt:
        prompt, _ = build_rank_prompt(
            queries[0],
            candidate_dicts(query_ids[0]),
            anonymize=args.anonymize_candidates,
        )
        prompt_path = output_dir / f"dry_run_prompt_{args.split}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"Wrote {prompt_path}")
        return

    repeats = max(1, args.repeats)
    rerank_outputs_path = output_dir / f"{args.split}_rerank_outputs.jsonl"
    raw_rows = read_existing_jsonl(rerank_outputs_path) if args.resume else []
    ranked_by_repeat: dict[int, dict[str, list[dict[str, Any]]]] = {
        r: {} for r in range(repeats)
    }
    for row in raw_rows:
        repeat = int(row.get("repeat", 0))
        if row.get("ranked") and repeat < repeats:
            ranked_by_repeat.setdefault(repeat, {})[str(row["query_id"])] = row["ranked"]
    completed = {
        (str(row["query_id"]), int(row.get("repeat", 0)))
        for row in raw_rows
        if row.get("ranked")
    }

    try:
        client = MockRankClient() if args.mock else make_client(args.model)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc))
    graph = build_graph(client, anonymize=args.anonymize_candidates)

    metadata = {
        "architecture": "single_agent_langgraph_rerank_baseline",
        "graph": "START -> rank_node -> END",
        "stage": "rerank_over_deterministic_retriever_topn",
        "retriever_file": _relative_retriever_path(retriever_file),
        "candidate_text_policy": (
            "leave_one_out_historical_profile_context"
            if args.historical_context
            else (
                (
                    "official_topic_scope_targeted_thin_only"
                    if args.enrich_thin_only
                    else "official_topic_scope_enriched"
                )
                if enriched_map is not None
                else "information_floor_call_label_plus_descriptions"
            )
        ),
        "candidate_descriptions_source": args.candidate_descriptions,
        "desc_char_budget": args.desc_char_budget if enriched_map is not None else None,
        "enrich_thin_only": args.enrich_thin_only if enriched_map is not None else None,
        "thin_threshold": args.thin_threshold if args.enrich_thin_only else None,
        "historical_context": args.historical_context,
        "historical_char_budget": (
            args.historical_char_budget if args.historical_context else None
        ),
        "excluded_leakage_fields": ["fundingScheme", "masterCall"],
        "candidate_id_policy": (
            "anonymized_alias_year_scrubbed"
            if args.anonymize_candidates
            else "real_subcall_label"
        ),
        "split": args.split,
        "n_queries": len(query_ids),
        "top_n": top_n,
        "repeats": repeats,
        "batch_size": args.batch_size,
        "resume": args.resume,
        "model": "mock" if args.mock else args.model,
        "provider": None if args.mock else provider_of(args.model),
        "temperature": (
            GEMINI_TEMPERATURE
            if (not args.mock and provider_of(args.model) == "google")
            else None
        ),
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "rerank_outputs": f"{args.split}_rerank_outputs.jsonl",
            "per_query_comparison": f"{args.split}_per_query_comparison.csv",
            "summary_metrics": f"{args.split}_summary_metrics.json",
        },
    }

    if completed:
        print(f"Resuming with {len(completed)} completed (query, repeat) units.")

    batch_size = max(1, args.batch_size)
    batch_new_count = 0
    total_units = len(query_ids) * repeats
    unit_index = 0
    for repeat in range(repeats):
        for query_row in queries:
            unit_index += 1
            query_id = str(query_row["query_id"])
            if (query_id, repeat) in completed:
                print(
                    f"[{unit_index}/{total_units}] skip done q={query_id} r={repeat}"
                )
                continue

            initial_state: RQ2State = {
                "query": query_row,
                "candidates": candidate_dicts(query_id),
                "scored": [],
                "ranked": [],
            }
            try:
                final_state = graph.invoke(initial_state)
            except Exception:
                if raw_rows:
                    write_checkpoint(
                        output_dir=output_dir,
                        split=args.split,
                        raw_rows=raw_rows,
                        ranked_by_repeat=ranked_by_repeat,
                        retriever_records=retriever_records,
                        truths=truths,
                        top_n=top_n,
                        query_ids=query_ids,
                        metadata=metadata,
                    )
                    print(f"Checkpointed {len(completed)} units before error.")
                raise

            ranked_by_repeat.setdefault(repeat, {})[query_id] = final_state["ranked"]
            raw_output = final_state.get("raw_output", "")
            raw_rows.append(
                {
                    "query_id": query_id,
                    "repeat": repeat,
                    "ranked": final_state["ranked"],
                    "raw_output": raw_output,
                    "parse_ok": raw_output_parse_ok(raw_output),
                    "usage_metadata": getattr(client, "last_usage_metadata", {}),
                }
            )
            completed.add((query_id, repeat))
            batch_new_count += 1
            print(f"[{unit_index}/{total_units}] ranked q={query_id} r={repeat}")

            if args.sleep_between_calls > 0:
                time.sleep(args.sleep_between_calls)

            if batch_new_count >= batch_size:
                write_checkpoint(
                    output_dir=output_dir,
                    split=args.split,
                    raw_rows=raw_rows,
                    ranked_by_repeat=ranked_by_repeat,
                    retriever_records=retriever_records,
                    truths=truths,
                    top_n=top_n,
                    query_ids=query_ids,
                    metadata=metadata,
                )
                print(f"Checkpointed {len(completed)} units.")
                batch_new_count = 0
                if args.sleep_between_batches > 0:
                    time.sleep(args.sleep_between_batches)

    summary = write_checkpoint(
        output_dir=output_dir,
        split=args.split,
        raw_rows=raw_rows,
        ranked_by_repeat=ranked_by_repeat,
        retriever_records=retriever_records,
        truths=truths,
        top_n=top_n,
        query_ids=query_ids,
        metadata=metadata,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
