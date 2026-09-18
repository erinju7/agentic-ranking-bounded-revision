from __future__ import annotations

import argparse
import csv
import json
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
    GeminiClient,
    RQ2State,
    build_rank_prompt,
    load_candidate_pool,
    load_frozen_split,
    load_ground_truth,
    model_visible_query,
    per_query_rows,
    rank_metrics,
    rank_with_llm,
)


DEFAULT_OUTPUT_DIR = RESULT_ROOT / "single_agent_langgraph_baseline"
MOCK_OUTPUT_DIR = RESULT_ROOT / "single_agent_langgraph_baseline_mock"


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


def build_graph(client: Any):
    def rank_node(state: RQ2State) -> RQ2State:
        ranked, raw_output = rank_with_llm(
            client,
            state["query"],
            state["candidates"],
        )
        return {
            **state,
            "ranked": ranked,
            "raw_output": raw_output,
        }

    graph = StateGraph(RQ2State)
    graph.add_node("rank_node", rank_node)
    graph.set_entry_point("rank_node")
    graph.set_finish_point("rank_node")
    return graph.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RQ2 single-agent LangGraph ranking baseline.",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of one-query LLM calls to run before checkpointing outputs.",
    )
    parser.add_argument(
        "--sleep-between-calls",
        type=float,
        default=0.0,
        help="Seconds to sleep after each query call. Useful for free-tier rate limits.",
    )
    parser.add_argument(
        "--sleep-between-batches",
        type=float,
        default=0.0,
        help="Seconds to sleep after each checkpointed batch.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing ranked outputs in the output directory.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="Debug only. Keep unset for the real architecture comparison.",
    )
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock output.")
    parser.add_argument(
        "--dry-run-prompt",
        action="store_true",
        help="Write the first assembled prompt and exit without calling the graph.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_checkpoint(
    *,
    output_dir: Path,
    split: str,
    raw_rows: list[dict[str, Any]],
    ranked_by_query: dict[str, list[dict[str, Any]]],
    truths: dict[str, list[str]],
    metadata: dict[str, Any],
) -> None:
    summary = rank_metrics(ranked_by_query, truths)
    per_query = per_query_rows(ranked_by_query, truths)
    write_jsonl(output_dir / f"{split}_ranked_outputs.jsonl", raw_rows)
    write_csv(output_dir / f"{split}_per_query_results.csv", per_query)
    (output_dir / f"{split}_summary_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"{split}_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.mock and output_dir == DEFAULT_OUTPUT_DIR:
        output_dir = MOCK_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = load_frozen_split(args.split, FROZEN_DIR)
    if args.limit is not None:
        queries = queries[: args.limit]

    candidates = load_candidate_pool(FROZEN_DIR)
    if args.candidate_limit is not None:
        candidates = candidates[: args.candidate_limit]

    if args.dry_run_prompt:
        prompt = build_rank_prompt(model_visible_query(queries[0]), candidates)
        prompt_path = output_dir / f"dry_run_prompt_{args.split}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"Wrote {prompt_path}")
        return

    truths = load_ground_truth(args.split, FROZEN_DIR)
    ranked_outputs_path = output_dir / f"{args.split}_ranked_outputs.jsonl"
    raw_rows = read_existing_jsonl(ranked_outputs_path) if args.resume else []
    ranked_by_query: dict[str, list[dict[str, Any]]] = {
        str(row["query_id"]): row["ranked"]
        for row in raw_rows
        if row.get("ranked")
    }
    completed_ids = set(ranked_by_query)

    try:
        client = MockRankClient() if args.mock else GeminiClient(model_name=args.model)
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    graph = build_graph(client)

    metadata = {
        "architecture": "single_agent_langgraph_baseline",
        "graph": "START -> rank_node -> END",
        "split": args.split,
        "frozen_dir": str(FROZEN_DIR.relative_to(FROZEN_DIR.parents[2])),
        "n_queries": len(queries),
        "n_candidates": len(candidates),
        "candidate_limit_debug": args.candidate_limit,
        "batch_size": args.batch_size,
        "sleep_between_calls": args.sleep_between_calls,
        "sleep_between_batches": args.sleep_between_batches,
        "resume": args.resume,
        "model": "mock" if args.mock else args.model,
        "temperature": None if args.mock else GEMINI_TEMPERATURE,
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "ranked_outputs": f"{args.split}_ranked_outputs.jsonl",
            "per_query_results": f"{args.split}_per_query_results.csv",
            "summary_metrics": f"{args.split}_summary_metrics.json",
        },
    }

    if completed_ids:
        print(f"Resuming with {len(completed_ids)} completed queries.")

    batch_size = max(1, args.batch_size)
    batch_new_count = 0
    for index, query_row in enumerate(queries, start=1):
        query = model_visible_query(query_row)
        query_id = str(query["query_id"])
        if query_id in completed_ids:
            print(f"[{index}/{len(queries)}] skipped completed query {query_id}")
            continue

        initial_state: RQ2State = {
            "query": query,
            "candidates": candidates,
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
                    ranked_by_query=ranked_by_query,
                    truths=truths,
                    metadata=metadata,
                )
                print(f"Checkpointed {len(ranked_by_query)} completed queries before error.")
            raise
        ranked_by_query[query_id] = final_state["ranked"]
        raw_rows.append(
            {
                "query_id": query_id,
                "ranked": final_state["ranked"],
                "raw_output": final_state.get("raw_output", ""),
                "usage_metadata": getattr(client, "last_usage_metadata", {}),
            }
        )
        print(f"[{index}/{len(queries)}] ranked query {query_id}")
        batch_new_count += 1
        completed_ids.add(query_id)

        if args.sleep_between_calls > 0:
            time.sleep(args.sleep_between_calls)

        if batch_new_count >= batch_size:
            write_checkpoint(
                output_dir=output_dir,
                split=args.split,
                raw_rows=raw_rows,
                ranked_by_query=ranked_by_query,
                truths=truths,
                metadata=metadata,
            )
            print(f"Checkpointed {len(ranked_by_query)} queries.")
            batch_new_count = 0
            if args.sleep_between_batches > 0:
                time.sleep(args.sleep_between_batches)

    write_checkpoint(
        output_dir=output_dir,
        split=args.split,
        raw_rows=raw_rows,
        ranked_by_query=ranked_by_query,
        truths=truths,
        metadata=metadata,
    )
    summary = rank_metrics(ranked_by_query, truths)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
