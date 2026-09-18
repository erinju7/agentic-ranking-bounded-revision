"""RQ2b / RQ2c agent skeletons on the Experiment-1 reranking task.

Design axis = CONTROL-FLOW AUTONOMY. The three levels share ONE decision core
(prompt + parser) so any performance difference is attributable to the control
structure wrapped around it, not to prompt wording. Same model, retriever,
split, seed, and evaluation throughout; token/round cost is logged for every
call so each level's gain can be weighed against its spend.

  RQ2a  (Experiment 1)  reorder a fixed top-k. No control-flow autonomy.
  RQ2b  (here)          single pass; the agent OWNS the output decision:
                        commit a variable-size shortlist, or ABSTAIN. No
                        re-retrieval.
  RQ2c  (here)          the agent controls RETRIEVAL: it may reformulate the
                        query and search again over bounded rounds, then
                        finalize. Only this level can break the retriever's
                        recall ceiling.

Runnable skeleton: graph structure, state, tool interface, decision schema,
abstention-aware metric, and budget guards are concrete. Two integration points
are marked TODO: (1) wiring `search()` to the retriever artifacts for RQ2c, and
(2) any prompt tuning. Everything else reuses rq2_core.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import StateGraph, END

from rq2_core import (
    make_client,
    model_visible_query,
    compact_candidate,
    extract_json_object,
    load_candidate_pool,
)

# ---------------------------------------------------------------------------
# Shared decision core: one prompt + one parser used by BOTH agent levels.
# It differs from the RQ2a rank prompt only by allowing abstention and a
# variable-length shortlist -- that difference *is* the RQ2b delegation.
# ---------------------------------------------------------------------------

DECISION_SCHEMA = {
    "abstain": "bool - true iff no candidate is the funding call for this project",
    "labels": "ranked shortlist of subCall labels, best first; [] if abstain",
    "confidence": "float 0..1",
}


def build_decision_prompt(query: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    payload = {
        "query": model_visible_query(query),
        "candidates": [compact_candidate(c) for c in candidates],
    }
    return f"""You are matching one Horizon 2020 project to its funding call.

Task:
- Decide which candidate subCall (if any) funded this project.
- You MAY return a short ranked shortlist rather than a single label.
- You MAY ABSTAIN if you believe the correct call is not among the candidates.
- Do not invent labels. Use only labels present in candidates.
- Return JSON only, matching this schema:
{json.dumps(DECISION_SCHEMA, indent=2)}

Input:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}
"""


def parse_decision(raw: str, candidate_labels: list[str]) -> dict[str, Any]:
    """Defensive parse -> {abstain, labels, confidence}. Salvages like RQ2a."""
    allowed = set(candidate_labels)
    try:
        obj = extract_json_object(raw)
    except (json.JSONDecodeError, ValueError):
        obj = {}
    if not isinstance(obj, dict):
        obj = {}
    labels = [x for x in (obj.get("labels") or []) if x in allowed]
    abstain = bool(obj.get("abstain", False)) and not labels
    try:
        confidence = float(obj.get("confidence"))
    except (TypeError, ValueError):
        confidence = None
    return {"abstain": abstain, "labels": labels, "confidence": confidence}


# ---------------------------------------------------------------------------
# RQ2b: single-pass decision agent.  START -> decide -> END
# Autonomy delegated: the output decision (commit shortlist / abstain). One call.
# ---------------------------------------------------------------------------

class DecisionState(TypedDict, total=False):
    query: dict[str, Any]
    candidates: list[dict[str, Any]]
    decision: dict[str, Any]
    raw_output: str
    usage: dict[str, Any]


def build_single_pass_agent(client: Any):
    def decide(state: DecisionState) -> DecisionState:
        prompt = build_decision_prompt(state["query"], state["candidates"])
        raw = client.generate(prompt)
        labels = [c["call_label"] for c in state["candidates"]]
        return {
            **state,
            "decision": parse_decision(raw, labels),
            "raw_output": raw,
            "usage": getattr(client, "last_usage_metadata", {}),
        }

    g = StateGraph(DecisionState)
    g.add_node("decide", decide)
    g.set_entry_point("decide")
    g.set_finish_point("decide")
    return g.compile()


# ---------------------------------------------------------------------------
# RQ2c: self-correcting retrieval agent.
#   START -> agent -> [search -> agent]*  (bounded) -> finalize -> END
# Autonomy delegated: retrieval control (reformulate + re-search). The only
# level that can recover a true call outside the initial top-k.
# ---------------------------------------------------------------------------

# Retriever tool interface. TODO: wire to the frozen TF-IDF retriever
# (retriever_artifacts/ + rq2_build_deterministic_topn machinery): vectorize
# `query_text`, score all candidates, return the top-k pool records. Keep it a
# pure function so rounds are reproducible.
SearchFn = Callable[[str, int], list[dict[str, Any]]]


@dataclass
class LoopBudget:
    max_rounds: int = 3          # hard cap on re-retrieval rounds
    stop_confidence: float = 0.8  # stop early once the agent is this confident
    top_k: int = 10              # tighten stage-1 so the loop has headroom


class LoopState(TypedDict, total=False):
    query: dict[str, Any]
    query_text: str              # current (possibly reformulated) query string
    candidates: list[dict[str, Any]]
    round: int
    searches: list[str]          # reformulations issued, for cost/behaviour audit
    decision: dict[str, Any]
    usage_log: list[dict[str, Any]]


def build_self_correcting_agent(client: Any, search: SearchFn, budget: LoopBudget):
    def agent(state: LoopState) -> LoopState:
        # One decision-core call over the current candidate set; the agent may
        # commit, abstain, or (via the router below) ask to reformulate.
        prompt = build_decision_prompt(state["query"], state["candidates"])
        raw = client.generate(prompt)
        labels = [c["call_label"] for c in state["candidates"]]
        decision = parse_decision(raw, labels)
        # TODO: also parse an optional {"reformulate": "<new query text>"} field
        # so the agent proposes the next search; fall back to a heuristic below.
        return {
            **state,
            "decision": decision,
            "usage_log": state.get("usage_log", []) + [getattr(client, "last_usage_metadata", {})],
        }

    def do_search(state: LoopState) -> LoopState:
        new_query = state.get("query_text") or state["query"].get("query_text", "")
        # TODO: use the agent-proposed reformulation; heuristic placeholder here.
        candidates = search(new_query, budget.top_k)
        return {
            **state,
            "candidates": candidates,
            "round": state.get("round", 0) + 1,
            "searches": state.get("searches", []) + [new_query],
        }

    def route(state: LoopState) -> str:
        d = state.get("decision") or {}
        confident = (d.get("confidence") or 0.0) >= budget.stop_confidence
        committed = bool(d.get("labels")) or d.get("abstain")
        if (committed and confident) or state.get("round", 0) >= budget.max_rounds:
            return "finalize"
        return "search"  # reformulate + retrieve again

    def finalize(state: LoopState) -> LoopState:
        return state

    g = StateGraph(LoopState)
    g.add_node("agent", agent)
    g.add_node("search", do_search)
    g.add_node("finalize", finalize)
    g.set_entry_point("search")          # initial retrieval
    g.add_edge("search", "agent")
    g.add_conditional_edges("agent", route, {"search": "search", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Abstention-aware evaluation. On queries where the true call is ABSENT from the
# candidate set, abstaining is CORRECT; committing is a false alarm. On queries
# where it is present, abstaining is a miss. This is what distinguishes RQ2b/c
# from RQ2a -- plain Recall@k cannot see abstention, so report the trade-off.
# ---------------------------------------------------------------------------

@dataclass
class DecisionOutcome:
    true_present: bool           # is the true call in the candidate set?
    abstained: bool
    top1_correct: bool           # committed and rank-1 == truth
    true_rank: int               # rank of truth in labels; len+1 if absent/abstain


def abstention_metrics(outcomes: list[DecisionOutcome]) -> dict[str, float]:
    n = len(outcomes) or 1
    committed = [o for o in outcomes if not o.abstained]
    # coverage: fraction of queries the agent committed on
    # selective accuracy: of committed queries, fraction correct at rank 1
    # correct abstention: of truly-absent queries, fraction it abstained on
    absent = [o for o in outcomes if not o.true_present]
    return {
        "coverage": round(len(committed) / n, 3),
        "selective_recall@1": round(
            sum(o.top1_correct for o in committed) / (len(committed) or 1), 3
        ),
        "correct_abstention_rate": round(
            sum(o.abstained for o in absent) / (len(absent) or 1), 3
        ),
        "false_commit_rate": round(
            sum((not o.abstained) for o in absent) / (len(absent) or 1), 3
        ),
    }


# ---------------------------------------------------------------------------
# Wiring sketch (not a CLI yet): build a client with make_client(model), then
#   RQ2b: graph = build_single_pass_agent(client); graph.invoke({query, candidates})
#   RQ2c: graph = build_self_correcting_agent(client, search, LoopBudget())
# Keep the SAME model across a/b/c; vary only which graph runs. Log usage_log /
# searches for the cost axis, and score with abstention_metrics on a split that
# includes some true-call-absent queries (e.g. a tightened top_k).
# ---------------------------------------------------------------------------
