"""Experiment 2 — agent ARCHITECTURE comparison (single-pass generalist vs
structured multi-agent), on the memorization-controlled governed setup.

The only independent variable is architecture. Both arms use:
  * the same anonymized Top-50 pool (per-query retriever set, real labels hidden
    behind stable governed-profile ids),
  * the same governed evidence per candidate (scope / specific_challenge /
    expected_outcome / action_type / call_scale) -- year- and code-scrubbed by
    build_guarded_candidate_profiles.py (scrub-guard enforced there),
  * the same backend, seed, split, and evaluation harness.

Arm A -- single-pass generalist: ONE agent, ONE call, sees the FULL governed
profile for all 50 candidates, returns a ranking.

Arm B -- specialists + coordinator: three specialists each see one slice of the
governed evidence and score all 50 candidates on their dimension; an LLM
coordinator reasons over the three score sets and emits the final ranking. The
union of the specialists' evidence equals exactly what the generalist sees, so
any difference is attributable to decomposition, not information.

Token usage, wall-clock latency, and cost are logged per query for both arms.
Consumes governed profiles + retriever sets; the backend is the only API cost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

from rq2_core import (
    FROZEN_DIR,
    RESULT_ROOT,
    extract_json_object,
    load_ground_truth,
    load_frozen_split,
    make_client,
    model_visible_query,
    per_query_rows,
    provider_of,
    rank_metrics,
    usage_cost_usd,
)

CURATED = FROZEN_DIR / "curated_evidence_subset"
PROFILES_PATH = CURATED / "candidate_profiles.json"
IDMAP_PATH = CURATED / "candidate_id_map.json"
RETRIEVER = {
    "dev": RESULT_ROOT / "deterministic_topn_retriever" / "dev_historical_top50_candidates.jsonl",
    "test": RESULT_ROOT / "deterministic_topn_retriever" / "test_historical_top50_candidates.jsonl",
}

# Governed evidence actually shown. submission_stage is intentionally excluded so
# that the union of the three specialists' fields is EXACTLY the generalist's
# field set (matched information; architecture is the only IV).
THEME_FIELDS = ["scope", "specific_challenge", "expected_outcome"]
ACTION_FIELDS = ["action_type"]
BUDGET_FIELDS = ["call_scale"]
ALL_FIELDS = THEME_FIELDS + ACTION_FIELDS + BUDGET_FIELDS
# Compact vs full for the selective-inspection tool agent. The compact profile is
# deliberately THIN: only the instrument (action type) and call-scale band, with NO
# scope text, so the agent cannot judge scientific fit without inspecting. inspect()
# reveals the full profile (scope + challenge + outcome + action + scale), which
# equals what the single-pass generalist sees. This makes inspection a real
# information lever rather than a near-duplicate of the compact view.
MAX_INSPECT = 5


def compact_view(cid: str, profile: dict) -> dict:
    view = {"id": cid}
    for f in ACTION_FIELDS + BUDGET_FIELDS:  # action_type, call_scale only
        v = profile.get(f)
        if v:
            view[f] = v
    return view


def full_view(cid: str, profile: dict) -> dict:
    return governed_view(cid, profile, ALL_FIELDS)


def load_profiles() -> tuple[dict[str, dict], dict[str, str], dict[str, str]]:
    profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    id_map = json.loads(IDMAP_PATH.read_text(encoding="utf-8"))       # cid -> label
    label_to_cid = {label: cid for cid, label in id_map.items()}
    return profiles, id_map, label_to_cid


def load_retriever_top50(split: str) -> dict[str, list[str]]:
    """query_id -> [real call_label ...] in retriever rank order."""
    out: dict[str, list[str]] = {}
    for line in RETRIEVER[split].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ranked = sorted(row["candidates"], key=lambda c: c["rank"])
        out[str(row["query_id"])] = [c["call_label"] for c in ranked]
    return out


def governed_view(cid: str, profile: dict, fields: list[str]) -> dict[str, Any]:
    view: dict[str, Any] = {"id": cid}
    for f in fields:
        v = profile.get(f)
        if v:
            view[f] = v
    return view


def query_block(query: dict, *, budget_only: bool = False) -> dict[str, Any]:
    v = model_visible_query(query)
    if budget_only:
        return {
            "totalCost_eur": v.get("totalCost_eur"),
            "ecMaxContribution_eur": v.get("ecMaxContribution_eur"),
            "duration_months": v.get("duration_months"),
        }
    return v


def _timed_generate(client: Any, prompt: str) -> tuple[str, dict, float]:
    t0 = time.perf_counter()
    raw = client.generate(prompt)
    dt = time.perf_counter() - t0
    return raw, dict(getattr(client, "last_usage_metadata", {}) or {}), dt


def _accumulate(into: dict, usage: dict, dt: float, calls: int = 1) -> None:
    into["prompt_token_count"] = into.get("prompt_token_count", 0) + (usage.get("prompt_token_count") or 0)
    into["candidates_token_count"] = into.get("candidates_token_count", 0) + (usage.get("candidates_token_count") or 0)
    into["latency_s"] = into.get("latency_s", 0.0) + dt
    into["n_calls"] = into.get("n_calls", 0) + calls


def _rank_from_ids(ordered_ids: list[str], all_cids: list[str], id_map: dict[str, str]) -> list[dict]:
    """Parsed id order -> ranked [{rank, call_label}]; back-fill missing ids in
    retriever order; drop unknown ids. Maps cid -> real label for scoring."""
    seen: list[str] = []
    for cid in ordered_ids:
        if cid in id_map and cid not in seen and cid in all_cids:
            seen.append(cid)
    for cid in all_cids:
        if cid not in seen:
            seen.append(cid)
    return [{"rank": i, "call_label": id_map[cid]} for i, cid in enumerate(seen, 1)]


def _parse_ranked_ids(raw: str) -> list[str]:
    try:
        obj = extract_json_object(raw)
    except Exception:
        return []
    if isinstance(obj, dict):
        obj = obj.get("ranked_ids") or obj.get("ranking") or obj.get("ranked") or []
    if isinstance(obj, list):
        return [str(x.get("id") if isinstance(x, dict) else x) for x in obj]
    return []


def _parse_scores(raw: str, valid: set[str]) -> dict[str, float]:
    try:
        obj = extract_json_object(raw)
    except Exception:
        return {}
    if isinstance(obj, dict) and "scores" in obj and isinstance(obj["scores"], dict):
        obj = obj["scores"]
    scores: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in valid:
                try:
                    scores[k] = float(v)
                except (TypeError, ValueError):
                    pass
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and item.get("id") in valid:
                try:
                    scores[item["id"]] = float(item.get("score"))
                except (TypeError, ValueError):
                    pass
    return scores


# --------------------------------------------------------------------------- #
# Arm A: single-pass generalist
# --------------------------------------------------------------------------- #
def run_generalist(client, query, cids, profiles, id_map) -> tuple[list[dict], dict]:
    candidates = [governed_view(c, profiles.get(c, {}), ALL_FIELDS) for c in cids]
    payload = {"project": query_block(query), "candidates": candidates}
    prompt = f"""You match one Horizon 2020 project to the funding call it was funded under.
You see a fixed pool of candidate calls, each identified by an opaque id and a
governed profile (scientific scope, instrument/action type, and funding scale).

Task:
- Rank ALL candidate ids from best match to worst match for this project.
- Reason jointly over scientific fit, instrument fit, and funding scale.
- Use only ids present in candidates. Return JSON only, no commentary.

Schema: {{"ranked_ids": ["id1", "id2", ...]}}

Input:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}
"""
    raw, usage, dt = _timed_generate(client, prompt)
    ranked = _rank_from_ids(_parse_ranked_ids(raw), cids, id_map)
    acc: dict = {}
    _accumulate(acc, usage, dt)
    return ranked, acc, None


# --------------------------------------------------------------------------- #
# Arm B: specialists + coordinator
# --------------------------------------------------------------------------- #
def _parse_specialist(raw: str, valid: set[str]) -> dict[str, dict]:
    """Return {cid: {score, confidence, evidence}} and {cid: note}."""
    try:
        obj = extract_json_object(raw)
    except Exception:
        return {"assessments": {}, "notes": {}}
    assessments: dict[str, dict] = {}
    notes: dict[str, str] = {}
    if isinstance(obj, dict):
        a = obj.get("assessments")
        if isinstance(a, dict):
            for k, v in a.items():
                if k in valid and isinstance(v, dict):
                    score = v.get("score")
                    try:
                        score = float(score) if score is not None else None
                    except (TypeError, ValueError):
                        score = None
                    assessments[k] = {
                        "score": score,
                        "confidence": v.get("confidence"),
                        "evidence": v.get("evidence"),
                    }
        nt = obj.get("notes")
        if isinstance(nt, dict):
            for k, v in nt.items():
                if k in valid and v:
                    notes[k] = str(v)[:200]
    return {"assessments": assessments, "notes": notes}


def _specialist(client, *, role, instruction, project, cids, profiles, fields, acc) -> dict:
    candidates = [governed_view(c, profiles.get(c, {}), fields) for c in cids]
    payload = {"project": project, "candidates": candidates}
    prompt = f"""You are the {role} specialist in a funding-call matching system.
{instruction}
Assess EVERY candidate id on YOUR dimension only:
- If the candidate carries no evidence on your dimension, ABSTAIN: set
  "evidence":"absent" and "score":null. Never invent a middle score for a
  candidate you cannot judge.
- Otherwise set "evidence":"present", "score" 0-100 (fit on your dimension), and
  "confidence":"low"|"medium"|"high".
Also give a concise justification (<=20 words) for up to your 5 best-fitting
candidates. Return JSON only.

Schema:
{{"assessments": {{"<id>": {{"score": <0-100 or null>, "confidence": "low|medium|high", "evidence": "present|absent"}}}},
 "notes": {{"<id>": "<short reason>"}}}}

Input:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}
"""
    raw, usage, dt = _timed_generate(client, prompt)
    _accumulate(acc, usage, dt)
    return _parse_specialist(raw, set(cids))


def run_multiagent(client, query, cids, profiles, id_map) -> tuple[list[dict], dict]:
    acc: dict = {}
    theme = _specialist(
        client, role="Theme (scientific fit)",
        instruction="Judge how well the project's scientific content matches each call's scope, specific challenge, and expected outcome.",
        project=query_block(query), cids=cids, profiles=profiles, fields=THEME_FIELDS, acc=acc,
    )
    action = _specialist(
        client, role="Action/Instrument",
        instruction="Infer the project's funding instrument (e.g. fellowship, research/innovation action, network, SME, career-stage grant) from its text, and judge instrument compatibility with each call's action type.",
        project=query_block(query), cids=cids, profiles=profiles, fields=ACTION_FIELDS, acc=acc,
    )
    budget = _specialist(
        client, role="Budget/Call-Scale",
        instruction="Judge how well the project's total cost, EC contribution, and duration fit each call's funding scale band.",
        project=query_block(query, budget_only=True), cids=cids, profiles=profiles, fields=BUDGET_FIELDS, acc=acc,
    )
    # Coordinator input in RETRIEVER ORDER (a list, not a sorted dict), so the
    # coordinator keeps the Stage-1 prior instead of an arbitrary alphabetical one.
    def _abstain():
        return {"score": None, "confidence": None, "evidence": "absent"}
    per_candidate = []
    for c in cids:
        entry = {
            "id": c,
            "theme": theme["assessments"].get(c, _abstain()),
            "action": action["assessments"].get(c, _abstain()),
            "budget": budget["assessments"].get(c, _abstain()),
        }
        note = {}
        for dim, spec in (("theme", theme), ("action", action), ("budget", budget)):
            if c in spec["notes"]:
                note[dim] = spec["notes"][c]
        if note:
            entry["notes"] = note
        per_candidate.append(entry)

    prompt = f"""You are the coordinator of a funding-call matching system. Three
specialists have each assessed every candidate on one dimension: theme
(scientific fit), action (funding-instrument type), and budget (funding scale).
Each assessment has a score, a confidence, and an evidence flag. An
"evidence":"absent" flag or a null score means the specialist ABSTAINED because it
had no evidence; treat that as abstention, never as a medium or neutral score.

Produce a final ranking of ALL candidate ids, best match first, by REASONING over
the assessments---do not average the scores. Apply this hierarchy:
- THEME is the PRIMARY signal: scientific/thematic fit drives the ranking.
- ACTION TYPE is a CONDITIONAL COMPATIBILITY CONSTRAINT, not an additive score: if
  a candidate's instrument type is clearly incompatible with the project, demote
  it even when theme fit is high; if action evidence is absent, do not penalize on
  this dimension.
- CALL SCALE is only a WEAK TIE-BREAKER between candidates that are otherwise
  comparable on theme and action.
- Where the specialists conflict (e.g. strong theme but incompatible action),
  reason about the conflict explicitly before deciding the order.
Use only ids present below. Return JSON only.

Schema: {{"ranked_ids": ["<id>", ...], "rationale": "<=60 words"}}

Candidate assessments (in retrieval order):
{json.dumps(per_candidate, ensure_ascii=False)}
"""
    raw, usage, dt = _timed_generate(client, prompt)
    _accumulate(acc, usage, dt)
    ranked = _rank_from_ids(_parse_ranked_ids(raw), cids, id_map)
    internals = {
        "assessments": {c: per_candidate[i] for i, c in enumerate(cids)},
        "coordinator_raw": raw,
        "coordinator_ranked_ids": _parse_ranked_ids(raw),
    }
    return ranked, acc, internals


# --------------------------------------------------------------------------- #
# Arm C: single agent with a bounded inspect() tool (selective access)
# --------------------------------------------------------------------------- #
def _parse_ids_field(raw: str, key: str) -> list[str]:
    try:
        obj = extract_json_object(raw)
    except Exception:
        return []
    if isinstance(obj, dict):
        val = obj.get(key) or []
        if isinstance(val, list):
            return [str(x.get("id") if isinstance(x, dict) else x) for x in val]
    return []


def run_tool_agent(client, query, cids, profiles, id_map) -> tuple[list[dict], dict]:
    acc: dict = {}
    compact = [compact_view(c, profiles.get(c, {})) for c in cids]
    proj = query_block(query)

    # Call 1: triage from compact profiles -> initial ranking + up to 5 inspect picks.
    p1 = f"""You are ranking Horizon 2020 funding call candidates for one project.
You see a THIN profile for each of the {len(cids)} candidates: only the normalized
action (instrument) type and the call-scale band---no scientific scope. Candidates
are in retrieval order and identified by opaque ids.

You may INSPECT up to {MAX_INSPECT} candidates to reveal their FULL profile (scope,
specific challenge, and expected outcome) before finalizing. Inspection is the only
way to obtain scientific/thematic evidence, so choose carefully.

Do two things and return JSON only:
1. "initial_ranked_ids": an initial ranking of ALL ids, best match first, from the
   compact profiles.
2. "inspect": up to {MAX_INSPECT} ids whose full profile would most likely change
   your ranking---the uncertain or high-potential candidates.

Schema: {{"initial_ranked_ids": ["<id>", ...], "inspect": ["<id>", ...]}}

Project:
{json.dumps(proj, ensure_ascii=False, sort_keys=True)}

Compact candidates (retrieval order):
{json.dumps(compact, ensure_ascii=False)}
"""
    raw1, u1, dt1 = _timed_generate(client, p1)
    _accumulate(acc, u1, dt1)
    valid = set(cids)
    initial_ids = _parse_ids_field(raw1, "initial_ranked_ids")
    inspect_ids, seen = [], set()
    for c in _parse_ids_field(raw1, "inspect"):
        if c in valid and c not in seen:
            inspect_ids.append(c); seen.add(c)
        if len(inspect_ids) >= MAX_INSPECT:
            break

    # inspect() tool: reveal full profiles for the chosen candidates.
    inspected = {c: full_view(c, profiles.get(c, {})) for c in inspect_ids}

    # Call 2: revise the ranking once, given the inspected full profiles.
    p2 = f"""You previously produced an initial ranking of {len(cids)} funding-call
candidates from their compact profiles, and chose to inspect some of them. Their
FULL profiles are now revealed (complete scope, specific challenge, expected
outcome). Revise your ranking ONCE using this detail, then return the final ranking
of ALL ids, best match first. Keep ids you did not inspect ordered sensibly from
their compact profiles and your initial ranking. Return JSON only.

Schema: {{"ranked_ids": ["<id>", ...]}}

Project:
{json.dumps(proj, ensure_ascii=False, sort_keys=True)}

Your initial ranking:
{json.dumps(initial_ids, ensure_ascii=False)}

Full profiles of the candidates you inspected:
{json.dumps(inspected, ensure_ascii=False)}

Compact profiles of all candidates (retrieval order):
{json.dumps(compact, ensure_ascii=False)}
"""
    raw2, u2, dt2 = _timed_generate(client, p2)
    _accumulate(acc, u2, dt2)
    ranked = _rank_from_ids(_parse_ranked_ids(raw2), cids, id_map)
    internals = {
        "inspected": inspect_ids,
        "n_inspected": len(inspect_ids),
        "initial_ranked_ids": initial_ids,
        "coordinator_raw": raw2,
        "coordinator_ranked_ids": _parse_ranked_ids(raw2),
    }
    return ranked, acc, internals


def run_compact_only(client, query, cids, profiles, id_map) -> tuple[list[dict], dict]:
    """Floor condition: rank from the thin compact profiles only, no inspection."""
    compact = [compact_view(c, profiles.get(c, {})) for c in cids]
    payload = {"project": query_block(query), "candidates": compact}
    prompt = f"""You are ranking Horizon 2020 funding call candidates for one project.
You see only a THIN profile for each candidate: the normalized action (instrument)
type and the call-scale band---no scientific scope. Rank ALL candidate ids from best
match to worst using only this evidence. Use only ids present. Return JSON only.

Schema: {{"ranked_ids": ["<id>", ...]}}

Input:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}
"""
    raw, usage, dt = _timed_generate(client, prompt)
    ranked = _rank_from_ids(_parse_ranked_ids(raw), cids, id_map)
    acc: dict = {}
    _accumulate(acc, usage, dt)
    return ranked, acc, None


# --------------------------------------------------------------------------- #
# Arm D: reviewer (worker -> reviewer -> one revision)
# --------------------------------------------------------------------------- #
def _worker_rank(client, query, candidates, acc, want_argument=False):
    extra = (',\n  "argument": "<=30 words: why your top pick"' if want_argument else "")
    prompt = f"""You match one Horizon 2020 project to the funding call it was funded
under. Rank ALL candidate ids from best match to worst, reasoning jointly over
scientific fit, instrument type, and funding scale. Use only ids present. Return
JSON only.

Schema: {{"ranked_ids": ["<id>", ...]{extra}}}

Input:
{json.dumps({"project": query_block(query), "candidates": candidates}, ensure_ascii=False, sort_keys=True)}
"""
    raw, u, dt = _timed_generate(client, prompt)
    _accumulate(acc, u, dt)
    return raw


def run_reviewer(client, query, cids, profiles, id_map) -> tuple[list[dict], dict]:
    acc: dict = {}
    candidates = [governed_view(c, profiles.get(c, {}), ALL_FIELDS) for c in cids]
    raw1 = _worker_rank(client, query, candidates, acc)
    worker_ids = _parse_ranked_ids(raw1)
    prompt = f"""You are a REVIEWER in a funding-call matching system. A worker agent
produced the ranking below. Independently check the worker's top candidates against
the evidence: verify scientific fit, instrument compatibility, and funding scale;
look for a better candidate ranked too low, or a top candidate that does not truly
fit. Then produce a corrected FINAL ranking of ALL ids. Return JSON only.

Schema: {{"ranked_ids": ["<id>", ...], "changes": "<=40 words on what you changed"}}

Project:
{json.dumps(query_block(query), ensure_ascii=False, sort_keys=True)}

Candidate profiles:
{json.dumps(candidates, ensure_ascii=False)}

Worker's ranking (best first):
{json.dumps(worker_ids, ensure_ascii=False)}
"""
    raw2, u, dt = _timed_generate(client, prompt)
    _accumulate(acc, u, dt)
    ranked = _rank_from_ids(_parse_ranked_ids(raw2), cids, id_map)
    internals = {"worker_ranked_ids": worker_ids, "coordinator_raw": raw2,
                 "coordinator_ranked_ids": _parse_ranked_ids(raw2)}
    return ranked, acc, internals


# --------------------------------------------------------------------------- #
# Arm E: debate (agent A vs agent B -> judge)
# --------------------------------------------------------------------------- #
def _topk_with_arg(raw, k=8):
    ids = _parse_ranked_ids(raw)[:k]
    try:
        arg = extract_json_object(raw).get("argument", "") if raw.strip().startswith("{") else ""
    except Exception:
        arg = ""
    return ids, str(arg)[:400]


def run_debate(client, query, cids, profiles, id_map) -> tuple[list[dict], dict]:
    acc: dict = {}
    candidates = [governed_view(c, profiles.get(c, {}), ALL_FIELDS) for c in cids]
    proj = json.dumps(query_block(query), ensure_ascii=False, sort_keys=True)
    cand_json = json.dumps(candidates, ensure_ascii=False)
    # Agent A
    rawA = _worker_rank(client, query, candidates, acc, want_argument=True)
    a_ids, a_arg = _topk_with_arg(rawA)
    # Agent B: critique A and argue its own ranking
    pB = f"""You are agent B in a debate about which Horizon 2020 funding call matches
a project. Agent A proposed the top candidates and an argument below. Critically
assess A: where do you disagree, and why? Argue for your own ranking of ALL ids.
Return JSON only.

Schema: {{"ranked_ids": ["<id>", ...], "argument": "<=50 words rebutting or supporting A"}}

Project:
{proj}

Candidate profiles:
{cand_json}

Agent A's top picks: {json.dumps(a_ids, ensure_ascii=False)}
Agent A's argument: {a_arg}
"""
    rawB, u, dt = _timed_generate(client, pB); _accumulate(acc, u, dt)
    b_ids, b_arg = _topk_with_arg(rawB)
    # Judge: resolve
    pJ = f"""You are the judge of a debate between two agents about which Horizon 2020
funding call matches a project. Weigh both arguments against the evidence and
produce the FINAL ranking of ALL ids, best match first. Return JSON only.

Schema: {{"ranked_ids": ["<id>", ...]}}

Project:
{proj}

Candidate profiles:
{cand_json}

Agent A top picks + argument: {json.dumps(a_ids, ensure_ascii=False)} | {a_arg}
Agent B top picks + argument: {json.dumps(b_ids, ensure_ascii=False)} | {b_arg}
"""
    rawJ, u, dt = _timed_generate(client, pJ); _accumulate(acc, u, dt)
    ranked = _rank_from_ids(_parse_ranked_ids(rawJ), cids, id_map)
    internals = {"a_ids": a_ids, "b_ids": b_ids, "a_arg": a_arg, "b_arg": b_arg,
                 "coordinator_raw": rawJ, "coordinator_ranked_ids": _parse_ranked_ids(rawJ)}
    return ranked, acc, internals


ARMS = {
    "generalist": run_generalist,
    "multiagent": run_multiagent,
    "tool_agent": run_tool_agent,
    "compact_only": run_compact_only,
    "reviewer": run_reviewer,
    "debate": run_debate,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--model", default="gemini-flash-latest")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeat-tag", default=None, help="Suffix for repeated-run output files (variance pilot).")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--log-internals", action="store_true",
                    help="Also dump per-query specialist scores + coordinator output "
                         "(multiagent only) for error analysis.")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to touch the sealed test split without --allow-test.")

    profiles, id_map, label_to_cid = load_profiles()
    top50 = load_retriever_top50(args.split)
    rows = load_frozen_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    truths = load_ground_truth(args.split)

    out_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / "arch_comparison" / f"{args.arm}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.repeat_tag}" if args.repeat_tag else ""
    per_query_path = out_dir / f"{args.split}{tag}_arch_outputs.jsonl"

    internals_handle = None
    if args.log_internals:
        internals_handle = (out_dir / f"{args.split}{tag}_internals.jsonl").open("a", encoding="utf-8")

    done: dict[str, dict] = {}
    if per_query_path.exists():
        for line in per_query_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[str(r["query_id"])] = r

    client = make_client(args.model)
    fn = ARMS[args.arm]
    ranked_by_query: dict[str, list[dict]] = {}
    totals: dict = {}
    with per_query_path.open("a", encoding="utf-8") as handle:
        for i, row in enumerate(rows, 1):
            qid = str(row["query_id"])
            if qid in done:
                ranked_by_query[qid] = done[qid]["ranked"]
                _accumulate(totals, {
                    "prompt_token_count": done[qid]["usage"]["prompt_token_count"],
                    "candidates_token_count": done[qid]["usage"]["candidates_token_count"],
                }, done[qid]["usage"]["latency_s"], done[qid]["usage"]["n_calls"])
                continue
            cids = [label_to_cid[l] for l in top50[qid] if l in label_to_cid]
            ranked, usage, internals = fn(client, row, cids, profiles, id_map)
            ranked_by_query[qid] = ranked
            rec = {"query_id": qid, "true_call_label": truths[qid][0],
                   "top1": ranked[0]["call_label"] if ranked else "", "ranked": ranked, "usage": usage}
            done[qid] = rec
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            if internals_handle is not None and internals is not None:
                internals_handle.write(json.dumps({
                    "query_id": qid,
                    "true_call_label": truths[qid][0],
                    "true_cid": label_to_cid.get(truths[qid][0]),
                    "cids": cids,
                    "cid_to_label": {c: id_map[c] for c in cids},
                    **internals,
                }, ensure_ascii=False) + "\n")
                internals_handle.flush()
            _accumulate(totals, usage, usage.get("latency_s", 0.0), usage.get("n_calls", 0))
            if i % 10 == 0:
                print(f"  [{args.arm}/{args.split}] {i}/{len(rows)}")

    metrics = rank_metrics(ranked_by_query, truths)
    n = len(ranked_by_query)
    cost = usage_cost_usd(args.model, {
        "prompt_token_count": totals.get("prompt_token_count", 0),
        "candidates_token_count": totals.get("candidates_token_count", 0),
    })
    summary = {
        "arm": args.arm, "model": args.model, "provider": provider_of(args.model),
        "split": args.split, "n_queries": n, "condition": "anonymized_governed_profiles",
        "metrics": metrics,
        "total_tokens": totals.get("prompt_token_count", 0) + totals.get("candidates_token_count", 0),
        "prompt_tokens": totals.get("prompt_token_count", 0),
        "output_tokens": totals.get("candidates_token_count", 0),
        "total_latency_s": round(totals.get("latency_s", 0.0), 2),
        "mean_latency_s_per_query": round(totals.get("latency_s", 0.0) / n, 3) if n else None,
        "n_llm_calls": totals.get("n_calls", 0),
        "mean_calls_per_query": round(totals.get("n_calls", 0) / n, 2) if n else None,
        "cost_usd": round(cost, 4) if cost is not None else None,
    }
    (out_dir / f"{args.split}{tag}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / f"{args.split}{tag}_per_query_rows.json").write_text(
        json.dumps(per_query_rows(ranked_by_query, truths), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
