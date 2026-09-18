"""Level 0: static enriched reranker over the GOVERNED, anonymized candidate profiles.

The lowest-autonomy rung and the fair A/B opponent for the tool agent: both see
the identical governed representation (candidate_profiles.json) under anonymized
ids. Dates are context-gated OFF here. The model ranks anon ids; the true label
is recovered only for scoring (candidate_id_map.json).

Reuses the rq2 harness: model_visible_query (query guard), the Gemini/Claude
clients, parse_ranked_output, and rank_metrics. Only the candidate rendering and
prompt change (governed profile instead of raw call_label + descriptions).

Smoke test:  python rq2_level0_enriched_reranker.py --limit 5
Full run:    python rq2_level0_enriched_reranker.py --model gemini-2.5-flash-lite
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import rq2_core as C

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "frozen" / "rq2_v1_seed42_description_filtered"
CURATED = FROZEN / "curated_evidence_subset"
DET = ROOT / "results" / "rq2_architecture" / "deterministic_topn_retriever"
OUT = ROOT / "results" / "rq2_architecture" / "level0_enriched_reranker"

import re

# Full profile: everything admissible (the shared L0/L1-2 governed representation).
PROFILE_FIELDS = ["title", "specific_challenge", "scope", "expected_outcome",
                  "action_type", "call_scale", "submission_stage"]
# Compact profile: 91% of the full payload is the three long sections. For L0
# ranking, drop specific_challenge + expected_outcome and summarise scope, keeping
# the discriminative structured signals. ~78% fewer candidate tokens.
COMPACT_SCOPE_CAP = 250


def scope_summary(text: str, cap: int = COMPACT_SCOPE_CAP) -> str | None:
    if not text:
        return None
    s = text[:cap]
    bounds = list(re.finditer(r"[.;]\s", s))
    return (s[: bounds[-1].end()].strip() if bounds else s.strip()) or None


def load_env() -> None:
    for env in (ROOT / ".env", ROOT.parent / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_truths_and_queries():
    truths, queries = {}, {}
    for fn in ("dev_queries.jsonl", "test_queries.jsonl"):
        for row in C.read_jsonl(FROZEN / fn):
            qid = str(row["query_id"])
            queries[qid] = row
            truths[qid] = [row["true_call_label"]]
    return truths, queries


def load_top50():
    top = {}
    for fn in ("dev_historical_top50_candidates.jsonl", "test_historical_top50_candidates.jsonl"):
        for o in C.read_jsonl(DET / fn):
            top[str(o["query_id"])] = [c["call_label"] for c in o["candidates"]]
    return top


def governed_candidates(labels, profiles, lab2id, mode="compact"):
    """Render retrieved call_labels as governed, anonymized candidate dicts.
    `call_label` carries the ANON id so the harness parser ranks ids, not labels.

    mode="full"    -> all admissible fields (heavy: ~443 tok/candidate).
    mode="compact" -> title + scope summary + action_type + call_scale +
                      submission_stage (~78% fewer tokens; L0 default).
    """
    out = []
    for lbl in labels:
        cid = lab2id[lbl]
        p = profiles[cid]
        cand = {"call_label": cid}
        if mode == "full":
            for f in PROFILE_FIELDS:
                if p.get(f):
                    cand[f] = p[f]
        else:
            if p.get("title"):
                cand["title"] = p["title"]
            ss = scope_summary(p.get("scope"))
            if ss:
                cand["scope_summary"] = ss
            for f in ("action_type", "call_scale", "submission_stage"):
                if p.get(f):
                    cand[f] = p[f]
        out.append(cand)
    return out


def build_prompt(query, candidates):
    visible = {k: v for k, v in C.model_visible_query(query).items()}
    payload = {"query": visible, "candidates": candidates}
    # guard: candidate ids must be anonymized, query must carry no blocked field
    assert all(c["call_label"].startswith("C_") for c in candidates), "non-anon candidate id"
    assert not (set(visible) & C.BLOCKED_MODEL_INPUT_FIELDS), "blocked query field leaked"
    return f"""You are ranking Horizon 2020 funding call candidates for one project.

Task:
- Use the project query and the candidate profiles.
- Each candidate is identified by an opaque id (e.g. "C_1a2b3c4d").
- Rank candidate ids from best match to worst match for this project.
- Judge fit from the profile fields (scope, specific challenge, expected outcome,
  instrument/action type, funding scale, submission stage), not the id.
- Do not invent ids. Use only ids present in candidates. Return JSON only.

Expected JSON schema:
{{
  "ranked_call_labels": ["C_id1", "C_id2"]
}}

Input:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--limit", type=int, default=None, help="Smoke test: cap #queries.")
    ap.add_argument("--profile-mode", choices=["compact", "full"], default="compact")
    ap.add_argument("--dry-run", action="store_true", help="Build prompts, no API calls.")
    ap.add_argument("--fresh", action="store_true", help="Ignore checkpoint, start over.")
    args = ap.parse_args()
    load_env()

    profiles = json.loads((CURATED / "candidate_profiles.json").read_text())
    id_map = json.loads((CURATED / "candidate_id_map.json").read_text())
    lab2id = {v: k for k, v in id_map.items()}
    subset = json.loads((CURATED / "subset_query_ids.json").read_text())
    curated_qids = list(subset["dev"]) + list(subset["test"])
    truths, queries = load_truths_and_queries()
    top50 = load_top50()

    if args.limit:
        curated_qids = curated_qids[: args.limit]

    # per-(model, profile) run dir so full vs compact never overwrite each other
    run_dir = OUT / f"{re.sub(r'[^A-Za-z0-9._-]', '_', args.model)}__{args.profile_mode}"
    ckpt = run_dir / "rankings_checkpoint.json"
    ranked_by_query = {}
    if ckpt.exists() and not args.fresh and not args.dry_run:
        ranked_by_query = json.loads(ckpt.read_text())
    todo = [q for q in curated_qids if q not in ranked_by_query]
    print(f"Level 0 enriched reranker | model={args.model} | profile={args.profile_mode} "
          f"| queries={len(curated_qids)} (done {len(ranked_by_query)}, todo {len(todo)}) "
          f"| dates=OFF (context-gated)\n")

    client = None if args.dry_run else C.make_client(args.model)
    fails = []
    for i, qid in enumerate(todo, 1):
        labels = top50.get(qid, [])
        cands = governed_candidates(labels, profiles, lab2id, mode=args.profile_mode)
        prompt = build_prompt(queries[qid], cands)
        if args.dry_run:
            if i == 1:
                print("--- sample prompt (query 1) ---")
                print(prompt[:1600], "\n...[truncated]\n")
            continue
        try:
            raw = client.generate(prompt)
            ranked_ids = C.parse_ranked_output(raw, cands)
            ranked = [{"rank": r["rank"], "call_label": id_map[r["call_label"]]} for r in ranked_ids]
        except Exception as exc:  # one bad/hung call must not kill the run
            fails.append(qid)
            print(f"[{i}/{len(todo)}] q={qid} FAILED: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        ranked_by_query[qid] = ranked
        true = truths[qid][0]
        tr = next((r["rank"] for r in ranked if r["call_label"] == true), ">50")
        print(f"[{i}/{len(todo)}] q={qid} true={true} true_rank={tr}")
        if i % 10 == 0:  # checkpoint every 10 queries
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt.write_text(json.dumps(ranked_by_query, indent=1))

    if args.dry_run or not ranked_by_query:
        return
    metrics = C.rank_metrics(ranked_by_query, truths)
    metrics["n_scored"] = len(ranked_by_query)
    metrics["n_failed"] = len(fails)
    print("\n=== metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    if fails:
        print(f"  failed queries ({len(fails)}): {fails}")
    if not args.limit:
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(json.dumps(ranked_by_query, indent=1))
        (run_dir / "curated_summary_metrics.json").write_text(json.dumps(metrics, indent=2))
        (run_dir / "curated_rankings.json").write_text(json.dumps(ranked_by_query, indent=1))
        print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
