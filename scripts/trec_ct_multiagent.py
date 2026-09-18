"""Structured eligibility-decomposition multi-agent on the frozen TREC CT 2022
benchmark (NEW arm; does not modify benchmark / pools / prompts / generalist).

Four agents, strict information parity with the single-pass generalist (union of
specialist evidence = generalist evidence: conditions+summary+eligibility):

  1. Clinical Match agent  -- medical relevance (disease/stage/biomarker/prior
     treatment/intervention) from conditions + summary.
  2. Inclusion Verification agent -- reads the FULL eligibility text; does the
     patient satisfy inclusion?
  3. Exclusion Verification agent -- reads the SAME full eligibility text; is the
     patient excluded? (identical info to Inclusion; only the objective differs)
  4. Coordinator -- a genuine REASONING agent (not a rule engine, no hard AND gate,
     no averaging): reasons over the specialists' outputs and the trial, weighs
     disagreements/overlooked evidence, and produces the final ranking.

Every specialist output (score/flag, confidence, justification, evidence) and the
coordinator output are logged per candidate. Reports retrieval metrics + calls/
latency/cost. Prompts frozen before evaluation; no tuning on the benchmark.
Reads data/trec_ct_2022/; writes results/trec_ct_2022/multiagent/.
"""
from __future__ import annotations
import json, math, os, time, statistics as st
from pathlib import Path
from rq2_core import make_client, extract_json_object, usage_cost_usd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "trec_ct_2022" / "benchmark.jsonl"
OUTDIR = BASE / "results" / "trec_ct_2022" / "multiagent"
OUTDIR.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-flash-latest"


def timed(client, prompt, acc):
    t0 = time.perf_counter()
    raw = client.generate(prompt)
    dt = time.perf_counter() - t0
    u = dict(getattr(client, "last_usage_metadata", {}) or {})
    acc["prompt_tokens"] = acc.get("prompt_tokens", 0) + (u.get("prompt_token_count") or 0)
    acc["output_tokens"] = acc.get("output_tokens", 0) + (u.get("candidates_token_count") or 0)
    acc["latency_s"] = acc.get("latency_s", 0.0) + dt
    acc["calls"] = acc.get("calls", 0) + 1
    return raw


def parse_assess(raw, valid):
    try:
        obj = extract_json_object(raw)
    except Exception:
        return {}
    a = obj.get("assessments", obj) if isinstance(obj, dict) else {}
    return {k: v for k, v in a.items() if k in valid and isinstance(v, dict)} if isinstance(a, dict) else {}


def clinical_agent(client, patient, cands, acc):
    payload = [{"id": c["alias"], "conditions": c["conditions"], "summary": c["summary"]} for c in cands]
    p = f"""You are the CLINICAL MATCH specialist in a clinical-trial matching system.
For each trial, judge how medically relevant it is to the patient -- consider the
disease/condition, disease stage, biomarkers, previous treatment history, and the
relevance of the trial's intervention. Do NOT judge eligibility rules (age limits,
exclusions); judge medical relevance only. Return JSON only.

Schema: {{"assessments": {{"<id>": {{"relevance": 0-100, "confidence": "low|medium|high",
"justification": "<=20 words", "evidence": "<key signals used>"}}}}}}

Patient:
{patient}

Trials:
{json.dumps(payload, ensure_ascii=False)}
"""
    return parse_assess(timed(client, p, acc), {c["alias"] for c in cands})


def inclusion_agent(client, patient, cands, acc):
    payload = [{"id": c["alias"], "eligibility": c["eligibility"]} for c in cands]
    p = f"""You are the INCLUSION VERIFICATION specialist. Read the FULL eligibility
text of each trial and judge whether the patient SATISFIES the trial's inclusion
criteria (what a participant must be/have to qualify). Judge inclusion only. Return
JSON only.

Schema: {{"assessments": {{"<id>": {{"inclusion_satisfied": "Yes|Probably|Uncertain|No",
"confidence": "low|medium|high", "justification": "<=20 words", "evidence": "<criteria checked>"}}}}}}

Patient:
{patient}

Trials (full eligibility text):
{json.dumps(payload, ensure_ascii=False)}
"""
    return parse_assess(timed(client, p, acc), {c["alias"] for c in cands})


def exclusion_agent(client, patient, cands, acc):
    payload = [{"id": c["alias"], "eligibility": c["eligibility"]} for c in cands]
    p = f"""You are the EXCLUSION VERIFICATION specialist. Read the FULL eligibility
text of each trial and judge whether the patient VIOLATES any exclusion criterion
(any condition that would disqualify the patient). Judge exclusion only. Return
JSON only.

Schema: {{"assessments": {{"<id>": {{"exclusion_triggered": "Yes|Probably|Uncertain|No",
"confidence": "low|medium|high", "justification": "<=20 words", "evidence": "<criteria checked>"}}}}}}

Patient:
{patient}

Trials (full eligibility text):
{json.dumps(payload, ensure_ascii=False)}
"""
    return parse_assess(timed(client, p, acc), {c["alias"] for c in cands})


def coordinator(client, patient, cands, cl, inc, exc, acc):
    combined = []
    for c in cands:
        i = c["alias"]
        combined.append({
            "id": i, "conditions": c["conditions"], "summary": c["summary"],
            "clinical_match": cl.get(i, {}), "inclusion": inc.get(i, {}), "exclusion": exc.get(i, {}),
        })
    p = f"""You are the COORDINATOR and final judge in a clinical-trial matching
system. Three specialists have assessed each trial: a clinical-match specialist
(medical relevance), an inclusion specialist (does the patient meet inclusion?), and
an exclusion specialist (is the patient excluded?). Each gave a verdict, confidence,
and justification.

You are NOT a rule engine: do not mechanically AND the verdicts and do not average
scores. Reason like an expert: where specialists disagree or are uncertain, weigh
their evidence and confidence, consider whether any specialist overlooked something
given the trial and patient, and decide how eligible the patient truly is for each
trial. A trial the patient is genuinely eligible for should rank above trials that
are on-topic but for which the patient fails inclusion or is excluded.

Rank ALL trial ids from best to worst match. Return JSON only.

Schema: {{"ranked_ids": ["<id>", ...], "confidence": "low|medium|high", "rationale": "<=60 words"}}

Patient:
{patient}

Per-trial information and specialist assessments:
{json.dumps(combined, ensure_ascii=False)}
"""
    raw = timed(client, p, acc)
    try:
        obj = extract_json_object(raw); ids = obj.get("ranked_ids", [])
    except Exception:
        obj, ids = {}, []
    valid = {c["alias"] for c in cands}; ranked = []
    for x in ids:
        v = str(x.get("id") if isinstance(x, dict) else x)
        if v in valid and v not in ranked:
            ranked.append(v)
    for c in cands:
        if c["alias"] not in ranked:
            ranked.append(c["alias"])
    return ranked, raw, combined


def dcg(rels): return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
def ndcg(labels, k):
    d = dcg(labels[:k]); i = dcg(sorted(labels, reverse=True)[:k]); return d / i if i else 0.0


def main():
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    lim = int(os.getenv("LIMIT", "0"))
    if lim:
        rows = rows[:lim]
    client = make_client(MODEL)
    r1 = err1 = err0 = 0; mrrs = []; nd5 = []; nd10 = []; rk2 = []; rk1 = []; acc = {}
    intern_path = OUTDIR / "internals.jsonl"
    intern_path.write_text("")
    for n, r in enumerate(rows, 1):
        cands = r["candidates"]
        for i, c in enumerate(cands):
            c["alias"] = f"TRIAL-{i:02d}"
        lab = {c["alias"]: c["label"] for c in cands}
        cl = clinical_agent(client, r["patient"], cands, acc)
        inc = inclusion_agent(client, r["patient"], cands, acc)
        exc = exclusion_agent(client, r["patient"], cands, acc)
        ranked, craw, combined = coordinator(client, r["patient"], cands, cl, inc, exc, acc)
        labels = [lab[x] for x in ranked]; top1 = lab[ranked[0]]
        r1 += top1 == 2
        if top1 != 2:
            err1 += top1 == 1; err0 += top1 == 0
        fg = next((i + 1 for i, x in enumerate(ranked) if lab[x] == 2), len(ranked) + 1)
        mrrs.append(1 / fg); nd5.append(ndcg(labels, 5)); nd10.append(ndcg(labels, 10))
        rk2.append(st.mean([i + 1 for i, x in enumerate(ranked) if lab[x] == 2]))
        rk1.append(st.mean([i + 1 for i, x in enumerate(ranked) if lab[x] == 1]))
        rec = {"topic_id": r["topic_id"], "ranked": ranked, "alias_label": lab,
               "top1_label": top1, "first_gold_rank": fg,
               "clinical": cl, "inclusion": inc, "exclusion": exc, "coordinator_raw": craw}
        with intern_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{n}/{len(rows)}] topic {r['topic_id']}: top1=qrel{top1} first_gold={fg}", flush=True)
    N = len(rows); nerr = N - r1
    cost = usage_cost_usd(MODEL, {"prompt_token_count": acc.get("prompt_tokens", 0),
                                  "candidates_token_count": acc.get("output_tokens", 0)})
    summary = {
        "arm": "eligibility_decomposition_multiagent_v2_reasoning_coordinator", "n_topics": N,
        "R@1_qrel2": round(r1 / N, 3), "MRR_first_qrel2": round(st.mean(mrrs), 3),
        "nDCG@5": round(st.mean(nd5), 3), "nDCG@10": round(st.mean(nd10), 3),
        "mean_rank_qrel2": round(st.mean(rk2), 2), "mean_rank_qrel1": round(st.mean(rk1), 2),
        "separation_q1_minus_q2": round(st.mean(rk1) - st.mean(rk2), 2),
        "calls_per_topic": round(acc.get("calls", 0) / N, 2),
        "mean_latency_s_per_topic": round(acc.get("latency_s", 0) / N, 2),
        "total_tokens": acc.get("prompt_tokens", 0) + acc.get("output_tokens", 0),
        "cost_usd": round(cost, 4) if cost is not None else None,
        "top1_errors": {"total": nerr, "qrel1_excluded": err1, "qrel0_irrelevant": err0,
                        "prop_qrel1": round(err1 / nerr, 3) if nerr else None},
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
