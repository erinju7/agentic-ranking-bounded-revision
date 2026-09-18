"""COST-PROFILING REPLICATION (clearly separate from ranking-performance runs).
Re-executes each system A/B/C/D on a shared frozen 12-query subset with the IDENTICAL
configuration (prompts, model, temp 0, pools) purely to record real computational cost:
model calls, input/output/total tokens, wall-clock latency, and monetary cost per query.
Prompts are imported verbatim from bright_finalize_measure.py. Nothing is tuned.
This does NOT report ranking metrics; those come from the full frozen runs.
"""
from __future__ import annotations
import json, time, statistics as st
from pathlib import Path
from rq2_core import make_client, extract_json_object, usage_cost_usd
from bright_finalize_measure import (p_rerank_plain, p_concept_B, p_rerank_B,
                                     p_a1, p_a2, p_a3_full, p_a3_anchor)

ROOT = Path(__file__).resolve().parents[1]
POOLS = json.loads((ROOT/"data"/"bright_hardpool"/"biology"/"pools.json").read_text())
OUT = ROOT/"results"/"bright_finalize"; OUT.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-flash-latest"
PROBE_N = 12


def call(client, prompt):
    t = time.perf_counter()
    raw = client.generate(prompt)
    dt = time.perf_counter() - t
    u = client.last_usage_metadata or {}
    it = u.get("prompt_token_count") or 0
    ot = u.get("candidates_token_count") or 0
    tt = u.get("total_token_count") or (it + ot)
    cost = usage_cost_usd(MODEL, u) or 0.0
    try: obj = extract_json_object(raw)
    except Exception: obj = {}
    return obj, dt, it, ot, tt, cost


def main():
    client = make_client(MODEL)
    ids = list(POOLS)[:PROBE_N]
    agg = {s: {"lat": [], "it": [], "ot": [], "tt": [], "cost": []} for s in "ABCD"}
    for qid in ids:
        pl = POOLS[qid]; q = pl["query"]; a2t = pl["alias_to_text"]
        body = [{"id": a, "text": a2t[a]} for a in a2t]

        def rec(s, calls):
            lat = sum(c[1] for c in calls); it = sum(c[2] for c in calls)
            ot = sum(c[3] for c in calls); tt = sum(c[4] for c in calls); cost = sum(c[5] for c in calls)
            agg[s]["lat"].append(lat); agg[s]["it"].append(it); agg[s]["ot"].append(ot)
            agg[s]["tt"].append(tt); agg[s]["cost"].append(cost)

        # A: 1 call
        rec("A", [call(client, p_rerank_plain(q, body))])
        # B: 2 calls (concept + concept-guided rerank)
        r = call(client, p_concept_B(q)); ca = r[0]
        rec("B", [r, call(client, p_rerank_B(q, body, ca))])
        # C: 3 calls (A1, A2, A3 full re-rank)
        rA1 = call(client, p_a1(q)); rA2 = call(client, p_a2(q))
        rec("C", [rA1, rA2, call(client, p_a3_full(q, rA1[0], rA2[0], body))])
        # D: 4 calls = baseline anchor (A) + A1 + A2 + A3 anchor-edit.
        # D's output depends on System A's ranking (the anchor), so its honest end-to-end
        # cost includes the baseline call; A1/A2 could run in parallel with it in deployment.
        anchor = call(client, p_rerank_plain(q, body))
        rA1 = call(client, p_a1(q)); rA2 = call(client, p_a2(q))
        rec("D", [anchor, rA1, rA2, call(client, p_a3_anchor(q, rA1[0], rA2[0], body))])

    calls_per = {"A": 1, "B": 2, "C": 3, "D": 4}
    prof = {}
    for s in "ABCD":
        a = agg[s]
        prof[s] = {"calls_per_query": calls_per[s],
                   "input_tokens_per_query": round(st.mean(a["it"]), 1),
                   "output_tokens_per_query": round(st.mean(a["ot"]), 1),
                   "total_tokens_per_query": round(st.mean(a["tt"]), 1),
                   "latency_s_per_query": round(st.mean(a["lat"]), 2),
                   "cost_usd_per_query": round(st.mean(a["cost"]), 6)}
    out = {"kind": "cost_profiling_replication", "probe_n": PROBE_N, "model": MODEL,
           "pricing_usd_per_mtok": {"input": 0.30, "output": 2.50}, "systems": prof}
    (OUT/"cost_profile.json").write_text(json.dumps(out, indent=1))
    print(f"=== COST PROFILE (replication, n={PROBE_N} shared queries, {MODEL}) ===")
    print(f"{'sys':<4}{'calls':>6}{'in_tok':>9}{'out_tok':>9}{'tot_tok':>9}{'lat_s':>8}{'$/query':>11}")
    for s in "ABCD":
        p = prof[s]
        print(f"{s:<4}{p['calls_per_query']:>6}{p['input_tokens_per_query']:>9.0f}{p['output_tokens_per_query']:>9.0f}"
              f"{p['total_tokens_per_query']:>9.0f}{p['latency_s_per_query']:>8.2f}{p['cost_usd_per_query']:>11.6f}")
    print(f"\nwrote {OUT/'cost_profile.json'}")


if __name__ == "__main__":
    main()
