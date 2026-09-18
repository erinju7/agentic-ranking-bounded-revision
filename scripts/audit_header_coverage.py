"""Coverage + mapping audit for the fetched portal headers (curated 101 calls).

Answers the four questions gating whether the multi-signal agent is viable:
  1. Did DATE coverage jump from the ~12%-in-text baseline?
  2. Is BUDGET the call total or per-project contribution? (report magnitudes)
  3. Does ACTION TYPE map stably to a small class vocabulary?
  4. How many calls cannot be uniquely / confidently mapped (need programme PDF)?

Aggregates topic-level headers up to the subCall level (a subCall spans several
topic codes), then reports per-signal coverage over the 101 curated true calls
and, weighted, over the curated queries.
"""
from __future__ import annotations

import csv
import json
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "frozen" / "rq2_v1_seed42_description_filtered" / "curated_evidence_subset"
HEADERS = CURATED / "topic_headers_raw.csv"
ELIG = CURATED / "eligibility_table.csv"
SUBCALL_CODES = CURATED / "subcall_to_codes.json"


def load_headers() -> list[dict]:
    return list(csv.DictReader(open(HEADERS)))


def nonempty(v) -> bool:
    return v not in (None, "", "nan", "None")


def main() -> None:
    rows = load_headers()
    by_code = {r["topic_code"]: r for r in rows}
    # authoritative one-to-many mapping: subCall -> its topic codes
    subcall_to_codes = json.loads(SUBCALL_CODES.read_text())
    by_call = {sc: [by_code[c] for c in codes if c in by_code]
               for sc, codes in subcall_to_codes.items()}
    calls = sorted(by_call)
    n = len(calls)
    covered = sum(bool(by_call[c]) for c in calls)
    print(f"topic rows: {len(rows)} | curated subCalls: {n} | with >=1 topic header: {covered}")

    # ---- fetch/mapping status ----
    fs = collections.Counter(r["fetch_status"] for r in rows)
    mc = collections.Counter(r["mapping_confidence"] for r in rows)
    print("\n[fetch_status]  ", dict(fs))
    print("[mapping_confidence]", dict(mc))

    # a subCall is 'covered' for a field if ANY of its topics has that field
    def call_has(call, field):
        return any(nonempty(r.get(field)) for r in by_call[call])

    fields = ["opening_date", "deadline_date", "raw_action_type",
              "topic_budget_eur", "submission_stage", "topic_status"]
    print(f"\n[per-call coverage over {n} curated true calls] (>=1 topic has the field)")
    for f in fields:
        c = sum(call_has(call, f) for call in calls)
        print(f"  {f:20s} {c:3d}/{n}  {c/n:.0%}")

    # ---- Q1: dates vs the ~12% in-text baseline ----
    dates = sum(call_has(c, "deadline_date") or call_has(c, "opening_date") for c in calls)
    print(f"\nQ1 DATES: {dates}/{n} ({dates/n:.0%}) curated calls now have a structured "
          f"opening/deadline date  (in-text baseline was ~12%).")

    # ---- Q2: budget magnitude / granularity ----
    budgets = []
    for r in rows:
        v = r.get("topic_budget_eur")
        if nonempty(v):
            try:
                budgets.append(float(v))
            except ValueError:
                pass
    if budgets:
        budgets.sort()
        med = budgets[len(budgets) // 2]
        print(f"\nQ2 BUDGET: {len(budgets)} topics carry a budget number. "
              f"min {min(budgets):,.0f}  median {med:,.0f}  max {max(budgets):,.0f} EUR.")
        big = sum(b > 50_000_000 for b in budgets)
        print(f"   {big}/{len(budgets)} exceed EUR 50M -> these are CALL-LEVEL totals, "
              f"not per-project contributions. Per-project size must come from the "
              f"description text or programme PDF; header budget is a call-total signal.")

    # ---- Q3: action type -> class vocabulary stability ----
    raw_types = collections.Counter()
    for r in rows:
        v = r.get("raw_action_type")
        if nonempty(v):
            for t in v.split(" | "):
                raw_types[t.strip()] += 1
    print(f"\nQ3 ACTION TYPE: {len(raw_types)} distinct raw typeOfAction strings across topics.")
    for t, c in raw_types.most_common(20):
        print(f"   {c:3d}  {t}")
    if len(raw_types) > 20:
        print(f"   ... (+{len(raw_types)-20} more)")

    # ---- Q4: unmappable / low-confidence calls ----
    call_conf = {}
    for call in calls:
        confs = [r["mapping_confidence"] for r in by_call[call]]
        call_conf[call] = ("high" if "high" in confs else
                           "medium" if "medium" in confs else
                           "low" if "low" in confs else "none")
    conf_dist = collections.Counter(call_conf.values())
    print(f"\nQ4 MAPPING: per-call best confidence -> {dict(conf_dist)}")
    weak = [c for c, v in call_conf.items() if v in ("low", "none")]
    if weak:
        print(f"   {len(weak)} calls need review / programme PDF: {weak}")

    # ---- weighted by curated queries ----
    elig = [r for r in csv.DictReader(open(ELIG)) if r["in_curated_subset"] == "1"]
    q_by_call = collections.Counter(r["true_call"] for r in elig)
    total_q = sum(q_by_call.values())
    print(f"\n[query-weighted coverage over {total_q} curated queries]")
    for f in ["deadline_date", "opening_date", "raw_action_type", "topic_budget_eur"]:
        w = sum(q_by_call[c] for c in calls if call_has(c, f))
        print(f"  {f:20s} {w}/{total_q}  {w/total_q:.0%}")


if __name__ == "__main__":
    main()
