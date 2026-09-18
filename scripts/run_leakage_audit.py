"""Leakage audit for candidate-side header features (pre-registered thresholds).

For each feature we ask: how much does this feature ALONE determine the true
call? A feature that clears the provenance test but nearly hands over the answer
is FLAGGED FOR REVIEW (exclude / coarsen / justify) per admissibility_manifest.yaml.

Per feature we report:
  coverage                     - queries whose true call has a value
  single_feature_recall_at_1   - queries where the true call is the ONLY call
                                 sharing its value (feature pins it down)
  median/mean/p90_remaining    - size of the confusable set (calls sharing a value)
  entropy_reduction            - normalised MI: 1 - H(call|feature)/H(call)
  decision                     - OK / FLAG-FOR-REVIEW vs manifest thresholds

Scope note: portal headers exist for the 101 curated true calls, so the candidate
universe for header features is those 101 calls (the evidence-rich eval set).
Action type additionally gets a full 413-candidate test via dominant_fundingScheme
(its structured equivalent), since it is the feature with the real leakage risk.
"""
from __future__ import annotations

import csv
import json
import re
import math
import collections
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "frozen" / "rq2_v1_seed42_description_filtered"
CURATED = FROZEN / "curated_evidence_subset"
HEADERS = CURATED / "topic_headers_raw.csv"
SUBCALL_CODES = CURATED / "subcall_to_codes.json"
NORM = CURATED / "action_type_normalization.json"
ELIG = CURATED / "eligibility_table.csv"

THRESH = {"recall_at_1": 0.90, "median_remaining": 1, "p90_remaining": 3}


def budget_band(v) -> str | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x < 50e6:   return "small (<50M)"
    if x < 200e6:  return "medium (50-200M)"
    if x < 1e9:    return "large (200M-1B)"
    return "very-large (>1B)"


def build_call_values():
    """subCall -> {feature: set(values)} aggregated over its topic codes."""
    rows = {r["topic_code"]: r for r in csv.DictReader(open(HEADERS))}
    subcall_to_codes = json.loads(SUBCALL_CODES.read_text())
    norm = json.loads(NORM.read_text())
    calls = {}
    for sc, codes in subcall_to_codes.items():
        v = collections.defaultdict(set)
        for c in codes:
            r = rows.get(c)
            if not r:
                continue
            for raw in (r.get("raw_action_type") or "").split(" | "):
                if raw.strip() and raw.strip() in norm:
                    v["normalized_action_type"].add(norm[raw.strip()]["normalized_action_type"])
            for y in re.findall(r"((?:19|20)\d{2})", r.get("opening_date") or ""):
                v["opening_year"].add(y)
            for y in re.findall(r"((?:19|20)\d{2})", r.get("deadline_date") or ""):
                v["deadline_year"].add(y)
            for d in (r.get("deadline_date") or "").split(" | "):
                if d.strip():
                    v["full_deadline"].add(d.strip())
            b = budget_band(r.get("topic_budget_eur"))
            if b:
                v["call_scale"].add(b)
            for s in (r.get("topic_status") or "").split(" | "):
                if s.strip():
                    v["topic_status"].add(s.strip())
            for s in (r.get("submission_stage") or "").split(" | "):
                if s.strip():
                    v["submission_stage"].add(s.strip())
        calls[sc] = v
    return calls


def audit_feature(feature, call_values, query_calls):
    """query_calls: list of true-call labels, one per curated query (weighted)."""
    # confusable set for a call C = calls sharing >=1 value with C
    value_to_calls = collections.defaultdict(set)
    for sc, feats in call_values.items():
        for val in feats.get(feature, set()):
            value_to_calls[val].add(sc)
    covered_calls = {sc for sc, f in call_values.items() if f.get(feature)}

    remaining, r1, cov = [], 0, 0
    for true in query_calls:
        vals = call_values.get(true, {}).get(feature, set())
        if not vals:
            continue
        cov += 1
        confusable = set()
        for val in vals:
            confusable |= value_to_calls[val]
        remaining.append(len(confusable))
        if len(confusable) == 1:
            r1 += 1
    n = len(query_calls)
    if cov == 0:
        return None

    # entropy reduction over the query-weighted call distribution, using a single
    # representative value per call (sorted-first) to define the partition.
    call_rep = {sc: (sorted(f[feature])[0] if f.get(feature) else None) for sc, f in call_values.items()}
    call_freq = collections.Counter(query_calls)
    total = sum(call_freq.values())
    H_call = -sum((c / total) * math.log2(c / total) for c in call_freq.values())
    part = collections.defaultdict(lambda: collections.Counter())
    for sc, c in call_freq.items():
        part[call_rep.get(sc)][sc] += c
    H_cond = 0.0
    for val, calls_in in part.items():
        pv = sum(calls_in.values()) / total
        Hv = -sum((c / sum(calls_in.values())) * math.log2(c / sum(calls_in.values()))
                  for c in calls_in.values())
        H_cond += pv * Hv
    ent_red = (H_call - H_cond) / H_call if H_call > 0 else 0.0

    recall1 = r1 / cov
    med = statistics.median(remaining)
    p90 = sorted(remaining)[max(0, math.ceil(0.9 * len(remaining)) - 1)]
    flag = (recall1 >= THRESH["recall_at_1"] or med <= THRESH["median_remaining"]
            or p90 <= THRESH["p90_remaining"])
    return {
        "feature": feature, "coverage": cov / n,
        "recall_at_1": recall1,
        "median_remaining": med, "mean_remaining": statistics.mean(remaining),
        "p90_remaining": p90, "entropy_reduction": ent_red,
        "decision": "FLAG-FOR-REVIEW" if flag else "OK",
    }


def audit_action_type_over_413():
    """Full-pool robustness test for action type via dominant_fundingScheme."""
    pool = json.load(open(FROZEN / "candidate_pool.json"))["candidates"]
    fam = {}
    for c in pool:
        s = (c.get("dominant_fundingScheme") or "").upper()
        base = ("ERC" if s.startswith("ERC") else "MSCA" if s.startswith("MSCA")
                else "RIA" if s == "RIA" else "IA" if s == "IA" else "CSA" if s == "CSA"
                else "SME" if s.startswith("SME") else s or "OTHER")
        fam[c["call_label"]] = base
    qs = [json.loads(l)["true_call_label"] for l in open(FROZEN / "dev_queries.jsonl")]
    qs += [json.loads(l)["true_call_label"] for l in open(FROZEN / "test_queries.jsonl")]
    value_to_calls = collections.defaultdict(set)
    for lbl, f in fam.items():
        value_to_calls[f].add(lbl)
    remaining, r1 = [], 0
    for t in qs:
        conf = value_to_calls[fam.get(t)]
        remaining.append(len(conf))
        if len(conf) == 1:
            r1 += 1
    return {"feature": "normalized_action_type [413-pool via fundingScheme]",
            "coverage": 1.0, "recall_at_1": r1 / len(qs),
            "median_remaining": statistics.median(remaining),
            "mean_remaining": statistics.mean(remaining),
            "p90_remaining": sorted(remaining)[max(0, math.ceil(0.9 * len(remaining)) - 1)],
            "entropy_reduction": float("nan"),
            "decision": "FLAG-FOR-REVIEW" if (r1 / len(qs) >= THRESH["recall_at_1"]
                        or statistics.median(remaining) <= THRESH["median_remaining"]) else "OK"}


def main() -> None:
    call_values = build_call_values()
    elig = [r for r in csv.DictReader(open(ELIG)) if r["in_curated_subset"] == "1"]
    query_calls = [r["true_call"] for r in elig]
    features = ["normalized_action_type", "opening_year", "deadline_year",
                "full_deadline", "call_scale", "topic_status", "submission_stage"]
    print(f"curated queries: {len(query_calls)} | unique true calls: {len(set(query_calls))} "
          f"| candidate universe: {len(call_values)} curated calls\n")
    hdr = ["feature", "coverage", "recall_at_1", "median_remaining",
           "mean_remaining", "p90_remaining", "entropy_reduction", "decision"]
    print(f"{'feature':40s} {'cov':>5s} {'R@1':>5s} {'med':>4s} {'mean':>5s} {'p90':>4s} {'entR':>5s}  decision")
    results = []
    for f in features:
        r = audit_feature(f, call_values, query_calls)
        if not r:
            continue
        results.append(r)
        print(f"{r['feature']:40s} {r['coverage']:5.0%} {r['recall_at_1']:5.0%} "
              f"{r['median_remaining']:4.0f} {r['mean_remaining']:5.1f} {r['p90_remaining']:4.0f} "
              f"{r['entropy_reduction']:5.2f}  {r['decision']}")
    r413 = audit_action_type_over_413()
    results.append(r413)
    print(f"{r413['feature']:40s} {r413['coverage']:5.0%} {r413['recall_at_1']:5.0%} "
          f"{r413['median_remaining']:4.0f} {r413['mean_remaining']:5.1f} {r413['p90_remaining']:4.0f} "
          f"{'  nan':>5s}  {r413['decision']}")

    out = CURATED / "leakage_audit.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(results)
    print(f"\nthresholds (pre-registered): R@1>={THRESH['recall_at_1']}, "
          f"median<={THRESH['median_remaining']}, p90<={THRESH['p90_remaining']} -> FLAG")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
