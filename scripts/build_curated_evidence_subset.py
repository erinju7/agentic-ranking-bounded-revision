#!/usr/bin/env python3
"""
Construct the evidence-rich curated subset from the frozen benchmark.

Selection is PREDEFINED and does NOT use any model/reranker output. A query is
eligible iff its true call satisfies four gates:

  A. Mapping available   - true call has real EU F&T Portal text (>= MIN_DESC_CHARS).
  B. Stage-1 retrieval   - true call is in the frozen historical top-50 set.
  C. Evidence richness   - true call description exposes >= MIN_EVIDENCE signal
                           dimensions among {scope, specific_challenge,
                           expected_impact, budget, duration, dates, action_type}.
  D. No text leakage     - the exact true subCall label does not appear verbatim
                           in the exposed description.

Eligible queries are then STRATIFIED by (scheme family x call size) and the full
eligible pool is retained (stratification is reported for balance, not used to
cherry-pick). Outputs an auditable per-query table plus the subset id list.
"""
import json, re, csv, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(ROOT, "data/frozen/rq2_v1_seed42_description_filtered")
DET = os.path.join(ROOT, "results/rq2_architecture/deterministic_topn_retriever")
OUT = os.path.join(ROOT, "data/frozen/rq2_v1_seed42_description_filtered/curated_evidence_subset")

MIN_DESC_CHARS = 300
MIN_EVIDENCE = 3    # of 7 signal dimensions
CAP_PER_CALL = 8   # max queries kept per unique true call, to stop a few
                   # high-frequency calls (e.g. MSCA-IF fellowship years) from
                   # dominating the metric. Set to None to keep the full pool.

SIGNALS = {
    "scope":              r"\bscope\b",
    "specific_challenge": r"specific challenge",
    "expected_impact":    r"expected (impact|outcome)",
    "budget":             r"eur\s?[\d\s]{4,}|€\s?[\d\s]{4,}|budget",
    "duration":           r"\b\d{1,3}\s?(months|years)\b",
    "dates":              r"deadline|opening date|closing date",
    "action_type":        r"type of action|research and innovation action|innovation action|coordination and support",
}

def load_desc():
    d = json.load(open(os.path.join(ROOT, "data/reference/subcall_official_descriptions.json")))
    return {k: (" ".join(x for x in v if x) if isinstance(v, list) else (v or "")) for k, v in d.items()}

def load_queries(fn):
    return [json.loads(l) for l in open(os.path.join(FROZEN, fn))]

def load_top50(fn):
    m = {}
    for l in open(os.path.join(DET, fn)):
        o = json.loads(l)
        m[str(o["query_id"])] = o
    return m

def scheme_family(scheme, label):
    s = (scheme or label or "").upper()
    if s.startswith("ERC"): return "ERC"
    if s.startswith("MSCA"): return "MSCA"
    if any(t in s for t in ("-RIA", "RIA")): return "RIA"
    if any(t in s for t in ("-IA", "IA-")):  return "IA"
    if "CSA" in s:  return "CSA"
    return "OTHER"

def size_band(n):
    if n is None: return "unknown"
    if n <= 25:  return "small"
    if n >= 150: return "large"
    return "medium"

def main():
    desc = load_desc()
    pool = {c["call_label"]: c for c in
            json.load(open(os.path.join(FROZEN, "candidate_pool.json")))["candidates"]}
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for split, qfn, tfn in [("dev", "dev_queries.jsonl", "dev_historical_top50_candidates.jsonl"),
                            ("test", "test_queries.jsonl", "test_historical_top50_candidates.jsonl")]:
        top = load_top50(tfn)
        for q in load_queries(qfn):
            qid = str(q["query_id"])
            true = q["true_call_label"]
            text = desc.get(true, "")
            low = text.lower()
            present = {name: bool(re.search(rx, low)) for name, rx in SIGNALS.items()}
            ev = sum(present.values())
            cand = pool.get(true, {})
            n_proj = cand.get("n_projects")
            t = top.get(qid, {})
            retrieved = bool(t.get("true_in_top_n"))
            leak_free = true.lower() not in low
            gates = {
                "A_mapping":   len(text) >= MIN_DESC_CHARS,
                "B_retrieved": retrieved,
                "C_evidence":  ev >= MIN_EVIDENCE,
                "D_leakfree":  leak_free,
            }
            eligible = all(gates.values())
            rows.append({
                "query_id": qid, "split": split, "true_call": true,
                "n_projects": n_proj, "desc_chars": len(text),
                "scheme_family": scheme_family(cand.get("dominant_fundingScheme"), true),
                "size_band": size_band(n_proj),
                "true_rank": t.get("true_rank_in_full_candidate_pool"),
                **{f"has_{k}": int(v) for k, v in present.items()},
                "evidence_score": ev,
                **{k: int(v) for k, v in gates.items()},
                "eligible": int(eligible),
            })

    elig = [r for r in rows if r["eligible"]]
    # per-call cap (deterministic by query_id) to balance high-frequency calls
    if CAP_PER_CALL is not None:
        by_call = collections.defaultdict(list)
        for r in sorted(elig, key=lambda x: x["query_id"]):
            by_call[r["true_call"]].append(r)
        capped = [r for rs in by_call.values() for r in rs[:CAP_PER_CALL]]
        capped_ids = {r["query_id"] for r in capped}
        for r in rows:
            r["in_curated_subset"] = int(r["query_id"] in capped_ids)
        elig = capped
    else:
        for r in rows:
            r["in_curated_subset"] = r["eligible"]
    ids = {"dev": [r["query_id"] for r in elig if r["split"] == "dev"],
           "test": [r["query_id"] for r in elig if r["split"] == "test"],
           "cap_per_call": CAP_PER_CALL, "min_evidence": MIN_EVIDENCE}
    json.dump(ids, open(os.path.join(OUT, "subset_query_ids.json"), "w"), indent=1)

    # write full auditable table (incl. in_curated_subset flag)
    cols = list(rows[0].keys())
    with open(os.path.join(OUT, "eligibility_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    # ---- report ----
    def gate_yield(key):
        return sum(r[key] for r in rows)
    N = len(rows)
    print(f"total queries: {N}  (dev {sum(r['split']=='dev' for r in rows)}, "
          f"test {sum(r['split']=='test' for r in rows)})")
    print("gate pass-rates (independent):")
    for g in ["A_mapping", "B_retrieved", "C_evidence", "D_leakfree"]:
        c = gate_yield(g); print(f"  {g:12s} {c:4d}/{N}  {c/N:.0%}")
    n_elig = sum(r["eligible"] for r in rows)
    print(f"ALL GATES (eligible): {n_elig}/{N}  {n_elig/N:.0%}")
    print(f"FINAL curated subset (cap_per_call={CAP_PER_CALL}): {len(elig)}  "
          f"(dev {len(ids['dev'])}, test {len(ids['test'])}), "
          f"unique true calls {len(set(r['true_call'] for r in elig))}")

    print("\nstratification of curated subset (scheme_family x size_band):")
    strata = collections.Counter((r["scheme_family"], r["size_band"]) for r in elig)
    for (fam, band), c in sorted(strata.items()):
        print(f"  {fam:6s} {band:7s} {c}")
    print("\nevidence-threshold sensitivity (queries passing A,B,D and evidence>=k):")
    abd = [r for r in rows if r["A_mapping"] and r["B_retrieved"] and r["D_leakfree"]]
    for k in range(1, 8):
        c = sum(1 for r in abd if r["evidence_score"] >= k)
        print(f"  evidence>={k}: {c}")
    print(f"\nwrote {OUT}/eligibility_table.csv and subset_query_ids.json")

if __name__ == "__main__":
    main()
