"""Build the FULL frozen TREC Clinical Trials 2022 eligibility-ranking benchmark.

Hard-negative-dominated pools so the benchmark tests ELIGIBILITY reasoning, not
broad topical relevance. Target composition per topic (20 candidates):
    2  x qrel=2  (eligible)
    14 x qrel=1  (topically relevant but eligibility-EXCLUDED -- hard negatives)
    4  x qrel=0  (irrelevant)
Deterministic sampling (seed below); if a topic lacks enough in a category, a
logged fallback tops up the deficit from surplus categories in the priority order
qrel=1 -> qrel=0 -> qrel=2 (preserving hard-negative dominance). Nothing is changed
silently: per-topic composition and every fallback are written to a log.

Frozen: topic ids, candidate trial ids, candidate ORDER, qrels/labels, the fetched
ClinicalTrials.gov fields, and the sampling seed/logic. Kept separate from CORDIS.
Trials fetched on demand from the public ClinicalTrials.gov API v2 (cached).
"""
from __future__ import annotations
import json, re, random, time, csv, urllib.request, collections
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "data"
SRC = BASE / "trec_ct_mvp"                    # topics2022.xml, qrels2022.txt, cache
OUT = BASE / "trec_ct_2022"                   # frozen benchmark (separate dir)
OUT.mkdir(parents=True, exist_ok=True)
CACHE = SRC / "trials_cache.json"
SEED = 42
TARGET = {2: 2, 1: 14, 0: 4}                  # eligible / excluded-hard / irrelevant
FILL_PRIORITY = [1, 0, 2]                     # deterministic top-up order
POOL = 20
ELIG_CAP, SUM_CAP = 1500, 400
API = "https://clinicaltrials.gov/api/v2/studies/{}?fields=protocolSection.identificationModule.briefTitle,protocolSection.conditionsModule.conditions,protocolSection.descriptionModule.briefSummary,protocolSection.eligibilityModule.eligibilityCriteria"


def clean(t): return " ".join(str(t or "").split())


def parse_topics():
    xml = (SRC / "topics2022.xml").read_text()
    return {n: clean(t) for n, t in re.findall(r'<topic number="(\d+)">(.*?)</topic>', xml, re.S)}


def parse_qrels():
    q = collections.defaultdict(dict)
    for line in (SRC / "qrels2022.txt").read_text().splitlines():
        p = line.split()
        if len(p) == 4:
            q[p[0]][p[2]] = int(p[3])
    return q


def fetch_trial(nct, cache):
    if nct in cache:
        return cache[nct]
    rec = None
    try:
        with urllib.request.urlopen(API.format(nct), timeout=25) as r:
            ps = json.load(r).get("protocolSection", {})
        rec = {
            "title": clean(ps.get("identificationModule", {}).get("briefTitle")),
            "conditions": ps.get("conditionsModule", {}).get("conditions", []),
            "summary": clean(ps.get("descriptionModule", {}).get("briefSummary"))[:SUM_CAP],
            "eligibility": clean(ps.get("eligibilityModule", {}).get("eligibilityCriteria"))[:ELIG_CAP],
        }
        time.sleep(0.04)
    except Exception:
        rec = None
    cache[nct] = rec
    return rec


def has_text(rec): return bool(rec and (rec["eligibility"] or rec["summary"]))


def main():
    topics, qrels = parse_topics(), parse_qrels()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    rows, log = [], []
    fetch_fail = 0
    for t in sorted(qrels, key=int):
        rng = random.Random(f"{SEED}-{t}")
        by = collections.defaultdict(list)
        for nct, lab in qrels[t].items():
            by[lab].append(nct)
        for lab in by:
            by[lab].sort(); rng.shuffle(by[lab])          # deterministic order per label
        cursor = {lab: 0 for lab in (0, 1, 2)}
        picked = {2: [], 1: [], 0: []}

        def take(lab, n):
            got = []
            while len(got) < n and cursor[lab] < len(by[lab]):
                nct = by[lab][cursor[lab]]; cursor[lab] += 1
                rec = fetch_trial(nct, cache)
                if has_text(rec):
                    got.append(nct)
            return got

        for lab in (2, 1, 0):
            picked[lab] = take(lab, TARGET[lab])
        # deterministic fallback: top up to POOL from surplus, priority 1 -> 0 -> 2
        deficit = POOL - sum(len(v) for v in picked.values())
        fb = 0
        for lab in FILL_PRIORITY:
            while deficit > 0:
                extra = take(lab, 1)
                if not extra:
                    break
                picked[lab] += extra; deficit -= 1; fb += 1
        n2, n1, n0 = len(picked[2]), len(picked[1]), len(picked[0])
        if n2 == 0 or (n2 + n1 + n0) < 8:
            log.append({"topic": t, "n2": n2, "n1": n1, "n0": n0, "fallback": fb, "status": "SKIPPED"})
            continue
        cands = []
        for lab in (2, 1, 0):
            for nct in picked[lab]:
                rec = cache[nct]
                cands.append({"nct": nct, "label": lab, "title": rec["title"],
                              "conditions": rec["conditions"], "summary": rec["summary"],
                              "eligibility": rec["eligibility"]})
        rng.shuffle(cands)                                  # freeze candidate order
        for i, c in enumerate(cands):
            c["order"] = i
        rows.append({"topic_id": t, "patient": topics[t], "candidates": cands,
                     "gold": [c["nct"] for c in cands if c["label"] == 2]})
        log.append({"topic": t, "n2": n2, "n1": n1, "n0": n0, "fallback": fb,
                    "status": "fallback" if fb else "ok"})
        CACHE.write_text(json.dumps(cache))
        print(f"topic {t}: n2={n2} n1={n1} n0={n0} pool={n2+n1+n0} fallback={fb}")

    (OUT / "benchmark.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    with (OUT / "composition_log.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["topic", "n2", "n1", "n0", "fallback", "status"]); w.writeheader(); w.writerows(log)
    agg = {k: sum(r[k] for r in log if r["status"] != "SKIPPED") for k in ("n2", "n1", "n0", "fallback")}
    manifest = {
        "source": "TREC Clinical Trials 2022 (topics2022.xml, qrels2022.txt)",
        "trial_text": "ClinicalTrials.gov API v2; eligibility<=%d, summary<=%d chars" % (ELIG_CAP, SUM_CAP),
        "seed": SEED, "target_composition": TARGET, "pool_size": POOL,
        "fill_priority": FILL_PRIORITY, "sampling": "per-topic Random(f'{seed}-{topic}'); sort+shuffle each label; take target; fallback top-up; final shuffle freezes order",
        "n_topics_included": len(rows),
        "n_topics_skipped": sum(r["status"] == "SKIPPED" for r in log),
        "aggregate_candidate_counts": agg,
        "gold_definition": "qrel==2 (patient eligible)",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {OUT}/benchmark.jsonl  ({len(rows)} topics)")
    print("aggregate:", agg)
    print("manifest:", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
