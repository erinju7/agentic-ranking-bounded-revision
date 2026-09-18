"""Build a SMALL frozen TREC Clinical Trials 2022 matching benchmark (MVP).

Task: patient case (query) -> rank candidate clinical trials; a trial is a correct
match iff the patient is ELIGIBLE (qrel label 2). Label 1 = topically relevant but
EXCLUDED by eligibility (hard distractor); label 0 = irrelevant.

Pools are kept small and eligibility-focused: per topic we take a few eligible
(label 2 = gold), several topically-relevant-but-excluded (label 1 = hard), and a
few irrelevant (label 0), so the deciding factor is eligibility reasoning. Trial
text is fetched on demand from the public ClinicalTrials.gov API v2 (no 1.7 GB
corpus download); results are cached.
"""
from __future__ import annotations
import json, re, random, time, urllib.request, urllib.parse, collections
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "data" / "trec_ct_mvp"
CACHE = DIR / "trials_cache.json"
OUT = DIR / "mvp.jsonl"
N_TOPICS = 10
N_GOLD, N_HARD, N_EASY = 5, 10, 5      # label 2 / label 1 / label 0 per pool
SEED = 42
API = "https://clinicaltrials.gov/api/v2/studies/{}?fields=protocolSection.identificationModule.briefTitle,protocolSection.conditionsModule.conditions,protocolSection.descriptionModule.briefSummary,protocolSection.eligibilityModule.eligibilityCriteria"


def clean(t): return " ".join(str(t or "").split())


def parse_topics():
    xml = (DIR / "topics2022.xml").read_text()
    return {n: clean(t) for n, t in re.findall(r'<topic number="(\d+)">(.*?)</topic>', xml, re.S)}


def parse_qrels():
    q = collections.defaultdict(dict)
    for line in (DIR / "qrels2022.txt").read_text().splitlines():
        p = line.split()
        if len(p) == 4:
            q[p[0]][p[2]] = int(p[3])
    return q


def fetch_trial(nct, cache):
    if nct in cache:
        return cache[nct]
    try:
        with urllib.request.urlopen(API.format(nct), timeout=25) as r:
            d = json.load(r)
        ps = d.get("protocolSection", {})
        rec = {
            "title": clean(ps.get("identificationModule", {}).get("briefTitle")),
            "conditions": ps.get("conditionsModule", {}).get("conditions", []),
            "summary": clean(ps.get("descriptionModule", {}).get("briefSummary"))[:600],
            "eligibility": clean(ps.get("eligibilityModule", {}).get("eligibilityCriteria"))[:900],
        }
    except Exception as e:
        rec = None
    cache[nct] = rec
    return rec


def main():
    rng = random.Random(SEED)
    topics, qrels = parse_topics(), parse_qrels()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    chosen = [t for t in sorted(qrels, key=int)
              if sum(v == 2 for v in qrels[t].values()) >= N_GOLD
              and sum(v == 1 for v in qrels[t].values()) >= N_HARD][:N_TOPICS]
    rows = []
    for t in chosen:
        by = {2: [], 1: [], 0: []}
        for nct, lab in qrels[t].items():
            by.setdefault(lab, []).append(nct)
        for lab in by:
            rng.shuffle(by[lab])
        picks = by[2][:N_GOLD] + by[1][:N_HARD] + by[0][:N_EASY]
        cands = []
        for nct in picks:
            rec = fetch_trial(nct, cache)
            if not rec or not (rec["eligibility"] or rec["summary"]):
                continue
            cands.append({"nct": nct, "label": qrels[t][nct], **rec})
            time.sleep(0.05)
        rng.shuffle(cands)
        gold = [c["nct"] for c in cands if c["label"] == 2]
        if gold and len(cands) >= 8:
            rows.append({"topic_id": t, "patient": topics[t], "candidates": cands, "gold": gold})
        CACHE.write_text(json.dumps(cache))
        print(f"topic {t}: {len(cands)} candidates ({len(gold)} eligible)")
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT}  ({len(rows)} topics)")


if __name__ == "__main__":
    main()
