"""MVP generalist reranker on the small TREC CT 2022 benchmark.

Per patient case, rank the candidate trials by eligibility fit; a match = the
patient is ELIGIBLE (qrel label 2). Candidates are shown with opaque ids so the
model ranks by evidence, not by recognising trial identifiers. One LLM call per
topic. Reports Recall@1/@5, MRR over the eligible (gold) trials, plus how the
model separates eligible (label 2) from topically-relevant-but-excluded (label 1).
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
from rq2_core import make_client, extract_json_object

DIR = Path(__file__).resolve().parents[1] / "data" / "trec_ct_mvp"
MODEL = "gemini-flash-latest"


def build_prompt(patient, cands):
    lines = []
    for i, c in enumerate(cands):
        lines.append({
            "id": c["alias"],
            "conditions": c["conditions"],
            "summary": c["summary"],
            "eligibility": c["eligibility"],
        })
    return f"""You match a patient to the clinical trials they are ELIGIBLE for.
Given the patient case and a list of candidate trials (opaque ids), rank ALL ids
from most to least appropriate: a trial ranks high only if the patient both fits
the trial's topic AND satisfies its eligibility criteria (inclusion met, exclusion
not triggered). A trial that is on-topic but for which the patient is excluded must
rank below eligible trials. Return JSON only.

Schema: {{"ranked_ids": ["<id>", ...]}}

Patient case:
{patient}

Candidate trials:
{json.dumps(lines, ensure_ascii=False)}
"""


def parse_ids(raw, valid):
    try:
        obj = extract_json_object(raw)
    except Exception:
        return []
    ids = obj.get("ranked_ids", obj) if isinstance(obj, dict) else obj
    out = []
    if isinstance(ids, list):
        for x in ids:
            v = str(x.get("id") if isinstance(x, dict) else x)
            if v in valid and v not in out:
                out.append(v)
    return out


def main():
    rows = [json.loads(l) for l in (DIR / "mvp.jsonl").read_text().splitlines() if l.strip()]
    client = make_client(MODEL)
    r1 = r5 = 0; mrrs = []; gold_mean_ranks = []; hard_above = []
    for r in rows:
        cands = r["candidates"]
        for i, c in enumerate(cands):
            c["alias"] = f"TRIAL-{i:02d}"
        alias2lab = {c["alias"]: c["label"] for c in cands}
        gold = {c["alias"] for c in cands if c["label"] == 2}
        raw = client.generate(build_prompt(r["patient"], cands))
        ranked = parse_ids(raw, set(alias2lab))
        for c in cands:  # backfill any missing in input order
            if c["alias"] not in ranked:
                ranked.append(c["alias"])
        # rank of first gold
        first_gold = next((i + 1 for i, a in enumerate(ranked) if a in gold), len(ranked) + 1)
        r1 += first_gold <= 1; r5 += first_gold <= 5; mrrs.append(1.0 / first_gold)
        gr = [i + 1 for i, a in enumerate(ranked) if a in gold]
        gold_mean_ranks.append(st.mean(gr) if gr else len(ranked) + 1)
        # separation: mean rank of eligible (2) vs excluded (1)
        r2 = [i + 1 for i, a in enumerate(ranked) if alias2lab[a] == 2]
        r1_ = [i + 1 for i, a in enumerate(ranked) if alias2lab[a] == 1]
        hard_above.append((st.mean(r2) if r2 else 99, st.mean(r1_) if r1_ else 99))
        print(f"topic {r['topic_id']}: first-eligible rank {first_gold}, "
              f"mean eligible rank {st.mean(r2):.1f} vs excluded {st.mean(r1_):.1f}")
    n = len(rows)
    print(f"\n=== MVP generalist on TREC CT 2022 (n={n} topics, 20 candidates each) ===")
    print(f"  Recall@1 (an eligible trial at rank 1): {r1}/{n} = {r1/n:.2f}")
    print(f"  Recall@5:                               {r5}/{n} = {r5/n:.2f}")
    print(f"  MRR (first eligible):                   {st.mean(mrrs):.3f}")
    me = st.mean(m[0] for m in hard_above); mh = st.mean(m[1] for m in hard_above)
    print(f"  mean rank: eligible(2) {me:.1f}  vs  topically-relevant-but-excluded(1) {mh:.1f}")
    print(f"  -> separates eligible above excluded by {mh-me:.1f} ranks on average")


if __name__ == "__main__":
    main()
