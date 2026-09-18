"""15-topic pilot of the frozen single-pass generalist on the hard TREC CT 2022
benchmark. Deterministic hard-subset selection (most qrel=1 hard negatives first,
ties by topic id) -- NOT by model performance. Generalist only; no reviewer/debate/
injection. Behavioural gating: suitable iff meaningful errors AND errors are
predominantly qrel=1 (excluded near-miss) rather than qrel=0 (irrelevant).
"""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path
from rq2_core import make_client, extract_json_object

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "trec_ct_2022" / "benchmark.jsonl"
OUTDIR = BASE / "results" / "trec_ct_2022" / "generalist_pilot15"
OUTDIR.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-flash-latest"
N_PILOT = 15


def prompt(patient, cands):
    lines = [{"id": c["alias"], "conditions": c["conditions"],
              "summary": c["summary"], "eligibility": c["eligibility"]} for c in cands]
    return f"""You match a patient to the clinical trials they are ELIGIBLE for.
Given the patient case and candidate trials (opaque ids), rank ALL ids from most to
least appropriate. A trial ranks high only if the patient fits its topic AND meets
its eligibility (inclusion satisfied, exclusion not triggered). A trial that is
on-topic but for which the patient is EXCLUDED must rank below eligible trials.
Return JSON only.

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


def dcg(rels): return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
def ndcg(labels, k):
    d = dcg(labels[:k]); i = dcg(sorted(labels, reverse=True)[:k]); return d / i if i else 0.0
def n1(r): return sum(c["label"] == 1 for c in r["candidates"])


def main():
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    pilot = sorted(rows, key=lambda r: (-n1(r), int(r["topic_id"])))[:N_PILOT]
    print("selected topics (id: n2/n1/n0):",
          ", ".join(f"{r['topic_id']}:{sum(c['label']==2 for c in r['candidates'])}/"
                    f"{n1(r)}/{sum(c['label']==0 for c in r['candidates'])}" for r in pilot))
    client = make_client(MODEL)
    r1 = err_q1 = err_q0 = 0; mrrs = []; ndcg5 = []; rank2 = []; rank1 = []; per = []
    for r in pilot:
        cands = r["candidates"]
        for i, c in enumerate(cands):
            c["alias"] = f"TRIAL-{i:02d}"
        lab = {c["alias"]: c["label"] for c in cands}
        raw = client.generate(prompt(r["patient"], cands))
        ranked = parse_ids(raw, set(lab))
        for c in cands:
            if c["alias"] not in ranked:
                ranked.append(c["alias"])
        labels = [lab[a] for a in ranked]
        top1 = lab[ranked[0]]
        r1 += top1 == 2
        if top1 != 2:
            err_q1 += top1 == 1; err_q0 += top1 == 0
        fg = next((i + 1 for i, a in enumerate(ranked) if lab[a] == 2), len(ranked) + 1)
        mrrs.append(1 / fg); ndcg5.append(ndcg(labels, 5))
        mr2 = st.mean([i + 1 for i, a in enumerate(ranked) if lab[a] == 2])
        mr1 = st.mean([i + 1 for i, a in enumerate(ranked) if lab[a] == 1])
        rank2.append(mr2); rank1.append(mr1)
        per.append({"topic": r["topic_id"], "top1_label": top1, "first_gold_rank": fg,
                    "mean_rank_q2": round(mr2, 1), "mean_rank_q1": round(mr1, 1), "nDCG@5": round(ndcg5[-1], 3)})
    (OUTDIR / "pilot_outputs.json").write_text(json.dumps(per, indent=2))
    n = len(pilot); nerr = n - r1
    print("\n=== per-topic ===")
    print(f"{'topic':>6} {'top1':>5} {'1stGold':>8} {'mrkQ2':>6} {'mrkQ1':>6} {'nDCG@5':>7}")
    for p in per:
        flag = "" if p["top1_label"] == 2 else ("  <-err:qrel1" if p["top1_label"] == 1 else "  <-err:qrel0")
        print(f"{p['topic']:>6} {p['top1_label']:>5} {p['first_gold_rank']:>8} {p['mean_rank_q2']:>6} {p['mean_rank_q1']:>6} {p['nDCG@5']:>7}{flag}")
    summary = {
        "n_topics": n,
        "R@1_qrel2": round(r1 / n, 3),
        "MRR_first_qrel2": round(st.mean(mrrs), 3),
        "nDCG@5": round(st.mean(ndcg5), 3),
        "mean_rank_qrel2": round(st.mean(rank2), 2),
        "mean_rank_qrel1": round(st.mean(rank1), 2),
        "separation_q1_minus_q2": round(st.mean(rank1) - st.mean(rank2), 2),
        "top1_errors": {"total": nerr, "qrel1_excluded": err_q1, "qrel0_irrelevant": err_q0,
                        "prop_qrel1": round(err_q1 / nerr, 3) if nerr else None},
    }
    (OUTDIR / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    # behavioural gating
    print("\n=== gating verdict ===")
    if nerr <= 1:
        print(f"NEAR-CEILING: only {nerr} top-1 error in {n} topics -> pool still too EASY. "
              f"Stop; do not modify silently.")
    elif nerr and (err_q1 / nerr) >= 0.6:
        print(f"SUITABLE: {nerr} errors, {err_q1}/{nerr} are qrel=1 eligibility near-misses "
              f"(not qrel=0). Failures are genuine eligibility reasoning -> proceed to full 50.")
    else:
        print(f"MIXED: {nerr} errors but only {err_q1} qrel=1 vs {err_q0} qrel=0 -> "
              f"errors not predominantly eligibility near-misses; report, do not modify silently.")


if __name__ == "__main__":
    main()
