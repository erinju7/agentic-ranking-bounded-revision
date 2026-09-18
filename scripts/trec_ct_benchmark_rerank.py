"""Frozen single-pass generalist on the full TREC CT 2022 eligibility benchmark.

One LLM call per topic; ranks the frozen candidate pool by eligibility fit using
opaque ids. Reports the requested diagnostics. No reviewer/debate/injection.
Kept separate from CORDIS: reads data/trec_ct_2022/, writes results/trec_ct_2022/.
"""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path
from rq2_core import make_client, extract_json_object

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "trec_ct_2022" / "benchmark.jsonl"
OUTDIR = BASE / "results" / "trec_ct_2022" / "generalist"
OUTDIR.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-flash-latest"


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


def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg(order_labels, k):
    d = dcg(order_labels[:k])
    idcg = dcg(sorted(order_labels, reverse=True)[:k])
    return d / idcg if idcg else 0.0


def main():
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    client = make_client(MODEL)
    r1 = 0; mrrs = []; ndcg5 = []; ndcg10 = []
    rank2_all, rank1_all = [], []
    err_q1 = err_q0 = 0
    comp = {2: [], 1: [], 0: []}
    out_records = []
    for r in rows:
        cands = r["candidates"]
        for i, c in enumerate(cands):
            c["alias"] = f"TRIAL-{i:02d}"
        lab = {c["alias"]: c["label"] for c in cands}
        for l in (2, 1, 0):
            comp[l].append(sum(1 for c in cands if c["label"] == l))
        raw = client.generate(prompt(r["patient"], cands))
        ranked = parse_ids(raw, set(lab))
        for c in cands:
            if c["alias"] not in ranked:
                ranked.append(c["alias"])
        order_labels = [lab[a] for a in ranked]
        top1 = lab[ranked[0]]
        r1 += top1 == 2
        if top1 != 2:
            err_q1 += top1 == 1; err_q0 += top1 == 0
        fg = next((i + 1 for i, a in enumerate(ranked) if lab[a] == 2), len(ranked) + 1)
        mrrs.append(1.0 / fg)
        ndcg5.append(ndcg(order_labels, 5)); ndcg10.append(ndcg(order_labels, 10))
        r2 = [i + 1 for i, a in enumerate(ranked) if lab[a] == 2]
        r1r = [i + 1 for i, a in enumerate(ranked) if lab[a] == 1]
        rank2_all += r2; rank1_all += r1r
        out_records.append({"topic_id": r["topic_id"], "ranked_aliases": ranked,
                            "alias_label": lab, "top1_label": top1, "first_gold_rank": fg})
    (OUTDIR / "outputs.jsonl").write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out_records))
    n = len(rows); n_err = n - r1
    summary = {
        "n_topics": n,
        "pool_composition_mean": {f"qrel{l}": round(st.mean(comp[l]), 2) for l in (2, 1, 0)},
        "pool_composition_total": {f"qrel{l}": sum(comp[l]) for l in (2, 1, 0)},
        "1_recall@1_top_is_qrel2": round(r1 / n, 3),
        "2_MRR_first_qrel2": round(st.mean(mrrs), 3),
        "3_nDCG@5": round(st.mean(ndcg5), 3), "3_nDCG@10": round(st.mean(ndcg10), 3),
        "4_mean_rank_qrel2": round(st.mean(rank2_all), 2),
        "5_mean_rank_qrel1": round(st.mean(rank1_all), 2),
        "6_eligible_vs_excluded_separation": round(st.mean(rank1_all) - st.mean(rank2_all), 2),
        "7_top1_errors": {"n_errors": n_err,
                          "chose_qrel1_excluded": err_q1, "chose_qrel0_irrelevant": err_q0,
                          "prop_error_is_qrel1": round(err_q1 / n_err, 3) if n_err else None},
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
