"""Score Study 2 listwise runs against the expert's gold labels (study2_listwise_redesign_spec.md).

Metrics (all vs expert gold):
  - graded nDCG@{1,3,6}: relevance grades from the expert gold (match=2, borderline=1, no_match=0,
    linear gain). Scorable proposals = those with ideal DCG>0 (>=1 match or borderline) -> n=5
    (Mito_meeting all-zero, excluded).
  - Recall@1 / MRR (match-only): proposals with >=1 match -> n=4 (AZN_HTS + Mito excluded).
  - per-call grade kappa: variant's own grades vs the expert (quadratic-weighted, 3-level) over 36 pairs.
  - Mito top-1 false-positive: 1-case check that rank-1 has expert-grade 0 (probes D's preserve).
  - pointwise-vs-listwise kappa contrast: listwise per-call kappa vs the existing pairwise-run kappa.
"""
from __future__ import annotations
import os, json, csv, math, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LW = ROOT / "results" / "study2_validation" / "listwise" / "listwise_results.json"
GOLD = Path(os.environ.get("STUDY2_GOLD_CSV", ROOT/"data"/"study2_validation"/"expert_labels.csv"))
OUT = ROOT / "results" / "study2_validation" / "listwise" / "scores_vs_expert.json"

GRADE_VAL = {"match": 2, "borderline": 1, "no_match": 0}
CATS = ["no_match", "borderline", "match"]; IDX = {c: i for i, c in enumerate(CATS)}


def load_gold():
    g = {}
    for r in csv.DictReader(open(GOLD)):
        g[(r["proposal_id"], r["call_id"])] = r["label"].strip()
    return g


def dcg(gains):
    return sum(gm / math.log2(i + 2) for i, gm in enumerate(gains))


def ndcg_at(ranked_calls, prop, gold, k):
    rel = [GRADE_VAL[gold[(prop, c)]] for c in ranked_calls]
    ideal = sorted(rel, reverse=True)
    idcg = dcg(ideal[:k])
    if idcg == 0:
        return None                                   # all-zero proposal: undefined
    return dcg(rel[:k]) / idcg


def weighted_kappa(pairs):
    # pairs: list of (expert_label, variant_grade)
    n = len(pairs); M = [[0] * 3 for _ in range(3)]
    for a, b in pairs:
        M[IDX[a]][IDX[b]] += 1
    ra = [sum(M[i]) for i in range(3)]; ca = [sum(M[i][j] for i in range(3)) for j in range(3)]
    W = [[((i - j) / 2) ** 2 for j in range(3)] for i in range(3)]
    O = sum(W[i][j] * M[i][j] for i in range(3) for j in range(3)) / n
    E = sum(W[i][j] * ra[i] * ca[j] / n / n for i in range(3) for j in range(3))
    return 1 - O / E if E else 0.0


def score_variant(results, key, gold):
    ndcgs = {1: [], 3: [], 6: []}
    r1_hits, mrr_vals = [], []
    kappa_pairs = []
    mito_fp = None
    for r in results:
        prop = r["proposal"]; v = r[key]
        ranked = v["ranked"]; grades = v["grades"]
        # nDCG (skip all-zero proposals)
        for k in (1, 3, 6):
            nd = ndcg_at(ranked, prop, gold, k)
            if nd is not None:
                ndcgs[k].append(nd)
        # R@1 / MRR: match-only, proposals with >=1 match
        golds = [c for c in ranked if gold[(prop, c)] == "match"]
        if golds:
            r1_hits.append(1 if gold[(prop, ranked[0])] == "match" else 0)
            first = next(i for i, c in enumerate(ranked, 1) if gold[(prop, c)] == "match")
            mrr_vals.append(1 / first)
        # per-call grade kappa
        for c in ranked:
            kappa_pairs.append((gold[(prop, c)], grades[c]))
        # Mito top-1 false positive (all-zero proposal)
        if all(gold[(prop, c)] == "no_match" for c in ranked):
            mito_fp = {"proposal": prop, "rank1_call": ranked[0],
                       "rank1_gold": gold[(prop, ranked[0])],
                       "false_positive": gold[(prop, ranked[0])] != "no_match"}
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    return {
        "nDCG@1": mean(ndcgs[1]), "nDCG@3": mean(ndcgs[3]), "nDCG@6": mean(ndcgs[6]),
        "nDCG_n": len(ndcgs[6]),
        "Recall@1": mean(r1_hits), "MRR": mean(mrr_vals), "rank_n": len(r1_hits),
        "per_call_weighted_kappa": round(weighted_kappa(kappa_pairs), 3),
        "mito_top1_fp": mito_fp,
    }


def pairwise_kappa(gold):
    """Existing pairwise run per-call kappa (binary match/no_match decision vs 3-level gold)."""
    p = ROOT / "results" / "study2_validation" / "pairs.json"
    if not p.exists():
        return {}
    pairs = json.loads(p.read_text())
    out = {}
    for v in ["A", "B", "D"]:
        kp = [(gold[(x["proposal"], x["call"])], x[v]["decision"]) for x in pairs]
        out[v] = round(weighted_kappa(kp), 3)
    return out


def main():
    results = json.loads(LW.read_text())
    gold = load_gold()
    variants = ["A", "B", "C", "D_cap1", "D_cap2"]
    scores = {v: score_variant(results, v, gold) for v in variants}
    pw = pairwise_kappa(gold)

    print(f"Study 2 listwise vs the expert (n_proposals={len(results)}; nDCG n=5, rank n=4)\n")
    hdr = f"{'variant':9s}{'nDCG@1':>8s}{'nDCG@3':>8s}{'nDCG@6':>8s}{'R@1':>7s}{'MRR':>7s}{'κ(call)':>9s}"
    print(hdr)
    for v in variants:
        s = scores[v]
        def f(x): return f"{x:.3f}" if isinstance(x, float) else "  -  "
        print(f"{v:9s}{f(s['nDCG@1']):>8s}{f(s['nDCG@3']):>8s}{f(s['nDCG@6']):>8s}"
              f"{f(s['Recall@1']):>7s}{f(s['MRR']):>7s}{f(s['per_call_weighted_kappa']):>9s}")
    print("\nMito top-1 false-positive check:")
    for v in variants:
        m = scores[v]["mito_top1_fp"]
        if m:
            print(f"  {v}: rank1={m['rank1_call']} gold={m['rank1_gold']} FP={m['false_positive']}")
    print("\npointwise vs listwise per-call weighted-kappa (vs the expert):")
    for v in ["A", "B", "D"]:
        lw = scores["D_cap1"]["per_call_weighted_kappa"] if v == "D" else scores[v]["per_call_weighted_kappa"]
        print(f"  {v}: pairwise={pw.get(v)}  listwise={lw}")

    OUT.write_text(json.dumps({"scores": scores, "pairwise_kappa": pw,
                               "notes": "graded nDCG grades from expert gold; LLM grades feed kappa only; "
                                        "nDCG n=5 (Mito excluded), R@1/MRR n=4 (AZN+Mito excluded)"}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
