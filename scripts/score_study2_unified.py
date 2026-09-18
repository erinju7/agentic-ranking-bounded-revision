"""Unified Study-2 comparison vs James expert gold, across ALL architectures on ONE backbone
(claude-haiku-4-5-20251001, temp=0). The common currency is per-(proposal,call) weighted kappa
(3-level: no_match/borderline/match) over the same 36 cells -- the only metric every architecture
produces, so listwise rankers (A/B/C/R), the anchor-edit (D), the decomposed matcher (E), and the
IND/DEP alignment fork are directly comparable. Listwise arms additionally get nDCG@6 / R@1 (ranking
metrics D/rankers have but pairwise matchers do not).

NOTE: in the listwise pipeline D reuses A's per-call grades (positional edit, not a re-grade), so
D's kappa == A's by construction; D's signal is in the ranking metrics, not kappa.
"""
from __future__ import annotations
import json
from pathlib import Path
from score_study2_listwise import weighted_kappa, load_gold, score_variant, CATS

ROOT = Path(__file__).resolve().parents[1]
M = "claude-haiku-4-5-20251001"
JV = ROOT / "results" / "james_validation"


def kappa_over(gold, triples):
    kp = [(gold[(p, c)], d if d in CATS else "no_match") for p, c, d in triples if (p, c) in gold]
    return round(weighted_kappa(kp), 3), len(kp)


def pairwise_triples(path, key):
    return [(x["proposal"], x["call"], str(x[key]["decision"]).lower()) for x in json.loads(path.read_text())]


def listwise_triples(lw, var):
    out = []
    for r in lw:
        for call, g in r[var]["grades"].items():
            out.append((r["proposal"], r["call"] if False else call, g))
    return out


def provenance(path, keys):
    try:
        d = json.loads(path.read_text())
        return {k: d.get(k) for k in keys}
    except Exception:
        return {}


def main():
    gold = load_gold()
    lw = json.loads((JV / "listwise" / M / "listwise_results.json").read_text())

    rows = []
    # listwise arms (own grades -> own kappa; plus ranking metrics)
    for var, label in [("A", "A"), ("B", "B"), ("C", "C"), ("D_cap1", "D (anchor)")]:
        k, n = kappa_over(gold, listwise_triples(lw, var))
        sv = score_variant(lw, var, gold)
        rows.append({"arch": label, "family": "listwise", "kappa": k, "n": n,
                     "nDCG@6": sv["nDCG@6"], "R@1": sv["Recall@1"]})
    # pairwise arms (own decision -> own kappa; no ranking metrics)
    k, n = kappa_over(gold, pairwise_triples(JV / f"decomposed_{M}" / "pairs_E.json", "E"))
    rows.append({"arch": "E (decomposed)", "family": "pairwise", "kappa": k, "n": n, "nDCG@6": None, "R@1": None})
    for key in ["IND", "DEP"]:
        k, n = kappa_over(gold, pairwise_triples(JV / f"study2a_dependency_{M}" / "pairs.json", key))
        rows.append({"arch": key, "family": "pairwise", "kappa": k, "n": n, "nDCG@6": None, "R@1": None})

    print(f"=== Study 2 vs James gold — unified, backbone {M}, temp=0 ===")
    print(f"{'arch':16s}{'family':10s}{'kappa':>8s}{'n':>5s}{'nDCG@6':>9s}{'R@1':>8s}")
    for r in rows:
        nd = f"{r['nDCG@6']:.3f}" if r["nDCG@6"] is not None else "  -  "
        r1 = f"{r['R@1']:.3f}" if r["R@1"] is not None else "  -  "
        print(f"{r['arch']:16s}{r['family']:10s}{r['kappa']:>8.3f}{r['n']:>5d}{nd:>9s}{r1:>8s}")

    prov = {"listwise": provenance(JV / "listwise" / M / "summary.json", ["resolved_model", "temperature", "run_utc"]),
            "E": provenance(JV / f"decomposed_{M}" / "summary.json", ["resolved_model", "temperature", "run_utc"]),
            "IND_DEP": provenance(JV / f"study2a_dependency_{M}" / "summary.json", ["resolved_model", "temperature", "run_utc"])}
    print("\nprovenance (must be one model/temp/window):")
    for k, v in prov.items():
        print(f"  {k}: {v}")

    out = JV / f"unified_vs_james_{M}.json"
    out.write_text(json.dumps({"backbone": M, "temperature": 0, "rows": rows, "provenance": prov,
                               "note": "kappa = per-(proposal,call) quadratic-weighted 3-level agreement vs James; "
                                       "D reuses A grades so kappa(D)==kappa(A) by construction (D signal is in ranking)."},
                              indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
