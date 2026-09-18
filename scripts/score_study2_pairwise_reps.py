"""Kappa-vs-James distribution across k=3 temp=0 replicates for the pairwise Study-2 arms
(E decomposed, IND, DEP). Mirrors the BRIGHT replicate treatment: reports per-rep kappa + mean+-sd,
so the low pairwise-cluster agreement is shown as a distribution, not a single draw (E already
wobbled 0.332->0.298 across two runs). Reads decomposed_<M>_rep<k>/ and study2a_dependency_<M>_rep<k>/.
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
from score_study2_unified import kappa_over, pairwise_triples, load_gold, M, JV

REPS = list(range(1, 11))


def main():
    gold = load_gold()
    arms = {"E": [], "IND": [], "DEP": []}
    prov = {}
    for k in REPS:
        ed = JV / f"decomposed_{M}_rep{k}" / "pairs_E.json"
        dd = JV / f"study2a_dependency_{M}_rep{k}" / "pairs.json"
        if ed.exists():
            arms["E"].append(kappa_over(gold, pairwise_triples(ed, "E"))[0])
        for key in ["IND", "DEP"]:
            if dd.exists():
                arms[key].append(kappa_over(gold, pairwise_triples(dd, key))[0])
        for tag, s in [("E", JV / f"decomposed_{M}_rep{k}" / "summary.json"),
                       ("INDDEP", JV / f"study2a_dependency_{M}_rep{k}" / "summary.json")]:
            if s.exists():
                d = json.loads(s.read_text())
                prov[f"{tag}_rep{k}"] = {x: d.get(x) for x in ["resolved_model", "temperature", "run_utc"]}

    print(f"=== Study 2 pairwise arms vs James — k=3 temp=0 replicates ({M}) ===")
    print(f"{'arm':6s}{'kappa per rep':>28s}{'mean':>8s}{'sd':>7s}")
    out = {}
    for a in ["E", "IND", "DEP"]:
        xs = arms[a]
        if not xs:
            print(f"{a:6s}{'(no data yet)':>28s}"); continue
        mu = st.mean(xs); sd = st.pstdev(xs) if len(xs) > 1 else 0.0
        out[a] = {"values": xs, "mean": round(mu, 3), "sd": round(sd, 3), "n": len(xs)}
        print(f"{a:6s}{str([round(x,3) for x in xs]):>28s}{mu:>8.3f}{sd:>7.3f}")
    print("\nprovenance:")
    for k, v in sorted(prov.items()):
        print(f"  {k}: {v}")

    res = JV / f"pairwise_reps_vs_james_{M}.json"
    res.write_text(json.dumps({"backbone": M, "temperature": 0, "reps": REPS,
                               "kappa_dist": out, "provenance": prov}, indent=1))
    print(f"\nwrote {res}")


if __name__ == "__main__":
    main()
