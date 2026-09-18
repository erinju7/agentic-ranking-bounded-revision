"""Clean, parity-controlled re-run of ALL FOUR arms (A/B/C/D) on gemini-2.5-flash-lite over the
frozen biology 100-doc pool. Motivation: the original A/D panel and the B/C add-on used DIFFERENT
callers with DIFFERENT failure handling, so their 504/parse-failure ("fallback") counts are not
comparable -- and on gemini-lite B/C carried 16/11 fallbacks while A/D logged none. A fallback is
recorded as no-change, so it only DEPRESSES the arm that carries it: unequal fallback counts across
arms confound the D>B>C ordering on this row.

This run fixes that: one robust caller for every arm, with aggressive retry to drive fallbacks
toward zero, and PER-ARM PER-QUERY logging of whether the decisive call actually succeeded. It then
reports (1) each arm's residual fallback count and (2) the D/B/C-vs-A comparison on the COMPLETE-CASE
subset -- queries where A, B, C AND D all produced a real ranking -- so every arm shares one
denominator and no arm is penalised by transient API failures the others escaped.

Backbone pinned: gemini-2.5-flash-lite. Verbatim prompts reused. Writes results/bright_backbones/
gemini-2.5-flash-lite/clean_abcd.json (does NOT overwrite the original summary/per_query files).
"""
from __future__ import annotations
import os, sys, json, time, math, statistics as st
from pathlib import Path
from rq2_core import extract_json_object
from bright_backbone import (client_for, POOLS, mcnemar, ndcg, rank_from_ids,
                             p_rerank_plain, p_a1, p_a2, p_a3_anchor, ROOT)
from bright_finalize_measure import p_concept_B, p_rerank_B, p_a3_full

for _env in (ROOT / ".env", ROOT.parent / ".env"):
    if _env.exists():
        for _l in _env.read_text().splitlines():
            _l = _l.strip()
            if _l and not _l.startswith("#") and "=" in _l:
                _k, _v = _l.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemini-2.5-flash-lite"
OUT = ROOT / "results" / "bright_backbones" / MODEL
OUT.mkdir(parents=True, exist_ok=True)
TRIES = 10                                   # outer retries on top of the client's own internal retries


def rcall(cl, prompt, need):
    """Robust call. need in {'rank','dec','aux'}. Returns (obj, ok). ok=False only after TRIES
    genuine failures (exception, or a response with no usable field)."""
    for i in range(TRIES):
        o = None
        try:
            o = extract_json_object(cl.generate(prompt))
        except Exception:
            o = None
        if o:
            if need == "rank" and o.get("ranked_ids"):
                return o, True
            if need == "dec" and o.get("decision"):
                return o, True
            if need == "aux":
                return o, True
        time.sleep(min(4 * (i + 1), 20))
    return ({}, False)


def main():
    cl = client_for(MODEL)
    per = []
    fail = {"A": 0, "B": 0, "C": 0, "D": 0}
    for qi, (qid, pl) in enumerate(POOLS.items(), 1):
        q = pl["query"]; a2t = pl["alias_to_text"]; goldset = set(pl["gold_aliases"]); order = list(a2t)
        body = [{"id": a, "text": a2t[a]} for a in order]

        # --- shared hypotheses (aux; C and D consume the SAME pair so they differ only in coordinator)
        h1, _ = rcall(cl, p_a1(q), "aux")
        h2, _ = rcall(cl, p_a2(q), "aux")

        # --- A: single-pass
        oa, aok = rcall(cl, p_rerank_plain(q, body), "rank")
        A = rank_from_ids(oa.get("ranked_ids", []), order) if aok else list(order)
        if not aok: fail["A"] += 1

        # --- B: concept -> concept-conditioned rerank (concept is aux; rerank is decisive)
        cb, _ = rcall(cl, p_concept_B(q), "aux")
        ob, bok = rcall(cl, p_rerank_B(q, body, cb), "rank")
        B = rank_from_ids(ob.get("ranked_ids", []), order) if bok else list(A)   # fail -> preserve A
        if not bok: fail["B"] += 1

        # --- C: competing hypotheses -> coordinator rebuilds full ranking
        oc, cok = rcall(cl, p_a3_full(q, h1, h2, body), "rank")
        C = rank_from_ids(oc.get("ranked_ids", []), order) if cok else list(A)
        if not cok: fail["C"] += 1

        # --- D: anchor-and-edit; coordinator decision drives a mechanical promote of <=2 to front of A
        od, dok = rcall(cl, p_a3_anchor(q, h1, h2, body), "dec")
        if dok:
            dec = od.get("decision") or "surface_sufficient"
            prom = [str(x.get("id") if isinstance(x, dict) else x) for x in (od.get("promote_ids") or [])]
            prom = [a for a in prom if a in a2t][:2]
            D = list(A) if (dec == "surface_sufficient" or not prom) else prom + [a for a in A if a not in prom]
        else:
            fail["D"] += 1; dec = "FAIL"; D = list(A)

        def top1(r): return r[0] in goldset
        per.append({"id": qid, "ok": {"A": aok, "B": bok, "C": cok, "D": dok}, "D_decision": dec,
                    "top1": {k: top1(v) for k, v in {"A": A, "B": B, "C": C, "D": D}.items()}})
        print(f"[{qi}/{len(POOLS)}] {qid} ok(A{int(aok)}B{int(bok)}C{int(cok)}D{int(dok)}) dec={dec}", flush=True)

    # ---------- complete-case subset (all four arms succeeded) ----------
    cc = [x for x in per if all(x["ok"].values())]
    n_all = len(per); n_cc = len(cc)

    def arm_stats(rows, arm):
        r1 = sum(1 for x in rows if x["top1"][arm]) / len(rows)
        fx = sum(1 for x in rows if not x["top1"]["A"] and x["top1"][arm])
        bk = sum(1 for x in rows if x["top1"]["A"] and not x["top1"][arm])
        return {"R@1": round(r1, 3), "fix": fx, "break": bk, "mcnemar_p": round(mcnemar(fx, bk), 4)}

    report = {
        "model": MODEL, "n_all": n_all, "n_complete_case": n_cc,
        "fallbacks_per_arm": fail,
        "full_set": {a: arm_stats(per, a) for a in ["A", "B", "C", "D"]},
        "complete_case": {a: arm_stats(cc, a) for a in ["A", "B", "C", "D"]},
        "note": "A fallback => that arm defaults to A's ranking (B/C/D) or to shuffle order (A). "
                "complete_case = queries where all four decisive calls succeeded (equal denominator, "
                "no fallback confound). This is the parity-clean basis for the D>B>C ordering on this row.",
    }
    (OUT / "clean_abcd.json").write_text(json.dumps({"report": report, "per_query": per}, indent=1))
    print("\n=== gemini-2.5-flash-lite CLEAN A/B/C/D ===")
    print(f"fallbacks per arm: {fail}   (n_all={n_all}, complete_case n={n_cc})")
    for basis in ["full_set", "complete_case"]:
        print(f"-- {basis} --")
        for a in ["A", "B", "C", "D"]:
            s = report[basis][a]
            extra = "" if a == "A" else f"  fix={s['fix']} break={s['break']} McNemar p={s['mcnemar_p']}"
            print(f"   {a}: R@1={s['R@1']}{extra}")
    print(f"\nwrote {OUT/'clean_abcd.json'}")


if __name__ == "__main__":
    main()
