"""Parity-clean A/B/C/D on ANY BRIGHT domain for ANY backbone, with per-arm fallback logging,
complete-case analysis, and RESOLVED MODEL VERSION capture. This is the promotion vehicle for the
new primary backbone (claude-haiku-4-5-20251001): one robust caller for every arm, so fallback
counts are comparable across arms, and the exact served model version is recorded (response.model)
so the run is version-identifiable -- the property the old gemini-flash-latest primary lacked.

Usage: python bright_clean_split.py <model> <domain>
  <model>  e.g. claude-haiku-4-5-20251001   (pass the DATED snapshot, not the alias)
  <domain> one of: biology earth_science psychology sustainable_living
Writes results/bright_backbones/<model>/<domain>/clean_abcd.json  (+ resolved_model in the report).
"""
from __future__ import annotations
import os, sys, json, time
from datetime import datetime, timezone
from pathlib import Path
from rq2_core import extract_json_object
from bright_backbone import (client_for, mcnemar, rank_from_ids,
                             p_rerank_plain, p_a1, p_a2, p_a3_anchor, ROOT)
from bright_finalize_measure import p_concept_B, p_rerank_B, p_a3_full

for _env in (ROOT / ".env", ROOT.parent / ".env"):
    if _env.exists():
        for _l in _env.read_text().splitlines():
            _l = _l.strip()
            if _l and not _l.startswith("#") and "=" in _l:
                _k, _v = _l.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

MODEL = sys.argv[1]
DOMAIN = sys.argv[2]
REP = sys.argv[3] if len(sys.argv) > 3 else None            # optional replicate tag (temp=1.0 -> non-det)
POOLS = json.loads((ROOT / "data" / "bright_hardpool" / DOMAIN / "pools.json").read_text())
OUT = ROOT / "results" / "bright_backbones" / MODEL / DOMAIN
if REP:
    OUT = OUT / f"rep_{REP}"
OUT.mkdir(parents=True, exist_ok=True)
TRIES = 10


def rcall(cl, prompt, need):
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
        time.sleep(min(3 * (i + 1), 15))
    return ({}, False)


def main():
    cl = client_for(MODEL)
    per = []
    fail = {"A": 0, "B": 0, "C": 0, "D": 0}
    for qi, (qid, pl) in enumerate(POOLS.items(), 1):
        q = pl["query"]; a2t = pl["alias_to_text"]; goldset = set(pl["gold_aliases"]); order = list(a2t)
        body = [{"id": a, "text": a2t[a]} for a in order]
        h1, _ = rcall(cl, p_a1(q), "aux")
        h2, _ = rcall(cl, p_a2(q), "aux")

        oa, aok = rcall(cl, p_rerank_plain(q, body), "rank")
        A = rank_from_ids(oa.get("ranked_ids", []), order) if aok else list(order)
        if not aok: fail["A"] += 1

        cb, _ = rcall(cl, p_concept_B(q), "aux")
        ob, bok = rcall(cl, p_rerank_B(q, body, cb), "rank")
        B = rank_from_ids(ob.get("ranked_ids", []), order) if bok else list(A)
        if not bok: fail["B"] += 1

        oc, cok = rcall(cl, p_a3_full(q, h1, h2, body), "rank")
        C = rank_from_ids(oc.get("ranked_ids", []), order) if cok else list(A)
        if not cok: fail["C"] += 1

        od, dok = rcall(cl, p_a3_anchor(q, h1, h2, body), "dec")
        if dok:
            dec = od.get("decision") or "surface_sufficient"
            prom = [str(x.get("id") if isinstance(x, dict) else x) for x in (od.get("promote_ids") or [])]
            prom = [a for a in prom if a in a2t][:2]
            D = list(A) if (dec == "surface_sufficient" or not prom) else prom + [a for a in A if a not in prom]
        else:
            fail["D"] += 1; dec = "FAIL"; D = list(A)

        def top1(r): return r[0] in goldset

        def firstgold(r): return next((i + 1 for i, x in enumerate(r) if x in goldset), len(r) + 1)
        per.append({"id": qid, "ok": {"A": aok, "B": bok, "C": cok, "D": dok}, "D_decision": dec,
                    "top1": {k: top1(v) for k, v in {"A": A, "B": B, "C": C, "D": D}.items()},
                    "first_gold": {k: firstgold(v) for k, v in {"A": A, "B": B, "C": C, "D": D}.items()}})
        print(f"[{DOMAIN} {qi}/{len(POOLS)}] {qid} ok(A{int(aok)}B{int(bok)}C{int(cok)}D{int(dok)}) dec={dec}", flush=True)

    cc = [x for x in per if all(x["ok"].values())]

    def arm_stats(rows, arm):
        r1 = sum(1 for x in rows if x["top1"][arm]) / len(rows)
        fx = sum(1 for x in rows if not x["top1"]["A"] and x["top1"][arm])
        bk = sum(1 for x in rows if x["top1"]["A"] and not x["top1"][arm])
        return {"R@1": round(r1, 3), "fix": fx, "break": bk, "mcnemar_p": round(mcnemar(fx, bk), 4)}

    report = {
        "model": MODEL, "resolved_model": getattr(cl, "last_model", None), "domain": DOMAIN,
        "temperature": getattr(cl, "temperature", None), "run_utc": datetime.now(timezone.utc).isoformat(),
        "n_all": len(per), "n_complete_case": len(cc), "fallbacks_per_arm": fail,
        "full_set": {a: arm_stats(per, a) for a in ["A", "B", "C", "D"]},
        "complete_case": {a: arm_stats(cc, a) for a in ["A", "B", "C", "D"]},
    }
    (OUT / "clean_abcd.json").write_text(json.dumps({"report": report, "per_query": per}, indent=1))
    print(f"\n=== {MODEL} / {DOMAIN}  resolved={report['resolved_model']} ===")
    print(f"fallbacks {fail}  (n_all={report['n_all']}, complete_case={report['n_complete_case']})")
    for a in ["A", "B", "C", "D"]:
        s = report["complete_case"][a]
        ex = "" if a == "A" else f"  fix={s['fix']} break={s['break']} p={s['mcnemar_p']}"
        print(f"  {a}: R@1={s['R@1']}{ex}")
    print(f"wrote {OUT/'clean_abcd.json'}")


if __name__ == "__main__":
    main()
