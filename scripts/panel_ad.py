"""Cross-model panel runner: A and D only, BRIGHT biology, pool 100 (same task as the primary),
k replicates. For every query records whether A's ranking needed deterministic REPAIR (did not
cover the full pool = format failure) and whether any arm CALL FAILED (exhausted retries). A query
whose A or D call fails has no usable response and is excluded from the paired comparison
(complete-case); it is tallied in call-fail-count but not scored via a fallback ranking. Reports
A/D R@1, D-vs-A fix/break/McNemar, plus repair-count and call-fail-count and the decoding provenance
(temperature settable? resolved version) -- the columns the panel table needs so nothing is hidden.

Any backbone works: claude/gemini via their own clients, open models via the OpenAI-compatible
client (set OAI_BASE_URL + OAI_API_KEY_ENV). Output: results/cross_model/<model>/rep<k>.json.

Usage: python panel_ad.py <model> [--pool 100] [--reps 2] [--n 97]
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from rq2_core import extract_json_object
from bright_backbone import POOLS, rank_from_ids, p_rerank_plain, p_a1, p_a2, p_a3_anchor, mcnemar, ROOT
from model_precheck import build_client, subsample

for _e in (ROOT / ".env", ROOT.parent / ".env"):
    if _e.exists():
        for _l in _e.read_text().splitlines():
            _l = _l.strip()
            if _l and not _l.startswith("#") and "=" in _l:
                _k, _v = _l.split("=", 1); os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


def run_rep(model, pool, items, rep, outdir):
    cl = build_client(model)

    def rcall(prompt, need):
        for i in range(8):
            try:
                o = extract_json_object(cl.generate(prompt))
            except Exception:
                o = None
            if o and ((need == "rank" and o.get("ranked_ids")) or (need == "dec" and o.get("decision")) or need == "any"):
                return o, True
        return {}, False

    per = []; repair = 0; callfail = 0
    for qi, (qid, pl) in enumerate(items, 1):
        q = pl["query"]; a2t = pl["alias_to_text"]; goldset = set(pl["gold_aliases"])
        order = subsample(pl, pool)
        body = [{"id": a, "text": a2t[a]} for a in order]
        oa, aok = rcall(p_rerank_plain(q, body), "rank")
        if not aok:
            # Complete-case: a query with no usable A response is excluded from the paired
            # comparison (counted only in call_fail_count), never scored via a fallback ranking.
            callfail += 1
            print(f"[{model} rep{rep} {qi}/{len(items)}] {qid} A_FAIL -> excluded", flush=True)
            continue
        ids = [str(x.get("id") if isinstance(x, dict) else x) for x in oa.get("ranked_ids", [])]
        a_repaired = len({x for x in ids if x in set(order)}) < len(order)
        if a_repaired:
            repair += 1
        A = rank_from_ids(oa.get("ranked_ids", []), order)
        h1, _ = rcall(p_a1(q), "any"); h2, _ = rcall(p_a2(q), "any")
        od, dok = rcall(p_a3_anchor(q, h1, h2, body), "dec")
        if not dok:
            # Complete-case: exclude queries with no usable D response as well.
            callfail += 1
            print(f"[{model} rep{rep} {qi}/{len(items)}] {qid} D_FAIL -> excluded", flush=True)
            continue
        dec = od.get("decision") or "surface_sufficient"
        prom = [str(x.get("id") if isinstance(x, dict) else x) for x in (od.get("promote_ids") or [])]
        prom = [a for a in prom if a in a2t][:2]
        D = list(A) if (dec == "surface_sufficient" or not prom) else prom + [a for a in A if a not in prom]
        per.append({"id": qid, "A_top1": A[0] in goldset, "D_top1": D[0] in goldset,
                    "A_ok": aok, "D_ok": dok, "A_repaired": a_repaired})
        print(f"[{model} rep{rep} {qi}/{len(items)}] {qid} A_ok={aok} D_ok={dok} repaired={a_repaired}", flush=True)

    n = len(per)
    a_r1 = sum(x["A_top1"] for x in per) / n
    d_r1 = sum(x["D_top1"] for x in per) / n
    fx = sum(1 for x in per if not x["A_top1"] and x["D_top1"])
    bk = sum(1 for x in per if x["A_top1"] and not x["D_top1"])
    rep_out = {"model": model, "rep": rep, "pool": pool, "n": n,
               "temperature_settable_0": cl.temperature == 0, "temperature": cl.temperature,
               "no_think": getattr(cl, "no_think", None),
               "resolved_model": getattr(cl, "last_model", None),
               "A_R@1": round(a_r1, 3), "D_R@1": round(d_r1, 3),
               "fix": fx, "break": bk, "mcnemar_p": round(mcnemar(fx, bk), 4),
               "repair_count": repair, "call_fail_count": callfail,
               "repair_rate": round(repair / n, 3)}
    (outdir / f"rep{rep}.json").write_text(json.dumps({"report": rep_out, "per_query": per}, indent=1))
    print(f"  rep{rep}: A={a_r1:.3f} D={d_r1:.3f} fix={fx} break={bk} p={rep_out['mcnemar_p']} "
          f"repair={repair} callfail={callfail}")
    return rep_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--reps", type=int, default=2); ap.add_argument("--n", type=int, default=len(POOLS))
    args = ap.parse_args()
    items = list(POOLS.items())[: args.n]
    outdir = ROOT / "results" / "cross_model" / args.model.replace("/", "_"); outdir.mkdir(parents=True, exist_ok=True)
    reps = []
    for k in range(1, args.reps + 1):
        if (outdir / f"rep{k}.json").exists():
            reps.append(json.loads((outdir / f"rep{k}.json").read_text())["report"]); continue
        reps.append(run_rep(args.model, args.pool, items, k, outdir))
    print(f"\n=== {args.model}: {len(reps)} reps ===")
    for r in reps:
        print(f"  rep{r['rep']}: A={r['A_R@1']} D={r['D_R@1']} fix={r['fix']} break={r['break']} "
              f"p={r['mcnemar_p']} repair={r['repair_count']} callfail={r['call_fail_count']}")


if __name__ == "__main__":
    main()
