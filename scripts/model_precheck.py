"""Pre-check a candidate cross-model backbone on BRIGHT biology BEFORE committing to a full run.
Runs a small sample (default 10 queries) of A and D at a configurable pool size and reports the
three gates from the panel design:
  1. temperature settable to 0?  (client.temperature after the run: 0 = yes, None = model rejected it)
  2. repair rate  -- fraction of A rankings that did NOT cover the full pool (format failure); the
     >20% reject gate lives here.
  3. version lock -- the resolved model string the provider returned (vs the alias requested).
Plus quick A/D R@1 on the sample. Writes results/cross_model/precheck_<model>.json. A model that
fails a gate should NOT be run at full scale -- and "excluded: repair rate X%" is itself reportable.

Usage: python model_precheck.py <model> [--pool 100] [--n 10]
  Open models need OAI_BASE_URL + OAI_API_KEY_ENV set (see oai_client.py). claude/gemini use their
  own clients automatically.
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from rq2_core import extract_json_object
from bright_backbone import POOLS, rank_from_ids, p_rerank_plain, p_a1, p_a2, p_a3_anchor, ROOT

for _e in (ROOT / ".env", ROOT.parent / ".env"):
    if _e.exists():
        for _l in _e.read_text().splitlines():
            _l = _l.strip()
            if _l and not _l.startswith("#") and "=" in _l:
                _k, _v = _l.split("=", 1); os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


def build_client(model):
    if model.startswith(("claude", "gemini", "gemma")):   # gemma is served on the Google API
        from bright_backbone import client_for
        return client_for(model)
    from oai_client import OAIClient
    return OAIClient(model)


def subsample(pl, pool):
    """Keep all gold + top negatives to fill `pool`, preserving order."""
    order = list(pl["alias_to_text"]); gold = set(pl["gold_aliases"])
    if pool >= len(order):
        return order
    negs = [a for a in order if a not in gold]
    keep = list(gold) + negs[: max(0, pool - len(gold))]
    return [a for a in order if a in set(keep)]        # preserve original order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--pool", type=int, default=100); ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    cl = build_client(args.model)

    def rcall(prompt, need):
        for i in range(6):
            try:
                o = extract_json_object(cl.generate(prompt))
            except Exception:
                o = None
            if o and ((need == "rank" and o.get("ranked_ids")) or (need == "dec" and o.get("decision"))):
                return o, True
        return {}, False

    n_repair = n_callfail = a_hit = d_hit = scored = 0
    items = list(POOLS.items())[: args.n]
    for qi, (qid, pl) in enumerate(items, 1):
        q = pl["query"]; a2t = pl["alias_to_text"]; goldset = set(pl["gold_aliases"])
        order = subsample(pl, args.pool)
        body = [{"id": a, "text": a2t[a]} for a in order]
        oa, aok = rcall(p_rerank_plain(q, body), "rank")
        if not aok:
            n_callfail += 1; A = list(order)
        else:
            ids = [str(x.get("id") if isinstance(x, dict) else x) for x in oa.get("ranked_ids", [])]
            valid_unique = len({x for x in ids if x in set(order)})
            if valid_unique < len(order):
                n_repair += 1                          # A did not cover the full pool -> repaired
            A = rank_from_ids(oa.get("ranked_ids", []), order)
        h1, _ = rcall(p_a1(q), "any"); h2, _ = rcall(p_a2(q), "any")
        od, dok = rcall(p_a3_anchor(q, h1, h2, body), "dec")
        if not dok:
            n_callfail += 1; D = list(A)
        else:
            dec = od.get("decision") or "surface_sufficient"
            prom = [str(x.get("id") if isinstance(x, dict) else x) for x in (od.get("promote_ids") or [])]
            prom = [a for a in prom if a in a2t][:2]
            D = list(A) if (dec == "surface_sufficient" or not prom) else prom + [a for a in A if a not in prom]
        a_hit += A[0] in goldset; d_hit += D[0] in goldset; scored += 1
        print(f"[{qi}/{len(items)}] {qid} A_ok={aok} D_ok={dok}", flush=True)

    rep_rate = round(n_repair / max(1, scored), 3)
    fail_rate = round(n_callfail / max(1, 2 * scored), 3)
    report = {"model": args.model, "pool": args.pool, "n_sample": scored,
              "temperature_settable_0": cl.temperature == 0, "temperature_recorded": cl.temperature,
              "resolved_model": getattr(cl, "last_model", None),
              "repair_rate_A": rep_rate, "call_fail_rate": fail_rate,
              "A_R@1_sample": round(a_hit / max(1, scored), 3), "D_R@1_sample": round(d_hit / max(1, scored), 3),
              "gate_repair_ok": rep_rate <= 0.20, "verdict": "RUN" if (rep_rate <= 0.20 and fail_rate <= 0.1) else "EXCLUDE"}
    out = ROOT / "results" / "cross_model"; out.mkdir(parents=True, exist_ok=True)
    (out / f"precheck_{args.model.replace('/', '_')}.json").write_text(json.dumps(report, indent=1))
    print("\n=== PRE-CHECK ===")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
