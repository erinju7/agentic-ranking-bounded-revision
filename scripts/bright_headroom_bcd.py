"""Run B/C/D over a domain's frozen 100-doc pool (built by bright_hardpool_baseline.py),
using the VERBATIM prompts imported from bright_finalize_measure.py, so the additional
splits use byte-identical prompts + the same frozen pool as biology (config parity for the
RQ3 headroom pre-registration). Writes per_query.json (with baseline_top1_gold +
arch_top1_gold) per variant, matching the biology output schema.

Usage: python bright_headroom_bcd.py <domain>
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from rq2_core import make_client
import bright_finalize_measure as F   # verbatim prompts + call() + rank_from_ids()

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "biology"
POOLS = json.loads((ROOT / "data" / "bright_hardpool" / DOMAIN / "pools.json").read_text())
BASE = {str(p["id"]): p for p in
        json.loads((ROOT / "results" / "bright_hardpool" / DOMAIN / "baseline_per_query.json").read_text())}
MODEL = "gemini-flash-latest"


def load_env():
    for env in (ROOT / ".env", ROOT.parent / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def first_gold(ranked, goldset):
    return next((i + 1 for i, a in enumerate(ranked) if a in goldset), len(ranked) + 1)


FAIL = {"B": 0, "C": 0, "D": 0}


def safe_call(client, prompt, tag):
    """Tolerate transient 504/timeout: the harness raises after 6 retries; a failed
    coordination step must fall back (no crash), not kill the whole split."""
    try:
        return F.call(client, prompt)
    except Exception:
        FAIL[tag] += 1
        return {}, "", 0.0, 0.0


def run():
    load_env()
    client = make_client(MODEL)
    outB, outC, outD = [], [], []
    for qid, pl in POOLS.items():
        q = pl["query"]; a2t = pl["alias_to_text"]; goldset = set(pl["gold_aliases"]); order = list(a2t)
        body = [{"id": a, "text": a2t[a]} for a in order]
        bt = BASE[qid]["baseline_top1_gold"]; bfr = BASE[qid]["baseline_first_gold_rank"]
        base_ranked = BASE[qid]["baseline_ranked"]

        # B: concept-guided rerank. On failure -> baseline ranking (neutral "no change").
        ca, _, _, _ = safe_call(client, F.p_concept_B(q), "B")
        objb, _, _, _ = safe_call(client, F.p_rerank_B(q, body, ca), "B")
        ids_b = objb.get("ranked_ids", [])
        rb = F.rank_from_ids(ids_b, order) if ids_b else list(base_ranked)
        outB.append({"id": qid, "concept": ca.get("concept", ""), "baseline_top1_gold": bt,
                     "arch_top1_gold": rb[0] in goldset, "baseline_first_gold_rank": bfr,
                     "arch_first_gold_rank": first_gold(rb, goldset)})

        # C: competing hypotheses -> coordinator full re-rank. On failure -> baseline.
        h1, _, _, _ = safe_call(client, F.p_a1(q), "C"); h2, _, _, _ = safe_call(client, F.p_a2(q), "C")
        objc, _, _, _ = safe_call(client, F.p_a3_full(q, h1, h2, body), "C")
        ids_c = objc.get("ranked_ids", [])
        rc = F.rank_from_ids(ids_c, order) if ids_c else list(base_ranked)
        outC.append({"id": qid, "baseline_top1_gold": bt, "arch_top1_gold": rc[0] in goldset,
                     "baseline_first_gold_rank": bfr, "arch_first_gold_rank": first_gold(rc, goldset)})

        # D: anchor-and-edit -> promote <=2 to FRONT of baseline ranking (else keep).
        h1d, _, _, _ = safe_call(client, F.p_a1(q), "D"); h2d, _, _, _ = safe_call(client, F.p_a2(q), "D")
        cod, _, _, _ = safe_call(client, F.p_a3_anchor(q, h1d, h2d, body), "D")
        dec = str(cod.get("decision", "surface_sufficient")).lower()
        promote = [str(x.get("id") if isinstance(x, dict) else x) for x in (cod.get("promote_ids") or [])]
        promote = [a for a in promote if a in a2t][:2]
        if dec == "surface_sufficient" or not promote:
            rd = list(base_ranked)
        else:
            rd = promote + [x for x in base_ranked if x not in promote]
        outD.append({"id": qid, "decision": dec, "promote_ids": promote, "baseline_top1_gold": bt,
                     "arch_top1_gold": rd[0] in goldset, "baseline_first_gold_rank": bfr,
                     "arch_first_gold_rank": first_gold(rd, goldset)})

    for name, out in [("bright_concept_hardpool", outB),
                      ("bright_competing_hypotheses", outC),
                      ("bright_ch_anchor", outD)]:
        d = ROOT / "results" / name / DOMAIN; d.mkdir(parents=True, exist_ok=True)
        (d / "per_query.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    def fb(out):
        fix = sum(1 for x in out if (not x["baseline_top1_gold"]) and x["arch_top1_gold"])
        brk = sum(1 for x in out if x["baseline_top1_gold"] and (not x["arch_top1_gold"]))
        return fix, brk
    print(f"domain={DOMAIN} n={len(POOLS)}  (504-fallbacks: {FAIL})")
    for nm, out in [("B", outB), ("C", outC), ("D", outD)]:
        f, b = fb(out); print(f"  {nm}: fix={f} break={b} net={f - b}  fallbacks={FAIL[nm]}")


if __name__ == "__main__":
    run()
