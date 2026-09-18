"""Add B (concept-guided) and C (competing hypotheses) to the BRIGHT backbone panel, so the
full A/B/C/D architecture comparison is cross-checked on version-pinned backbones. Reuses the
IDENTICAL frozen biology pool and each backbone's OWN stored A baseline
(results/bright_backbones/<model>/per_query.json, field A_top1_gold). Verbatim prompts. 504-tolerant:
a failed coordination call falls back to the baseline outcome (no crash). Resumable per model.

Usage: python bright_backbone_bc.py [<model> ...]   (default: the three pinned panel backbones)
"""
from __future__ import annotations
import os, sys, json, math, statistics as st
from pathlib import Path
from rq2_core import extract_json_object
from bright_backbone import client_for, POOLS, mcnemar, p_a1, p_a2, ROOT
from bright_finalize_measure import p_concept_B, p_rerank_B, p_a3_full, rank_from_ids

for _env in (ROOT / ".env", ROOT.parent / ".env"):
    if _env.exists():
        for _l in _env.read_text().splitlines():
            _l = _l.strip()
            if _l and not _l.startswith("#") and "=" in _l:
                _k, _v = _l.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

MODELS = sys.argv[1:] or ["gemini-2.5-flash-lite", "claude-haiku-4-5", "claude-sonnet-5"]
BB = ROOT / "results" / "bright_backbones"


def safe(cl, prompt):
    try:
        return extract_json_object(cl.generate(prompt))
    except Exception:
        return {}


def run_model(model):
    cl = client_for(model)
    mdir = BB / model
    A = {str(q["id"]): q for q in json.loads((mdir / "per_query.json").read_text())}  # backbone's own A
    outB, outC = [], []
    fb = {"B": [0, 0], "C": [0, 0]}   # [fix, break]
    fail = {"B": 0, "C": 0}
    for qid, pl in POOLS.items():
        q = pl["query"]; a2t = pl["alias_to_text"]; goldset = set(pl["gold_aliases"]); order = list(a2t)
        body = [{"id": a, "text": a2t[a]} for a in order]
        a_hit = bool(A[qid]["A_top1_gold"]); a_first = A[qid]["A_first_gold"]

        # B: concept -> concept-conditioned rerank
        ca = safe(cl, p_concept_B(q))
        ob = safe(cl, p_rerank_B(q, body, ca))
        ids_b = ob.get("ranked_ids", [])
        if ids_b:
            rb = rank_from_ids(ids_b, order); b_hit = rb[0] in goldset
            b_first = next((i + 1 for i, a in enumerate(rb) if a in goldset), len(rb) + 1)
        else:
            fail["B"] += 1; b_hit = a_hit; b_first = a_first          # 504 fallback -> baseline
        outB.append({"id": qid, "baseline_top1_gold": a_hit, "arch_top1_gold": b_hit,
                     "baseline_first_gold_rank": a_first, "arch_first_gold_rank": b_first})
        if not a_hit and b_hit: fb["B"][0] += 1
        if a_hit and not b_hit: fb["B"][1] += 1

        # C: two hypotheses -> coordinator rebuilds full ranking
        h1 = safe(cl, p_a1(q)); h2 = safe(cl, p_a2(q)); oc = safe(cl, p_a3_full(q, h1, h2, body))
        ids_c = oc.get("ranked_ids", [])
        if ids_c:
            rc = rank_from_ids(ids_c, order); c_hit = rc[0] in goldset
            c_first = next((i + 1 for i, a in enumerate(rc) if a in goldset), len(rc) + 1)
        else:
            fail["C"] += 1; c_hit = a_hit; c_first = a_first
        outC.append({"id": qid, "baseline_top1_gold": a_hit, "arch_top1_gold": c_hit,
                     "baseline_first_gold_rank": a_first, "arch_first_gold_rank": c_first})
        if not a_hit and c_hit: fb["C"][0] += 1
        if a_hit and not c_hit: fb["C"][1] += 1

    (mdir / "B_per_query.json").write_text(json.dumps(outB, indent=1))
    (mdir / "C_per_query.json").write_text(json.dumps(outC, indent=1))
    n = len(POOLS)
    a_r1 = sum(1 for x in outB if x["baseline_top1_gold"]) / n
    summ = {"model": model, "n": n, "A_R@1": round(a_r1, 3)}
    for v in ["B", "C"]:
        out = outB if v == "B" else outC
        r1 = sum(1 for x in out if x["arch_top1_gold"]) / n
        fx, bk = fb[v]
        summ[v] = {"R@1": round(r1, 3), "fixed": fx, "broke": bk, "net": fx - bk,
                   "mcnemar_p_R@1": round(mcnemar(fx, bk), 4), "fallbacks_504": fail[v]}
    (mdir / "bc_summary.json").write_text(json.dumps(summ, indent=1))
    print(f"=== {model} (n={n}, A R@1={a_r1:.3f}) ===")
    for v in ["B", "C"]:
        s = summ[v]
        print(f"  {v}: R@1={s['R@1']:.3f} fix={s['fixed']} break={s['broke']} "
              f"McNemar p={s['mcnemar_p_R@1']} (504-fallbacks={s['fallbacks_504']})")


def main():
    for M in MODELS:
        if (BB / M / "bc_summary.json").exists():
            print(f"# {M} already has bc_summary.json -- skipping")
            continue
        run_model(M)


if __name__ == "__main__":
    main()
