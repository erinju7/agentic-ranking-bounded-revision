"""Anchor-visibility perturbation check: hidden (Study-1 style) vs visible (Study-2 style) D.

Controls everything but anchor visibility. Same 53 Biology queries (A ranks a relevant doc
first), same degraded ranking, same model, same seed, SAME base coordinator prompt, and the
SAME promotion cap (1). The only difference between configurations is whether the coordinator
is shown the ranking:
  - HIDDEN  (Study 1 style): coordinator does NOT see A/degraded ranking; it only names <=1
             promotion; the code applies it to the saved ranking. Its input is anchor-agnostic,
             so one call is shared across the correct/degraded conditions.
  - VISIBLE (Study 2 style): coordinator IS shown the ranking with a preserve-unless-warranted
             default; separate calls for correct and degraded anchors.

Each configuration is scored on:
  - correct anchor  -> preservation rate (does it break a correct rank 1?)
  - degraded anchor -> recovery rate     (does it restore the relevant doc to rank 1?)
"""
from __future__ import annotations
import os, sys, json, time, statistics as st
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bright_backbone import POOLS, p_rerank_plain, p_a1, p_a2, rank_from_ids, client_for, ROOT
from rq2_core import extract_json_object

for _e in (ROOT/".env", ROOT.parent/".env"):
    if _e.exists():
        for _l in _e.read_text().splitlines():
            _l=_l.strip()
            if _l and not _l.startswith("#") and "=" in _l:
                _k,_v=_l.split("=",1); os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

MODEL = "claude-haiku-4-5-20251001"
CAP = 1  # unified promotion cap for BOTH configurations
OUT = ROOT / "results" / "bright_ablation" / MODEL / "anchor_visibility_out.json"; OUT.parent.mkdir(parents=True, exist_ok=True)
cl = client_for(MODEL)

def gen(p, tries=3):
    for t in range(tries):
        try:
            return extract_json_object(cl.generate(p))
        except Exception as e:
            if t == tries-1:
                print(f"   gen failed: {e}", flush=True); return {}
            time.sleep(2*(t+1))
    return {}

# Shared base coordinator prompt. anchor_ids=None -> HIDDEN; a list -> VISIBLE.
def p_coord(q, h1, h2, c, anchor_ids=None):
    if anchor_ids is None:
        situation = ("A single-pass baseline has ALREADY produced a good ranking of the "
                     "candidates (you do NOT see it).")
        task = ("Decide whether the baseline's surface reading suffices; if not, name the "
                "single document the correct latent reading promotes to the top.")
    else:
        situation = ("A single-pass baseline has ALREADY produced this ranking of the candidates "
                     f"(best first):\n{json.dumps(anchor_ids, ensure_ascii=False)}\n"
                     "Your DEFAULT is to PRESERVE this ranking.")
        task = ("Decide whether the shown ranking's top is correct; if not, name the single "
                "document that should be promoted to the top.")
    return f"""You are the COORDINATOR in an ANCHOR-AND-EDIT retrieval system. {situation}

Two competing hypotheses (neither privileged):
  SURFACE/LEXICAL: {json.dumps(h1, ensure_ascii=False)}
  LATENT CONCEPT : {json.dumps(h2, ensure_ascii=False)}

{task} Do NOT re-rank from scratch; edit only on strong evidence.
- decision="surface_sufficient": the current top is correct; promote_ids MUST be [].
- decision="concept_decisive"/"mixed": name AT MOST ONE id to promote to the top.
Reason from the EVIDENCE in the candidate texts. Return JSON only.
Schema: {{"decision":"surface_sufficient|concept_decisive|mixed","promote_ids":["<id>"],"rationale":"<short>"}}

Question: {q[:1500]}

Documents:
{json.dumps(c, ensure_ascii=False)}"""

def promote_ids_of(co, a2t):
    prom = [str(x.get("id") if isinstance(x, dict) else x) for x in (co.get("promote_ids") or [])]
    dec = co.get("decision") or "surface_sufficient"
    if dec == "surface_sufficient":
        return []
    return [a for a in prom if a in a2t][:CAP]

def apply(anchor, prom):
    return list(anchor) if not prom else prom + [a for a in anchor if a not in prom]

def top1_gold(order, goldset):
    return order[0] in goldset

def main():
    rows = []
    for i, (qid, pl) in enumerate(POOLS.items(), 1):
        q = pl["query"]; a2t = pl["alias_to_text"]; goldset = set(pl["gold_aliases"]); order = list(a2t)
        body = [{"id": a, "text": a2t[a]} for a in order]
        A = rank_from_ids(gen(p_rerank_plain(q, body)).get("ranked_ids", []), order)
        if A[0] not in goldset:
            print(f"[{i}/97] {qid} skip (A top1 not gold)", flush=True); continue
        nongolds = [a for a in A if a not in goldset]; golds = [a for a in A if a in goldset]
        if not nongolds:
            print(f"[{i}/97] {qid} skip (no non-gold)", flush=True); continue
        degraded = nongolds + golds  # non-gold leads; golds recoverable

        h1 = gen(p_a1(q)); h2 = gen(p_a2(q))

        # HIDDEN: one anchor-agnostic call, applied to both base rankings
        prom_h = promote_ids_of(gen(p_coord(q, h1, h2, body, anchor_ids=None)), a2t)
        hid_correct  = apply(A, prom_h)
        hid_degraded = apply(degraded, prom_h)

        # VISIBLE: separate calls (sees the ranking)
        prom_vc = promote_ids_of(gen(p_coord(q, h1, h2, body, anchor_ids=A)), a2t)
        prom_vd = promote_ids_of(gen(p_coord(q, h1, h2, body, anchor_ids=degraded)), a2t)
        vis_correct  = apply(A, prom_vc)
        vis_degraded = apply(degraded, prom_vd)

        rows.append({
            "id": qid,
            "hidden":  {"promote": prom_h,
                        "preserve_correct": top1_gold(hid_correct, goldset),
                        "recover_degraded": top1_gold(hid_degraded, goldset)},
            "visible": {"promote_correct": prom_vc, "promote_degraded": prom_vd,
                        "preserve_correct": top1_gold(vis_correct, goldset),
                        "recover_degraded": top1_gold(vis_degraded, goldset)},
        })
        print(f"[{i}/97] {qid} | hidden rec={rows[-1]['hidden']['recover_degraded']} "
              f"pres={rows[-1]['hidden']['preserve_correct']} | "
              f"visible rec={rows[-1]['visible']['recover_degraded']} "
              f"pres={rows[-1]['visible']['preserve_correct']}", flush=True)

    N = len(rows)
    def rate(cfg, key): return round(sum(r[cfg][key] for r in rows)/N, 3)
    # paired recovery discordance (McNemar-style)
    h_only = sum(1 for r in rows if r["hidden"]["recover_degraded"] and not r["visible"]["recover_degraded"])
    v_only = sum(1 for r in rows if r["visible"]["recover_degraded"] and not r["hidden"]["recover_degraded"])
    summary = {
        "model": MODEL, "cap": CAP, "n": N, "run_utc": datetime.now(timezone.utc).isoformat(),
        "hidden":  {"recovery_degraded": rate("hidden", "recover_degraded"),
                    "preservation_correct": rate("hidden", "preserve_correct")},
        "visible": {"recovery_degraded": rate("visible", "recover_degraded"),
                    "preservation_correct": rate("visible", "preserve_correct")},
        "paired_recovery": {"hidden_only": h_only, "visible_only": v_only,
                            "both": sum(1 for r in rows if r["hidden"]["recover_degraded"] and r["visible"]["recover_degraded"]),
                            "neither": sum(1 for r in rows if not r["hidden"]["recover_degraded"] and not r["visible"]["recover_degraded"])},
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("\n==== SUMMARY ===="); print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
