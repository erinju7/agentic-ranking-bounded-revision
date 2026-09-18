"""Study 2 LISTWISE reranking (A/B/D), James-aligned redesign (see study2_listwise_redesign_spec.md).

Per proposal, rank its FULL set of 6 candidate calls (no first-stage truncation -> no recall
ceiling). A single-pass listwise; B concept-guided listwise; D anchor-and-edit (coordinator
promotes <=cap ids to the FRONT of A's ranking, else preserves A). D coordinator runs once; the
cap is applied mechanically at cap=1 (primary) and cap=2 (sensitivity). Each variant emits a
ranking + per-call relevance grade; grades feed per-call kappa, the James gold feeds nDCG.
Reuses the per-proposal concept/surface/latent prompts from james_match.py. Backbone: same
gemini-flash-latest as the existing Study-2 runs and the Study-1 reference. LLM grades are NOT
gold; James labels remain authoritative.
"""
from __future__ import annotations
import os, json, random, argparse
from datetime import datetime, timezone
from pathlib import Path
import james_match as _jm
from james_match import p_concept, p_surface, p_latent, call, DOCS, ROOT, MODEL, PROP_CAP, CALL_CAP, client_for


def load_env():
    for env in (ROOT / ".env", ROOT.parent / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

OUT = ROOT / "results" / "james_validation" / "listwise"; OUT.mkdir(parents=True, exist_ok=True)
GRADES = ("match", "borderline", "no_match")
SEED = 42

# Shared ground-truth rubric, given verbatim to the model so grader and expert use the
# same criteria (annotation_rubric.md). Frozen; injected identically into every grading prompt.
RUBRIC = """GRADING CRITERIA (apply exactly; grade using ONLY the proposal and funding-call texts below, not outside knowledge of the sponsor or its strategy):
- "match": the proposal clearly falls within the call's stated therapeutic areas / scope / eligibility.
- "borderline": partial fit -- overlaps on some dimension (disease area, modality, mechanism) but has a scope, eligibility, or sponsor-specific mismatch that makes suitability genuinely uncertain.
- "no_match": outside the call's stated areas/scope, or violates a stated constraint (wrong disease area, wrong modality, an explicit exclusion, or a sponsor/institutional conflict the call rules out).
Decision procedure, in order: (1) read the call's stated areas of interest / scope / eligibility; (2) read the proposal's disease area, mechanism/modality, stage, and any constraints; (3) if a hard constraint is violated, grade "no_match"; (4) else if it squarely fits, grade "match"; (5) otherwise grade "borderline"."""


def render_calls(calls, cnames, rng):
    order = cnames[:]; rng.shuffle(order)                      # shuffle to remove position priors
    alias = {f"C{i}": cn for i, cn in enumerate(order)}
    cards = [{"id": a, "text": calls[cn]["text"][:CALL_CAP]} for a, cn in alias.items()]
    return alias, cards


def p_rank(prop, cards, concept_block=""):
    ids = ", ".join('"' + c["id"] + '"' for c in cards)
    return f"""You are matching a research PROPOSAL to candidate FUNDING CALLS / company wishlists.
Rank ALL {len(cards)} candidate ids from best to worst fit for this proposal, and assign each a
relevance grade.

{RUBRIC}

Return JSON only.
Schema: {{"ranked_call_ids": [all ids in your best-to-worst order, from: {ids}],
          "grades": {{"<id>": "match"|"borderline"|"no_match", ... for every id}},
          "rationale": "<=2 sentences"}}
{concept_block}
PROPOSAL:
{prop[:PROP_CAP]}

CANDIDATE CALLS:
{json.dumps(cards, ensure_ascii=False)}
"""


def p_coord_listwise(prop, cards, anchor_ids, surface, latent):
    return f"""You are the COORDINATOR in an ANCHOR-AND-EDIT reranker. A single-pass baseline has
ALREADY ranked the candidate calls (the ANCHOR, best first). Your DEFAULT is to PRESERVE it. Only
if the surface/latent interpretations give STRONG evidence that specific calls belong at the very
top should you name those ids (best first). Do NOT re-rank from scratch; do NOT edit on weak or
stylistic grounds.

ANCHOR ranking (best first): {json.dumps(anchor_ids, ensure_ascii=False)}
SURFACE interpretation: {json.dumps(surface, ensure_ascii=False)}
LATENT concept: {json.dumps(latent, ensure_ascii=False)}

Return JSON only.
Schema: {{"action":"keep"|"edit","promote_ids":["<id>", ... best first, or empty],"rationale":"<=2 sentences"}}
PROPOSAL:
{prop[:PROP_CAP]}

CANDIDATE CALLS:
{json.dumps(cards, ensure_ascii=False)}
"""


def p_coord_c(prop, cards, surface, latent):
    return f"""You are the COORDINATOR in a COMPETING-HYPOTHESES matcher. You are given the proposal,
TWO competing readings of what it is really about (neither privileged), and the candidate calls.
Decide from the evidence in the call texts which reading the best matches support, and produce the
FINAL ranking of ALL calls from best to worst fit, assigning each a relevance grade. Reason
call-by-call; do NOT average.

{RUBRIC}

Return JSON only.
Schema: {{"ranked_call_ids": [all ids best-to-worst], "grades": {{"<id>": "match"|"borderline"|"no_match", ... for every id}}, "rationale": "<=2 sentences"}}
SURFACE reading: {json.dumps(surface, ensure_ascii=False)}
LATENT reading: {json.dumps(latent, ensure_ascii=False)}
PROPOSAL:
{prop[:PROP_CAP]}

CANDIDATE CALLS:
{json.dumps(cards, ensure_ascii=False)}
"""


def clean_rank(obj, ids):
    r = [str(x) for x in (obj.get("ranked_call_ids") or []) if str(x) in ids]
    seen = set(); r = [x for x in r if not (x in seen or seen.add(x))]
    r += [i for i in ids if i not in r]                        # repair: append missing in pool order
    g = obj.get("grades") or {}
    grades = {i: (g.get(i) if g.get(i) in GRADES else "no_match") for i in ids}
    return r, grades


def promote(anchor, promote_ids, cap):
    p = [x for x in promote_ids if x in anchor][:cap]
    return p + [x for x in anchor if x not in p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--tag", type=str, default="")     # suffix output dir (e.g. "_rubric") to preserve prior runs
    args = ap.parse_args()
    seed = args.seed
    model = args.model
    _jm.MODEL = model                                  # so call()'s cost lookup uses this backbone
    # keep runs from different backbones in separate subdirs (do not overwrite the flash-latest record)
    base = OUT if model == MODEL else OUT / (model.replace("/", "_") + args.tag)
    out_dir = base if seed == SEED else base / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    load_env()
    client = client_for(model)
    props = DOCS["proposals"]; calls = DOCS["calls"]
    pnames = list(props); cnames = list(calls)
    rng = random.Random(seed)
    meter = {s: {"cost": 0.0, "lat": 0.0, "calls": 0, "in": 0, "out": 0} for s in ["A", "B", "C", "D"]}
    def add(s, m):
        meter[s]["cost"] += m["cost"]; meter[s]["lat"] += m["lat"]; meter[s]["calls"] += 1
        meter[s]["in"] += m["in"]; meter[s]["out"] += m["out"]

    results = []
    for pn in pnames:
        pt = props[pn]["text"]
        alias, cards = render_calls(calls, cnames, rng)
        ids = [c["id"] for c in cards]; id2call = {a: cn for a, cn in alias.items()}
        # per-proposal cached artifacts (concept for B; surface+latent for D)
        concept, mc = call(client, p_concept(pt)); add("B", mc)
        surface, ms = call(client, p_surface(pt)); add("D", ms)
        latent, ml = call(client, p_latent(pt)); add("D", ml)
        # A: single-pass listwise
        a_raw, am = call(client, p_rank(pt, cards)); add("A", am)
        A_rank, A_grades = clean_rank(a_raw, ids)
        # B: concept-guided listwise
        cb = ("\nProposal research-intent hypothesis (hint; judge on the actual texts): "
              f"concept={concept.get('concept','')}; reasoning={concept.get('reasoning','')}; "
              f"alternative={concept.get('alternative','')}\n")
        b_raw, bm = call(client, p_rank(pt, cards, cb)); add("B", bm)
        B_rank, B_grades = clean_rank(b_raw, ids)
        # C: competing hypotheses; coordinator rebuilds full ranking from surface+latent (reused from D)
        c_raw, cm = call(client, p_coord_c(pt, cards, surface, latent)); add("C", cm)
        C_rank, C_grades = clean_rank(c_raw, ids)
        # D: anchor-and-edit; coordinator once, caps applied mechanically
        co_raw, dm = call(client, p_coord_listwise(pt, cards, A_rank, surface, latent)); add("D", dm)
        action = "edit" if str(co_raw.get("action", "keep")).lower() == "edit" else "keep"
        prom = [str(x) for x in (co_raw.get("promote_ids") or []) if str(x) in ids] if action == "edit" else []
        D1, D2 = promote(A_rank, prom, 1), promote(A_rank, prom, 2)

        names = lambda r: [id2call[i] for i in r]
        gnames = lambda g: {id2call[i]: v for i, v in g.items()}
        results.append({
            "proposal": pn, "alias": alias,
            "A": {"ranked": names(A_rank), "grades": gnames(A_grades)},
            "B": {"ranked": names(B_rank), "grades": gnames(B_grades)},
            "C": {"ranked": names(C_rank), "grades": gnames(C_grades)},
            # D reorders A's ranking; per-call grades inherited from A (edit is positional, not a re-judgement)
            "D_cap1": {"ranked": names(D1), "grades": gnames(A_grades), "action": action, "promoted": names(prom[:1])},
            "D_cap2": {"ranked": names(D2), "grades": gnames(A_grades), "action": action, "promoted": names(prom[:2])},
            "concept": concept.get("concept", ""),
        })

    (out_dir / "listwise_results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    summary = {"n_proposals": len(pnames), "n_calls": len(cnames), "model": model,
               "resolved_model": getattr(client, "last_model", None),
               "temperature": getattr(client, "temperature", None),
               "run_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
               "promote_cap_primary": 1, "promote_cap_sensitivity": 2,
               "note": "LLM grades are NOT gold; James labels authoritative. D grades inherit A (positional edit).",
               "cost": {s: {"calls": meter[s]["calls"], "cost_usd": round(meter[s]["cost"], 4),
                            "lat_s": round(meter[s]["lat"], 1), "tok_in": meter[s]["in"], "tok_out": meter[s]["out"]}
                        for s in ["A", "B", "C", "D"]}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"=== Study 2 LISTWISE seed={seed} (A/B/C/D; D cap=1 primary + cap=2 sensitivity) ===")
    for s in ["A", "B", "C", "D"]:
        print(f"  {s}: calls={meter[s]['calls']} cost=${meter[s]['cost']:.4f} lat={meter[s]['lat']:.0f}s")
    print(f"  wrote listwise_results.json, summary.json to {out_dir}")


if __name__ == "__main__":
    main()
