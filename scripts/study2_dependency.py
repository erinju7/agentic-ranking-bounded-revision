"""Study 2a: Information Dependency (proposal--funding matching). Two systems that
differ ONLY in what the Alignment Agent consumes:
  DEP: Alignment reads ONLY the two structured representations (reasons over relationships).
  IND: Alignment reads the RAW proposal + raw call directly, NOT the structured reps.
Everything else identical: same docs, same backbone (gemini-flash-latest, temp 0), same
Proposal/Call agents, identical Final Judge (proposal+call+both summaries+alignment). Prompts
reused verbatim from study2_decomposed.py. No reviewer/debate/reflection/retrieval/optimisation.
Exploratory: no ground truth -> no accuracy claims. Reports distributions, changes, cost.
"""
from __future__ import annotations
import json, time, statistics as st
from pathlib import Path
import os, argparse
from datetime import datetime, timezone
from rq2_core import make_client, extract_json_object, usage_cost_usd
from study2_decomposed import p_agent1, p_agent2, p_agent3, p_agent4, clamp
from study2_match import client_for

ROOT = Path(__file__).resolve().parents[1]
def load_env():
    for e in (ROOT/".env", ROOT.parent/".env"):
        if e.exists():
            for l in e.read_text().splitlines():
                l=l.strip()
                if l and not l.startswith("#") and "=" in l:
                    k,v=l.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
DOCS = json.loads((ROOT/"data"/"study2_validation"/"docs.json").read_text())
OUT = ROOT/"results"/"study2_validation"/"study2a_dependency"; OUT.mkdir(parents=True, exist_ok=True)
MODEL="gemini-flash-latest"; PRICE_KEY=MODEL; PROP_CAP, CALL_CAP = (int(os.environ.get("PROP_CAP","0")) or 10**9), (int(os.environ.get("CALL_CAP","0")) or 10**9)
PROPS=list(DOCS["proposals"]); CALLS=list(DOCS["calls"])
DIMS=["thematic_fit","mechanistic_fit","technology_fit","stage_fit","constraint_fit"]

# Shared ground-truth rubric, given verbatim to the Final Judge so grader and expert use the
# same criteria (annotation_rubric.md). Frozen; prepended identically to the judge prompt.
RUBRIC = """GRADING CRITERIA (apply exactly; grade using ONLY the proposal and funding-call texts, not outside knowledge of the sponsor or its strategy):
- "match": the proposal clearly falls within the call's stated therapeutic areas / scope / eligibility.
- "borderline": partial fit -- overlaps on some dimension (disease area, modality, mechanism) but has a scope, eligibility, or sponsor-specific mismatch that makes suitability genuinely uncertain.
- "no_match": outside the call's stated areas/scope, or violates a stated constraint (wrong disease area, wrong modality, an explicit exclusion, or a sponsor/institutional conflict the call rules out).
Decision procedure, in order: (1) read the call's stated areas of interest / scope / eligibility; (2) read the proposal's disease area, mechanism/modality, stage, and any constraints; (3) if a hard constraint is violated, grade "no_match"; (4) else if it squarely fits, grade "match"; (5) otherwise grade "borderline".
"""


def p_align_ind(prop, call_txt):
    # IND alignment: identical schema to DEP alignment (p_agent3) but reads raw documents
    # directly and does NOT use the specialist structured representations.
    return f"""You are the ALIGNMENT agent. Read the PROPOSAL and the FUNDING CALL directly and
evaluate semantic alignment dimension-by-dimension (0-100 each). Return JSON only. Do NOT
produce a final decision.
Schema: {{"thematic_fit":0-100,"mechanistic_fit":0-100,"technology_fit":0-100,"stage_fit":0-100,"constraint_fit":0-100,"major_strengths":["..."],"major_mismatches":["..."],"uncertainties":["..."]}}

PROPOSAL:
{prop[:PROP_CAP]}

FUNDING CALL:
{call_txt[:CALL_CAP]}"""


def main():
    global MODEL, PRICE_KEY, OUT
    ap=argparse.ArgumentParser(); ap.add_argument("--model", default=MODEL); ap.add_argument("--rep", default=None); ap.add_argument("--tag", default=""); args=ap.parse_args()
    load_env(); MODEL=args.model
    PRICE_KEY=("claude-sonnet-5" if "sonnet" in MODEL else "claude-opus-4-8" if "opus" in MODEL
               else "claude-haiku-4-5" if "haiku" in MODEL else MODEL)
    if MODEL!="gemini-flash-latest":
        name="study2a_dependency_"+MODEL.replace("/","_")+args.tag+(f"_rep{args.rep}" if args.rep else "")
        OUT=OUT.parent/name; OUT.mkdir(parents=True, exist_ok=True)
    client=client_for(MODEL)
    meter={"P":[],"C":[],"align_DEP":[],"align_IND":[],"judge_DEP":[],"judge_IND":[]}
    def call(prompt,bucket):
        t=time.perf_counter(); raw=client.generate(prompt); dt=time.perf_counter()-t
        u=client.last_usage_metadata or {}
        meter[bucket].append({"lat":dt,"in":u.get("prompt_token_count") or 0,"out":u.get("candidates_token_count") or 0,"cost":usage_cost_usd(PRICE_KEY,u) or 0.0})
        try: return extract_json_object(raw)
        except Exception: return {}
    # shared specialist agents (identical in both systems)
    psum={p:call(p_agent1(DOCS["proposals"][p]["text"]),"P") for p in PROPS}
    csum={c:call(p_agent2(DOCS["calls"][c]["text"]),"C") for c in CALLS}

    def judge(prop,call_txt,ps,cs,align):
        o=call(RUBRIC+"\n"+p_agent4(prop,call_txt,ps,cs,align),"judge_DEP" if align.get("_sys")=="DEP" else "judge_IND")
        return {"match_score":clamp(o.get("match_score",0)),
                "decision":(str(o.get("decision","no_match")).lower() if str(o.get("decision","no_match")).lower() in ("match","no_match","borderline") else "no_match"),
                "confidence":clamp(o.get("confidence",0)),
                "reasoning":(o.get("reasoning","") or "")[:400],
                "key_supporting_dimensions":o.get("key_supporting_dimensions",[])}

    pairs=[]
    for p in PROPS:
        pt=DOCS["proposals"][p]["text"]
        for c in CALLS:
            ct=DOCS["calls"][c]["text"]
            aD=call(p_agent3(psum[p],csum[c]),"align_DEP"); aD["_sys"]="DEP"
            aI=call(p_align_ind(pt,ct),"align_IND"); aI["_sys"]="IND"
            JD=judge(pt,ct,psum[p],csum[c],aD)
            JI=judge(pt,ct,psum[p],csum[c],aI)
            dimsD={d:clamp(aD.get(d,0)) for d in DIMS}; dimsI={d:clamp(aI.get(d,0)) for d in DIMS}
            pairs.append({"proposal":p,"call":c,
                "DEP":{**JD,"alignment_dims":dimsD,"major_mismatches":aD.get("major_mismatches",[]),"major_strengths":aD.get("major_strengths",[])},
                "IND":{**JI,"alignment_dims":dimsI,"major_mismatches":aI.get("major_mismatches",[]),"major_strengths":aI.get("major_strengths",[])},
                "score_diff_DEP_minus_IND":JD["match_score"]-JI["match_score"],
                "decision_change":f'{JI["decision"]}->{JD["decision"]}' if JI["decision"]!=JD["decision"] else ""})
    (OUT/"pairs.json").write_text(json.dumps(pairs,indent=1,ensure_ascii=False))

    idx={(x["proposal"],x["call"]):x for x in pairs}
    def mat(sys,field):
        L=["proposal\\call,"+",".join(CALLS)]
        for p in PROPS: L.append(",".join([p]+[str(idx[(p,c)][sys][field]) for c in CALLS]))
        return "\n".join(L)
    for sys in ["IND","DEP"]:
        (OUT/f"matrix_{sys}_score.csv").write_text(mat(sys,"match_score"))
        (OUT/f"matrix_{sys}_decision.csv").write_text(mat(sys,"decision"))

    # distributions & aggregates
    def dist(sys):
        return {d:sum(1 for x in pairs if x[sys]["decision"]==d) for d in ["match","borderline","no_match"]}
    def scores(sys): return [x[sys]["match_score"] for x in pairs]
    dchanges=[x for x in pairs if x["decision_change"]]
    diffs=[x["score_diff_DEP_minus_IND"] for x in pairs]
    def dimmean(sys): return {d:round(st.mean([x[sys]["alignment_dims"][d] for x in pairs]),1) for d in DIMS}
    def cagg(k):
        m=meter[k]; return {"calls":len(m),"cost":round(sum(z["cost"] for z in m),4),"lat":round(sum(z["lat"] for z in m),1),"in":sum(z["in"] for z in m),"out":sum(z["out"] for z in m)}
    # per-SYSTEM cost = shared(P+C) + its own align + judge
    def sys_cost(sysalign,sysjudge):
        parts=[cagg("P"),cagg("C"),cagg(sysalign),cagg(sysjudge)]
        return {"calls":sum(p["calls"] for p in parts),"cost_usd":round(sum(p["cost"] for p in parts),4),
                "latency_s":round(sum(p["lat"] for p in parts),1),"tokens_in":sum(p["in"] for p in parts),"tokens_out":sum(p["out"] for p in parts)}
    summary={"disclaimer":"Exploratory; no expert ground truth; no accuracy claims. LLM outputs not gold.",
        "n_pairs":36,"model":MODEL,"resolved_model":getattr(client,"last_model",None),
        "temperature":getattr(client,"temperature",None),"run_utc":datetime.now(timezone.utc).isoformat(),
        "decision_distribution":{"IND":dist("IND"),"DEP":dist("DEP")},
        "score_distribution":{s:{"mean":round(st.mean(scores(s)),1),"median":st.median(scores(s)),"sd":round(st.pstdev(scores(s)),1),
                                 "min":min(scores(s)),"max":max(scores(s))} for s in ["IND","DEP"]},
        "decision_changes_IND_to_DEP":{"count":len(dchanges),"pairs":[{"pair":f'{x["proposal"]}->{x["call"]}',"change":x["decision_change"],
                                        "IND_score":x["IND"]["match_score"],"DEP_score":x["DEP"]["match_score"]} for x in dchanges]},
        "score_diff_DEP_minus_IND":{"mean_signed":round(st.mean(diffs),1),"mean_abs":round(st.mean([abs(d) for d in diffs]),1),
                                    "max_abs":max(abs(d) for d in diffs)},
        "mean_alignment_dims":{"IND":dimmean("IND"),"DEP":dimmean("DEP")},
        "cost":{"IND":sys_cost("align_IND","judge_IND"),"DEP":sys_cost("align_DEP","judge_DEP")},
        "cost_by_agent":{k:cagg(k) for k in meter}}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=1,ensure_ascii=False))

    # comparison table
    md=["# Study 2a Information Dependency — IND vs DEP (proposal--funding matching). EXPLORATORY, no ground truth.","",
        "| Proposal | Call | IND score/dec | DEP score/dec | ΔScore(D-I) | Decision change | IND dims (t/m/te/s/c) | DEP dims |",
        "|---|---|---|---|---|---|---|---|"]
    for x in pairs:
        di=x["IND"]["alignment_dims"]; dd=x["DEP"]["alignment_dims"]
        md.append(f"| {x['proposal']} | {x['call']} | {x['IND']['match_score']}/{x['IND']['decision']} | {x['DEP']['match_score']}/{x['DEP']['decision']} "
                  f"| {x['score_diff_DEP_minus_IND']:+d} | {x['decision_change'] or '—'} "
                  f"| {di['thematic_fit']}/{di['mechanistic_fit']}/{di['technology_fit']}/{di['stage_fit']}/{di['constraint_fit']} "
                  f"| {dd['thematic_fit']}/{dd['mechanistic_fit']}/{dd['technology_fit']}/{dd['stage_fit']}/{dd['constraint_fit']} |")
    (OUT/"comparison_table.md").write_text("\n".join(md))

    print("=== Study 2a: Information Dependency (IND vs DEP), n=36, EXPLORATORY ===")
    print(f"  decision dist IND: {dist('IND')}")
    print(f"  decision dist DEP: {dist('DEP')}")
    print(f"  score IND mean/med/sd: {summary['score_distribution']['IND']}")
    print(f"  score DEP mean/med/sd: {summary['score_distribution']['DEP']}")
    print(f"  decision changes IND->DEP: {len(dchanges)}  | mean signed Δ(D-I)={summary['score_diff_DEP_minus_IND']['mean_signed']}  mean|Δ|={summary['score_diff_DEP_minus_IND']['mean_abs']}")
    print(f"  mean alignment dims IND: {dimmean('IND')}")
    print(f"  mean alignment dims DEP: {dimmean('DEP')}")
    print(f"  cost IND: {summary['cost']['IND']}")
    print(f"  cost DEP: {summary['cost']['DEP']}")
    print("  decision-change pairs:")
    for x in dchanges: print(f"    {x['proposal']:12}->{x['call']:16} {x['decision_change']:24} IND={x['IND']['match_score']} DEP={x['DEP']['match_score']}")
    print(f"  wrote pairs.json, comparison_table.md, matrix_*.csv, summary.json to {OUT}")


if __name__ == "__main__":
    main()
