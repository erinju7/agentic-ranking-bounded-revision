"""System E: DECOMPOSED AGENTIC MATCHER for pairwise proposal--funding-call matching.
Separate from BRIGHT (unmodified) and NOT anchor-and-edit. Pipeline:
  Agent1 Proposal Understanding (proposal only)  -> structured summary (cached per proposal)
  Agent2 Call Understanding (call only)          -> structured summary (cached per call)
  Agent3 Alignment (ONLY the two summaries)      -> dimension fits + strengths/mismatches
  Agent4 Final Decision (docs + summaries + alignment) -> match_score/decision/confidence
Baseline = System A (single-pass) reused from results/study2_validation/pairs.json (NOT rerun).
No ground truth -> NO Accuracy/Precision/Recall/F1. Reports score/decision/rationale changes,
dimension analysis, cost/latency/tokens. Frozen exploratory external validation.
"""
from __future__ import annotations
import json, time, statistics as st
from pathlib import Path
import os, argparse
from datetime import datetime, timezone
from rq2_core import make_client, extract_json_object, usage_cost_usd
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
BASE = {(p["proposal"],p["call"]):p["A"] for p in json.loads((ROOT/"results"/"study2_validation"/"pairs.json").read_text())}
OUT = ROOT/"results"/"study2_validation"/"decomposed"; OUT.mkdir(parents=True, exist_ok=True)
MODEL="gemini-flash-latest"; PRICE_KEY=MODEL; PROP_CAP, CALL_CAP = (int(os.environ.get("PROP_CAP","0")) or 10**9), (int(os.environ.get("CALL_CAP","0")) or 10**9)
PROPS=list(DOCS["proposals"]); CALLS=list(DOCS["calls"])


def call(client,prompt):
    t=time.perf_counter(); raw=client.generate(prompt); dt=time.perf_counter()-t
    u=client.last_usage_metadata or {}
    m={"lat":dt,"in":u.get("prompt_token_count") or 0,"out":u.get("candidates_token_count") or 0,"cost":usage_cost_usd(PRICE_KEY,u) or 0.0}
    try: o=extract_json_object(raw)
    except Exception: o={}
    return o,m


def p_agent1(prop):
    return f"""You are the PROPOSAL UNDERSTANDING agent. Read ONLY the proposal. Do NOT consider
any funding call. Produce a structured semantic representation. Return JSON only.
Schema: {{"research_problem":"...","disease_area":"...","scientific_question":"...","mechanism":"...","technology_or_modality":"...","research_stage":"...","expected_impact":"...","keywords":["..."]}}

PROPOSAL:
{prop[:PROP_CAP]}"""

def p_agent2(call_txt):
    return f"""You are the FUNDING CALL UNDERSTANDING agent. Read ONLY the funding call. Do NOT
read any proposal. Extract what kind of projects the programme is seeking. Return JSON only.
Schema: {{"programme_goal":"...","target_disease":"...","preferred_mechanisms":["..."],"preferred_modalities":["..."],"technology_scope":"...","eligibility_constraints":["..."],"desired_research_stage":"...","success_criteria":["..."]}}

FUNDING CALL:
{call_txt[:CALL_CAP]}"""

def p_agent3(psum,csum):
    return f"""You are the ALIGNMENT agent. You receive ONLY two structured summaries. Do NOT
reread the original documents. Evaluate semantic alignment dimension-by-dimension (0-100 each).
Return JSON only. Do NOT produce a final decision.
Schema: {{"thematic_fit":0-100,"mechanistic_fit":0-100,"technology_fit":0-100,"stage_fit":0-100,"constraint_fit":0-100,"major_strengths":["..."],"major_mismatches":["..."],"uncertainties":["..."]}}

PROPOSAL SUMMARY:
{json.dumps(psum,ensure_ascii=False)}

FUNDING CALL SUMMARY:
{json.dumps(csum,ensure_ascii=False)}"""

def p_agent4(prop,call_txt,psum,csum,align):
    return f"""You are the FINAL DECISION agent for proposal--funding-call matching. Using all
inputs below, produce the final judgement. Return JSON only.
Schema: {{"match_score":0-100,"decision":"match"|"no_match"|"borderline","confidence":0-100,"reasoning":"<=3 sentences","key_supporting_dimensions":["..."]}}

PROPOSAL:
{prop[:PROP_CAP]}

FUNDING CALL:
{call_txt[:CALL_CAP]}

PROPOSAL SUMMARY:
{json.dumps(psum,ensure_ascii=False)}

FUNDING CALL SUMMARY:
{json.dumps(csum,ensure_ascii=False)}

ALIGNMENT ANALYSIS:
{json.dumps(align,ensure_ascii=False)}"""


def clamp(v):
    try: return max(0,min(100,int(round(float(v)))))
    except Exception: return 0


def main():
    global MODEL, PRICE_KEY, OUT
    ap=argparse.ArgumentParser(); ap.add_argument("--model", default=MODEL); ap.add_argument("--rep", default=None); args=ap.parse_args()
    load_env(); MODEL=args.model
    PRICE_KEY=("claude-sonnet-5" if "sonnet" in MODEL else "claude-opus-4-8" if "opus" in MODEL
               else "claude-haiku-4-5" if "haiku" in MODEL else MODEL)
    if MODEL!="gemini-flash-latest":
        name="decomposed_"+MODEL.replace("/","_")+(f"_rep{args.rep}" if args.rep else "")
        OUT=OUT.parent/name; OUT.mkdir(parents=True, exist_ok=True)
    client=client_for(MODEL)
    meter={"a1":[],"a2":[],"a3":[],"a4":[]}
    def add(k,m): meter[k].append(m)
    # Agent1 per proposal, Agent2 per call (cached)
    psum={}; csum={}
    for p in PROPS:
        o,m=call(client,p_agent1(DOCS["proposals"][p]["text"])); add("a1",m); psum[p]=o
    for c in CALLS:
        o,m=call(client,p_agent2(DOCS["calls"][c]["text"])); add("a2",m); csum[c]=o
    (OUT/"proposal_summaries.json").write_text(json.dumps(psum,indent=1,ensure_ascii=False))
    (OUT/"call_summaries.json").write_text(json.dumps(csum,indent=1,ensure_ascii=False))

    DIMS=["thematic_fit","mechanistic_fit","technology_fit","stage_fit","constraint_fit"]
    pairs=[]
    for p in PROPS:
        for c in CALLS:
            align,m3=call(client,p_agent3(psum[p],csum[c])); add("a3",m3)
            fin,m4=call(client,p_agent4(DOCS["proposals"][p]["text"],DOCS["calls"][c]["text"],psum[p],csum[c],align)); add("a4",m4)
            E={"match_score":clamp(fin.get("match_score",0)),
               "decision":str(fin.get("decision","no_match")).lower(),
               "confidence":clamp(fin.get("confidence",0)),
               "reasoning":(fin.get("reasoning","") or "")[:400],
               "key_supporting_dimensions":fin.get("key_supporting_dimensions",[])}
            if E["decision"] not in ("match","no_match","borderline"): E["decision"]="no_match"
            A=BASE[(p,c)]
            dims={d:clamp(align.get(d,0)) for d in DIMS}
            pairs.append({"proposal":p,"call":c,
                "A":{"match_score":A["match_score"],"decision":A["decision"],"rationale":A.get("rationale","")},
                "E":E,"alignment":{**dims,
                    "major_strengths":align.get("major_strengths",[]),
                    "major_mismatches":align.get("major_mismatches",[]),
                    "uncertainties":align.get("uncertainties",[])},
                "score_change_E_minus_A":E["match_score"]-A["match_score"],
                "decision_change":f'{A["decision"]}->{E["decision"]}' if A["decision"]!=E["decision"] else "",
                "pair_cost_usd":round(m3["cost"]+m4["cost"],6),
                "pair_latency_s":round(m3["lat"]+m4["lat"],2),
                "pair_tokens":{"in":m3["in"]+m4["in"],"out":m3["out"]+m4["out"]}})
    (OUT/"pairs_E.json").write_text(json.dumps(pairs,indent=1,ensure_ascii=False))

    idx={(x["proposal"],x["call"]):x for x in pairs}
    def mat(g):
        L=["proposal\\call,"+",".join(CALLS)]
        for p in PROPS: L.append(",".join([p]+[str(g(p,c)) for c in CALLS]))
        return "\n".join(L)
    (OUT/"matrix_E_score.csv").write_text(mat(lambda p,c: idx[(p,c)]["E"]["match_score"]))
    (OUT/"matrix_E_decision.csv").write_text(mat(lambda p,c: idx[(p,c)]["E"]["decision"]))
    for d in DIMS: (OUT/f"matrix_{d}.csv").write_text(mat(lambda p,c: idx[(p,c)]["alignment"][d]))

    # comparison table (baseline vs decomposed), all 36 pairs
    md=["# proposal--funding matching — Baseline (A) vs Decomposed Agentic Matcher (E). EXPLORATORY, no ground truth.","",
        "| Proposal | Call | A score/dec | E score/dec/conf | ΔScore | Decision change | Alignment (them/mech/tech/stage/constr) |",
        "|---|---|---|---|---|---|---|"]
    for x in pairs:
        a=x["alignment"]
        md.append(f"| {x['proposal']} | {x['call']} | {x['A']['match_score']}/{x['A']['decision']} "
                  f"| {x['E']['match_score']}/{x['E']['decision']}/{x['E']['confidence']} | {x['score_change_E_minus_A']:+d} "
                  f"| {x['decision_change'] or '—'} | {a['thematic_fit']}/{a['mechanistic_fit']}/{a['technology_fit']}/{a['stage_fit']}/{a['constraint_fit']} |")
    (OUT/"comparison_table.md").write_text("\n".join(md))

    # dimension dominance: correlation of each dim with E match_score; mean dim by E decision
    def pearson(xs,ys):
        n=len(xs); mx=st.mean(xs); my=st.mean(ys)
        num=sum((a-mx)*(b-my) for a,b in zip(xs,ys))
        dx=(sum((a-mx)**2 for a in xs))**0.5; dy=(sum((b-my)**2 for b in ys))**0.5
        return round(num/(dx*dy),3) if dx and dy else 0.0
    scores=[x["E"]["match_score"] for x in pairs]
    dom={d:pearson([x["alignment"][d] for x in pairs],scores) for d in DIMS}
    bydec={}
    for dec in ["match","borderline","no_match"]:
        grp=[x for x in pairs if x["E"]["decision"]==dec]
        if grp: bydec[dec]={"n":len(grp),**{d:round(st.mean([x["alignment"][d] for x in grp]),1) for d in DIMS}}
    # changes summary
    dec_changes=[x for x in pairs if x["decision_change"]]
    big=sorted(pairs,key=lambda x:abs(x["score_change_E_minus_A"]),reverse=True)[:6]
    def cagg(k):
        c=meter[k]; return {"calls":len(c),"cost":round(sum(m["cost"] for m in c),4),"lat":round(sum(m["lat"] for m in c),1),"in":sum(m["in"] for m in c),"out":sum(m["out"] for m in c)}
    Ecost=sum(sum(m["cost"] for m in meter[k]) for k in meter)
    Elat=sum(sum(m["lat"] for m in meter[k]) for k in meter)
    summary={"disclaimer":"Exploratory external validation; no expert ground truth; LLM outputs are NOT gold.",
        "n_pairs":36,"model":MODEL,"resolved_model":getattr(client,"last_model",None),
        "temperature":getattr(client,"temperature",None),"run_utc":datetime.now(timezone.utc).isoformat(),
        "E_total":{"calls":sum(len(meter[k]) for k in meter),"cost_usd":round(Ecost,4),"latency_s":round(Elat,1),
                   "cost_per_pair":round(Ecost/36,6),"tokens_in":sum(sum(m['in'] for m in meter[k]) for k in meter),
                   "tokens_out":sum(sum(m['out'] for m in meter[k]) for k in meter)},
        "E_by_agent":{k:cagg(k) for k in meter},
        "behaviour":{"decision_changes_vs_A":len(dec_changes),"mean_abs_score_change":round(st.mean([abs(x["score_change_E_minus_A"]) for x in pairs]),1),
                     "E_decisions":{d:sum(1 for x in pairs if x["E"]["decision"]==d) for d in ["match","borderline","no_match"]}},
        "dimension_correlation_with_E_score":dom,
        "mean_dimension_by_E_decision":bydec,
        "decision_change_pairs":[{"pair":f'{x["proposal"]}->{x["call"]}',"change":x["decision_change"],
                                  "A_score":x["A"]["match_score"],"E_score":x["E"]["match_score"],
                                  "mismatches":x["alignment"]["major_mismatches"][:2]} for x in dec_changes]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=1,ensure_ascii=False))
    print("=== System E (decomposed) vs A (baseline), n=36, EXPLORATORY (no ground truth) ===")
    print(f"  E decisions: {summary['behaviour']['E_decisions']} | decision changes vs A: {len(dec_changes)} | mean |Δscore|: {summary['behaviour']['mean_abs_score_change']}")
    print(f"  dimension correlation with E match_score: {dom}")
    print(f"  mean dimension by E decision: {bydec}")
    print(f"  E cost=${summary['E_total']['cost_usd']} lat={summary['E_total']['latency_s']}s calls={summary['E_total']['calls']} tok in/out={summary['E_total']['tokens_in']}/{summary['E_total']['tokens_out']}")
    print(f"  decision-change pairs ({len(dec_changes)}):")
    for x in dec_changes: print(f"    {x['proposal']:12}->{x['call']:16} {x['decision_change']:22} A={x['A']['match_score']} E={x['E']['match_score']}")
    print(f"  wrote pairs_E.json, comparison_table.md, matrix_*.csv, summary.json to {OUT}")


if __name__ == "__main__":
    main()
