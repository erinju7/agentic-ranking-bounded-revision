"""James proposal--funding-call matching: EXPLORATORY UNLABELED validation.
Systems (frozen prompts/logic, same design principle as BRIGHT):
  A single-pass matcher; B concept-guided matcher; D anchor-and-edit matcher.
For all 6x6=36 (proposal,call) pairs, saves A/B/D score+decision+rationale, whether B and D
changed the baseline, and per-call latency/tokens/cost. NO Accuracy/Precision/Recall/F1/AUROC/
McNemar -- there is no expert ground truth. LLM outputs are NOT gold. Produces a 6x6 review
matrix for James to label later.
"""
from __future__ import annotations
import os, json, time, statistics as st
from pathlib import Path
from rq2_core import make_client, extract_json_object, usage_cost_usd

ROOT = Path(__file__).resolve().parents[1]
# Load .env (ROOT/.env or ROOT.parent/.env) so open-model runs see OAI_BASE_URL /
# OAI_API_KEY_ENV / TOGETHER_API_KEY without the caller having to export them.
for _e in (ROOT/".env", ROOT.parent/".env"):
    if _e.exists():
        for _l in _e.read_text().splitlines():
            _l = _l.strip()
            if _l and not _l.startswith("#") and "=" in _l:
                _k, _v = _l.split("=", 1); os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
DOCS = json.loads((ROOT/"data"/"james_validation"/os.environ.get("DOCS_FILE","docs.json")).read_text())
OUT = ROOT/"results"/"james_validation"; OUT.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-flash-latest"
PROP_CAP, CALL_CAP = int(os.environ.get("PROP_CAP","7000")), int(os.environ.get("CALL_CAP","5000"))


class RawClaudeJSON:
    """Schema-free Claude caller for the James pipeline: unlike rq2_core.AnthropicClient it does
    NOT force the ranked_call_labels output schema, so the free-form concept/alignment/grade JSON
    works. Same generate()/last_usage_metadata surface. Records the served model version and a
    price_key (dated snapshots strip to their base id for cost lookup)."""
    def __init__(self, model):
        import anthropic
        self._c = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=180.0, max_retries=2)
        self.model_name = model; self.last_usage_metadata = {}; self.last_model = None
        self.temperature = 0
        self.price_key = "claude-sonnet-5" if "sonnet" in model else \
                         "claude-opus-4-8" if "opus" in model else "claude-haiku-4-5"
    def generate(self, prompt):
        import time as _t
        kw = {"model": self.model_name, "max_tokens": 8192,
              "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if not self.model_name.startswith("claude-haiku"):
            kw["thinking"] = {"type": "disabled"}
        last = None
        for attempt in range(6):
            try:
                r = self._c.messages.create(**kw)
                self.last_model = getattr(r, "model", None)
                u = getattr(r, "usage", None)
                if u is not None:
                    self.last_usage_metadata = {"prompt_token_count": getattr(u, "input_tokens", 0),
                                                "candidates_token_count": getattr(u, "output_tokens", 0)}
                return "".join(b.text for b in getattr(r, "content", []) if getattr(b, "type", None) == "text")
            except Exception as e:
                msg = str(e).lower()
                if "temperature" in msg and "temperature" in kw:   # model rejects temperature -> drop it, retry now
                    kw.pop("temperature"); self.temperature = None; continue
                last = e; _t.sleep(5 * (attempt + 1))
        raise last


def client_for(model):
    if model.startswith("claude"):
        return RawClaudeJSON(model)
    if model.startswith(("gemini", "gemma")):   # gemma is served on the Google API
        return make_client(model)
    from oai_client import OAIClient   # open-weight models via Together (OpenAI-compatible)
    return OAIClient(model)


def call(client, prompt):
    t=time.perf_counter(); raw=client.generate(prompt); dt=time.perf_counter()-t
    u=client.last_usage_metadata or {}
    it=u.get("prompt_token_count") or 0; ot=u.get("candidates_token_count") or 0
    cost=usage_cost_usd(getattr(client, "price_key", MODEL), u) or 0.0
    try: obj=extract_json_object(raw)
    except Exception: obj={}
    return obj,{"lat":dt,"in":it,"out":ot,"cost":cost}


def p_match(prop, call_txt, concept_block=""):
    return f"""You are matching a research PROPOSAL to a FUNDING CALL / company wishlist. Judge
how well the proposal's research fits the call's stated areas of interest. Return JSON only.
Schema: {{"match_score": <integer 0-100>, "decision": "match"|"no_match", "rationale": "<=2 sentences"}}
{concept_block}
PROPOSAL:
{prop[:PROP_CAP]}

FUNDING CALL:
{call_txt[:CALL_CAP]}
"""

def p_concept(prop):
    return f"""You are a research-intent abstraction agent. Identify the LATENT scientific
intent of this proposal -- its core therapeutic area, biological target/mechanism, and
modality/technology -- not just surface keywords. If uncertain, give one alternative framing.
Return JSON only.
Schema: {{"concept":"<core intent>","reasoning":"<1-2 sentences>","alternative":"<competing framing or empty>"}}
PROPOSAL:
{prop[:PROP_CAP]}
"""

def p_surface(prop):
    return f"""You are the SURFACE/LEXICAL agent. State what the proposal LITERALLY is about:
key terms, therapeutic area, target/modality, and any explicit constraints. Do NOT abstract to
deeper principles. Return JSON only.
Schema: {{"surface_intent":"<one sentence>","key_terms":["..."],"therapeutic_area":"<...>","modality":"<...>"}}
PROPOSAL:
{prop[:PROP_CAP]}
"""

def p_latent(prop):
    return f"""You are the LATENT CONCEPT agent. Identify the underlying scientific intent the
proposal really targets (disease biology, target class, mechanism, modality), not surface
wording. Alternatives are first-class. Return JSON only.
Schema: {{"primary_concept":"<...>","reasoning":"<1-2 sentences>","alternatives":[{{"concept":"<...>","confidence":"high|medium|low"}}]}}
PROPOSAL:
{prop[:PROP_CAP]}
"""

def p_coord(prop, call_txt, anchor, surface, latent):
    return f"""You are the COORDINATOR in an ANCHOR-AND-EDIT matcher. A single-pass baseline has
ALREADY judged this proposal-call pair (the ANCHOR). Your DEFAULT is to PRESERVE the baseline.
Only EDIT if the surface/latent interpretations reveal sufficiently STRONG additional evidence
that the baseline clearly mis-scored the fit. Do NOT regenerate the judgement from scratch and
do NOT edit on weak or stylistic grounds.

BASELINE (anchor): score={anchor['match_score']}, decision={anchor['decision']}, rationale={anchor.get('rationale','')}
SURFACE interpretation: {json.dumps(surface, ensure_ascii=False)}
LATENT concept: {json.dumps(latent, ensure_ascii=False)}

Return JSON only.
Schema: {{"action":"keep"|"edit","final_score":<integer 0-100>,"final_decision":"match"|"no_match","rationale":"<=2 sentences: if edit, cite the specific evidence>"}}

PROPOSAL:
{prop[:PROP_CAP]}

FUNDING CALL:
{call_txt[:CALL_CAP]}
"""


def norm_match(o):
    try: s=int(round(float(o.get("match_score",0))))
    except Exception: s=0
    s=max(0,min(100,s)); d=o.get("decision","no_match")
    d="match" if str(d).lower().startswith("match") else "no_match"
    return {"match_score":s,"decision":d,"rationale":(o.get("rationale","") or "")[:300]}


def main():
    client=make_client(MODEL)
    props=DOCS["proposals"]; calls=DOCS["calls"]
    pnames=list(props); cnames=list(calls)
    # per-proposal cached artifacts (concept for B; surface+latent for D)
    pre={}
    meter={s:{"lat":[],"in":[],"out":[],"cost":[],"calls":0} for s in ["A","B","D"]}
    def add(sys,m): meter[sys]["lat"].append(m["lat"]);meter[sys]["in"].append(m["in"]);meter[sys]["out"].append(m["out"]);meter[sys]["cost"].append(m["cost"]);meter[sys]["calls"]+=1
    for pn in pnames:
        pt=props[pn]["text"]
        c,mc=call(client,p_concept(pt)); add("B",mc)
        s,ms=call(client,p_surface(pt)); add("D",ms)
        l,ml=call(client,p_latent(pt)); add("D",ml)
        pre[pn]={"concept":c,"surface":s,"latent":l}
    pairs=[]
    for pn in pnames:
        pt=props[pn]["text"]
        for cn in cnames:
            ct=calls[cn]["text"]
            # A
            a_raw,am=call(client,p_match(pt,ct)); add("A",am); A=norm_match(a_raw)
            # B (concept-guided)
            cc=pre[pn]["concept"]
            cb=f"""\nProposal research-intent hypothesis (treat as a hint; judge on the actual texts):
  concept: {cc.get('concept','')}
  reasoning: {cc.get('reasoning','')}
  alternative: {cc.get('alternative','')}\n"""
            b_raw,bm=call(client,p_match(pt,ct,cb)); add("B",bm); B=norm_match(b_raw)
            # D (anchor-and-edit): anchor = A; coordinator decides keep/edit
            co_raw,dm=call(client,p_coord(pt,ct,A,pre[pn]["surface"],pre[pn]["latent"])); add("D",dm)
            action=str(co_raw.get("action","keep")).lower()
            action="edit" if action=="edit" else "keep"
            if action=="keep":
                D={"match_score":A["match_score"],"decision":A["decision"],"rationale":"(kept baseline)"}
            else:
                D=norm_match(co_raw)
            b_changed = (B["decision"]!=A["decision"]) or (abs(B["match_score"]-A["match_score"])>=10)
            d_changed = (action=="edit") and ((D["decision"]!=A["decision"]) or (D["match_score"]!=A["match_score"]))
            pairs.append({"proposal":pn,"call":cn,
                "A":A,"B":B,"D":{**D,"action":action},
                "B_changed_vs_baseline":b_changed,"D_changed_vs_baseline":d_changed,
                "D_decision_flip": action=="edit" and D["decision"]!=A["decision"],
                "concept":cc.get("concept",""),
                "pair_cost_usd":round(am["cost"]+bm["cost"]+dm["cost"],6),
                "pair_latency_s":round(am["lat"]+bm["lat"]+dm["lat"],2),
                "pair_tokens":{"in":am["in"]+bm["in"]+dm["in"],"out":am["out"]+bm["out"]+dm["out"]}})
    (OUT/"pairs.json").write_text(json.dumps(pairs,indent=1,ensure_ascii=False))

    # 6x6 matrices
    def matrix(getter):
        lines=["proposal\\call,"+",".join(cnames)]
        for pn in pnames:
            row=[pn]+[str(getter(pn,cn)) for cn in cnames]; lines.append(",".join(row))
        return "\n".join(lines)
    idx={(p["proposal"],p["call"]):p for p in pairs}
    (OUT/"matrix_A_score.csv").write_text(matrix(lambda p,c: idx[(p,c)]["A"]["match_score"]))
    (OUT/"matrix_B_score.csv").write_text(matrix(lambda p,c: idx[(p,c)]["B"]["match_score"]))
    (OUT/"matrix_D_score.csv").write_text(matrix(lambda p,c: idx[(p,c)]["D"]["match_score"]))
    (OUT/"matrix_D_action.csv").write_text(matrix(lambda p,c: idx[(p,c)]["D"]["action"]))

    # human review table (markdown) for James to label later
    md=["# James proposal--funding-call matching — exploratory (UNLABELED). LLM outputs are NOT gold.",
        "",
        "Score = LLM match_score (0-100); dec = match/no_match; D.act = keep/edit. `label` column is blank for James.",
        "",
        "| Proposal | Call | A score/dec | B score/dec | D act | D score/dec | Bchg | Dchg | James label |",
        "|---|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        md.append(f"| {p['proposal']} | {p['call']} | {p['A']['match_score']}/{p['A']['decision']} "
                  f"| {p['B']['match_score']}/{p['B']['decision']} | {p['D']['action']} "
                  f"| {p['D']['match_score']}/{p['D']['decision']} | {'Y' if p['B_changed_vs_baseline'] else ''} "
                  f"| {'Y' if p['D_changed_vs_baseline'] else ''} |  |")
    (OUT/"review_table.md").write_text("\n".join(md))

    # cost/behaviour summary (NO accuracy metrics)
    def agg(s):
        m=meter[s]; return {"calls_total":m["calls"],"cost_usd_total":round(sum(m["cost"]),4),
            "latency_s_total":round(sum(m["lat"]),1),"tokens_in_total":sum(m["in"]),"tokens_out_total":sum(m["out"]),
            "cost_usd_per_pair":round(sum(m["cost"])/36,6),"latency_s_per_pair":round(sum(m["lat"])/36,2)}
    nB=sum(1 for p in pairs if p["B_changed_vs_baseline"]); nD=sum(1 for p in pairs if p["D_changed_vs_baseline"])
    nEdit=sum(1 for p in pairs if p["D"]["action"]=="edit"); nFlip=sum(1 for p in pairs if p["D_decision_flip"])
    summary={"n_pairs":36,"model":MODEL,"note":"exploratory unlabeled; LLM outputs are NOT gold; no ground-truth metrics computed",
        "behaviour":{"B_changed_vs_baseline":nB,"D_edited":nEdit,"D_changed_vs_baseline":nD,"D_decision_flips":nFlip,
                     "D_kept_baseline":36-nEdit},
        "cost":{s:agg(s) for s in ["A","B","D"]}}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=1))
    print("=== James matching (exploratory, UNLABELED, n=36 pairs) ===")
    print(f"  B changed baseline: {nB}/36 | D edited: {nEdit}/36 (kept {36-nEdit}) | D decision flips: {nFlip}/36")
    print("  cost/behaviour:")
    for s in ["A","B","D"]:
        a=agg(s); print(f"    {s}: calls={a['calls_total']:>3} cost=${a['cost_usd_total']:.4f} lat={a['latency_s_total']:.0f}s in/out tok={a['tokens_in_total']}/{a['tokens_out_total']}")
    print(f"  wrote pairs.json, matrix_*.csv, review_table.md, summary.json to {OUT}")


if __name__ == "__main__":
    main()
