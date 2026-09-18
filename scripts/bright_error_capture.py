"""Study 1 error analysis capture: rerun A+D on BRIGHT biology (pinned haiku, pool 100, temp 0)
capturing, per query, what A picked, what D picked, the gold, D's decision/rationale/promotes,
and the ranks needed to categorize each discordant/failure case. Enables an error taxonomy that
the stored correctness-only artifacts cannot support. Output:
results/bright_ablation/claude-haiku-4-5-20251001/error_capture.json
"""
from __future__ import annotations
import os, sys, json, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bright_backbone import POOLS, p_rerank_plain, p_a1, p_a2, p_a3_anchor, rank_from_ids, client_for, ROOT
from rq2_core import extract_json_object
for _e in (ROOT/".env", ROOT.parent/".env"):
    if _e.exists():
        for _l in _e.read_text().splitlines():
            _l=_l.strip()
            if _l and not _l.startswith("#") and "=" in _l:
                _k,_v=_l.split("=",1); os.environ.setdefault(_k.strip(),_v.strip().strip('"').strip("'"))

MODEL="claude-haiku-4-5-20251001"
OUT=ROOT/"results"/"bright_ablation"/MODEL; OUT.mkdir(parents=True,exist_ok=True)
cl=client_for(MODEL)
def gen(p):
    raw=cl.generate(p)
    try: return extract_json_object(raw)
    except Exception: return {}

def rankpos(ranked, goldset):
    return next((i+1 for i,x in enumerate(ranked) if x in goldset), len(ranked)+1)

rows=[]
for i,(qid,pl) in enumerate(POOLS.items(),1):
    q=pl["query"]; a2t=pl["alias_to_text"]; goldset=set(pl["gold_aliases"]); order=list(a2t)
    body=[{"id":a,"text":a2t[a]} for a in order]
    A=rank_from_ids(gen(p_rerank_plain(q,body)).get("ranked_ids",[]),order)
    h1=gen(p_a1(q)); h2=gen(p_a2(q))
    co=gen(p_a3_anchor(q,h1,h2,body))
    dec=co.get("decision") or "surface_sufficient"
    prom=[str(x.get('id') if isinstance(x,dict) else x) for x in (co.get("promote_ids") or [])]
    prom=[a for a in prom if a in a2t][:2]
    D=list(A) if (dec=="surface_sufficient" or not prom) else prom+[a for a in A if a not in prom]
    a1=A[0] in goldset; d1=D[0] in goldset
    outcome=("fix" if (not a1 and d1) else "break" if (a1 and not d1) else "kept_correct" if a1 else "kept_wrong")
    rows.append({"id":qid,"outcome":outcome,"decision":dec,
                 "A_top1":A[0],"D_top1":D[0],"promote_ids":prom,
                 "gold":list(goldset),"gold_rank_A":rankpos(A,goldset),"gold_rank_D":rankpos(D,goldset),
                 "A_top1_gold":a1,"D_top1_gold":d1,
                 "rationale":(co.get("rationale") or "")[:400],
                 "concept":(h2.get("primary_concept") if isinstance(h2,dict) else "")})
    print(f"[{i}/{len(POOLS)}] {qid} {outcome} dec={dec}",flush=True)

json.dump({"model":MODEL,"resolved_model":getattr(cl,"last_model",None),"temperature":0,
           "run_utc":datetime.now(timezone.utc).isoformat(),"rows":rows}, open(OUT/"error_capture.json","w"),indent=1)
from collections import Counter
print("\noutcomes:",dict(Counter(r["outcome"] for r in rows)))
print("DONE")
