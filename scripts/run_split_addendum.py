"""ADDENDUM (API): run System A + System D at the reference backbone (gemini-flash-latest, t=0) on the
previously-excluded (>6-gold) queries built by build_split_pools_addendum.py. Identical prompts and
assembly to run_split_AD.py. Per-call cache (separate dir), global $10 HARD-ABORT. Writes
A_per_query_addendum.json / D_per_query_addendum.json per split. No frozen artefact is modified.
"""
import json, time, hashlib
from pathlib import Path
from rq2_core import make_client, extract_json_object, usage_cost_usd
from bright_backbone import p_rerank_plain, p_a1, p_a2, p_a3_anchor, rank_from_ids

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["earth_science", "sustainable_living", "psychology"]
MODEL = "gemini-flash-latest"; COST_CAP = 10.0
ledger = {"cost": 0.0, "calls": 0}
client = make_client(MODEL)

def call(split, system, agent, qid, prompt):
    cache = ROOT/"runs"/"splits"/split/"cache_addendum"; cache.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    cf = cache/f"{system}_{agent}_{qid}_{h}.json"
    if cf.exists():
        d = json.loads(cf.read_text())
    else:
        t = time.perf_counter(); raw = client.generate(prompt); dt = time.perf_counter()-t
        u = client.last_usage_metadata or {}
        d = {"raw": raw, "in": u.get("prompt_token_count") or 0, "out": u.get("candidates_token_count") or 0,
             "cost": usage_cost_usd(MODEL, u) or 0.0, "lat": dt}
        cf.write_text(json.dumps(d)); ledger["cost"] += d["cost"]; ledger["calls"] += 1
        if ledger["cost"] > COST_CAP: raise RuntimeError(f"COST HARD-ABORT ${ledger['cost']:.2f}")
    try: obj = extract_json_object(d["raw"])
    except Exception: obj = {}
    return obj, d

for split in SPLITS:
    pools = json.loads((ROOT/"runs"/"splits"/split/"pools_addendum.json").read_text())
    perA = []; perD = []; decisions = {}
    for qid, pl in pools.items():
        q = pl["query"]; a2t = pl["alias_to_text"]; goldset = set(pl["gold_aliases"]); order = list(a2t)
        body = [{"id": a, "text": a2t[a]} for a in order]
        objA, _ = call(split, "A", "rerank", qid, p_rerank_plain(q, body))
        rA = rank_from_ids(objA.get("ranked_ids", []), order)
        firstA = next((i+1 for i, a in enumerate(rA) if a in goldset), len(rA)+1)
        h1, _ = call(split, "D", "a1", qid, p_a1(q)); h2, _ = call(split, "D", "a2", qid, p_a2(q))
        co, _ = call(split, "D", "a3", qid, p_a3_anchor(q, h1, h2, body))
        dec = (co.get("decision") or "surface_sufficient"); decisions[dec] = decisions.get(dec, 0)+1
        prom = [str(x.get("id") if isinstance(x, dict) else x) for x in (co.get("promote_ids") or [])]
        prom = [a for a in prom if a in a2t][:2]
        rD = list(rA) if (dec == "surface_sufficient" or not prom) else prom + [a for a in rA if a not in prom]
        firstD = next((i+1 for i, a in enumerate(rD) if a in goldset), len(rD)+1)
        perA.append({"id": qid, "first": firstA, "top1": (rA[0] in goldset) if rA else False,
                     "n_golds": pl.get("n_golds_in_corpus")})
        perD.append({"id": qid, "first": firstD, "top1": (rD[0] in goldset) if rD else False, "decision": dec})
    (ROOT/"runs"/"splits"/split/"A_per_query_addendum.json").write_text(json.dumps(perA, indent=1))
    (ROOT/"runs"/"splits"/split/"D_per_query_addendum.json").write_text(json.dumps(perD, indent=1))
    a1 = sum(x["top1"] for x in perA)/len(perA); d1 = sum(x["top1"] for x in perD)/len(perD)
    print(f"{split:18} addendum n={len(perA)}  A R@1={a1:.3f}  D R@1={d1:.3f}  decisions={decisions}")

(ROOT/"runs"/"splits"/"addendum_ledger.json").write_text(json.dumps(ledger, indent=1))
print(f"\nTOTAL live spend = ${ledger['cost']:.4f} over {ledger['calls']} calls (cached free).")
