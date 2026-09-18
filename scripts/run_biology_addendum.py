"""ADDENDUM (API): build pools for + run A+D on the biology queries the MAX_GOLD=6 cap excluded
(>6 in-corpus golds; 6 queries). Construction is byte-identical to bright_hardpool_baseline.py
(POOL=100, insert <=6 golds, TF-IDF hard negatives, 600-char truncation, opaque aliases); prompts are
bright_backbone.p_rerank_plain / p_a1 / p_a2 / p_a3_anchor (rendered byte-identical to the frozen
biology A and bright_ch_anchor D). Independent RNG; frozen pools.json untouched. Separate cache.
Writes results/bright_hardpool/biology/addendum_per_query.json (A+D fields for the 6 queries).
"""
import json, time, hashlib, random
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rq2_core import make_client, extract_json_object, usage_cost_usd
from bright_backbone import p_rerank_plain, p_a1, p_a2, p_a3_anchor, rank_from_ids

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"data"/"bright_mvp"/"biology"
POOL, MAX_GOLD, DOC_CAP = 100, 6, 600
MODEL = "gemini-flash-latest"; COST_CAP = 10.0
ledger = {"cost": 0.0, "calls": 0}
client = make_client(MODEL)
CACHE = ROOT/"runs"/"biology_addendum_cache"; CACHE.mkdir(parents=True, exist_ok=True)

def call(system, agent, qid, prompt):
    h = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    cf = CACHE/f"{system}_{agent}_{qid}_{h}.json"
    if cf.exists():
        d = json.loads(cf.read_text())
    else:
        t = time.perf_counter(); raw = client.generate(prompt); dt = time.perf_counter()-t
        u = client.last_usage_metadata or {}
        d = {"raw": raw, "cost": usage_cost_usd(MODEL, u) or 0.0}
        cf.write_text(json.dumps(d)); ledger["cost"] += d["cost"]; ledger["calls"] += 1
        if ledger["cost"] > COST_CAP: raise RuntimeError(f"COST ABORT ${ledger['cost']:.2f}")
    try: obj = extract_json_object(d["raw"])
    except Exception: obj = {}
    return obj

docs = pd.read_parquet(SRC/"documents.parquet"); ex = pd.read_parquet(SRC/"examples.parquet")
id2doc = dict(zip(docs["id"].astype(str), docs["content"].astype(str)))
corpus_ids = list(id2doc); corpus_txt = [id2doc[i] for i in corpus_ids]
exl = ex.to_dict("records")
excl_ex = [e for e in exl if len([g for g in list(e["gold_ids"]) if g in id2doc]) > MAX_GOLD]
excl_ex.sort(key=lambda e: str(e["id"])); random.Random(42).shuffle(excl_ex)
vec = TfidfVectorizer(stop_words="english", max_features=50000); X = vec.fit_transform(corpus_txt)
rng = random.Random(7)
per = []
for e in excl_ex:
    q = e["query"]
    gold = [g for g in list(e["gold_ids"]) if g in id2doc][:MAX_GOLD]
    excl = set(list(e.get("excluded_ids") or [])) | set(gold)
    sims = linear_kernel(vec.transform([q]), X).ravel(); order = sims.argsort()[::-1]
    distract = []
    for row in order:
        cid = corpus_ids[row]
        if cid not in excl: distract.append(cid)
        if len(gold) + len(distract) >= POOL: break
    pool = gold + distract; rng.shuffle(pool)
    alias = {f"D{i:02d}": cid for i, cid in enumerate(pool)}
    a2t = {a: id2doc[cid][:DOC_CAP] for a, cid in alias.items()}
    goldset = {a for a, cid in alias.items() if cid in gold}
    orderA = list(a2t); body = [{"id": a, "text": a2t[a]} for a in orderA]
    qid = str(e["id"])
    objA = call("A", "rerank", qid, p_rerank_plain(q, body))
    rA = rank_from_ids(objA.get("ranked_ids", []), orderA)
    firstA = next((i+1 for i, a in enumerate(rA) if a in goldset), len(rA)+1)
    h1 = call("D", "a1", qid, p_a1(q)); h2 = call("D", "a2", qid, p_a2(q))
    co = call("D", "a3", qid, p_a3_anchor(q, h1, h2, body))
    dec = co.get("decision") or "surface_sufficient"
    prom = [str(x.get("id") if isinstance(x, dict) else x) for x in (co.get("promote_ids") or [])]
    prom = [a for a in prom if a in a2t][:2]
    rD = list(rA) if (dec == "surface_sufficient" or not prom) else prom + [a for a in rA if a not in prom]
    firstD = next((i+1 for i, a in enumerate(rD) if a in goldset), len(rD)+1)
    per.append({"id": qid, "n_golds_in_corpus": len([g for g in list(e["gold_ids"]) if g in id2doc]),
                "baseline_first_gold_rank": firstA, "baseline_top1_gold": (rA[0] in goldset) if rA else False,
                "arch_first_gold_rank": firstD, "arch_top1_gold": (rD[0] in goldset) if rD else False,
                "decision": dec})
(ROOT/"results"/"bright_hardpool"/"biology"/"addendum_per_query.json").write_text(json.dumps(per, indent=1))
print(f"biology addendum n={len(per)}  A R@1={sum(x['baseline_top1_gold'] for x in per)/len(per):.3f}  "
      f"D R@1={sum(x['arch_top1_gold'] for x in per)/len(per):.3f}")
print(f"spend ${ledger['cost']:.4f} / {ledger['calls']} calls")
