"""Compute-matched control for the anchor-and-edit (System D) finding.
BRIGHT Biology, pool 100, Haiku primary backbone, temperature 0, k reps.

Design: A-SC@m self-consistency. Per query, run the SAME plain single-pass
reranker (p_rerank_plain, identical to arm A) m times, each over an
independently shuffled candidate presentation order (temperature stays 0, so
diversity comes only from input order -- no new sampling channel). Aggregate the
m rankings by mean rank (Borda); the consensus top-1 is the prediction. No
concept/surface/anchor scaffolding: any gain over single-pass A is from extra
compute, not from D's architecture.

This gives D roughly the same call count (m=4 vs D's 4) with a larger token
budget (4 heavy body-calls vs D's 2 heavy + 2 cheap), so the control is
conservative -- it spends at least as much as D without the structure. Compared
against the saved A and full-D per-query outputs in
results/bright_ablation/<MODEL>/rep_*/per_query.json (matched queries).

Output: results/bright_selfconsistency/<MODEL>/rep_<k>/{scref.json,per_query.json}
Usage: python bright_selfconsistency_haiku.py --reps 5 --samples 4
"""
from __future__ import annotations
import os, sys, json, math, time, random, argparse, statistics as st
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bright_backbone import POOLS, p_rerank_plain, client_for, ROOT
from rq2_core import extract_json_object

# load .env-style key file the same way the ablation runner does (ROOT = main_experiment)
for _e in (ROOT / ".env", ROOT.parent / ".env"):
    if _e.exists():
        for _l in _e.read_text().splitlines():
            _l = _l.strip()
            if _l and not _l.startswith("#") and "=" in _l:
                _k, _v = _l.split("=", 1); os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

MODEL = "claude-haiku-4-5-20251001"; PRICE = (1.00, 5.00)  # $/Mtok in,out


def ndcg(r, k):
    def dcg(x): return sum(v / math.log2(i + 2) for i, v in enumerate(x))
    d = dcg(r[:k]); ide = dcg(sorted(r, reverse=True)[:k]); return d / ide if ide else 0.0


def mcnemar(f, b):
    n = f + b
    if n == 0: return 1.0
    k = min(f, b); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def rank_from_ids(ids, order):
    out = []
    for x in ids:
        v = str(x.get("id") if isinstance(x, dict) else x)
        if v in order and v not in out: out.append(v)
    for a in order:
        if a not in out: out.append(a)
    return out


def aggregate_meanrank(rankings, order):
    """Borda / mean-rank consensus over m rankings. Missing id -> penalty rank N+1.
    Tie-break: lower mean rank, then more first-place votes, then id."""
    N = len(order)
    pos = {a: [] for a in order}
    for r in rankings:
        rp = {a: i + 1 for i, a in enumerate(r)}
        for a in order:
            pos[a].append(rp.get(a, N + 1))
    mean = {a: sum(pos[a]) / len(pos[a]) for a in order}
    top1c = {a: sum(1 for r in rankings if r and r[0] == a) for a in order}
    return sorted(order, key=lambda a: (mean[a], -top1c[a], a))


def run_rep(cl, rep, samples, outdir):
    acc = {"cost": 0.0, "lat": 0.0, "calls": 0, "tin": 0, "tout": 0}
    per = []

    def call(prompt):
        t = time.perf_counter(); raw = cl.generate(prompt); dt = time.perf_counter() - t
        u = cl.last_usage_metadata or {}
        acc["cost"] += (u.get("prompt_token_count", 0) * PRICE[0] + u.get("candidates_token_count", 0) * PRICE[1]) / 1e6
        acc["lat"] += dt; acc["calls"] += 1
        acc["tin"] += u.get("prompt_token_count", 0); acc["tout"] += u.get("candidates_token_count", 0)
        try: o = extract_json_object(raw)
        except Exception: o = {}
        return o, raw

    for i, (qid, pl) in enumerate(POOLS.items(), 1):
        q = pl["query"]; a2t = pl["alias_to_text"]; goldset = set(pl["gold_aliases"])
        canonical = list(a2t)
        rankings = []
        for s in range(samples):
            shuf = list(canonical)
            random.Random(f"{rep}:{qid}:{s}").shuffle(shuf)
            body = [{"id": a, "text": a2t[a]} for a in shuf]
            o, _ = call(p_rerank_plain(q, body))
            rankings.append(rank_from_ids(o.get("ranked_ids", []), shuf))
        consensus = aggregate_meanrank(rankings, canonical)
        rels = [1 if x in goldset else 0 for x in consensus]
        first = next((j + 1 for j, x in enumerate(consensus) if x in goldset), len(consensus) + 1)
        per.append({"id": qid, "gold": list(goldset), "ranked": consensus[:20],
                    "top1": consensus[0], "top1_gold": rels[0] == 1, "first_gold": first,
                    "mrr": 1.0 / first, "ndcg10": ndcg(rels, 10),
                    "sample_top1": [r[0] for r in rankings]})
        print(f"[rep{rep} {i}/{len(POOLS)}] {qid} top1_gold={rels[0]==1}", flush=True)

    summ = {m: round(st.mean(x[k] for x in per), 4)
            for m, k in [("Hit@1", "top1_gold"), ("MRR", "mrr"), ("nDCG@10", "ndcg10")]}
    out = {"model": MODEL, "resolved_model": getattr(cl, "last_model", None), "arm": f"A-SC@{samples}",
           "temperature": 0, "pool": 100, "domain": "biology", "samples": samples, "n": len(per),
           "rep": rep, "run_utc": datetime.now(timezone.utc).isoformat(),
           "metrics": summ, "cost": {**acc, "cost": round(acc["cost"], 4)}}
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outdir / "scref.json", "w"), indent=1)
    json.dump(per, open(outdir / "per_query.json", "w"), indent=1, ensure_ascii=False)
    print(f"  rep{rep}: Hit@1={summ['Hit@1']} MRR={summ['MRR']} "
          f"nDCG@10={summ['nDCG@10']} cost=${out['cost']['cost']} calls={acc['calls']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()
    cl = client_for(MODEL)
    base = ROOT / "results" / "bright_selfconsistency" / MODEL
    for k in range(1, args.reps + 1):
        od = base / f"rep_{k}"
        if (od / "scref.json").exists():
            print(f"rep{k}: exists, skip"); continue
        run_rep(cl, k, args.samples, od)
    print("DONE self-consistency control")


if __name__ == "__main__":
    main()
