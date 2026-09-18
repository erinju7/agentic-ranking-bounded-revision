"""Step 6a (NO API): build frozen 100-doc pools for the pre-registered splits, identical
construction/seeds to biology (TF-IDF hard negatives, gold always inserted, MAX_GOLD=6,
600-char truncation, opaque D00.. aliases, seeds 42/7). Also fixes the 30-query gate sample.
Prints eligible counts + a projected API cost so Step 6b can respect the $10 hard-abort.
"""
import json, math, random, urllib.request
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT=Path(__file__).resolve().parents[1]
SPLITS=["earth_science","sustainable_living","psychology"]
POOL,MAX_GOLD,DOC_CAP=100,6,600
BASE="https://huggingface.co/datasets/xlangai/BRIGHT/resolve/refs%2Fconvert%2Fparquet"

def load_config(split, cfg, cache_dir):
    """Download + concatenate all parquet shards for documents/examples of a split."""
    frames=[]; i=0
    while True:
        url=f"{BASE}/{cfg}/{split}/{i:04d}.parquet"; p=cache_dir/f"{cfg}_{i:04d}.parquet"
        if not p.exists():
            try: urllib.request.urlretrieve(url,p)
            except Exception: break
        try: frames.append(pd.read_parquet(p))
        except Exception: break
        i+=1
        if i>50: break
    if not frames: raise RuntimeError(f"no parquet for {cfg}/{split}")
    return pd.concat(frames, ignore_index=True)

summary={}
for split in SPLITS:
    data_dir=ROOT/"data"/"bright_splits"/split; data_dir.mkdir(parents=True,exist_ok=True)
    out_dir=ROOT/"runs"/"splits"/split; out_dir.mkdir(parents=True,exist_ok=True)
    docs=load_config(split,"documents",data_dir); ex=load_config(split,"examples",data_dir)
    id2doc=dict(zip(docs["id"].astype(str), docs["content"].astype(str)))
    corpus_ids=list(id2doc); corpus_txt=[id2doc[i] for i in corpus_ids]
    exl=ex.to_dict("records")
    elig=[e for e in exl if 1<=len([g for g in list(e["gold_ids"]) if g in id2doc])<=MAX_GOLD]
    elig.sort(key=lambda e: str(e["id"])); random.Random(42).shuffle(elig)
    N=len(elig)
    vec=TfidfVectorizer(stop_words="english",max_features=50000); X=vec.fit_transform(corpus_txt)
    rng=random.Random(7)
    pools={}
    for e in elig:
        q=e["query"]; gold=[g for g in list(e["gold_ids"]) if g in id2doc][:MAX_GOLD]
        excl=set(list(e.get("excluded_ids") or []))|set(gold)
        sims=linear_kernel(vec.transform([q]),X).ravel(); order=sims.argsort()[::-1]
        distract=[]
        for row in order:
            cid=corpus_ids[row]
            if cid not in excl: distract.append(cid)
            if len(gold)+len(distract)>=POOL: break
        pool=gold+distract; rng.shuffle(pool)
        alias={f"D{i:02d}":cid for i,cid in enumerate(pool)}
        pools[str(e["id"])]={"query":q,"gold_aliases":[a for a,cid in alias.items() if cid in gold],
                             "alias_to_text":{a:id2doc[cid][:DOC_CAP] for a,cid in alias.items()},
                             "gold_answer":e.get("gold_answer","")}
    (out_dir/"pools.json").write_text(json.dumps(pools,ensure_ascii=False))
    # gate sample = first 30 of the seeded shuffle
    gate_ids=[str(e["id"]) for e in elig[:30]]
    (out_dir/"gate_sample.json").write_text(json.dumps(gate_ids))
    # projected cost: gate 30 short calls + A (N calls, 100-pool) + D (3N: A1,A2 short + A3 100-pool)
    est = 30*0.0006 + N*0.0043 + N*(0.0006+0.0006+0.0043)
    summary[split]={"corpus":len(corpus_ids),"examples":len(exl),"eligible_N":N,"proj_cost_usd":round(est,3)}
    print(f"{split:<20} corpus={len(corpus_ids):>6} examples={len(exl):>4} eligible_N={N:>4} proj_cost=${est:.2f}")

tot=sum(v["proj_cost_usd"] for v in summary.values())
(ROOT/"runs"/"splits"/"build_summary.json").write_text(json.dumps(summary,indent=1))
print(f"\nPROJECTED TOTAL (3 splits, A+D+gate at reference backbone) = ${tot:.2f}")
print("UNDER $10 hard-abort -> OK to proceed" if tot<10 else "OVER $10 -> ABORT, report")
