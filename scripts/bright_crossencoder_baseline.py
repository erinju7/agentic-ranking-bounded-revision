"""Cross-encoder reranker baseline on the BRIGHT pools (same 100-candidate pools
as A/B/C/D). Uses a small MS MARCO cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
as a cheap trained neural reranker reference. Reports Hit@1 and MRR. CPU-only."""
import json
from pathlib import Path
from sentence_transformers import CrossEncoder
ROOT = Path(__file__).resolve().parents[1]
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
def run(model, domain):
    P = json.loads((ROOT / "data" / "bright_hardpool" / domain / "pools.json").read_text())
    hit1 = 0; rr = 0.0; n = 0
    for qid, pl in P.items():
        gold = set(pl["gold_aliases"]); items = list(pl["alias_to_text"].items())
        pairs = [(pl["query"], t[:2000]) for _, t in items]
        scores = model.predict(pairs, batch_size=64, show_progress_bar=False)
        ranked = [a for (a, _), s in sorted(zip(items, scores), key=lambda x: -x[1])]
        n += 1
        if ranked[0] in gold: hit1 += 1
        rank = next((i for i, a in enumerate(ranked, 1) if a in gold), None)
        rr += (1 / rank) if rank else 0
    print(f"{domain:20s} n={n}  cross-encoder Hit@1={hit1/n:.3f}  MRR={rr/n:.3f}", flush=True)
if __name__ == "__main__":
    m = CrossEncoder(MODEL, max_length=512)
    for d in ["biology", "earth_science", "psychology", "sustainable_living"]:
        try: run(m, d)
        except Exception as e: print(d, "skip:", e)
