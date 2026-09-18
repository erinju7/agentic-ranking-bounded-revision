"""Combine retained + addendum per-query outputs for the three new splits and recompute the full-set
(no-MAX_GOLD-filter) statistics: R@1, MRR, fixed/broke, exact McNemar, paired-bootstrap MRR CI and
sign-flip permutation. Also recompute the gold-count vs baseline-hit relationship on the full set.
NO API. Prints a comparison of retained-only vs full-set so we can see whether the ordering / gains
survive lifting the eligibility cap.
"""
import json, math, random, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["earth_science", "sustainable_living", "psychology"]
BM25 = {"earth_science": 27.2, "sustainable_living": 15.0, "psychology": 12.5}

def mcnemar(f, b):
    n = f+b; k = min(f, b); return min(1.0, 2*sum(math.comb(n, i) for i in range(k+1))/(2**n)) if n else 1.0
def boot_perm(a, s, B=10000, seed=123):
    diffs = [si-ai for ai, si in zip(a, s)]; obs = st.mean(diffs); n = len(diffs)
    rng = random.Random(seed); means = [sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(B)]
    means.sort(); lo, hi = means[int(0.025*B)], means[int(0.975*B)]
    rng2 = random.Random(seed+1)
    c = sum(1 for _ in range(B) if abs(sum((d if rng2.random() < 0.5 else -d) for d in diffs)/n) >= abs(obs)-1e-12)
    return round(obs, 4), round(lo, 4), round(hi, 4), round((c+1)/(B+1), 4)

def load(split, add):
    suf = "_addendum" if add else ""
    A = {str(x["id"]): x for x in json.load(open(ROOT/"runs"/"splits"/split/f"A_per_query{suf}.json"))}
    D = {str(x["id"]): x for x in json.load(open(ROOT/"runs"/"splits"/split/f"D_per_query{suf}.json"))}
    return A, D

def stats(A, D):
    ids = list(A); n = len(ids)
    r1A = sum(A[q]["top1"] for q in ids)/n; r1D = sum(D[q]["top1"] for q in ids)/n
    mrrA = st.mean([1/A[q]["first"] for q in ids]); mrrD = st.mean([1/D[q]["first"] for q in ids])
    fixed = sum(1 for q in ids if (not A[q]["top1"]) and D[q]["top1"])
    broke = sum(1 for q in ids if A[q]["top1"] and (not D[q]["top1"]))
    a = [1/A[q]["first"] for q in ids]; s = [1/D[q]["first"] for q in ids]
    md, lo, hi, pp = boot_perm(a, s)
    return dict(n=n, A_r1=r1A, D_r1=r1D, dR1=r1D-r1A, mrrA=mrrA, mrrD=mrrD, fixed=fixed, broke=broke,
                mcn=mcnemar(fixed, broke), mrr_d=md, mrr_ci=(lo, hi), mrr_p=pp)

print(f"{'split':18} {'set':8} {'n':>3} {'A R@1':>6} {'D R@1':>6} {'dR@1':>7} {'fix/brk':>7} {'McN':>6} {'MRRd':>7} {'MRRp':>6}")
full_rows = {}
for split in SPLITS:
    rA, rD = load(split, False); aA, aD = load(split, True)
    ret = stats(rA, rD)
    full = stats({**rA, **aA}, {**rD, **aD})
    full_rows[split] = full
    for tag, r in [("retain", ret), ("FULL", full)]:
        print(f"{split:18} {tag:8} {r['n']:>3} {r['A_r1']:>6.3f} {r['D_r1']:>6.3f} {r['dR1']:>+7.3f} "
              f"{str(r['fixed'])+'/'+str(r['broke']):>7} {r['mcn']:>6.3f} {r['mrr_d']:>+7.3f} {r['mrr_p']:>6.3f}")

print("\n=== FULL-SET ordering check (baseline R@1 asc -> is gain monotone decreasing?) ===")
order = sorted(SPLITS, key=lambda s: full_rows[s]["A_r1"])
for s in order:
    print(f"  {s:18} baseline A R@1={full_rows[s]['A_r1']:.3f}  gain dR@1={full_rows[s]['dR1']:+.3f}  (BM25={BM25[s]})")
gains = [full_rows[s]["dR1"] for s in order]
mono = all(gains[i] > gains[i+1] for i in range(len(gains)-1))
print(f"  monotone-decreasing gain across 3 splits: {mono}")
print(f"  BM25 order (asc difficulty by nDCG@10): psychology(12.5)<sustainable(15.0)<earth(27.2)")
print(f"  our baseline order (asc): {' < '.join(order)}  -> inversion vs BM25 still holds: {order[0]=='earth_science' and order[-1]=='psychology'}")

# gold-count vs hit on FULL set (retained + addendum), pooled over 3 new splits
print("\n=== gold-count vs baseline hit (FULL 3-split set) ===")
rows = []
for split in SPLITS:
    rA, _ = load(split, False); aA, _ = load(split, True)
    pools = json.loads((ROOT/"runs"/"splits"/split/"pools.json").read_text())
    padd = json.loads((ROOT/"runs"/"splits"/split/"pools_addendum.json").read_text())
    for q, x in {**rA, **aA}.items():
        g = len(pools[q]["gold_aliases"]) if q in pools else len(padd[q]["gold_aliases"])
        ncorpus = padd[q].get("n_golds_in_corpus") if q in padd else g
        rows.append((g, ncorpus, 1 if x["top1"] else 0))
import statistics
gs = [r[0] for r in rows]; hs = [r[2] for r in rows]
mg = sum(gs)/len(gs); mh = sum(hs)/len(hs)
num = sum((g-mg)*(h-mh) for g, h in zip(gs, hs)); den = (sum((g-mg)**2 for g in gs)*sum((h-mh)**2 for h in hs))**0.5
print(f"  in-pool gold count vs hit: point-biserial r={num/den:+.3f} (n={len(rows)})")
add_only = [r for r in rows if r[1] and r[1] > 6]
if add_only:
    print(f"  previously-EXCLUDED (>6 corpus golds) baseline R@1 = {sum(r[2] for r in add_only)/len(add_only):.3f} (n={len(add_only)})")
    print(f"  retained (<=6) baseline R@1 = {sum(r[2] for r in rows if not (r[1] and r[1]>6))/sum(1 for r in rows if not (r[1] and r[1]>6)):.3f}")

(ROOT/"runs"/"splits"/"fullset_summary.json").write_text(json.dumps(
    {s: {k: (list(v) if isinstance(v, tuple) else v) for k, v in full_rows[s].items()} for s in SPLITS}, indent=1))
print("\nwrote runs/splits/fullset_summary.json")
