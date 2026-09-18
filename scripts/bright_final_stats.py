"""Final statistics + comparison table across the four frozen systems (A,B,C,D) on the
identical 100-doc pools. McNemar (exact) for R@1; paired bootstrap 95% CI + sign-flip
permutation test for MRR (per-query reciprocal rank). Primary analysis = complete frozen
set; NO samples removed, contestable golds retained. Run after bright_finalize_measure.py.
"""
from __future__ import annotations
import json, math, random, statistics as st
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
def load(p): return json.loads(Path(p).read_text())

# per-query first-gold-rank + top1, aligned by id
A = {str(x["id"]): x for x in load(R/"bright_hardpool"/"biology"/"baseline_per_query.json")}
B = {str(x["id"]): x for x in load(R/"bright_concept_hardpool"/"biology"/"per_query.json")}
C = {str(x["id"]): x for x in load(R/"bright_competing_hypotheses"/"biology"/"per_query.json")}
D = {str(x["id"]): x for x in load(R/"bright_ch_anchor"/"biology"/"per_query.json")}
ids = list(A)

def rank(sysd, i, base=False):
    x = sysd[i]
    return x["baseline_first_gold_rank"] if base else x["arch_first_gold_rank"]
def top1(sysd, i, base=False):
    x = sysd[i]
    return x["baseline_top1_gold"] if base else x["arch_top1_gold"]

def mcnemar_exact(fixed, broke):
    n = fixed + broke
    if n == 0: return 1.0
    k = min(fixed, broke)
    return min(1.0, 2*sum(math.comb(n, i) for i in range(k+1))/(2**n))

def paired_rr(sysd):
    a = [1.0/rank(A, i, base=True) for i in ids]
    s = [1.0/rank(sysd, i) for i in ids]
    return a, s

def bootstrap_perm(a, s, B_boot=10000, seed=123):
    diffs = [si - ai for ai, si in zip(a, s)]
    obs = st.mean(diffs); n = len(diffs)
    rng = random.Random(seed)
    # bootstrap CI
    means = []
    for _ in range(B_boot):
        samp = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(samp)/n)
    means.sort()
    lo, hi = means[int(0.025*B_boot)], means[int(0.975*B_boot)]
    # sign-flip permutation (two-sided)
    rng2 = random.Random(seed+1); cnt = 0
    for _ in range(B_boot):
        m = sum((d if rng2.random() < 0.5 else -d) for d in diffs)/n
        if abs(m) >= abs(obs) - 1e-12: cnt += 1
    return obs, lo, hi, (cnt+1)/(B_boot+1)

def r1_stats(sysd):
    fixed = sum(1 for i in ids if (not top1(A, i, base=True)) and top1(sysd, i))
    broke = sum(1 for i in ids if top1(A, i, base=True) and (not top1(sysd, i)))
    return fixed, broke, mcnemar_exact(fixed, broke)

# metric summaries
sA = load(R/"bright_hardpool"/"biology"/"baseline_summary.json")
sB = load(R/"bright_concept_hardpool"/"biology"/"summary.json")
sC = load(R/"bright_competing_hypotheses"/"biology"/"summary.json")["architecture"]
sD = load(R/"bright_ch_anchor"/"biology"/"summary.json")["architecture"]
cl = load(R/"bright_finalize"/"cost_profile.json")["systems"]

rows = {"A": sA, "B": sB, "C": sC, "D": sD}
def g(d, k, alt=None):
    return d.get(k, d.get(alt) if alt else None)

print("="*96)
print("FINAL COMPARISON (n=97, POOL=100, gemini-flash-latest, temp 0; identical frozen pools)")
print("="*96)
hdr = f"{'System':<34}{'R@1':>6}{'Rec@5':>7}{'MRR':>7}{'nDCG10':>8}{'gRank':>7}{'fix':>5}{'brk':>5}{'call':>5}{'inTok':>7}{'outTok':>7}{'lat_s':>7}{'$/q':>10}"
print(hdr)
names = {"A":"A single-pass baseline","B":"B concept-guided rerank","C":"C competing-hyp v1 (full)","D":"D competing-hyp anchor-edit *"}
report = {}
for s in "ABCD":
    d = rows[s]
    r1 = d["R@1"]; rec5 = d["Recall@5"]; mrr = d["MRR"]; ndcg = d["nDCG@10"]; gr = d["mean_gold_rank"]
    if s == "A":
        fx = bk = "-"; fxi = bki = 0
    else:
        fxi, bki, _ = r1_stats({"B":B,"C":C,"D":D}[s]); fx, bk = fxi, bki
    m = cl[s]
    print(f"{names[s]:<34}{r1:>6.2f}{rec5:>7.2f}{mrr:>7.3f}{ndcg:>8.3f}{gr:>7.2f}{str(fx):>5}{str(bk):>5}"
          f"{m['calls_per_query']:>5}{m['input_tokens_per_query']:>7.0f}{m['output_tokens_per_query']:>7.0f}"
          f"{m['latency_s_per_query']:>7.2f}{m['cost_usd_per_query']:>10.6f}")
    report[s] = {"R@1":r1,"Recall@5":rec5,"MRR":mrr,"nDCG@10":ndcg,"mean_gold_rank":gr,
                 "fixed":fxi,"broke":bki,**m}
print("* final system")

print("\n"+"="*96)
print("STATISTICAL SIGNIFICANCE vs System A (paired, n=97, complete set — no exclusions)")
print("="*96)
for s in "BCD":
    sysd = {"B":B,"C":C,"D":D}[s]
    fixed, broke, p_mc = r1_stats(sysd)
    a, sv = paired_rr(sysd)
    obs, lo, hi, p_perm = bootstrap_perm(a, sv)
    mrr_a, mrr_s = st.mean(a), st.mean(sv)
    print(f"\nSystem {s} vs A:")
    print(f"  R@1 McNemar: fixed={fixed} broke={broke}  exact two-sided p = {p_mc:.4f}")
    print(f"  MRR: {mrr_a:.3f} -> {mrr_s:.3f}  (Δ={obs:+.4f})  bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]  perm p = {p_perm:.4f}")
    report.setdefault("stats", {})[s] = {"R@1_fixed":fixed,"R@1_broke":broke,"mcnemar_p":p_mc,
        "MRR_A":mrr_a,"MRR_sys":mrr_s,"MRR_delta":obs,"MRR_CI":[lo,hi],"MRR_perm_p":p_perm}
(R/"bright_finalize"/"final_report.json").write_text(json.dumps(report, indent=1))
print(f"\nwrote {R/'bright_finalize'/'final_report.json'}")
