"""Diagnostic analysis: structured multi-agent vs the frozen single-pass generalist
on TREC CT 2022. Consumes existing outputs only (no API calls):
  generalist: results/trec_ct_2022/generalist/outputs.jsonl
  multiagent: results/trec_ct_2022/multiagent/internals.jsonl

Reports the five requested diagnostics. Verdict heuristics are stated explicitly:
  clinical relevant : relevance >= 60
  inclusion fail    : inclusion_satisfied == "No"
  inclusion ok      : inclusion_satisfied in {"Yes","Probably"}
  exclusion hit     : exclusion_triggered in {"Yes","Probably"}
Aliases are the frozen candidate index, identical across both arms.
"""
from __future__ import annotations
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "results" / "trec_ct_2022"
GEN = BASE / "generalist" / "outputs.jsonl"
MA = BASE / "multiagent" / "internals.jsonl"


def rel(a):
    try: return float(a.get("relevance")) if a and a.get("relevance") is not None else None
    except (TypeError, ValueError): return None
def clinical_low(a): r = rel(a); return r is not None and r < 60
def incl_fail(a): return bool(a) and a.get("inclusion_satisfied") == "No"
def excl_hit(a): return bool(a) and a.get("exclusion_triggered") in ("Yes", "Probably")


def main():
    gen = {r["topic_id"]: r for r in (json.loads(l) for l in GEN.read_text().splitlines() if l.strip())}
    ma = {r["topic_id"]: r for r in (json.loads(l) for l in MA.read_text().splitlines() if l.strip())}
    topics = sorted(set(gen) & set(ma), key=int)
    print(f"topics compared: {len(topics)}\n")

    # 3. top-1 error reduction
    gen_err = {t for t in topics if gen[t]["top1_label"] == 1}   # generalist errs are all qrel1
    ma_top1 = {t: ma[t]["top1_label"] for t in topics}
    ma_err = {t for t in topics if ma_top1[t] != 2}
    fixed = sorted(gen_err - ma_err, key=int)
    new_err = sorted((set(topics) - gen_err) - {t for t in topics if gen[t]["top1_label"] != 2} & set() , key=int)  # placeholder
    # new error = generalist correct (top1==2) but MA wrong
    gen_correct = {t for t in topics if gen[t]["top1_label"] == 2}
    new_err = sorted(gen_correct & ma_err, key=int)
    print("=== 3. top-1 error reduction ===")
    print(f"  generalist top-1 errors: {len(gen_err)} (all qrel1)")
    print(f"  multi-agent top-1 errors: {len(ma_err)}  "
          f"(qrel1={sum(ma_top1[t]==1 for t in ma_err)}, qrel0={sum(ma_top1[t]==0 for t in ma_err)})")
    print(f"  net change: {len(ma_err)-len(gen_err):+d}  | fixed {len(fixed)}, newly broken {len(new_err)}")

    # helper: specialist verdicts for a topic keyed by alias
    def verd(t):
        r = ma[t]
        return r["clinical"], r["inclusion"], r["exclusion"], r["alias_label"], r["ranked"]

    # 1. specialist disagreement with coordinator (on the coordinator's top-1 pick)
    dis = {"clinical": 0, "inclusion": 0, "exclusion": 0}
    for t in topics:
        cl, inc, exc, lab, ranked = verd(t)
        pick = ranked[0]
        dis["clinical"] += clinical_low(cl.get(pick, {}))
        dis["inclusion"] += incl_fail(inc.get(pick, {}))
        dis["exclusion"] += excl_hit(exc.get(pick, {}))
    print("\n=== 1. specialist disagreed with coordinator's top-1 pick ===")
    for k, v in dis.items():
        print(f"  {k}: {v}/{len(topics)}  (specialist's verdict was negative on the trial the coordinator ranked #1)")

    # 2. exclusion agent catches a qrel1 near-miss the generalist ranked #1
    saved = []
    for t in gen_err:  # generalist top1 was a qrel1 trial
        cl, inc, exc, lab, ranked = verd(t)
        gpick = gen[t]["ranked_aliases"][0]      # generalist's wrong (qrel1) pick, same alias space
        if lab.get(gpick) == 1 and excl_hit(exc.get(gpick, {})) and ranked and ranked[0] != gpick:
            saved.append((t, ma_top1[t]))
    print("\n=== 2. exclusion agent flagged the generalist's qrel1 near-miss AND it was demoted ===")
    print(f"  {len(saved)}/{len(gen_err)} generalist near-miss errors: exclusion flagged the trial and MA no longer ranks it #1")
    print(f"     of those, MA top-1 became qrel2 (correct): {sum(1 for _,l in saved if l==2)}")

    # 4. corrected queries -> decisive specialist
    def decisive(t):
        cl, inc, exc, lab, ranked = verd(t)
        gpick = gen[t]["ranked_aliases"][0]      # the qrel1 trial the generalist chose
        if excl_hit(exc.get(gpick, {})): return "exclusion"
        if incl_fail(inc.get(gpick, {})): return "inclusion"
        if clinical_low(cl.get(gpick, {})): return "clinical"
        return "coordinator_reasoning"
    print("\n=== 4. corrected queries: decisive specialist ===")
    from collections import Counter
    dec = Counter(decisive(t) for t in fixed)
    for t in fixed:
        print(f"  topic {t}: fixed by {decisive(t)}")
    print(f"  totals: {dict(dec)}")

    # 5. new errors -> culprit specialist (which one wrongly downgraded a true qrel2)
    def culprit(t):
        cl, inc, exc, lab, ranked = verd(t)
        golds = [a for a, l in lab.items() if l == 2]
        # a gold the MA demoted out of #1
        reasons = Counter()
        for g in golds:
            if excl_hit(exc.get(g, {})): reasons["exclusion_false_exclude"] += 1
            elif incl_fail(inc.get(g, {})): reasons["inclusion_false_reject"] += 1
            elif clinical_low(cl.get(g, {})): reasons["clinical_underrated"] += 1
            else: reasons["coordinator_misranked"] += 1
        return reasons.most_common(1)[0][0] if reasons else "unknown"
    print("\n=== 5. new errors: culprit specialist ===")
    cul = Counter(culprit(t) for t in new_err)
    for t in new_err:
        print(f"  topic {t}: MA top1=qrel{ma_top1[t]} -> {culprit(t)}")
    print(f"  totals: {dict(cul)}")


if __name__ == "__main__":
    main()
