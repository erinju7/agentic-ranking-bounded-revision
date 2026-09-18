"""Bootstrap 95% CIs for Study 2 agreement kappa over the 36 proposal-call pairs.
Reports kappa(non-expert second annotation vs expert gold), kappa(single-pass A
system vs expert gold), and their difference. n=36 gives very wide intervals, so
0.62 (system) and 0.49 (reference) cannot be distinguished. Reproduces the CIs
quoted in Study 2 (results section 4.3)."""
import csv, json, random
from pathlib import Path
random.seed(42)
CATS=["no_match","borderline","match"]; IDX={c:i for i,c in enumerate(CATS)}
norm=lambda x:{"no_match":"no_match","match":"match","borderline":"borderline"}[x.strip().lower().replace(" ","_")]
def wkappa(pairs):
    n=len(pairs); M=[[0]*3 for _ in range(3)]
    for a,b in pairs: M[IDX[a]][IDX[b]]+=1
    ra=[sum(M[i]) for i in range(3)]; ca=[sum(M[i][j] for i in range(3)) for j in range(3)]
    W=[[((i-j)**2)/4.0 for j in range(3)] for i in range(3)]
    po=sum(W[i][j]*M[i][j] for i in range(3) for j in range(3))/n
    pe=sum(W[i][j]*ra[i]*ca[j]/n for i in range(3) for j in range(3))/n
    return 1-po/pe if pe else 0.0
ROOT=Path(__file__).resolve().parents[1]
gold={(r["proposal_id"],r["call_id"]):norm(r["label"]) for r in csv.DictReader(open("/Users/macbook/Desktop/gold_standard_JDR.csv"))}
ne={(r["proposal_id"],r["call_id"]):norm(list(r.values())[-1]) for r in csv.DictReader(open(ROOT/"analysis/study2_annotation/annotation_sheet_FILLED.csv"))}
res=json.loads((ROOT/"results/james_validation/listwise/claude-haiku-4-5-20251001_rubric/listwise_results.json").read_text())
Asys={(rr["proposal"],c):norm(g) for rr in res for c,g in rr["A"]["grades"].items()}
tri=[(gold[k],ne[k],Asys[k]) for k in gold if k in ne and k in Asys]
kne=wkappa([(e,n) for e,n,a in tri]); ka=wkappa([(e,a) for e,n,a in tri])
def boot(f,B=10000):
    xs=sorted(f([random.choice(tri) for _ in range(len(tri))]) for _ in range(B))
    return xs[int(0.025*B)], xs[int(0.975*B)]
lo1,hi1=boot(lambda s:wkappa([(e,n) for e,n,a in s]))
lo2,hi2=boot(lambda s:wkappa([(e,a) for e,n,a in s]))
lod,hid=boot(lambda s:wkappa([(e,a) for e,n,a in s])-wkappa([(e,n) for e,n,a in s]))
print(f"n={len(tri)} pairs")
print(f"kappa(non-expert vs expert) = {kne:.2f}  95% CI [{lo1:.2f}, {hi1:.2f}]")
print(f"kappa(A system  vs expert)  = {ka:.2f}  95% CI [{lo2:.2f}, {hi2:.2f}]")
print(f"difference (A - non-expert) = {ka-kne:+.2f} 95% CI [{lod:+.2f}, {hid:+.2f}]  includes 0: {lod<0<hid}")
