"""Score the reference-backbone Study 2 outputs against the human annotation (single annotator).
NO API. Reports 3-way accuracy and Cohen's kappa (system-vs-human) for each variant, plus the two
RQ2-relevant intervention analyses: (i) on pairs where DEP != IND, does DEP move TOWARD the human
label? (ii) on pairs the 2b reviewer REVISEd, does the final move TOWARD the human label?
"""
import os, json, ast, csv, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
HUMAN_CSV = Path(os.environ.get("STUDY2_ANNOTATION_CSV", ROOT/"data"/"study2_validation"/"annotation_sheet.csv"))

NORM = {"match": "match", "no match": "no_match", "no_match": "no_match", "borderline": "borderline"}
def norm(s): return NORM[str(s).strip().lower()]

# ---- human labels ----
human = {}
with open(HUMAN_CSV) as f:
    r = csv.reader(f); next(r)
    for row in r:
        if len(row) >= 4 and row[0].startswith("P"):
            human[(row[1], row[2])] = norm(row[3])

def dec(field):  # parse stringified dict, return decision
    d = ast.literal_eval(field) if isinstance(field, str) else field
    return d.get("decision")

# ---- system decisions ----
base = {(x["proposal"], x["call"]): x for x in json.load(open(ROOT/"results/study2_validation/pairs.json"))}
dep = {(x["proposal"], x["call"]): x for x in json.load(open(ROOT/"results/study2_validation/study2a_dependency/pairs.json"))}
rev = {(x["proposal"], x["call"]): x for x in json.load(open(ROOT/"results/study2_validation/study2b_review/pairs.json"))}

variants = {}
for k in human:
    variants.setdefault("A", {})[k] = dec(base[k]["A"])
    variants.setdefault("B", {})[k] = dec(base[k]["B"])
    variants.setdefault("D", {})[k] = dec(base[k]["D"])
    variants.setdefault("IND", {})[k] = dec(dep[k]["IND"])
    variants.setdefault("DEP", {})[k] = dec(dep[k]["DEP"])
    variants.setdefault("2b-final", {})[k] = dec(rev[k]["final"])

CATS = ["match", "borderline", "no_match"]
def kappa(sys, hum):
    keys = list(hum); n = len(keys)
    po = sum(1 for k in keys if sys[k] == hum[k]) / n
    ps = {c: sum(1 for k in keys if sys[k] == c)/n for c in CATS}
    ph = {c: sum(1 for k in keys if hum[k] == c)/n for c in CATS}
    pe = sum(ps[c]*ph[c] for c in CATS)
    return po, (po-pe)/(1-pe) if pe < 1 else 0.0

print("Human label distribution:", {c: sum(1 for v in human.values() if v == c) for c in CATS}, f"(n={len(human)})")
print(f"\n{'variant':10} {'acc':>6} {'kappa':>7}   confusion(sys rows x human cols not shown; see below)")
accs = {}
for name, sysd in variants.items():
    acc, k = kappa(sysd, human); accs[name] = (acc, k)
    print(f"{name:10} {acc:>6.3f} {k:>7.3f}")

# ---- RQ2 (i): DEP vs IND -- does structuring move toward human? ----
print("\n=== RQ2.1  DEP vs IND (Study 2a): on pairs where they differ, which is closer to human? ====")
diff = [k for k in human if variants["IND"][k] != variants["DEP"][k]]
dep_closer = sum(1 for k in diff if variants["DEP"][k] == human[k] and variants["IND"][k] != human[k])
ind_closer = sum(1 for k in diff if variants["IND"][k] == human[k] and variants["DEP"][k] != human[k])
neither = len(diff) - dep_closer - ind_closer
print(f"IND!=DEP on {len(diff)} pairs. DEP matches human & IND doesn't: {dep_closer}; "
      f"IND matches & DEP doesn't: {ind_closer}; neither: {neither}")
print(f"overall: IND acc {accs['IND'][0]:.3f} (kappa {accs['IND'][1]:.3f})  vs  DEP acc {accs['DEP'][0]:.3f} (kappa {accs['DEP'][1]:.3f})")

# ---- RQ2 (ii): reviewer REVISE -- 2b-final vs DEP (=ref_2aDEP) ----
print("\n=== RQ2.2  reviewer REVISE (Study 2b): on pairs the final differs from pre-review DEP, closer to human? ===")
changed = [k for k in human if variants["2b-final"][k] != variants["DEP"][k]]
fin_closer = sum(1 for k in changed if variants["2b-final"][k] == human[k] and variants["DEP"][k] != human[k])
dep_closer2 = sum(1 for k in changed if variants["DEP"][k] == human[k] and variants["2b-final"][k] != human[k])
neither2 = len(changed) - fin_closer - dep_closer2
print(f"final!=DEP on {len(changed)} pairs. final matches human & DEP didn't: {fin_closer}; "
      f"DEP matched & final broke it: {dep_closer2}; neither: {neither2}")
print(f"overall: DEP acc {accs['DEP'][0]:.3f}  vs  2b-final acc {accs['2b-final'][0]:.3f}")
for k in changed:
    print(f"   {k[0]:14}->{k[1]:16} DEP={variants['DEP'][k]:11} final={variants['2b-final'][k]:11} human={human[k]}")

# ---- ordinal off-by-one (borderline is between) ----
order = {"no_match": 0, "borderline": 1, "match": 2}
print("\n=== ordinal mean |distance| to human (0=no_match,1=borderline,2=match) ===")
for name, sysd in variants.items():
    md = sum(abs(order[sysd[k]]-order[human[k]]) for k in human)/len(human)
    hard = sum(1 for k in human if abs(order[sysd[k]]-order[human[k]])==2)  # match<->no_match flips
    print(f"{name:10} mean|dist|={md:.3f}  hard-flips(match<->no_match)={hard}")
