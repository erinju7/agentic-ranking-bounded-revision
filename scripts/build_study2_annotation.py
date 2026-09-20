"""Build a BLINDED expert-reference annotation package for Study 2 (proposal--funding matching).
Exposes ONLY the proposal and call texts -- no System-A/E/DEP/IND/reviewer/judge outputs, no scores,
no decisions -- so human labels are not contaminated by the model. 6 proposals x 6 calls = 36 pairs.
Outputs to analysis/study2_annotation/: rubric, readable sheet (full texts + 6x6 grid), CSV for
structured entry, and a protocol. Mirrors the BRIGHT annotation practice (independent annotators,
consensus, agreement reported as-is).
"""
import json, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"analysis"/"study2_annotation"; OUT.mkdir(parents=True, exist_ok=True)
d = json.loads((ROOT/"data"/"study2_validation"/"docs.json").read_text())
props = d["proposals"]; calls = d["calls"]
PIDS = list(props); CIDS = list(calls)

RUBRIC = """# Study 2 --- expert-reference annotation rubric (proposal <-> funding call)

**Task.** For each (proposal, call) pair, judge whether the *proposal* is a suitable match for the
*funding call*'s stated interests, using ONLY the two documents. Assign one label:

- **match** --- the proposal clearly falls within the call's stated therapeutic areas / scope /
  eligibility; a funder reading both would plausibly consider it in scope.
- **borderline** --- partial fit: it overlaps on some dimension (disease area, modality, mechanism)
  but has a scope, eligibility, or sponsor-specific mismatch that makes suitability genuinely
  uncertain.
- **no_match** --- the proposal is outside the call's stated areas/scope, or violates a stated
  constraint (e.g. wrong disease area, wrong modality, an explicit exclusion, or an institutional /
  sponsor conflict the call rules out).

**Decision procedure (apply in order).**
1. Read the call's stated *areas of interest / scope / eligibility*.
2. Read the proposal's *disease area, mechanism/modality, stage, and any constraints* (e.g. named
   partner, exclusivity, institution).
3. If a hard constraint is violated (disease area outside scope, explicit exclusion, sponsor
   conflict) -> **no_match**.
4. Else if the proposal squarely fits the call's stated interests -> **match**.
5. Else (overlaps but with a real caveat) -> **borderline**.

**For each pair record:** the label, your confidence (high / medium / low), and a short
**evidence** note quoting the deciding text from BOTH documents (a phrase from the proposal and a
phrase from the call). Evidence is required for match and no_match.

**Blinding.** These sheets contain NO model outputs, scores, or decisions. Do not consult the
system's results while labelling. Label every pair before comparing to anything.
"""

PROTOCOL = """# Study 2 --- annotation protocol

- **Primary labels are human.** The machine (System E / DEP / reviewer / judge) outputs are sealed
  and are NOT to be viewed during annotation.
- **Independent annotation + consensus (recommended).** Have >=2 domain-qualified annotators label
  independently on separate copies of `annotation_sheet_blinded.csv`; then reconcile to a consensus
  label, keeping only pairs all annotators agree on as "clean" (BRIGHT's practice). Report raw
  inter-annotator agreement (Cohen's kappa) as-is, without adjudicating it away.
- **Scope.** 36 pairs = 6 proposals x 6 calls. Read the 6 proposals and 6 calls once (Section A/B of
  the readable sheet), then fill the 36-row grid.
- **After labelling.** The completed CSV becomes the expert reference. We then score the reference-backbone
  Study 2 outputs against it (does DEP vs IND, and the reviewer's REVISE actions, move decisions
  toward the human label?). Only if a real accuracy effect appears is a multi-backbone panel run.
- **No relabelling to fit results.** Labels are frozen before any scoring; disagreements are reported,
  not revised.
"""

# ---- CSV for structured entry ----
with open(OUT/"annotation_sheet_blinded.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pair_id", "proposal_id", "call_id", "label(match|borderline|no_match)",
                "confidence(high|medium|low)", "evidence_proposal_quote", "evidence_call_quote"])
    i = 0
    for p in PIDS:
        for c in CIDS:
            i += 1
            w.writerow([f"P{i:02d}", p, c, "", "", "", ""])

# ---- readable sheet: full texts + grid ----
L = ["# Study 2 --- blinded annotation sheet\n",
     "Read the rubric (`annotation_rubric.md`) first. Sections A and B give the full documents; "
     "Section C is the 36-pair grid to fill (or use `annotation_sheet_blinded.csv`).\n",
     "\n## Section A --- Proposals (6)\n"]
for p in PIDS:
    L.append(f"\n### Proposal: `{p}`\n\n```\n{props[p]['text'].strip()}\n```\n")
L.append("\n## Section B --- Funding calls (6)\n")
for c in CIDS:
    L.append(f"\n### Call: `{c}`\n\n```\n{calls[c]['text'].strip()}\n```\n")
L.append("\n## Section C --- 36-pair grid (fill label / confidence / one-line evidence)\n")
L.append("\n| pair | proposal | call | label | conf | evidence (proposal phrase -> call phrase) |")
L.append("|---|---|---|---|---|---|")
i = 0
for p in PIDS:
    for c in CIDS:
        i += 1
        L.append(f"| P{i:02d} | {p} | {c} |  |  |  |")
(OUT/"annotation_sheet_blinded.md").write_text("\n".join(L))
(OUT/"annotation_rubric.md").write_text(RUBRIC)
(OUT/"annotation_protocol.md").write_text(PROTOCOL)

# ---- blinding self-check: assert no result fields leaked ----
sheet = (OUT/"annotation_sheet_blinded.md").read_text().lower()
leaks = [w for w in ["match_score", "decision\":", "\"e\":", "dep\":", "ind\":", "reviewer\":", "arch_", "promote_ids"] if w in sheet]
print("files written to", OUT)
for fn in ["annotation_rubric.md", "annotation_protocol.md", "annotation_sheet_blinded.md", "annotation_sheet_blinded.csv"]:
    print("  ", fn, (OUT/fn).stat().st_size, "bytes")
print("blinding check (should be []):", leaks)
print(f"grid: {len(PIDS)}x{len(CIDS)} = {len(PIDS)*len(CIDS)} pairs")
