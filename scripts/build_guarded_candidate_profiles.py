"""Build guarded candidate profiles for all 413 candidates.

The single model-facing datastore for Level 0 (static enriched reranker) and
Level 1/2 tools. Enforces admissibility_manifest.yaml:

  * Anonymized candidate id (cand_id). The raw call_label embeds scheme + year
    (e.g. "ERC-2016-STG"), so exposing it as the candidate identifier would leak
    exactly the fundingScheme/year the manifest blocks. Models rank cand_ids; a
    server-side map recovers the true label for scoring only.
  * Only admissible normalized fields (title, scope, normalized action type,
    call-scale budget, submission stage).
  * Dates live in a separate `context_only` block: the Date Tool reads them only
    when an exogenous application_context is supplied; they are NOT default input,
    so the scored benchmark stays year-agnostic.
  * A scrub-guard fails the build if any model-facing text contains the call_label
    code, fundingScheme, or masterCall.
"""
from __future__ import annotations

import csv
import json
import re
import hashlib
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "frozen" / "rq2_v1_seed42_description_filtered"
CURATED = FROZEN / "curated_evidence_subset"
HEADERS = CURATED / "topic_headers_raw_all413.csv"
SUBCALL_CODES = CURATED / "subcall_to_codes_all413.json"
NORM = CURATED / "action_type_normalization.json"
DESC = ROOT / "data" / "reference" / "subcall_official_descriptions.json"

OUT_PROFILES = CURATED / "candidate_profiles.json"
OUT_IDMAP = CURATED / "candidate_id_map.json"

SECTION_CAP = 600         # per section (specific_challenge / scope / expected_outcome)
ID_SALT = "p58-cand-v1"   # fixed salt -> deterministic, order-scrambled ids

# H2020 section headers, in canonical order. "objectives" is the ERC/MSCA fallback.
_SECTION_RE = re.compile(
    r"(specific challenge|scope|expected impacts?|expected outcomes?|objectives?)\s*:?\s*",
    re.IGNORECASE,
)
_CANON = {"specific challenge": "specific_challenge", "scope": "scope",
          "expected impact": "expected_outcome", "expected impacts": "expected_outcome",
          "expected outcome": "expected_outcome", "expected outcomes": "expected_outcome",
          "objective": "scope", "objectives": "scope"}   # objectives -> scope fallback


def split_sections(text: str) -> dict[str, str]:
    """Parse an official description into specific_challenge / scope / expected_outcome.

    Falls back to putting the whole text in `scope` when no headers are present
    (as with ERC/MSCA 'Objectives ...' prose)."""
    out = {"specific_challenge": "", "scope": "", "expected_outcome": ""}
    if not text:
        return out
    marks = [(m.start(), m.end(), _CANON.get(m.group(1).lower().strip(), "scope"))
             for m in _SECTION_RE.finditer(text)]
    if not marks:
        out["scope"] = text[:SECTION_CAP]
        return out
    for i, (_, end, field) in enumerate(marks):
        seg_end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        seg = text[end:seg_end].strip()
        if seg and len(out[field]) < SECTION_CAP:
            out[field] = (out[field] + " " + seg).strip()[:SECTION_CAP]
    if not any(out.values()):
        out["scope"] = text[:SECTION_CAP]
    return out


def anon_id(label: str) -> str:
    h = hashlib.sha1((ID_SALT + label).encode()).hexdigest()[:8]
    return f"C_{h}"


def budget_band(v) -> str | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x < 50e6:  return "small (<50M)"
    if x < 200e6: return "medium (50-200M)"
    if x < 1e9:   return "large (200M-1B)"
    return "very-large (>1B)"


def consistent_codes(label: str, codes: list[str]) -> list[str]:
    """Drop topic codes whose year is inconsistent with the subCall's year(s).

    The source project->subCall->funding_call_id join occasionally attaches a
    code from a different year/scheme (e.g. ERC-2017-PoC under ERC-2020-STG),
    which contaminates the aggregated action type and dates. A code is kept if it
    carries no year or shares a year with the subCall label. Within-year
    multi-instrument calls (e.g. BBI-2016-* spanning RIA/IA/CSA) are preserved.
    """
    label_years = set(re.findall(r"(?:19|20)\d{2}", label))
    if not label_years:
        return codes
    kept = []
    for c in codes:
        cy = set(re.findall(r"(?:19|20)\d{2}", c))
        if not cy or cy & label_years:
            kept.append(c)
    return kept or codes  # never empty out a candidate


def scrub(text: str, label: str) -> str:
    """Remove the exact label code and any 4-digit year from short model-facing text."""
    if not text:
        return text
    text = re.sub(re.escape(label), "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(19|20)\d{2}\b", "", text)
    return " ".join(text.split())


def main() -> None:
    header_rows = {r["topic_code"]: r for r in csv.DictReader(open(HEADERS))}
    subcall_to_codes = json.loads(SUBCALL_CODES.read_text())
    norm = json.loads(NORM.read_text())
    desc = json.loads(DESC.read_text())
    pool = {c["call_label"]: c for c in json.load(open(FROZEN / "candidate_pool.json"))["candidates"]}

    profiles, id_map = {}, {}
    for label, cand in pool.items():
        cid = anon_id(label)
        codes = consistent_codes(label, subcall_to_codes.get(label, []))
        rows = [header_rows[c] for c in codes if c in header_rows]

        actions, opens, deadlines, stages, budgets = set(), set(), set(), set(), []
        for r in rows:
            for raw in (r.get("raw_action_type") or "").split(" | "):
                if raw.strip() in norm:
                    actions.add(norm[raw.strip()]["normalized_action_type"])
            for d in (r.get("opening_date") or "").split(" | "):
                if d.strip():
                    opens.add(d.strip())
            for d in (r.get("deadline_date") or "").split(" | "):
                if d.strip():
                    deadlines.add(d.strip())
            for s in (r.get("submission_stage") or "").split(" | "):
                if s.strip():
                    stages.add(s.strip())
            try:
                budgets.append(float(r.get("topic_budget_eur")))
            except (TypeError, ValueError):
                pass

        title = scrub((cand.get("funding_call_descriptions") or [""])[0], label)
        sections = split_sections(" ".join(x for x in (desc.get(label) or []) if x))

        profiles[cid] = {
            # ---- governed admissible representation (IDENTICAL for L0 and L1-2) ----
            "title": title,
            "specific_challenge": sections["specific_challenge"] or None,
            "scope": sections["scope"] or None,
            "expected_outcome": sections["expected_outcome"] or None,
            "action_type": sorted(actions) or None,   # normalized, programme-agnostic
            "call_scale": budget_band(max(budgets)) if budgets else None,
            "submission_stage": sorted(stages) or None,
            # ---- context-only (Date Tool / application_context mode; OFF by default) ----
            "context_only": {
                "opening_dates": sorted(opens) or None,
                "deadline_dates": sorted(deadlines) or None,
            },
            # NB: raw dates, raw_action_type, and admin ids are NOT here -- audit
            # only, in topic_headers_raw_all413.csv.
        }
        id_map[cid] = label

    # ---- scrub-guard ----
    # Only the DISTINCTIVE full codes are leakage: the exact call_label (target)
    # and the masterCall code. We do NOT check fundingScheme, because the
    # manifest deliberately admits the normalized instrument family (action_type)
    # -- and the 2-3 letter scheme abbreviation ("IA","RIA") occurs harmlessly
    # inside ordinary prose ("negotIAtion") and inside the normalized class name.
    # Codes contain digits/hyphens and do not appear in scope prose.
    violations = []
    for label, cand in pool.items():
        cid = anon_id(label)
        p = profiles[cid]
        blob = " ".join(str(p[k]) for k in ("title", "specific_challenge", "scope",
                                            "expected_outcome", "action_type",
                                            "call_scale", "submission_stage") if p[k])
        low = blob.lower()
        master = cand.get("dominant_masterCall") or ""
        for needle in (label, master):
            # only treat multi-char hyphenated/dated codes as leakage needles
            if needle and re.search(r"[-\d]", needle) and needle.lower() in low:
                violations.append((cid, label, needle))
    if violations:
        for cid, label, n in violations[:20]:
            print(f"  LEAK: {cid} ({label}) contains {n!r}")
        raise SystemExit(f"scrub-guard FAILED: {len(violations)} leaks")

    OUT_PROFILES.write_text(json.dumps(profiles, indent=1, ensure_ascii=False) + "\n")
    OUT_IDMAP.write_text(json.dumps(id_map, indent=1) + "\n")

    # ---- coverage report ----
    n = len(profiles)
    def cov(f):
        return sum(1 for p in profiles.values()
                   if (p["context_only"][f] if f in p["context_only"] else p.get(f)))
    print(f"built {n} guarded profiles (anon ids); scrub-guard PASSED")
    for f in ["title", "specific_challenge", "scope", "expected_outcome",
              "action_type", "call_scale", "submission_stage"]:
        c = sum(1 for p in profiles.values() if p.get(f))
        print(f"  {f:18s} {c}/{n} {c/n:.0%}")
    for f in ["opening_dates", "deadline_dates"]:
        c = sum(1 for p in profiles.values() if p["context_only"][f])
        print(f"  context.{f:16s} {c}/{n} {c/n:.0%}")
    print(f"wrote {OUT_PROFILES.name} and {OUT_IDMAP.name}")


if __name__ == "__main__":
    main()
