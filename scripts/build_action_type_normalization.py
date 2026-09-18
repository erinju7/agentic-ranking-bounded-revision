"""Build action_type_normalization.json from the raw portal typeOfAction strings.

The portal emits 41 fine-grained strings (e.g. "FCH2-RIA Research and Innovation
action", "MSCA-IF-EF-CAR Career Restart panel"). Per the governance decision, the
model may see only a programme-agnostic *instrument / career-stage class*:

  - strip the exact call/topic code and any year;
  - strip the thematic-programme prefix (FCH2 / BBI / IMI2 / SESAR / EuroHPC ...),
    so action type is a pure instrument signal and does not double as a
    thematic-programme leak (theme stays in the scope text);
  - collapse over-fine sub-panels (MSCA-IF-EF-ST/SE/RI/CAR/GF -> Individual
    Fellowship) that carry no applicant-facing instrument meaning;
  - KEEP distinctions a real applicant would recognise and could infer from
    content: RIA vs IA vs CSA; ERC career stage; SME phase.

The raw string is retained in the audit layer (topic_headers_raw.csv); only the
normalized class is admissible as model input.
"""
from __future__ import annotations

import csv
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "frozen" / "rq2_v1_seed42_description_filtered" / "curated_evidence_subset"
HEADERS = CURATED / "topic_headers_raw.csv"
OUT = CURATED / "action_type_normalization.json"


def classify(raw: str) -> tuple[str, str]:
    """raw typeOfAction string -> (normalized_action_type, instrument_family)."""
    code = raw.split(" ", 1)[0].upper()          # leading code token
    desc = raw.lower()

    # ERC: career stage is applicant-meaningful and content-inferable
    if code.startswith("ERC-STG"): return "ERC Starting Grant", "ERC"
    if code.startswith("ERC-COG"): return "ERC Consolidator Grant", "ERC"
    if code.startswith("ERC-ADG"): return "ERC Advanced Grant", "ERC"
    if code.startswith("ERC-SYG"): return "ERC Synergy Grant", "ERC"
    if code.startswith("ERC-POC"): return "ERC Proof of Concept Grant", "ERC"

    # MSCA: collapse sub-panels; keep the mobility scheme distinction
    if code.startswith("MSCA-IF"):     return "MSCA Individual Fellowship", "MSCA"
    if code.startswith("MSCA-RISE"):   return "MSCA Staff Exchanges (RISE)", "MSCA"
    if code.startswith("MSCA-ITN"):    return "MSCA Innovative Training Network", "MSCA"
    if code.startswith("MSCA-COFUND"): return "MSCA COFUND", "MSCA"

    # SME instrument: phase is meaningful (feasibility vs demonstration)
    if code.startswith("SME-1"):  return "SME Instrument Phase 1", "SME"
    if code.startswith("SME-2"):  return "SME Instrument Phase 2", "SME"

    # Procurement / cofund / partnership instruments
    if "PCP" in code:              return "Pre-Commercial Procurement (PCP)", "PCP/PPI"
    if code.startswith("COFUND-EJP") or "european joint programme" in desc:
        return "COFUND (European Joint Programme)", "COFUND"
    if code.startswith("ERA-NET"): return "ERA-NET Cofund", "COFUND"
    if code == "FPA" or "framework partnership agreement" in desc and "csa" not in desc:
        return "Framework Partnership Agreement", "FPA"

    # Generic instruments, programme-agnostic (matches across BBI/FCH2/IMI2/SESAR/EuroHPC)
    if "research and innovation action" in desc:  return "Research and Innovation Action", "RIA"
    if "innovation action" in desc:               return "Innovation Action", "IA"
    if "coordination" in desc:                    return "Coordination and Support Action", "CSA"

    # Code-suffix fallback for strings whose description just repeats the code
    # (e.g. "EuroHPC-RIA EuroHPC-RIA").
    if code.endswith("-RIA") or code.endswith("RIA"): return "Research and Innovation Action", "RIA"
    if code.endswith("-IA") or code.endswith("IA"):   return "Innovation Action", "IA"
    if code.endswith("-CSA") or code.endswith("CSA"): return "Coordination and Support Action", "CSA"

    return "Other / Unmapped", "OTHER"


def main() -> None:
    rows = list(csv.DictReader(open(HEADERS)))
    counts = collections.Counter()
    for r in rows:
        for t in (r.get("raw_action_type") or "").split(" | "):
            if t.strip():
                counts[t.strip()] += 1

    mapping = {}
    for raw, n in counts.most_common():
        norm, fam = classify(raw)
        mapping[raw] = {
            "normalized_action_type": norm,
            "instrument_family": fam,
            "n_topics": n,
            "year": None,          # stripped
            "exact_code": None,    # stripped
        }

    unmapped = [r for r, v in mapping.items() if v["normalized_action_type"] == "Other / Unmapped"]
    normset = sorted({v["normalized_action_type"] for v in mapping.values()})
    OUT.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{len(mapping)} raw strings -> {len(normset)} normalized classes")
    for nm in normset:
        raws = [r for r, v in mapping.items() if v["normalized_action_type"] == nm]
        print(f"  {nm:36s} <- {len(raws)} raw")
    if unmapped:
        print("UNMAPPED (review):", unmapped)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
