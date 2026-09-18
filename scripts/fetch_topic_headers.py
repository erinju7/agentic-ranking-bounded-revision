"""Fetch structured *header* fields from the EU Funding & Tenders Portal.

Companion to ``fetch_topic_descriptions.py``. That script kept only the free-text
``description`` (scope) and threw away the rest of the TopicDetails payload. The
non-semantic signals we need for multi-dimensional matching -- opening/deadline
dates, type of action, call budget, submission stage, status -- live in the
``actions`` and ``budgetOverviewJSONItem`` sub-objects, which it discarded.

Scope: ONLY the topic codes under the curated-subset true calls (101 subCalls)
are fetched, per the "audit coverage before building tools" plan. We do NOT
touch the frozen retriever or candidate pool.

Order of operations (deliberate):
  1. fetch raw header  2. store raw fields + raw text  3. (separate) coverage audit
  4. only then normalization / admissibility.

We keep BOTH the raw and (later) normalized action type. Nothing here strips the
year or normalizes -- that is a downstream, guarded step. This stage is raw +
auditable only.

    https://ec.europa.eu/info/funding-tenders/opportunities/data/topicDetails/{id}.json
"""
from __future__ import annotations

import argparse
import html
import json
import re
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "data" / "cleaned" / "project_objectives_with_call_descriptions.csv"
CURATED_DIR = ROOT / "data" / "frozen" / "rq2_v1_seed42_description_filtered" / "curated_evidence_subset"
CURATED_CALLS = CURATED_DIR / "curated_true_calls.json"
HEADER_CACHE = ROOT / "data" / "topic_headers_cache"
HEADER_TABLE = CURATED_DIR / "topic_headers_raw.csv"
SUBCALL_CODES = CURATED_DIR / "subcall_to_codes.json"

TOPIC_URL = "https://ec.europa.eu/info/funding-tenders/opportunities/data/topicDetails/{}.json"
PORTAL_HUMAN_URL = "https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/{}"


def strip_html(raw: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", raw or "")).split())


def safe_name(code: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", code)


def curated_topic_codes(scope: str = "curated") -> dict[str, list[str]]:
    """subCall -> list of topic codes (funding_call_id).

    scope="curated": the 101 curated true calls (default).
    scope="all":     the full 413 candidate pool. Needed for candidate profiles,
                     so distractors are as richly described as true calls and
                     'has a header' cannot become a proxy for 'is the true call'.
    """
    if scope == "all":
        pool = json.loads((FROZEN_DIR := CURATED_DIR.parent / "candidate_pool.json").read_text())
        labels = {c["call_label"] for c in pool["candidates"]}
    else:
        labels = set(json.loads(CURATED_CALLS.read_text()))
    df = pd.read_csv(SOURCE_CSV, low_memory=False, encoding_errors="replace")
    df = df[df["subCall"].astype(str).isin(labels)].dropna(subset=["funding_call_id"])
    out: dict[str, list[str]] = {}
    for subcall, group in df.groupby("subCall", sort=False):
        out[str(subcall)] = list(dict.fromkeys(group["funding_call_id"].astype(str).tolist()))
    return out


def fetch_json(url: str, timeout: float, retries: int) -> tuple[Any, str]:
    """Returns (payload_or_None, status). status in {ok, 404, error}."""
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", "replace")), "ok"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None, "404"
            last = exc
        except Exception as exc:
            last = exc
        if attempt < retries:
            time.sleep(0.5 * attempt)
    return None, f"error:{type(last).__name__ if last else 'unknown'}"


def resolve_details(code: str, timeout: float, retries: int) -> dict[str, Any]:
    """Try id variants; return {resolved_id, fetch_status, details|None}."""
    statuses = []
    for variant in (code.lower(), f"h2020-{code.lower()}"):
        payload, status = fetch_json(TOPIC_URL.format(variant), timeout, retries)
        statuses.append(status)
        details = (payload or {}).get("TopicDetails") if payload else None
        if details:
            return {"resolved_id": variant, "fetch_status": "ok", "details": details}
        if status.startswith("error"):
            # network error, not a clean miss -- surface it
            return {"resolved_id": None, "fetch_status": status, "details": None}
    return {"resolved_id": None, "fetch_status": "404", "details": None}


def extract_header(details: dict[str, Any]) -> dict[str, Any]:
    """Pull the priority header fields out of a TopicDetails payload. Raw only."""
    actions = details.get("actions") or []
    a0 = actions[0] if actions else {}
    # action types can be a list across the action objects
    action_types: list[str] = []
    opening_dates: list[str] = []
    deadline_dates: list[str] = []
    submission_stages: list[str] = []
    statuses: list[str] = []
    for a in actions:
        for t in a.get("types") or []:
            toa = t.get("typeOfAction")
            if toa:
                action_types.append(toa)
        if a.get("plannedOpeningDate"):
            opening_dates.append(a["plannedOpeningDate"])
        for d in a.get("deadlineDates") or []:
            deadline_dates.append(d)
        sp = (a.get("submissionProcedure") or {}).get("abbreviation")
        if sp:
            submission_stages.append(sp)
        st = (a.get("status") or {}).get("abbreviation")
        if st:
            statuses.append(st)

    # topic total budget: budgetYearMap sums across years, across action entries
    topic_budget = None
    budget = details.get("budgetOverviewJSONItem") or {}
    try:
        total = 0
        found = False
        for entries in (budget.get("budgetTopicActionMap") or {}).values():
            for e in entries:
                for _, amount in (e.get("budgetYearMap") or {}).items():
                    if isinstance(amount, (int, float)):
                        total += amount
                        found = True
        topic_budget = total if found else None
    except Exception:
        topic_budget = None

    fw = details.get("frameworkProgramme") or {}
    return {
        "topic_id": details.get("identifier"),
        "call_identifier": details.get("callIdentifier"),
        "call_title": details.get("callTitle"),
        "framework_programme": fw.get("abbreviation"),
        "raw_action_type": " | ".join(dict.fromkeys(action_types)) or None,
        "opening_date": " | ".join(dict.fromkeys(opening_dates)) or None,
        "deadline_date": " | ".join(dict.fromkeys(deadline_dates)) or None,
        "submission_stage": " | ".join(dict.fromkeys(submission_stages)) or None,
        "topic_status": " | ".join(dict.fromkeys(statuses)) or None,
        "topic_budget_eur": topic_budget,
        # keep the raw sub-objects for re-derivation / audit
        "_raw_actions": actions,
        "_raw_budgetOverview": budget,
        "description_chars": len(strip_html(details.get("description") or "")),
    }


def mapping_confidence(subcalls: list[str], header: dict[str, Any], fetch_status: str) -> str:
    """A topic code can back several subCalls (many-to-many). High confidence if
    the portal callIdentifier exactly equals ANY of the subCalls it serves."""
    if fetch_status != "ok":
        return "none"
    ci = (header.get("call_identifier") or "").strip().lower()
    if ci and any(ci == sc.lower() for sc in subcalls):
        return "high"           # portal call id exactly equals one of our subCall labels
    if ci:
        return "medium"         # resolved, but call id is the parent/multi-year call
    return "low"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--timeout", type=float, default=25.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--scope", choices=["curated", "all"], default="curated",
                   help="curated = 101 true calls; all = full 413 candidate pool.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    socket.setdefaulttimeout(args.timeout)
    HEADER_CACHE.mkdir(parents=True, exist_ok=True)

    subcall_to_codes = curated_topic_codes(args.scope)
    codes_path = SUBCALL_CODES if args.scope == "curated" else CURATED_DIR / "subcall_to_codes_all413.json"
    table_path = HEADER_TABLE if args.scope == "curated" else CURATED_DIR / "topic_headers_raw_all413.csv"
    codes_path.write_text(json.dumps(subcall_to_codes, indent=1) + "\n", encoding="utf-8")
    all_codes = sorted({c for codes in subcall_to_codes.values() for c in codes})
    # many-to-many: a topic code can serve several curated subCalls
    code_to_subcalls: dict[str, list[str]] = {}
    for sc, codes in subcall_to_codes.items():
        for c in codes:
            code_to_subcalls.setdefault(c, []).append(sc)
    if args.limit is not None:
        all_codes = all_codes[: args.limit]
    print(f"{len(all_codes)} topic codes under {len(subcall_to_codes)} curated subCalls.")

    for i, code in enumerate(all_codes, 1):
        cache_path = HEADER_CACHE / f"{safe_name(code)}.json"
        if cache_path.exists() and not args.refresh:
            if i % 50 == 0:
                print(f"[{i}/{len(all_codes)}] cached")
            continue
        res = resolve_details(code, args.timeout, args.retries)
        details = res["details"]
        header = extract_header(details) if details else {}
        subcalls = code_to_subcalls.get(code, [])
        record = {
            "topic_code": code,
            "subcalls": subcalls,
            "resolved_id": res["resolved_id"],
            "fetch_status": res["fetch_status"],
            "source_url": PORTAL_HUMAN_URL.format(res["resolved_id"]) if res["resolved_id"] else None,
            **header,
            "mapping_confidence": mapping_confidence(subcalls, header, res["fetch_status"]),
        }
        cache_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        flag = record.get("mapping_confidence")
        print(f"[{i}/{len(all_codes)}] {code} -> {res['resolved_id'] or res['fetch_status']}"
              f" | action={record.get('raw_action_type')} dates={record.get('deadline_date')}"
              f" budget={record.get('topic_budget_eur')} conf={flag}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    build_table(all_codes, code_to_subcalls, table_path)


def build_table(all_codes: list[str], code_to_subcalls: dict[str, list[str]],
                table_path=HEADER_TABLE) -> None:
    """Topic-code-keyed raw header table. `subcalls` is the authoritative
    one-to-many mapping (recomputed here, not read from the possibly-stale
    per-record field), so a code shared across subCalls is never dropped."""
    flat_cols = ["topic_code", "subcalls", "resolved_id", "fetch_status", "mapping_confidence",
                 "topic_id", "call_identifier", "call_title", "framework_programme",
                 "raw_action_type", "opening_date", "deadline_date", "submission_stage",
                 "topic_status", "topic_budget_eur", "description_chars", "source_url"]
    rows = []
    for code in all_codes:
        p = HEADER_CACHE / f"{safe_name(code)}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        rec["subcalls"] = " | ".join(code_to_subcalls.get(code, []))
        rows.append({k: rec.get(k) for k in flat_cols})
    pd.DataFrame(rows, columns=flat_cols).to_csv(table_path, index=False)
    print(f"\nWrote {table_path} ({len(rows)} topic rows, "
          f"{len({sc for code in all_codes for sc in code_to_subcalls.get(code, [])})} subCalls).")


if __name__ == "__main__":
    main()
