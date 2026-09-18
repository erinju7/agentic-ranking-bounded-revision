"""Fetch official H2020 topic scope text from the EU Funding & Tenders Portal.

Lever 2 (candidate-description enrichment). The CORDIS dump only carries the
short topic *title* (e.g. "ERC Starting Grant"); the discriminative official
*scope* text lives on the F&T Portal:

    https://ec.europa.eu/info/funding-tenders/opportunities/data/topicDetails/{id}.json

We fetch the ``description`` field (the Scope / Specific challenge / Objectives
text), strip HTML, and aggregate per subCall so the reranker can be shown the
real call scope instead of a degenerate title.

Design:
- Only the topic codes under the FROZEN candidate subCalls are fetched, so the
  retriever and frozen candidate pool are never touched. The enriched map is a
  pure rerank-time override (clean ablation: retrieval held constant).
- Resumable: each topic is cached to disk; re-runs skip cached topics.
- Robust id resolution: try ``lower(code)`` then ``h2020-<lower(code)>``.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from rq2_core import FROZEN_DIR, load_candidate_pool


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "data" / "cleaned" / "project_objectives_with_call_descriptions.csv"
CACHE_DIR = ROOT / "data" / "topic_details_cache"
TOPIC_CSV = ROOT / "data" / "reference" / "topic_official_descriptions.csv"
SUBCALL_JSON = ROOT / "data" / "reference" / "subcall_official_descriptions.json"

TOPIC_URL = (
    "https://ec.europa.eu/info/funding-tenders/opportunities/data/topicDetails/{}.json"
)
# Only the ``description`` field carries scope text; the rest is admin boilerplate.
ADMIN_FIELDS = {"conditions", "supportInfo", "sepTemplate"}


def strip_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    text = html.unescape(text)
    text = re.sub(r"^\s*Scope\s*:\s*", "", text)  # drop leading "Scope :" label
    return " ".join(text.split())


def safe_name(code: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", code)


def candidate_topic_codes() -> tuple[list[str], dict[str, list[str]]]:
    """Distinct topic codes under the frozen candidate subCalls + subCall->codes."""
    candidate_labels = {c["call_label"] for c in load_candidate_pool(FROZEN_DIR)}
    df = pd.read_csv(SOURCE_CSV, low_memory=False, encoding_errors="replace")
    df = df[df["subCall"].astype(str).isin(candidate_labels)]
    df = df.dropna(subset=["funding_call_id"])
    subcall_to_codes: dict[str, list[str]] = {}
    for subcall, group in df.groupby("subCall", sort=False):
        codes = list(dict.fromkeys(group["funding_call_id"].astype(str).tolist()))
        subcall_to_codes[str(subcall)] = codes
    all_codes = sorted({c for codes in subcall_to_codes.values() for c in codes})
    return all_codes, subcall_to_codes


def fetch_json(url: str, timeout: float, retries: int) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None  # genuine miss, do not retry
            last_error = exc
        except Exception as exc:  # network hiccup
            last_error = exc
        if attempt < retries:
            time.sleep(0.5 * attempt)
    if last_error is not None:
        print(f"    ! {url} failed: {last_error}")
    return None


def fetch_topic(code: str, *, timeout: float, retries: int) -> dict[str, Any] | None:
    """Resolve a topic code to its TopicDetails via id variants; None if unresolved."""
    for variant in (code.lower(), f"h2020-{code.lower()}"):
        payload = fetch_json(TOPIC_URL.format(variant), timeout, retries)
        details = (payload or {}).get("TopicDetails") if payload else None
        if details:
            return {"resolved_id": variant, "details": details}
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap topic count.")
    parser.add_argument("--sleep", type=float, default=0.15, help="Seconds between topics.")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch even if a topic is already cached.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TOPIC_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_codes, subcall_to_codes = candidate_topic_codes()
    if args.limit is not None:
        all_codes = all_codes[: args.limit]
    print(f"{len(all_codes)} topic codes under {len(subcall_to_codes)} candidate subCalls.")

    resolved = 0
    for index, code in enumerate(all_codes, start=1):
        cache_path = CACHE_DIR / f"{safe_name(code)}.json"
        if cache_path.exists() and not args.refresh:
            resolved += 1
            if index % 100 == 0:
                print(f"[{index}/{len(all_codes)}] cached, skipping")
            continue
        result = fetch_topic(code, timeout=args.timeout, retries=args.retries)
        record = {
            "topic_code": code,
            "resolved_id": result["resolved_id"] if result else None,
            "title": (result["details"].get("title") if result else None),
            "description_clean": (
                strip_html(result["details"].get("description") or "") if result else ""
            ),
        }
        cache_path.write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if record["resolved_id"]:
            resolved += 1
        status = "ok" if record["description_clean"] else "no-desc/404"
        print(
            f"[{index}/{len(all_codes)}] {code} -> {record['resolved_id'] or '404'}"
            f" ({len(record['description_clean'])}c, {status})"
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    build_outputs(all_codes, subcall_to_codes)
    print(f"\nResolved {resolved}/{len(all_codes)} topics.")
    print(f"Wrote {TOPIC_CSV.name} and {SUBCALL_JSON.name}")


def build_outputs(
    all_codes: list[str],
    subcall_to_codes: dict[str, list[str]],
) -> None:
    """Consolidate the per-topic cache into a topic CSV and a subCall->scope JSON."""
    topic_desc: dict[str, dict[str, Any]] = {}
    for code in all_codes:
        cache_path = CACHE_DIR / f"{safe_name(code)}.json"
        if cache_path.exists():
            topic_desc[code] = json.loads(cache_path.read_text(encoding="utf-8"))

    rows = [
        {
            "topic_code": code,
            "resolved_id": rec.get("resolved_id"),
            "title": rec.get("title"),
            "description_chars": len(rec.get("description_clean") or ""),
            "description_clean": rec.get("description_clean") or "",
        }
        for code, rec in sorted(topic_desc.items())
    ]
    pd.DataFrame(rows).to_csv(TOPIC_CSV, index=False)

    # Aggregate per subCall: ordered, de-duplicated official scope texts.
    subcall_desc: dict[str, list[str]] = {}
    for subcall, codes in subcall_to_codes.items():
        seen: set[str] = set()
        descriptions: list[str] = []
        for code in codes:
            text = (topic_desc.get(code, {}) or {}).get("description_clean") or ""
            if text and text not in seen:
                seen.add(text)
                descriptions.append(text)
        subcall_desc[subcall] = descriptions
    SUBCALL_JSON.write_text(
        json.dumps(subcall_desc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
