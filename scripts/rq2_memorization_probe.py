"""No-candidate memorization probe (review point 2b).

Question: can a backend name the correct Horizon 2020 subCall for a test project
*without any candidate list* -- i.e. purely from the query fields it already sees
in the reranker? If it can, a high Recall@1 in the reranking table is retrieval
from model weights (memorization of the public CORDIS corpus), not reasoning over
the provided candidates.

Design
- Input: ONLY the model-visible query fields (title, objective text, keywords,
  budget, contribution, duration). No candidate pool, no label list.
- Output: the model's single best guess at the CORDIS subCall label.
- Scoring against the frozen ground truth:
    exact       normalized string identity with the true label
    year_scheme label matches once the 4-digit year is removed from both sides
                (i.e. right programme/scheme, maybe wrong year)
- Chance baseline: uniform over the candidate pool (|pool| labels) ~ 0.2%, so any
  non-trivial exact rate with zero candidates is dispositive of memorization.

Same GeminiClient / AnthropicClient path as the reranker, so this measures the
same backend under the same API settings. Resumable: existing (model, query_id)
rows in the output file are reused.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from rq2_core import (
    RESULT_ROOT,
    extract_json_object,
    load_frozen_split,
    load_ground_truth,
    make_client,
    model_visible_query,
)

_YEAR = re.compile(r"(?:19|20)\d{2}")


def normalize(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(label or "").lower())


def strip_year(label: str) -> str:
    return normalize(_YEAR.sub("", str(label or "")))


def build_probe_prompt(query: dict[str, Any]) -> str:
    visible = model_visible_query(query)
    return f"""You are given a European Union Horizon 2020 research project.
Name the single Horizon 2020 funding call (CORDIS "subCall" identifier) under
which this project was funded.

Labels follow CORDIS conventions, e.g. "ERC-2016-STG", "H2020-SFS-2014-2",
"H2020-MSCA-ITN-2015". Return your single best guess even if unsure.

Return JSON only, no commentary:
{{"predicted_call_label": "<one CORDIS subCall identifier>"}}

Project:
{json.dumps(visible, ensure_ascii=False, sort_keys=True)}
"""


def parse_prediction(raw: str) -> str:
    try:
        obj = extract_json_object(raw)
    except Exception:
        return ""
    if isinstance(obj, dict):
        for key in ("predicted_call_label", "call_label", "label", "subCall"):
            if obj.get(key):
                return str(obj[key]).strip()
    if isinstance(obj, str):
        return obj.strip()
    return ""


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row["query_id"])] = row
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--allow-test", action="store_true",
                        help="Required guard to touch the sealed test split.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Probe only the first N queries (debug).")
    parser.add_argument("--output-dir",
                        default=str(RESULT_ROOT / "rq2_memorization_probe"))
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to probe the test split without --allow-test.")

    rows = load_frozen_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    truths = load_ground_truth(args.split)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = out_dir / f"{args.split}_{args.model}_predictions.jsonl"
    done = load_done(per_query_path)

    client = make_client(args.model)

    with per_query_path.open("a", encoding="utf-8") as handle:
        for i, row in enumerate(rows, 1):
            qid = str(row["query_id"])
            if qid in done:
                continue
            true_label = truths[qid][0]
            raw = client.generate(build_probe_prompt(row))
            pred = parse_prediction(raw)
            rec = {
                "query_id": qid,
                "model": args.model,
                "true_call_label": true_label,
                "predicted_call_label": pred,
                "exact": int(normalize(pred) == normalize(true_label)),
                "year_scheme": int(
                    bool(pred) and strip_year(pred) == strip_year(true_label)
                ),
                "usage": getattr(client, "last_usage_metadata", {}),
            }
            done[qid] = rec
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            if i % 10 == 0:
                print(f"  [{args.model}] {i}/{len(rows)}")

    results = [done[str(r["query_id"])] for r in rows if str(r["query_id"]) in done]
    n = len(results)
    exact = sum(r["exact"] for r in results)
    year_scheme = sum(r["year_scheme"] for r in results)
    n_pool = 0
    try:
        from rq2_core import load_candidate_pool
        n_pool = len(load_candidate_pool())
    except Exception:
        pass
    summary = {
        "model": args.model,
        "split": args.split,
        "n": n,
        "exact_label_accuracy": exact / n if n else None,
        "year_agnostic_scheme_accuracy": year_scheme / n if n else None,
        "chance_exact_uniform_over_pool": (1.0 / n_pool) if n_pool else None,
        "candidate_pool_size": n_pool,
        "interpretation": (
            "no candidate list was provided; any exact rate far above "
            "chance_exact_uniform_over_pool indicates the label is recovered from "
            "model weights (CORDIS memorization) rather than from prompt candidates"
        ),
    }
    summary_path = out_dir / f"{args.split}_{args.model}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
