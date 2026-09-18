from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Optional, TypedDict


ROOT = Path(__file__).resolve().parents[1]
FROZEN_DIR = ROOT / "data" / "frozen" / "rq2_v1_seed42_description_filtered"
RESULT_ROOT = ROOT / "results" / "rq2_architecture"

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_TEMPERATURE = 0.0
GEMINI_MAX_OUTPUT_TOKENS = 32768
GEMINI_MAX_RETRIES = 6
GEMINI_RETRY_SECONDS = 5.0
# Per-call client deadline. The SDK default is short enough that large enriched /
# historical prompts intermittently trip 504 DEADLINE_EXCEEDED; 600s removes it.
GEMINI_REQUEST_TIMEOUT = 600

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
ANTHROPIC_MAX_OUTPUT_TOKENS = 8192
ANTHROPIC_MAX_RETRIES = 3
ANTHROPIC_RETRY_SECONDS = 2.0

# List price in USD per 1,000,000 tokens as (input, output). Used only for the
# budget accounting (RQ2 cost table); it does not affect ranking. Gemini figures
# are the public paid-tier text prices; Claude figures are the published list
# prices (Sonnet 5 carries a lower $2/$10 introductory rate through 2026-08-31,
# but we bill at list to keep the comparison conservative). Update if prices move.
MODEL_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    # "latest" aliases resolve to the current flash / pro tiers; priced at those
    # tiers (approximate — record the alias, not a pinned version).
    "gemini-flash-latest": (0.30, 2.50),
    "gemini-pro-latest": (1.25, 10.00),
    "gemini-3-flash-preview": (0.30, 2.50),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
}


def provider_of(model_name: str) -> str:
    """Route a model string to its API provider."""
    if model_name == "mock":
        return "mock"
    if model_name.startswith("claude"):
        return "anthropic"
    if model_name.startswith("gemini") or model_name.startswith("gemma"):
        return "google"
    raise ValueError(f"Unknown model provider for {model_name!r}")


def usage_cost_usd(model_name: str, usage: dict[str, Any]) -> Optional[float]:
    """Dollar cost of one call from its usage metadata, or None if unpriced.

    Reads the provider-neutral token keys written by the clients below
    (prompt_token_count / candidates_token_count). Cache-read tokens, when
    present, are treated as full-price input here: none of these runs enable
    prompt caching, so the field is either absent or zero.
    """
    price = MODEL_PRICING_USD_PER_MTOK.get(model_name)
    if price is None:
        return None
    prompt_tokens = usage.get("prompt_token_count") or 0
    output_tokens = usage.get("candidates_token_count") or 0
    input_price, output_price = price
    return (prompt_tokens * input_price + output_tokens * output_price) / 1_000_000

PROMPT_VERSION = "rq2_ranker_prompt_v8_no_queryid_no_nprojects"
PARSER_VERSION = "rq2_ranker_parser_v1"
EVALUATOR_VERSION = "rq2_evaluate_v1"
# fundingScheme / dominant_fundingScheme / masterCall / dominant_masterCall are
# excluded from the reranker input: they share the subCall naming structure
# (near-label leakage per RQ1), so exposing them would let the reranker pattern-
# match the answer instead of reasoning over funding fit.
# query_id is the CORDIS project primary key (grant-agreement number). It is a
# bookkeeping handle only -- nothing in the prompt asks the model to use it -- and
# for any model whose training data includes the (public, heavily mirrored) CORDIS
# corpus it is a direct leakage channel to the project->call mapping. It is tracked
# outside the payload (see rq2_rerank_baseline.py), so it is withheld from model input.
# n_projects is a post-award count (projects aggregated into a subCall); it is never
# a legitimate query-time input and is the exact quantity the call-size bias analysis
# is about, so it is likewise withheld.
MODEL_VISIBLE_QUERY_FIELDS = [
    "project_title",
    "query_text",
    "project_keywords",
    "totalCost_eur",
    "ecMaxContribution_eur",
    "duration_months",
]
MODEL_VISIBLE_CANDIDATE_FIELDS = [
    "call_label",
    "funding_call_descriptions",
]
BLOCKED_MODEL_INPUT_FIELDS = {
    "query_id",
    "project_id",
    "true_call_label",
    "subCall",
    "masterCall",
    "dominant_masterCall",
    "fundingScheme",
    "dominant_fundingScheme",
    "funding_call_id",
    "funding_call_ids",
    "funding_call_description",
    "n_projects",
}


class RQ2State(TypedDict, total=False):
    query: dict[str, Any]
    candidates: list[dict[str, Any]]
    scored: list[dict[str, Any]]
    ranked: list[dict[str, Any]]
    raw_output: str
    errors: list[str]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_frozen_split(split: str, frozen_dir: Path = FROZEN_DIR) -> list[dict[str, Any]]:
    if split not in {"dev", "test"}:
        raise ValueError("split must be 'dev' or 'test'")
    return read_jsonl(frozen_dir / f"{split}_queries.jsonl")


def load_candidate_pool(frozen_dir: Path = FROZEN_DIR) -> list[dict[str, Any]]:
    pool = read_json(frozen_dir / "candidate_pool.json")
    return pool["candidates"]


def load_ground_truth(split: str, frozen_dir: Path = FROZEN_DIR) -> dict[str, list[str]]:
    if split == "test":
        return read_json(frozen_dir / "ground_truth.json")["truths"]
    if split == "dev":
        rows = load_frozen_split("dev", frozen_dir)
        return {str(row["query_id"]): [row["true_call_label"]] for row in rows}
    raise ValueError("split must be 'dev' or 'test'")


def model_visible_query(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_title": row.get("project_title", ""),
        "query_text": row.get("query_text", ""),
        "project_keywords": row.get("project_keywords", ""),
        "totalCost_eur": row.get("totalCost_eur"),
        "ecMaxContribution_eur": row.get("ecMaxContribution_eur"),
        "duration_months": row.get("duration_months"),
    }


def shorten(text: Any, limit: int | None) -> str:
    compact = " ".join(str(text or "").split())
    if limit is None or len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def compact_candidate(
    candidate: dict[str, Any],
    *,
    max_descriptions: int | None = None,
    description_chars: int | None = None,
) -> dict[str, Any]:
    # Information-floor policy: expose call_label + the FULL set of
    # funding_call_descriptions (no per-item or count truncation). Exact
    # duplicates are dropped (call descriptions repeat heavily across the
    # projects aggregated into one subCall) while preserving order.
    seen: set[str] = set()
    descriptions: list[str] = []
    for description in candidate.get("funding_call_descriptions", []) or []:
        text = shorten(description, description_chars)
        if text and text not in seen:
            seen.add(text)
            descriptions.append(text)
    if max_descriptions is not None:
        descriptions = descriptions[:max_descriptions]
    return {
        "call_label": candidate["call_label"],
        "funding_call_descriptions": descriptions,
    }


def assert_no_blocked_model_fields(payload: dict[str, Any]) -> None:
    query_fields = set(payload["query"])
    candidate_fields = {
        field
        for candidate in payload["candidates"]
        for field in candidate
    }
    blocked_query_fields = query_fields.intersection(BLOCKED_MODEL_INPUT_FIELDS)
    blocked_candidate_fields = candidate_fields.intersection(
        BLOCKED_MODEL_INPUT_FIELDS - {"funding_call_description"}
    )
    if blocked_query_fields or blocked_candidate_fields:
        raise ValueError(
            "Blocked fields in model input: "
            f"query={sorted(blocked_query_fields)}, "
            f"candidates={sorted(blocked_candidate_fields)}"
        )


# Candidate-label anonymization (memorization control). The real subCall label
# (e.g. "ERC-2016-STG") is itself a leakage channel: a backend that has memorized
# the public CORDIS corpus can map project text -> exact label from weights, and
# the embedded year lets it separate temporal siblings by string. Replacing each
# candidate's label with an opaque, per-query alias forces the model to express
# its choice as an alias it cannot recover from memory, and removes the year from
# the label. To stop the channel reopening through the description text, the same
# scrub strips 4-digit years and any embedded call-code / literal label from the
# funding_call_descriptions. Aliases are mapped back to real labels for scoring.
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_CALLCODE_RE = re.compile(
    r"\b(?:H2020|HORIZON|FP7|ERC|MSCA|ECSEL|CS2|JTI|IMI2?)[A-Za-z0-9]*(?:-[A-Za-z0-9]+)+",
    re.IGNORECASE,
)


def scrub_candidate_text(text: str, real_label: str) -> str:
    """Remove the literal label, embedded call-codes, and 4-digit years so an
    anonymized candidate cannot be re-identified from its description text."""
    cleaned = text or ""
    if real_label:
        cleaned = cleaned.replace(real_label, " ")
    cleaned = _CALLCODE_RE.sub(" ", cleaned)
    cleaned = _YEAR_RE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def anonymize_candidates(
    compact_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return (anonymized candidates, alias -> real_label). The alias is opaque
    ("CALL-###") and its number is decorrelated from retriever rank via a shuffle
    seeded deterministically by the real-label set, so the alias token carries no
    rank, scheme, or year signal. Candidates stay in retriever order (a legitimate
    Stage-1 prior); only their identifier and description text are anonymized."""
    real_labels = [c["call_label"] for c in compact_candidates]
    seed = int(hashlib.sha256(" ".join(real_labels).encode()).hexdigest(), 16)
    aliases = [f"CALL-{i:03d}" for i in range(len(compact_candidates))]
    random.Random(seed).shuffle(aliases)
    alias_to_label: dict[str, str] = {}
    anonymized: list[dict[str, Any]] = []
    for candidate, alias in zip(compact_candidates, aliases):
        real = candidate["call_label"]
        alias_to_label[alias] = real
        scrubbed = [
            s for s in (
                scrub_candidate_text(d, real)
                for d in candidate.get("funding_call_descriptions", []) or []
            ) if s
        ]
        anonymized.append(
            {**candidate, "call_label": alias, "funding_call_descriptions": scrubbed}
        )
    return anonymized, alias_to_label


def build_rank_prompt(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    anonymize: bool = False,
) -> tuple[str, Optional[dict[str, str]]]:
    """Return (prompt, alias_to_label). alias_to_label is None unless anonymize."""
    visible_query = model_visible_query(query)
    compact_candidates = [compact_candidate(candidate) for candidate in candidates]
    alias_to_label: Optional[dict[str, str]] = None
    if anonymize:
        compact_candidates, alias_to_label = anonymize_candidates(compact_candidates)
    payload = {
        "query": visible_query,
        "candidates": compact_candidates,
    }
    assert_no_blocked_model_fields(payload)
    id_noun = "candidate id" if anonymize else "subCall label"
    id_note = (
        "- Candidates are identified by opaque ids (e.g. \"CALL-007\"); rank those ids.\n"
        if anonymize
        else "- The evaluated target is subCall.\n"
    )
    prompt = f"""You are ranking Horizon 2020 funding call candidates for one project.

Task:
- Use the project query and the fixed candidate pool.
- Rank the {id_noun}s from best match to worst match.
- Do not invent ids. Use only ids present in candidates.
- Return JSON only. No markdown, no commentary.

Expected JSON schema:
{{
  "ranked_call_labels": ["{id_noun} 1", "{id_noun} 2"]
}}

Notes:
{id_note}- Query-side target metadata is intentionally excluded from model input.
- Return ids only; do not include scores, reasoning, candidate descriptions, or extra keys.

Input:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}
"""
    return prompt, alias_to_label


def extract_json_object(text: str) -> Any:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.I)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = min(
            [idx for idx in [cleaned.find("{"), cleaned.find("[")] if idx >= 0],
            default=-1,
        )
        if start < 0:
            raise
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end < start:
            raise
        return json.loads(cleaned[start : end + 1])


def raw_output_parse_ok(raw_output: str) -> bool:
    """Whether the model output is valid JSON we can parse without salvage."""
    try:
        extract_json_object(raw_output)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def parse_ranked_output(
    raw_output: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_labels = [candidate["call_label"] for candidate in candidates]
    candidate_set = set(candidate_labels)
    try:
        parsed: Any = extract_json_object(raw_output)
    except (json.JSONDecodeError, ValueError):
        # Truncated / malformed model output (e.g. a runaway repetition that
        # blows past the token budget). Never crash the whole run: salvage any
        # candidate labels that appear verbatim, in order, before the corruption
        # and let the loop below back-fill the remainder in retriever order.
        parsed = None

    scores: dict[str, Any] = {}
    ranked_labels: list[str] = []

    if isinstance(parsed, dict):
        scores = parsed.get("scores", {}) or {}
        labels = (
            parsed.get("ranked_call_labels")
            or parsed.get("ranked")
            or parsed.get("ranking")
            or parsed.get("call_labels")
            or []
        )
    elif isinstance(parsed, list):
        labels = parsed
    else:
        labels = re.findall(r'"([^"]+)"', raw_output or "")

    for item in labels:
        label = item.get("call_label") if isinstance(item, dict) else item
        if label in candidate_set and label not in ranked_labels:
            ranked_labels.append(label)

    for label in candidate_labels:
        if label not in ranked_labels:
            ranked_labels.append(label)

    ranked = []
    for rank, label in enumerate(ranked_labels, start=1):
        score_value = scores.get(label) if isinstance(scores, dict) else None
        try:
            score = float(score_value) if score_value is not None else None
        except (TypeError, ValueError):
            score = None
        ranked.append({"rank": rank, "call_label": label, "score": score})
    return ranked


class GeminiClient:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_GEMINI_MODEL,
        temperature: float = GEMINI_TEMPERATURE,
        max_output_tokens: int = GEMINI_MAX_OUTPUT_TOKENS,
        max_retries: int = GEMINI_MAX_RETRIES,
    ) -> None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY before running Gemini.")

        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self._model = genai.GenerativeModel(model_name)
        self.last_usage_metadata: dict[str, Any] = {}

    @staticmethod
    def _usage_metadata(response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {}
        keys = [
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "cached_content_token_count",
        ]
        return {
            key: getattr(usage, key)
            for key in keys
            if getattr(usage, key, None) is not None
        }

    def generate(self, prompt: str) -> str:
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json",
        }
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": GEMINI_REQUEST_TIMEOUT},
                )
                self.last_usage_metadata = self._usage_metadata(response)
                return getattr(response, "text", "") or ""
            except Exception as exc:  # pragma: no cover - external API path
                last_error = exc
                error_text = str(exc).lower()
                if "429" in error_text or "quota" in error_text:
                    break
                if attempt == self.max_retries:
                    break
                time.sleep(GEMINI_RETRY_SECONDS * attempt)
        raise RuntimeError(f"Gemini call failed after {self.max_retries} attempts: {last_error}")


class AnthropicClient:
    """Claude reranker with the same generate()/last_usage_metadata surface as
    GeminiClient, so the LangGraph runner is model-agnostic. Thinking is
    disabled for a deterministic, cheap single-shot rank; Haiku 4.5 has no
    thinking by default and rejects the explicit disable, so it is omitted
    there. Output is coerced to JSON via structured outputs when the SDK/model
    supports it, with a graceful fall-back to the shared defensive parser."""

    RANK_SCHEMA = {
        "type": "object",
        "properties": {
            "ranked_call_labels": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["ranked_call_labels"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_ANTHROPIC_MODEL,
        max_output_tokens: int = ANTHROPIC_MAX_OUTPUT_TOKENS,
        max_retries: int = ANTHROPIC_MAX_RETRIES,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY before running Claude.")

        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self._use_structured = True
        self.last_usage_metadata: dict[str, Any] = {}

    def _request_kwargs(self, prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Haiku 4.5 runs without thinking by default and rejects {"type":
        # "disabled"}; Opus 4.8 / Sonnet 5 need it set explicitly (Sonnet 5
        # would otherwise think adaptively). Keeps the rank a single cheap pass.
        if not self.model_name.startswith("claude-haiku"):
            kwargs["thinking"] = {"type": "disabled"}
        if self._use_structured:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": self.RANK_SCHEMA}
            }
        return kwargs

    @staticmethod
    def _usage_metadata(response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        prompt_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        cache_read = getattr(usage, "cache_read_input_tokens", None)
        cache_write = getattr(usage, "cache_creation_input_tokens", None)
        total = None
        if prompt_tokens is not None and output_tokens is not None:
            total = prompt_tokens + output_tokens
        metadata = {
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": output_tokens,
            "total_token_count": total,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
        }
        return {key: value for key, value in metadata.items() if value is not None}

    @staticmethod
    def _response_text(response: Any) -> str:
        parts = [
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        return "".join(parts)

    def generate(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.messages.create(**self._request_kwargs(prompt))
                self.last_usage_metadata = self._usage_metadata(response)
                return self._response_text(response)
            except self._anthropic.BadRequestError as exc:  # pragma: no cover - API path
                # If structured outputs are unsupported for this model/SDK, drop
                # them once and let the defensive parser handle raw JSON text.
                if self._use_structured and "output_config" in str(exc).lower():
                    self._use_structured = False
                    continue
                raise RuntimeError(f"Anthropic bad request: {exc}") from exc
            except Exception as exc:  # pragma: no cover - external API path
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(ANTHROPIC_RETRY_SECONDS * attempt)
        raise RuntimeError(
            f"Anthropic call failed after {self.max_retries} attempts: {last_error}"
        )


def make_client(model_name: str, **kwargs: Any) -> Any:
    """Return a reranker client for a Gemini or Claude model string."""
    provider = provider_of(model_name)
    if provider == "anthropic":
        return AnthropicClient(model_name=model_name, **kwargs)
    if provider == "google":
        return GeminiClient(model_name=model_name, **kwargs)
    raise ValueError(f"make_client cannot build a client for {model_name!r}")


def rank_with_llm(
    client: Any,
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    anonymize: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    prompt, alias_to_label = build_rank_prompt(query, candidates, anonymize=anonymize)
    raw_output = client.generate(prompt)
    if alias_to_label is not None:
        # Parse over the anonymized candidate view, then map aliases back to real
        # subCall labels so scoring downstream is unchanged.
        anon_candidates = [
            {"call_label": alias} for alias in alias_to_label
        ]
        ranked = parse_ranked_output(raw_output, anon_candidates)
        for row in ranked:
            row["call_label"] = alias_to_label.get(row["call_label"], row["call_label"])
    else:
        ranked = parse_ranked_output(raw_output, candidates)
    return ranked, raw_output


def rank_metrics(ranked_by_query: dict[str, list[dict[str, Any]]], truths: dict[str, list[str]]) -> dict[str, Any]:
    true_ranks = []
    for query_id, ranked in ranked_by_query.items():
        truth = set(truths[str(query_id)])
        rank = next(
            (row["rank"] for row in ranked if row["call_label"] in truth),
            len(ranked) + 1,
        )
        true_ranks.append(rank)

    if not true_ranks:
        return {"n_queries": 0}

    return {
        "n_queries": len(true_ranks),
        "Recall@1": sum(rank <= 1 for rank in true_ranks) / len(true_ranks),
        "Recall@5": sum(rank <= 5 for rank in true_ranks) / len(true_ranks),
        "Recall@10": sum(rank <= 10 for rank in true_ranks) / len(true_ranks),
        "MRR": sum(1.0 / rank for rank in true_ranks) / len(true_ranks),
        "NDCG@5": sum((1.0 / math.log2(rank + 1)) if rank <= 5 else 0.0 for rank in true_ranks)
        / len(true_ranks),
        "mean_true_rank": sum(true_ranks) / len(true_ranks),
    }


def per_query_rows(
    ranked_by_query: dict[str, list[dict[str, Any]]],
    truths: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows = []
    for query_id, ranked in ranked_by_query.items():
        truth = set(truths[str(query_id)])
        true_rank = next(
            (row["rank"] for row in ranked if row["call_label"] in truth),
            len(ranked) + 1,
        )
        rows.append(
            {
                "query_id": query_id,
                "true_call_label": ";".join(truths[str(query_id)]),
                "true_rank": true_rank,
                "top1_call_label": ranked[0]["call_label"] if ranked else "",
                "top5_call_labels": json.dumps(
                    [row["call_label"] for row in ranked[:5]],
                    ensure_ascii=False,
                ),
            }
        )
    return rows
