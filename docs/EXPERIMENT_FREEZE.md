# Experimental Freeze — BRIGH​T Biology Reranking / Agentic Architecture Study

*Frozen on the `exp2-architecture` branch. No further tuning. This file is the single
source of truth for reproduction.*

## 0. One-line summary
Reranking a fixed 100-document candidate pool for 97 BRIGHT-biology reasoning queries with
a single-pass LLM baseline and three agentic architectures, under strict information parity
(opaque document ids, identical frozen pools). Independent variable = **architecture only**.

---

## 1. Dataset
- **Benchmark:** BRIGHT (`xlangai/BRIGHT` on HuggingFace), the reasoning-intensive retrieval
  benchmark. Configs used: `documents` (corpus) and `examples` (queries + gold labels).
- **Domain:** `biology` (single domain, frozen).
- **Source files:** parquet from `refs/convert/parquet` —
  `documents/biology/0000.parquet` (57,359 docs), `examples/biology/0000.parquet` (103 examples).
  Cached at `data/bright_mvp/documents.parquet`, `data/bright_mvp/examples.parquet`.
- **Gold labels:** each example's `gold_ids` are the relevant documents; `excluded_ids`
  are held out of the candidate pool. `gold_answer` (free text) is used only for the
  closed-book memorization probe, never for ranking.

## 2. Query set (n = 97)
- **Eligibility filter:** examples whose number of `gold_ids` present in the biology corpus
  is between 1 and `MAX_GOLD = 6` inclusive → **97 eligible** of 103.
- **Deterministic selection:** `elig.sort(key=lambda e: str(e["id"]))`, then
  `random.Random(42).shuffle(elig)`, then take the first `N`.
  - Final frozen evaluation: **N = 97** (all eligible).
  - Exploratory 20-pool runs used the first 30 (and 15) of the *same* shuffle.
- Selection is **model-independent** (before any model is run); no selection on performance.

## 3. Candidate pool (POOL = 100), frozen
For each query, in the shuffled sample order:
1. `gold` = `gold_ids` present in corpus, capped at `MAX_GOLD = 6`.
2. `excl` = `excluded_ids ∪ gold`.
3. Hard negatives = corpus documents in **descending TF-IDF cosine similarity** to the query,
   skipping any id in `excl`, appended until `len(gold) + len(distract) >= 100`.
4. `pool = gold + distract`; shuffled once with a **single `random.Random(7)`** advanced
   query-by-query across the sample (so the whole sequence of pools is deterministic).
5. Aliases `D00 … D99` assigned in shuffled order. Documents truncated to `DOC_CAP = 600`
   chars in every prompt. Ids/labels/scores are never shown to any agent.
- **Frozen artefact:** `data/bright_hardpool/biology/pools.json`
  (`{qid: {query, gold_aliases, alias_to_text}}`, 97 pools, ~3.7 MB). All four systems
  rerank byte-identical inputs from this file.

## 4. Retriever (hard-negative mining only)
- `sklearn.feature_extraction.text.TfidfVectorizer(stop_words="english", max_features=50000)`
  fit on the **full** biology corpus (57,359 docs); query transformed with the same
  vectorizer; cosine similarity via `linear_kernel`; descending sort.
- Purely lexical, deterministic, no neural retriever, no training, no random state.
- Role: selects the hard distractors that fill the pool. It does **not** rank at eval time —
  ranking is done by the LLM systems over the frozen pool.

## 5. Backend model
- **Model:** `gemini-flash-latest` (Google), alias recorded (not pinned).
- **Temperature:** `0.0`. **max_output_tokens:** `32768`. **max_retries:** `6`.
- **Pricing used for cost accounting:** $0.30 / 1M input tokens, $2.50 / 1M output tokens
  (`MODEL_PRICING_USD_PER_MTOK["gemini-flash-latest"]`). Cost computed from real
  `usage_metadata` (`prompt_token_count`, `candidates_token_count`).
- No Claude/Anthropic model is used as a backend in this study (Claude is the assistant, not
  the ranker).

## 6. Seeds
| Purpose | Seed |
|---|---|
| Query sample shuffle | `random.Random(42)` |
| Pool shuffle (single RNG across queries) | `random.Random(7)` |
| TF-IDF | deterministic, no seed |
| LLM | temperature 0 (no sampling seed) |

## 7. Systems (frozen prompts)
All systems consume the frozen pool and output a ranking of all 100 ids. Missing ids in any
model output are appended in pool order (identical fallback across systems).

### A. Single-pass baseline
One call: "rank ALL ids from most to least relevant." (`p_rerank_plain`).
Frozen ranking + per-query first-gold-rank stored at
`results/bright_hardpool/biology/baseline_per_query.json`.

### B. Concept-guided reranking
Two calls: (1) **concept-abstraction agent** → `{concept, reasoning, alternative}`;
(2) the baseline rerank prompt **plus a "Latent concept hypothesis" block** carrying that
concept. One dominant concept prepended to the reranker.

### C. Competing-Hypotheses v1 (full re-rank)
Three calls: **A1 surface/lexical** → `{surface_intent, key_terms, expected_answer_type,
explicit_constraints}`; **A2 latent concept** → `{primary_concept, reasoning, concept_terms,
alternatives:[{concept, confidence}]}`; **A3 coordinator** sees both hypotheses + all 100
candidate texts and **regenerates the full ranking** (`{decision, rationale, per_candidate,
ranked_ids}`). `decision` is *not enforced* — it is descriptive only.

### D. Competing-Hypotheses Anchor-and-Edit  ← FINAL SYSTEM
Three calls: **A1** and **A2** identical to C; **A3 coordinator (anchor variant)** outputs
`{decision, promote_ids (≤2), rationale}` and is **blind to the baseline ranking**.
**Mechanical enforcement (in the harness, not the LLM):**
- `decision == "surface_sufficient"` *or* empty `promote_ids` → final ranking = **System A's
  frozen ranking, unchanged**.
- otherwise → `promote_ids` moved to the **front** of System A's ranking; all other documents
  keep their baseline order (surgical edit, ≤2 documents moved).

### Coordinator logic (final, D) — precise
- Reasoning arbiter, **not** a rule engine, **no** score averaging, **no** hard AND/OR gates.
- Chooses one of `{surface_sufficient, concept_decisive, mixed}` and, unless deferring, names
  ≤2 promotion ids justified by evidence in the candidate texts.
- The **anchor** (System A's ranking) is the strong prior; the coordinator only *edits* it.
  This enforcement is the single change from C to D.

### Hypothesis-agent logic
- **A1 (surface/lexical)** and **A2 (latent concept)** are symmetric, independent, run per
  query with no ordering dependence; neither is privileged. A2's `alternatives` are
  first-class inputs to the coordinator.

### Evidence-agent logic
- The final system has **no standalone evidence agent**. Evidence assessment over the
  candidate documents is performed **by the coordinator (A3)**, which reads the candidate
  texts and selects `promote_ids` on the evidence. (In the C design the same role appears as
  the coordinator's `per_candidate` annotations.) This is recorded here to avoid ambiguity:
  the "evidence" function is a responsibility of A3, not a separate agent.

## 8. Evaluation metrics
- **R@1** (top-1 is a gold), **Recall@5**, **MRR** (first gold), **nDCG@10** (binary gain,
  `gain = 1` per gold; `dcg = Σ rel/log2(i+2)`), **mean gold rank** (mean rank of all golds).
- **Paired** against System A on the identical pools: per-query fixed (A wrong→system right
  @1) and broke (A right→system wrong @1).
- **Significance:** exact McNemar on R@1 discordant pairs; paired bootstrap/permutation on
  per-query reciprocal rank for MRR. **No samples removed; contestable gold labels retained.**

## 9. Reproduction
```
# 1. pools + baseline (System A), freezes pools.json
venv/bin/python scripts/bright_hardpool_baseline.py
# 2. System C
venv/bin/python scripts/bright_competing_hypotheses.py
# 3. System D (final)
venv/bin/python scripts/bright_ch_anchor.py
# 4. System B on frozen pool + latency/cost probe
venv/bin/python scripts/bright_finalize_measure.py
# 5. final table + statistics
venv/bin/python scripts/bright_final_stats.py
```
Environment: project venv (Python 3.11); `GROQ_API_KEY`/`GOOGLE_API_KEY` from `.env`
(whitespace-trimmed loader). Dataset-gating scripts (`bright_mvp.py`, closed-book probe)
document how the benchmark was accepted.

## 10. Provenance of the design
`competing_hypotheses_design.md` (design rationale + I/O schemas, pre-registered success
criterion). The architecture was designed from the n=97 diagnostic of System B, then frozen
before the final hard-pool evaluation.
