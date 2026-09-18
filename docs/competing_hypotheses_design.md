# Competing-Hypotheses + Coordinator Architecture — Design Rationale & Schemas

*Design document only. No implementation in this file. Evaluated later against a
harder-pool baseline on BRIGHT biology under strict information parity.*

---

## 1. Why a different architecture (evidence, not intuition)

The full n=97 paired diagnostic of the `Concept→Rerank` pipeline (baseline single-pass
generalist vs. concept-hint-prepended reranker) established:

| Finding | Number | Implication |
|---|---|---|
| Baseline already correct @1 | 70/97 (72%) | Most queries need **no** intervention |
| Errors that are genuinely surface→latent | 12/27 (44%) | Only ~12% of all queries are addressable by concept abstraction |
| Errors that are gold-noise / ambiguous | 9/27 (33%) | Un-fixable by *any* architecture |
| Fix rate among errors | 3/27 (11%) | Concept signal converts a few |
| Break rate among baseline-correct | 5/70 (7%) | Universal injection **damages** easy queries |
| Mean Δrank on errors | **−1.52** (gold moves up) | Concept signal is genuinely informative |
| Mean Δrank on baseline-correct | **+0.20** (gold moves down) | Pure perturbation, no upside |

**Two conclusions drive the design:**

1. **The concept signal is real but mis-delivered.** A single dominant concept hint
   prepended to every query lifts the gold by 1.52 ranks on hard queries, yet perturbs
   already-correct rankings (5 demotions from rank 1, including one — id76 — where the
   concept was *correct*). The harm is **over-commitment / perturbation**, not wrong
   abstraction.

2. **The missing capability is arbitration.** All 5 regressions are "should have
   deferred to the surface reading." The pipeline has no mechanism to decide, per query,
   *whether the latent concept should override the surface reading at all.* And the
   correct answer was repeatedly sitting in the **alternative** hypothesis (id89's gold
   was the alternative; id64/id77 the concept beat the gold), which a single-hint pipeline
   discards.

The correct response is not to patch the hint (conditional gating, softer prompt) but to
change the **unit of reasoning**: from *one hypothesis imposed on the reranker* to
*several competing hypotheses arbitrated per candidate by a coordinator that can defer.*

---

## 2. Architecture (minimal: two hypothesis agents + one coordinator)

```
                         ┌─────────────────────────┐
      question  ───────► │  A1: Surface/Lexical     │──┐
          │              │      Hypothesis Agent    │  │
          │              └─────────────────────────┘  │
          │              ┌─────────────────────────┐  │   ┌──────────────────────┐
          ├────────────► │  A2: Latent Concept      │──┼─► │  A3: Coordinator      │ ─► ranked_ids
          │              │      Hypothesis Agent    │  │   │  (reasoning arbiter)  │
          │              └─────────────────────────┘  │   └──────────────────────┘
          │                                            │            ▲
   candidate pool ─────────────────────────────────────────────────┘
   (frozen, identical to baseline)
```

- **A1 and A2 run in parallel and are symmetric competitors** — neither is privileged.
  A1 represents "answer the question as literally asked"; A2 represents "answer the
  underlying principle." They deliberately encode the two readings that the diagnostic
  showed pull in different directions.
- **A3 is a genuine reasoning coordinator**, not a rule engine and not a score-averager
  (an explicit constraint carried over from the TREC-CT design). It reads *both*
  hypotheses and the *actual candidate documents*, and decides — per candidate and for
  the query as a whole — which reading the evidence supports, **including the option that
  the surface reading alone suffices and the concept should be ignored.** This is the
  arbitration capability the previous pipeline lacked, and it is what protects the 72%
  of already-correct queries.

### Design invariants (what this is NOT)
- **No averaging of scores**, no hard AND/OR gates, no weighted sums.
- **No debate, no reflection loop, no re-retrieval, no additional specialists.**
- **No modification of the retriever or the candidate pool** — the pool is frozen and
  byte-identical to the baseline's; the only independent variable is the architecture.
- **The coordinator may return the surface-only ranking unchanged** — deferring is a
  first-class, expected outcome, not a failure.

### Why each element maps to a diagnosed failure
| Diagnostic failure | Architectural remedy |
|---|---|
| Over-commitment: one concept perturbs easy queries | Coordinator can decide `surface_sufficient` and leave the ranking alone |
| Correct answer hidden in the *alternative* | A2 emits ranked alternatives; coordinator treats them as live competitors |
| No surface-vs-latent adjudication | A3 arbitrates per candidate with the documents in hand |
| Concept gains land at rank 2–5, not #1 | A3 reasons about the *top* explicitly rather than reordering globally on a hint |

---

## 3. Input / Output schemas (concrete, frozen contract)

All agents: `gemini-flash-latest`, temperature 0, JSON-only output. Candidate documents
are presented with **opaque aliases** (`D00…`) and the same `DOC_CAP=600` truncation as
the baseline, so no agent sees ids, labels, or scores.

### A1 — Surface / Lexical Hypothesis Agent
```json
// INPUT
{ "question": "<raw question text, ≤1500 chars>" }

// OUTPUT
{
  "surface_intent": "<what the question literally asks, one sentence>",
  "key_terms": ["<salient surface terms/entities the answer doc likely contains>"],
  "expected_answer_type": "<e.g. 'a chemical compound' | 'a mechanism' | 'an organism' | 'a method'>",
  "explicit_constraints": ["<hard constraints stated in the question, e.g. 'visible to naked eye, not microscopic'>"]
}
```
*Rationale:* `explicit_constraints` captures cases like id42 ("not a microscopic type")
that a concept-only reading ignores. `key_terms` gives the coordinator a lexical anchor
so it can recognise when the surface reading already picks the right doc.

### A2 — Latent Concept Hypothesis Agent
```json
// INPUT
{ "question": "<raw question text, ≤1500 chars>" }

// OUTPUT
{
  "primary_concept": "<the core latent principle/mechanism/compound/law>",
  "reasoning": "<1–2 sentences: why this, not the surface topic>",
  "concept_terms": ["<terms the concept-bearing doc likely contains>"],
  "alternatives": [
    { "concept": "<competing latent concept>", "confidence": "high|medium|low" }
  ]
}
```
*Rationale:* `alternatives` are **first-class**, not a throwaway field — the diagnostic
showed the gold often lives here. `confidence` lets the coordinator weigh, not average.

### A3 — Coordinator (reasoning arbiter)
```json
// INPUT
{
  "question": "<raw question text>",
  "surface_hypothesis": { ...A1 output... },
  "concept_hypothesis": { ...A2 output... },
  "candidates": [ { "id": "D00", "text": "<≤600 chars>" }, ... ]   // frozen pool, shuffled
}

// OUTPUT
{
  "decision": "surface_sufficient | concept_decisive | mixed",
  "rationale": "<why this decision, referencing the evidence in the candidates>",
  "per_candidate": [
    { "id": "D07", "satisfies": "surface | concept | alternative | both | neither",
      "note": "<short evidence-based justification>" }
  ],
  "ranked_ids": ["<all candidate ids, most→least relevant>"]
}
```
*Rationale:* `decision` externalises the arbitration and makes the "defer to surface"
path auditable; `per_candidate.satisfies` forces the coordinator to reason document-by-
document (not average a global hint); `ranked_ids` is the only field scored. If
`decision == "surface_sufficient"`, the coordinator is expected to return essentially the
surface-driven order — the mechanism that should eliminate the 7% break rate.

---

## 4. Evaluation protocol (to be run after the harder-pool baseline)

- **Same 97 biology queries, same frozen candidate pools** as the harder-pool baseline
  (pools serialized to disk so the architecture reranks byte-identical inputs).
- **Metrics:** R@1, Recall@5, MRR, nDCG@10, mean gold rank — paired vs the harder-pool
  single-pass baseline.
- **Significance:** exact McNemar on R@1 discordant pairs; report fixed/broken ids.
- **Mechanism checks:** (a) break rate among baseline-correct queries — the design's
  central promise is that this drops toward 0 via `surface_sufficient`; (b) how often the
  gold matched an `alternative`; (c) distribution of `decision` values.
- **Cost/latency:** 3 calls/query (A1, A2 parallel; A3), reported against the 1 call/query
  baseline — the architecture must justify its cost, not only its accuracy.

**Success criterion (pre-registered):** the coordinator recovers the surface→latent
errors *without* re-introducing the perturbation cost — i.e., positive net R@1 driven by
`fixed > broke`, with break rate among baseline-correct meaningfully below the 7%
observed for universal Concept→Rerank.
