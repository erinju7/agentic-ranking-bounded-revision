# Agentic Architecture — Experiment Design

**Status:** proposal for approval. No implementation until the open decisions
(§18) are resolved. The only executable step approved in principle is the small
backend validation (§17), and even that waits for sign-off on its exact query
sample, model ids, and prompt.
**Scope:** a controlled 2×2 factorial comparison of reasoning architecture ×
evidence-access strategy over the existing leakage-controlled Horizon-call
matching benchmark. The goal is clean isolation of *architecture* and *access
strategy*, not a maximally capable autonomous system.

---

## 1. Motivation

The dissertation asks whether increasing agentic control-flow autonomy improves
retrieval/reranking of Horizon funding calls. The profile ablation showed that no
fixed candidate projection dominates (Compact wins Recall@1, Full wins deeper
recall), because long descriptive text is *diluting* for broad calls but
*discriminating* for specialised and temporal-sibling calls. This motivates
separating two orthogonal design choices and testing them factorially:

- **Reasoning architecture** — one model reasons over everything (monolithic /
  generalist) vs. the problem is split across single-purpose specialists that a
  coordinator aggregates (specialist-decomposed multi-agent).
- **Evidence-access strategy** — all admissible detail is materialised up front
  (eager Full) vs. a compact view is materialised up front and full detail is
  fetched on demand for a bounded number of candidates (selective Compact +
  `inspect`).

The manipulated variables are these two factors. Everything else — retriever,
candidate pool, retrieved top-50, backend, temperature, ids, query
representation, guarded datastore, evaluation — is held constant.

---

## 2. Experimental design — 2×2 factorial

Two design dimensions, four primary cells:

| Reasoning architecture ↓ / Access strategy → | Eager **Full** access | Selective **Compact + inspect** |
|---|---|---|
| **Monolithic / generalist** | **L0 Full** | **L1** |
| **Specialist multi-agent** | **L2** | **L3** |

These are **controlled factorial conditions, not a monotonic autonomy ladder.**
L0 Full is the least agentic and L3 the most operationally agentic, but the
scientific value is in the *crossed* comparisons, not a single ordering.

**L0 Compact** is retained as an **auxiliary efficiency baseline** *outside* the
2×2 (monolithic × selective-but-no-inspect). It is the cheapest fixed projection
and is reported for cost context, not as a factorial cell.

---

## 3. Shared datastore

All systems read exactly one **guarded candidate datastore** and may access
nothing else. Admissible candidate fields (per `admissibility_manifest.yaml`):

`title`, `scope` (full), `specific_challenge` (full), `expected_outcome` (full),
`normalized_action_type` (year-stripped instrument class), `call_scale`
(call-level budget-envelope band; **not** a per-project ceiling), `submission_stage`.

**Excluded everywhere (leakage controls):** dates (opening/deadline — context-gated
off), raw call/topic identifiers, `fundingScheme`, `masterCall`, topic codes, any
post-award field. Candidates are addressed only by **anonymized opaque ids**
(`C_xxxxxxxx`). `topic_status` is admissible but signal-free (all "Closed") and
excluded by default.

**Envelope invariant.** The *collective admissible information envelope* — every
admissible field for all 50 candidates — is **identical across all four systems**.
The cells differ only in how much of that envelope is *materialised* and *when*:
eager cells materialise the full envelope up front; selective cells materialise
the Compact projection up front and up to five candidates' full semantic fields on
demand. No system can reach information another cannot.

---

## 4. Shared query representation

Identical across all systems (guarded query fields only): `project_title`,
`query_text` (objective/abstract), `project_keywords`, `totalCost_eur`,
`ecMaxContribution_eur`, `duration_months`. No query-side date/year,
`fundingScheme`, `masterCall`, `subCall`, or topic code is ever present. Query
fields are fully available to whichever component needs them; the Compact/Full
distinction below concerns **candidate** fields only.

---

## 5. Exact projections per component

The Compact vs Full distinction only bites on the **semantic** candidate fields
(`scope`/`specific_challenge`/`expected_outcome`), which have a long form (Full)
and a summary form (`scope_summary`, ≤250 chars, sentence-aware). The structured
fields (`normalized_action_type`, `call_scale`, `submission_stage`) have no long
form, so their Compact projection equals their Full projection.

| Component | **Compact projection** (initial) | **Full semantic fields** (eager in L0 Full / L2; via `inspect` in L1 / L3) |
|---|---|---|
| Generalist agent (L1) | `title`, `scope_summary`, `normalized_action_type`, `call_scale`, `submission_stage` | `scope`, `specific_challenge`, `expected_outcome` |
| Theme Agent | `title`, `scope_summary` | `scope`, `specific_challenge`, `expected_outcome` |
| Action Agent | `normalized_action_type`, `submission_stage` | — (no long form; Compact = Full) |
| Budget-Scale Agent | `call_scale` [+ query `totalCost_eur`/`ecMaxContribution_eur`/`duration_months`] | — (no long form; Compact = Full) |
| Coordinator | candidate index (`title`, `scope_summary`, `normalized_action_type`, `call_scale`, `submission_stage`) + specialist outputs | `scope`, `specific_challenge`, `expected_outcome` (via `inspect`, **L3 only**) |

The eager cells (L0 Full, L2-Theme) render the Full semantic fields for **all 50**
candidates. The selective cells (L1, L3) render Compact for all 50 and recover Full
for **≤5** via `inspect`. That difference **is** the access-strategy factor.

---

## 6. The four cells

### 6.1 L0 Full — monolithic × eager Full
One monolithic model call. Receives the Full candidate projection for all 50
eagerly. No tools, no revision.

### 6.2 L1 — generalist × selective Compact + inspect
One generalist tool-using agent. Starts from the Compact projection of all 50. May
`inspect` at most **5** distinct candidates and perform **one** final revision.
No planner, critic, memory, external retrieval, or query reformulation.

### 6.3 L2 — specialist × eager Full
Theme, Action, and Budget-Scale specialists + Coordinator. Each specialist
receives its **approved Full field projection eagerly** and runs **once**. The
Coordinator runs **once** over the Compact index + specialist outputs. No tools,
no revision, no reflection, no repeated specialist calls.

### 6.4 L3 — specialist × selective Compact + inspect
Same specialist roles and coordinator architecture as L2, with the access strategy
switched to selective: specialists initially receive their **Compact** field
projections (Theme sees `scope_summary`, not full semantic detail; Action and
Budget-Scale are unchanged since they have no long form). The **Coordinator** may
`inspect` at most **5** distinct candidates and perform **one** final revision.
Nothing else changes (specialists still run once, no reflection, no repeats).

### 6.5 Per-cell access tables

| Attribute | L0 Full | L1 | L2 | L3 |
|---|---|---|---|---|
| Reasoning architecture | monolithic | generalist | specialist | specialist |
| Access strategy | eager Full | selective | eager Full | selective |
| Candidate fields visible initially | Full (all 50) | Compact (all 50) | Full per specialist (all 50) | Compact per specialist (all 50) |
| Full semantic fields eagerly available | **yes** | no | **yes** (to Theme) | no |
| `inspect` available | no | yes (agent) | no | yes (coordinator) |
| `inspect` budget | 0 | 5 candidates | 0 | 5 candidates |
| Revision allowance | 0 | 1 | 0 | 1 |
| Model reasoning calls / query | 1 | 2 (rank → revise) | 4 (3 specialists + coordinator) | 5 (3 specialists + coordinator aggregate + revise) |

**Field-routing (who receives what):**

| Field | L0 Full | L1 | L2 | L3 |
|---|---|---|---|---|
| `title` | model | agent | Theme | Theme |
| `scope_summary` | — | agent | — | Theme (initial) |
| `scope`/`specific_challenge`/`expected_outcome` | model (eager) | agent (via inspect ≤5) | Theme (eager) | Coordinator (via inspect ≤5) |
| `normalized_action_type`, `submission_stage` | model | agent | Action | Action |
| `call_scale` (+ query budget/duration) | model | agent | Budget-Scale | Budget-Scale |
| Compact index + specialist outputs | — | — | Coordinator | Coordinator |

---

## 7. The four controlled comparisons

| # | Comparison | Factor held | Factor varied |
|---|---|---|---|
| 1 | **L0 Full vs L2** | access = eager Full | architecture: monolithic → specialist |
| 2 | **L1 vs L3** | access = selective Compact + inspect | architecture: generalist → specialist |
| 3 | **L0 Full vs L1** | architecture = monolithic/generalist | access: eager Full → selective |
| 4 | **L2 vs L3** | architecture = specialist | access: eager Full → selective |

Comparisons 1–2 estimate the **architecture** main effect at each access level;
3–4 estimate the **access-strategy** main effect at each architecture level; the
difference between them is the **interaction**.

**Explicit note on comparison 4 (L2 vs L3):** this changes the *evidence-access
strategy*, not merely the presence of a tool. In L2 the specialists already hold
the Full semantic fields eagerly, so an `inspect` tool would be redundant; the L3
change is that specialists start from Compact and full detail is recovered
selectively by the coordinator. L2 vs L3 therefore isolates eager-Full vs
selective access under a fixed specialist architecture.

---

## 8. Fairness

**Held constant across all systems:** retriever, candidate pool (413), retrieved
top-50, model backend, temperature, candidate anonymized ids, evaluation protocol,
the `inspect` tool and its budget, the query representation, and the guarded
datastore. The **admissible information envelope is identical** (§3). Only the two
design factors differ.

The eager cells materialise Full detail for all 50; the selective cells materialise
Compact for all 50 + Full for ≤5. This is **not** an unfair information advantage —
it is precisely the access-strategy treatment. The experiment asks whether bounded
selective access can match or beat eager full access (and at what cost), under each
architecture.

---

## 9. Tool specification

A single read-only tool, available to the L1 agent and the L3 coordinator only.

```
inspect(candidate_id: str, fields: list[str]) -> InspectResult
```
- `candidate_id` — one anonymized id in the retrieved set.
- `fields` — subset of `{"scope", "specific_challenge", "expected_outcome"}`.
- **Budget:** ≤ **5 distinct candidates** per query; re-inspecting the same id or
  requesting multiple fields in one call does **not** consume extra budget (per
  §18-D4, recommended). One inspection phase → exactly **one** revision. No loops.
- **Guard:** passes through the admissibility manifest; can only ever return the
  three allowed full-text fields, never dates, raw codes, `fundingScheme`/
  `masterCall`, or a de-anonymized id. Returned text is byte-identical to L0 Full.

---

## 10. Specialist definitions

Each specialist is single-purpose and sees only the fields for its dimension.

- **Theme Agent.** Semantic match only. Input: query (`project_title`,
  `query_text`, `project_keywords`) + candidate semantic fields (Full in L2,
  `scope_summary` in L3). Output: per-candidate relevance in [0,1] + brief evidence.
- **Action Agent.** Action-type compatibility only. Input: full query + candidate
  `{normalized_action_type, submission_stage}`. **Infers** the project's likely
  instrument from query content (single-PI frontier → ERC; fellowship → MSCA-IF;
  consortium innovation → IA) and judges compatibility. Never receives a
  project-side scheme (leakage-blocked).
- **Budget-Scale Feasibility Agent** (renamed per D2). Coarse feasibility only.
  Input: query `{totalCost_eur, ecMaxContribution_eur, duration_months}` +
  candidate `call_scale`. Constraints:
  - assesses only whether the requested budget (where available) is *broadly*
    compatible with the candidate `call_scale` band;
  - **must not** treat `call_scale` as a precise project-level ceiling (no false
    precision);
  - **must not** claim duration compatibility — no approved candidate-side duration
    field exists in the guarded datastore (dates gated); duration is reported for
    context only, never as a compatibility verdict, unless such a field is later
    approved;
  - if query budget is missing → `missing_evidence = true` and a **neutral**
    contribution;
  - **may not independently veto** a candidate;
  - default influence **lower** than Theme and Action.
  Its structural weakness is documented as an experimental limitation and a
  possible finding.

Specialists do not communicate; each emits a structured judgment consumed only by
the Coordinator.

---

## 11. Coordinator

- **Input (L2):** candidate ids + Compact index + the three specialists' outputs.
- **Input (L3):** the same, plus `inspect` (≤5) and one revision.
- **Aggregation.** Reconciles the three dimensions into a single 50-ranking, using
  an **interpretable default influence** (recommended development default):
  - Theme: **0.60**
  - Action: **0.25**
  - Budget-Scale: **0.15**
  These are a starting point only; the **final weights are frozen on the
  development set before any sealed-test evaluation.** The Budget-Scale agent
  cannot veto; a `missing_evidence` Budget-Scale contribution is treated as
  neutral (its weight is redistributed or zeroed — see §18-D2b). The aggregation
  mechanism itself (LLM coordinator guided by these weights vs. a deterministic
  weighted score-blend vs. both, with the deterministic blend as an ablation) is
  §18-D2c.
- **L3 behaviour.** After aggregating, the coordinator inspects contested
  candidates (e.g. specialist disagreement or near-identical clusters), then
  revises once.
- **Output.** The final 50-ranking (§12), the sole scored artefact, schema-identical
  across all systems.

---

## 12. JSON schemas

All model calls use JSON-constrained output; candidate ids are always anonymized.

**inspect() request / result**
```json
{ "candidate_id": "C_08b81709", "fields": ["scope", "specific_challenge"] }
```
```json
{ "candidate_id": "C_08b81709",
  "fields": { "scope": "…", "specific_challenge": "…" },
  "budget_remaining": 4 }
```

**Theme Agent**
```json
{ "assessments": [
  { "candidate_id": "C_08b81709", "relevance": 0.82, "evidence": "…" } ] }
```

**Action Agent**
```json
{ "inferred_project_instrument": "MSCA Individual Fellowship",
  "assessments": [
    { "candidate_id": "C_08b81709", "candidate_action_type": "MSCA Individual Fellowship",
      "compatibility": "match", "confidence": 0.9 } ] }
```

**Budget-Scale Feasibility Agent**
```json
{ "project_cost_eur": 1998000,
  "project_duration_months": 24,
  "missing_evidence": false,
  "assessments": [
    { "candidate_id": "C_08b81709", "candidate_call_scale": "large (200M-1B)",
      "scale_compatibility": "compatible",   // compatible | unclear | mismatch
      "veto": false,                          // must always be false
      "rationale": "coarse: project cost within a large envelope" } ] }
```

**Coordinator / Final ranking (shared scored schema)**
```json
{ "ranked_candidate_ids": ["C_08b81709", "C_5159f5f6", "…50 ids…"],
  "weights_used": { "theme": 0.60, "action": 0.25, "budget_scale": 0.15 },
  "inspected": ["C_5159f5f6"],   // [] for L0/L2
  "revised": true,                // false for L0/L2
  "notes": "optional aggregate rationale" }
```
Only `ranked_candidate_ids` is used for scoring (mapped back to labels). Other
fields support logging.

---

## 13. Logging

Per query, per system, per split — one JSONL record + a run-level summary:
metrics (true label, true rank, Recall@{1,5,10} indicator, reciprocal rank);
tokens (per call, per agent, per-query total); latency (per call, per query); cost
(tokens × backend price); tool usage (inspect count, candidates inspected, fields,
budget used/remaining); inspection decisions (which candidates + stated reason);
specialist outputs (full structured Theme/Action/Budget-Scale); ranking changes
(initial vs revised; specialist-implied vs coordinator-final); failures (parse,
schema-retry, budget exhaustion, tool error, specialist non-response, coordinator
failure, non-termination). Runs checkpointed and resumable. Outputs under
`results/rq2_architecture/<system>__<model>/`.

---

## 14. Evaluation

Identical protocol to L0. Data: curated subset **N=178** (dev 97 design / test 81
sealed; 101 unique true calls). Primary metrics: Recall@1, Recall@5, Recall@10,
MRR, Mean Rank. Cost metrics: token usage, latency, API cost, inspection
statistics (mean inspects/query, % queries using the tool, budget utilisation).
Backend: one fixed model across all cells (§17). Temperature 0; optional k repeats
for the stochastic agentic cells with mean ± std (§18-D7).

---

## 15. Error analysis

Stratified, tied to the ablation mechanisms: **broad calls** (does selective access
avoid Full's dilution?); **specialised calls** (does bounded `inspect` recover the
deep recall Compact lost?); **temporal siblings** — **negative control**: siblings
are byte-identical on every admissible field and the year is gated, so no system
should resolve them; confirming L1/L3 do not helps validate the leakage design;
**budget-scale mismatch** and **action mismatch** decisive/erroneous cases;
**inspection success vs failure** (did an inspection move a candidate toward or
away from its true rank; wasted budget on already-correct/irrelevant candidates).

---

## 16. Failure handling

Parse/schema violations → defensive parser + salvage, ≤N schema retries, then
fallback. Budget exhaustion → auto-submit best-known ranking; un-ranked ids filled
in Compact-index order. Tool error/invalid id → typed error returned, no budget
consumed, logged. Specialist non-response → Coordinator proceeds with available
specialists; missing dimension logged. Coordinator failure → deterministic merge of
specialist scores; else Compact-index order. Non-termination → hard step cap → force
submit. Checkpoint every N queries; hung/failed queries retried on resume without
losing completed work.

---

## 17. Backend validation plan (D3 — the only next executable step)

Before implementing L0–L3, run **one** small controlled backend validation on the
current guarded pipeline. **Compare only two backends:** one **pinned** current
Gemini model and Claude Haiku at an **exact pinned** version. No broad sweep.

Held identical between the two: development queries, frozen top-50, anonymized ids,
guarded query, candidate projection, prompt, decoding settings (where the providers
allow), and evaluation code.

Backend chosen on: Recall@1/5/10, MRR, mean true rank, **parse/schema reliability**,
**latency**, and **estimated cost**. The winner is **frozen** for all primary L0–L3
comparisons.

**This validation itself waits for approval of:** (a) exact query sample (proposed:
all 97 curated dev queries, Compact projection); (b) exact pinned model ids
(proposed: a specific pinned Gemini snapshot — *not* the drifting `flash-latest`
alias — and `claude-haiku-4-5` at a pinned snapshot); (c) exact prompt (proposed:
the current L0 Compact prompt, unchanged). See §18-D3.

---

## 18. Open decisions (require approval before implementation)

Resolved by your rulings: **D1** (2×2 factorial adopted), **D2** (Budget-Scale
Feasibility Agent retained + narrowed), **D3** (single pinned two-backend
validation). Remaining items:

- **D2a — Weights.** Confirm the development-default influence Theme 0.60 / Action
  0.25 / Budget-Scale 0.15, to be frozen on dev before sealed test.
- **D2b — Missing Budget-Scale handling.** When `missing_evidence = true`, zero its
  weight and renormalise Theme/Action to 0.706/0.294, or keep 0.15 as neutral mass?
- **D2c — Aggregation mechanism.** LLM coordinator guided by the weights, a
  deterministic weighted score-blend, or both (blend as an ablation)?
- **D3 — Validation specifics.** Approve the exact (a) query sample, (b) pinned
  model ids, (c) prompt in §17 before the validation runs.
- **D4 — Tool-budget accounting.** Confirm ≤5 *distinct candidates*; multi-field /
  re-inspect calls are free (recommended) vs. each field counts.
- **D5 — Coordinator input in L2.** Confirm Coordinator sees Compact index +
  specialist outputs and does **not** read raw candidate text in L2 (raw access is
  the L2→L3 difference).
- **D6 — Specialist set.** Confirm exactly three specialists (Theme, Action,
  Budget-Scale) given D2's documented weakness.
- **D7 — Repeats.** Temperature 0 throughout; run k repeats (e.g. k=3, mean ± std)
  for the stochastic agentic cells, or single deterministic pass?
- **D8 — Abstention.** L0/L1 pure ranking (no abstention), preserving the metric.
  Confirm no system abstains.
- **D9 — Cost ceiling.** Set a maximum spend for the full L0–L3 × dev/test matrix
  (multi-agent fan-out on Opus-tier backends is expensive).
- **D10 — Scored subset.** Confirm the agentic tier is scored on the same curated
  N=178 (dev design / test sealed) as L0.
```
