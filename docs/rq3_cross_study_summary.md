# RQ3 — Cross-study transfer of coordination architectures

**Question.** Do the coordination mechanisms (concept-guided decomposition, competing
hypotheses, anchor-and-edit verification) transfer across tasks, and which task properties
determine when they help? Two studies instantiate the **same control-flow topology** (A/B/C/D)
with domain-reimplemented prompts: Study 1 = BRIGHT reasoning-intensive document reranking;
Study 2 = funding proposal→call matching (listwise), evaluated against the expert's labels.
Backbone held fixed at `gemini-flash-latest` in both, so RQ3 compares task, not model+task.

## The pivotal task property: single-pass headroom

| | Study 1 (BRIGHT, 100-doc hard pool) | Study 2 (funding, 6-call pool) |
|---|---|---|
| Single-pass A | R@1 = **0.54** (large headroom) | R@1 ≈ **0.83–1.0** (near-ceiling) |
| Pool construction | gold + 99 TF-IDF hard negatives | full candidate set (6 calls) |
| Evaluation | R@1 / MRR, paired fix/break | graded nDCG + per-call κ vs expert |

BRIGHT leaves the single pass far from ceiling; the small funding pool does not. Coordination
can only help where the baseline has room to improve.

## Per-variant results

**Study 1 — BRIGHT hard pool (n=97; A baseline R@1=0.536, MRR=0.646):**

| Variant | Mechanism | R@1 | fix / break | McNemar p | Read |
|---|---|---|---|---|---|
| B | concept-guided | 0.577 | 6 / 2 | 0.29 | favourable asymmetry, not sig. |
| C | competing hypotheses | — | **7 / 7** | 1.00 | **net zero, no benefit** |
| D | anchor-and-edit | — | **10 / 3** | **0.09** | best asymmetry, low break rate (5.8%), trending +ve |

**Study 2 — funding listwise (3 seeds [7,42,123], nDCG n=5, rank n=4):**

| Variant | nDCG@1 | R@1 | per-call κ vs the expert | Behaviour |
|---|---|---|---|---|
| A | 0.93±0.05 | 0.83±0.12 | 0.66±0.02 | baseline |
| B | 0.90±0.00 | 0.75±0.00 | 0.66±0.01 | marginally below A (overlapping) |
| D_cap1 | 0.93±0.05 | 0.83±0.12 | 0.66±0.02 | **≡ A** — keeps anchor on all proposals |
| D_cap2 | 0.93±0.05 | 0.83±0.12 | 0.66±0.02 | **≡ A** — cap-invariant ⇒ not a budget artefact |

All four variants are statistically indistinguishable; D preserves the (already near-optimal)
anchor everywhere; zero top-1 false positives on the all-negative Mito proposal across every
seed and variant. κ ≈ 0.66 sits above the human–human ceiling (0.49), so the LLM's per-call
judgement matches the expert at human level.

## The transfer conclusion, in three layers

1. **Architecture (runnability): transfers.** The same control-flow topology instantiates and
   runs correctly on both tasks; on Study 2, anchor-and-edit is now a genuine ranking-level
   mechanism (it preserves the anchor and produces zero false positives), not the mis-instantiated
   per-pair operation it was under the earlier pairwise design.

2. **Effectiveness: transfers conditionally, and the effect is small.** Coordination only has
   room to act where the single pass has headroom. On BRIGHT (headroom), the *constrained*
   verification D shows the strongest favourable asymmetry (10 fix / 3 break, p≈0.09) and concept
   guidance B a weaker one (6/2, p=0.29); the *unconstrained* competing-hypotheses C is net-zero
   (7/7). On the saturated funding pool, every variant ties — because there is no headroom, not
   because the mechanism is BRIGHT-specific.

3. **RQ3 answer: coordination's usefulness is a function of a task property — single-pass
   headroom / candidate-pool difficulty — not a universal gain.** The Study-2 null is a
   *finding about the task*, not a transfer failure: contrasting a headroom task with a saturated
   one isolates the moderator. This is a principled, generalisable result and directly rebuts the
   "overfit to BRIGHT" concern — the mechanism runs identically on a different task and behaves
   correctly; it simply has nothing to fix where the baseline is already right.

## Honest framing (do not overclaim)

- No variant reaches p<0.05 even on BRIGHT; B and D **trend** positive with low break rates, C is
  null. The claim is "constrained coordination helps *modestly and conditionally*," not "coordination
  works."
- Heavier coordination does **not** help: unconstrained C (7/7) and the Study-2 decompose+review
  pipeline (per its pairwise analysis, the review layer *lowered* expert-consistency, wκ 0.52→0.37)
  both fail to beat the simpler options.
- Single backbone (`gemini-flash-latest`); Study 2 ranking is illustrative (n=4–5, ceiling);
  Study 2's quantitative weight is the κ-consistency, not the ranking deltas.

## One-line conclusion

> The architecture transfers (runnable, correctly instantiated on both tasks); coordination's
> *effectiveness* does not transfer unconditionally — it is bounded by single-pass headroom.
> Where headroom exists (BRIGHT), only the constrained anchor-and-edit shows a modest positive
> trend (p≈0.09); where the pool is saturated (funding), all variants tie. RQ3's answer is that
> candidate-pool difficulty, not the coordination mechanism per se, governs when coordination pays.
