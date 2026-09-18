# RQ3 headroom test — pre-registration

**Registered: 2026-08-28, BEFORE running the three additional BRIGHT splits.**
Biology is already run; earth_science / psychology / sustainable_living are not. The
analysis plan below is fixed now and reported as-is regardless of outcome (per Method §3.5).

## Hypothesis

Coordination's benefit over the single pass scales with **single-pass headroom** (task
difficulty): where the baseline already answers, coordination has little to add and can only
risk breaking it; where the baseline fails, coordination has room to recover. Constrained
verification (D) is predicted to exploit headroom **safely** (fixes concentrate on A-wrong,
low break on A-right); unconstrained competing-hypotheses (C) is predicted to act
indiscriminately (no concentration, net zero).

## Design (config parity is mandatory)

- **Variants:** A (single-pass), B (concept-guided), C (competing hypotheses), D (anchor-edit).
- **Splits:** biology (done) + earth_science, psychology, sustainable_living (to run). Four
  same-corpus splits; **only task difficulty varies** — pool size, label source and pool
  construction are held constant, so split differences are pure difficulty (stated in the paper).
- **Held identical across every split and variant** (any mismatch ⇒ discard the run, do not
  reconcile): backbone `gemini-flash-latest`; POOL=100 (gold + top-99 TF-IDF hard negatives);
  MAX_GOLD=6; DOC_CAP=600; same RNG seeds; same prompts (hash-locked); the pool is built once by
  A per split and **reused unchanged** by B/C/D (pool + baseline parity across variants).

## Analysis — fixed before seeing results

1. **PRIMARY — pooled query-level logistic (n≈400).** Pool all four splits' queries. For each
   variant fit
   `logit P(flip = 1) = β0 + β1·A_correct + β2·split + β3·(A_correct × split)`,
   where `flip = 1` iff the variant changes the top-1 correctness (A-wrong→right = fix;
   A-right→wrong = break). **β1** tests whether the variant's action concentrates on A-wrong
   (query-level headroom concentration); **β3 (the A_correct × split interaction)** is the direct
   test of whether that concentration **varies with task difficulty** — i.e. the task-level
   headroom effect. This n supports the test; the 4-point regression does not, so the interaction
   term (not a split-level slope) is the primary cross-task evidence.

2. **DESCRIPTIVE supplement — 4-point table.** One row per split: n, A R@1 (headroom proxy = 1−R@1),
   D fix, break, net, fix_rate (fix / A-wrong), break_rate (break / A-right). Report **monotonicity
   only** (does fix_rate rise as A R@1 falls?); **no regression p-value on 4 points.**

3. **B and C run identically; the slope comparison is core evidence.** If only D's fix_rate tracks
   difficulty while B/C do not, the conclusion attaches to the **constrained design**, not to
   "coordination" in general. Report all three variants' concentration and interaction.

4. **Study 2 enters no regression.** Its 4–6 queries are an independent qualitative contrast point,
   never a fifth data point.

## Mechanistic note (strengthens, does not replace, the test)

`break` is a-priori easier than `fix`: among 100 candidates there are 99 wrong ways to seat the
top-1 and one right way. That D still shows fix 22% / break 6% on biology (Fisher p=0.033) means
its promotions are **not random** — the concentration is a property of the mechanism, not of the
base rates. This is reported alongside the statistic so p=0.033 reads as a mechanistically-grounded
test, not a bare number.

## Boundary of the query-level result

Query-level concentration ("D acts where A fails") does **not** entail the task-level headroom
effect ("harder tasks ⇒ coordination more useful"); they are different levels. The **interaction
term (β3)** is what tests the task-level claim. Absent a significant interaction, the cross-task
headroom conclusion rests on only two points (BRIGHT vs funding) and must be reported as such.

## Outcome handling — pre-committed

1. **β3 significant (+ monotonic 4-point) →** headroom has within-dataset support; RQ3 states
   "candidate-pool difficulty determines whether coordination has room," with the cross-dataset
   funding contrast reported as consistent.
2. **β1 holds but β3 / monotonicity do not →** use the query-level concentration conclusion; the
   split-level is a noisy supplement, not force-interpreted.
3. **Neither holds →** report honestly that headroom is one explanation *consistent with* the two
   datasets but *not separable* from covarying factors; downgrade to future work. **This is written
   in the Discussion body, not hidden in limitations** — a proposed-and-tested-and-unconfirmed
   explanation is among the most credible passages an audit-style paper can contain.
