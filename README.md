# Agentic Document Ranking with Bounded Revision — code

Experiment code for the MSc dissertation *A Controlled and Applied Evaluation of
Agentic Document Ranking with Bounded Revision*. It implements four listwise
coordination designs and evaluates them on a public benchmark (Study 1) and an
applied proposal–funding matching task (Study 2).

**Designs.** A = single-pass baseline; B = concept-guided reranking; C =
alternative-interpretation (competing-hypotheses) coordination; D =
**anchor-and-edit**, the proposed bounded-revision architecture (treats A's
ranking as an anchor and makes a small number of capped promotions).

## Repository layout

```
scripts/   all experiment code (runners, clients, scorers)
configs/   experiment configuration
docs/      experiment design, competing-hypotheses design, freeze notes,
           pre-registration, cross-study summary
```

## Key scripts

**Study 1 — BRIGHT reranking (public benchmark)**
- `scripts/bright_backbone.py` — A/D prompts and pool loading for BRIGHT.
- `scripts/bright_backbone_bc.py` — adds the B and C arms.
- `scripts/bright_clean_split.py` — parity-clean A/B/C/D on any domain/backbone.
- `scripts/panel_ad.py` — cross-model A/D panel (any backbone; open models via the
  OpenAI-compatible client).
- `scripts/score_*` — scoring and analysis.

**Study 2 — applied proposal–funding matching**
- `scripts/james_match.py`, `scripts/james_match_listwise.py` — listwise A/B/C/D matcher.
- `scripts/rq2_core.py` — provider routing, clients, cost accounting.
- `scripts/oai_client.py` — provider-agnostic OpenAI-compatible client for
  open-weight models (Together / Fireworks / Groq / …).
- `scripts/score_study2_listwise*.py` — metrics vs the expert gold.
- `scripts/bootstrap_study2_kappa.py` — bootstrap 95% CIs for the agreement κ.

## Data availability

- **Study 1 (BRIGHT).** BRIGHT is a public benchmark
  (Su et al., 2025; https://huggingface.co/datasets/xlangai/BRIGHT). The candidate
  pools used here are derived from it. Large data files are not committed; see the
  script paths under `data/` to reconstruct them.
- **Study 2 (applied).** The proposal–funding matching dataset is **proprietary
  non-public partnering data** and is **not included**. Scripts reference paths
  under `data/james_validation/` and a gold-label CSV that are withheld; the
  code is provided for transparency of method, not to redistribute the data.

## Setup

```bash
python3 -m pip install anthropic google-generativeai certifi
cp .env.example .env    # then fill in your own API keys
```

Set the keys you need in `.env` (never commit real keys):

```
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
TOGETHER_API_KEY=...        # open-weight models
OAI_BASE_URL=https://api.together.xyz/v1
OAI_API_KEY_ENV=TOGETHER_API_KEY
```

Decoding is temperature 0 where settable; reasoning-capable open-weight models
run with `enable_thinking=false` for parity with the direct-answer closed models.
Long runs are wrapped in `caffeinate` to prevent sleep.

## Notes

- The reference backbone is the pinned `claude-haiku-4-5-20251001` snapshot.
- Study 1 uses five replications; the applied Study 2 uses a single
  presentation-order seed (seed 42) for the main comparison and panel.
- Seeds control candidate presentation order only; they are not passed to the
  provider, so runs are not bit-reproducible from a seed alone.
