# Literature currency sweep (2024-2026) -- 2026-06-19

Two parallel literature scouts (targeted direct-competitor + broad landscape),
each finding independently verified against arXiv before any citation was added.

## Headline verdict

**The novelty conjunction is SAFE.** No 2024-2026 paper does BOTH of DreamLog's
defining axes together: (A) an LLM-proposes + symbolic-SLD-verifies learner over
an UNANNOTATED, self-querying fact base with an MDL acceptance gate, and (B) an
invented-PREDICATE-vocabulary memorization control with a raw-LLM baseline used to
attribute each recovered rule type to compression vs. recall. Every candidate
breaks at least one axis. The paper's own disclaimers (not first to do
propose-then-verify; not first memorization control) are accurate and well-placed.

## Added to the paper (4, all arXiv-verified)

Integrated into the propose-then-verify related-work paragraph with one-sentence
distinctions, because a reviewer who knows them will ask "isn't this already done?":

- **peng2025ilpcot** -- ILP-CoT (arXiv:2509.21874, 2025): MLLM proposes rule
  skeletons, ILP solver verifies. Axis A, but consumes text/visual tasks, not an
  unannotated symbolic KB; no compression objective.
- **yang2025languagebias** -- LLM-automated language bias for ILP (arXiv:2505.21486,
  2025): multi-agent LLMs design the predicate vocabulary from raw text. Axis A, but
  raw-text language-bias automation, not cross-predicate bodies over an unannotated KB.
- **patsantzis2026poker** -- Self-Supervised ILP / Poker (AAAI 2026, arXiv:2507.16405):
  auto-generates and labels its own pos/neg examples -- the closest analog to
  DreamLog's self-supervision, but purely symbolic (no LLM), needs >=1 labeled
  positive + second-order metarule bias, no compression. Same venue family (AAAI).
- **xie2024memorization** -- On Memorization of LLMs in Logical Reasoning
  (arXiv:2410.23123, 2024): counterfactual-perturbation memorization control. Added
  to the axis-B disclaimer alongside HtT as a memorization-control precedent.

## Optional / supporting candidates the scouts found (NOT added -- coauthor call)

The paper is already long (~20pp); these strengthen but are not protective, so
they are left for the coauthors to weigh against the page budget:

- **InductionBench** (arXiv:2502.15823, 2025): LLMs fail at simplest-complexity-class
  induction -- external corroboration of the raw-LLM 0% result. Strong supportive cite.
- **RLAD** (arXiv:2510.02263, 2025): RL-trained LLM abstraction discovery -- a modern
  LLM analog of DreamCoder wake-sleep. Context for the library-learning lineage.
- **Bridging Kolmogorov Complexity and Deep Learning** (arXiv:2509.22445, 2025):
  description-length training objectives for transformers -- a 2025 anchor for the
  bits-DL contribution (complements Solomonoff/Rissanen/deletang2024). NOT yet
  independently verified; verify before adding.
- **RLIE** (arXiv:2510.19698, 2025), **On the Role of Model Prior in Inductive
  Reasoning** (arXiv:2412.13645, 2024), **Predicate Renaming via LLMs**
  (arXiv:2510.25517, 2025, maps to DreamLog's LLM predicate-naming step), **Reducer**
  (AAAI 2025, arXiv:2502.01232, newest Popper-lineage ILP): all citable context.

## Camera-ready hygiene flags (from both scouts + the editorial review)

- he2025reasoning: confirm venue is TMLR 2025 (currently cited as arXiv); verify before
  changing.
- desouza2025inductive: already correctly cited as AAAI 2025 (no change needed).
- Verify (before camera-ready) the venues/author lists flagged in the editorial review:
  lotfi2022pac, grand2024lilo, ellis2023dreamcoder.
