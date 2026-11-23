# ECSD Next-Stage Experimental Design

## Research hypothesis

Semantic diversity is not sufficient for OVSS. A set of captions should cover complementary semantic facts and roles while avoiding unsupported or incomplete fragments. The downstream hypothesis is that better caption complementarity may improve region-concept alignment in Talk2DINO.

The implementation deliberately separates three questions:

1. Is the candidate pool linguistically usable and traceable?
2. Which selection objective best balances fact coverage, CLIP diversity, and semantic-role novelty?
3. Do the resulting captions improve OVSS consistently across training seeds?

## Stage A: Candidate quality

### Legacy diagnosis

The existing `experiments/5k_ecsd/ecsd_core.py::build_candidates()` is called without changing its behavior. The analysis reports the complete candidate pool and the five selected captions separately.

The following automatic signals are reported:

- candidate type counts: `original`, `noun_phrase`, `clause`
- invalid surface-form warnings: leading connectors, dangling prepositions/conjunctions, too few content tokens, overlong fragments
- exact normalized duplicate frequency within each image
- lexical unsupported-content proxy: candidate content tokens absent from the source Raw Captions
- content-token and sentence-token lengths
- multi-label role distribution and per-image role coverage
- selection frequency of each candidate type and candidate string

These are diagnostics only. A low unsupported-token ratio does not prove visual grounding; a high invalidity warning does not replace human judgment.

### Improved candidate construction

The v2 builder uses only original Raw Caption strings:

- full original captions are always candidates
- clauses are split at explicit punctuation
- conjunction splits are conservative: both sides must contain an action lexicon match
- noun phrases are extracted from determiner-led spans and stopped at action, preposition, conjunction, auxiliary, or a new determiner
- one-token noun phrases are retained only when the local entity lexicon supports the head
- every candidate stores exact source caption index, start/end offsets, and source span text
- duplicates are retained in the saved pool for diagnosis; the common selection filter removes duplicate normalized strings before B1-B4 selection

The builder does not invent visual facts. The `A photo shows ...` wrapper is a neutral grammatical frame; content tokens are checked against the extracted source span.

### Rule-based role classification

Each candidate can receive multiple labels:

- Entity
- Attribute
- Action
- Relation
- Context
- Unknown

The output stores matched lexical terms per category. This makes the prediction auditable and makes ambiguous cases explicit. It is not a POS tagger and does not claim semantic correctness.

### Manual audit

A fixed-seed sample of 100 images is exported with all Raw Captions, all candidates, source spans, role predictions, and selection status. Annotators label:

- Valid / Partially Valid / Invalid
- Grounded / Unsupported
- Complete / Incomplete
- Category Correct / Incorrect
- Redundant / Complementary

The CSV is intentionally blank. The human-label analyzer excludes blank labels and reports all-candidate and selected-candidate subsets separately.

## Stage B: Selection ablation

All methods use:

- one saved v2 candidate pool
- the same common validity filter
- the same CLIP ViT-B/16 embeddings
- the same inverse-frequency semantic-fact weights
- the same deterministic anchor
- the same stable candidate-ID tie-break
- exactly five selected captions per image

### Anchor

The anchor is the valid original caption with the greatest inverse-frequency-weighted fact mass. Ties are resolved by ascending `candidate_id`. The same anchor is used in B1-B4.

### Greedy objectives

For selected set `S` and candidate `c`:

\[
\Delta C(c|S)=
\frac{\sum_{t\in F(c)\setminus F(S)}w_t}
{\sum_{t\in F_{source}}w_t}
\]

\[
D(c|S)=\frac{1}{|S|}\sum_{s\in S}\frac{1-\cos(z_c,z_s)}{2}
\]

\[
R(c|S)=
\frac{|Roles(c)\setminus Roles(S)|}{|Roles(c)|}
\]

If a candidate has no known role, its role novelty is zero.

| Method | Greedy objective |
|---|---|
| B1 | \(\Delta C\) |
| B2 | \(D\) |
| B3 | \(0.5\Delta C + 0.5D\) |
| B4 | \(0.4\Delta C + 0.4D + 0.2R\) |

The score components are normalized to [0,1]. No quality bonus, redundancy penalty, or hidden term is added to one arm only. If fewer than five unique valid candidates remain, the trace records explicit original-caption fallback steps to preserve the five-caption contract.

## Stage C: Multi-seed OVSS

Four caption arms are compared:

- Raw
- Rich
- Rich-Diversity
- ECSD v2 (B4)

One shared 4,500/500 image split is generated with split seed 123. This split is fixed for every caption arm and every training seed. Training seeds are 123, 456, and 789.

The original Talk2DINO model config, DINO features, CLIP text features, projector, InfoNCE objective, optimizer, batch size, epochs, and evaluation configs are reused. Only the caption arm and training seed differ.

PAMR is disabled for all four evaluation datasets. The original Raw VOC20 PAMR-enabled result must not be mixed into the final comparison.

## Reproducibility and safety

- Candidate pool, traces, config snapshots, split IDs, logs, checkpoints, and evaluation results are stored separately from the legacy experiment.
- Existing checkpoints and manual annotation files are protected by default.
- The seed patch is limited to exposing the existing seed argument and explicitly seeding DataLoader workers.
- Full training/evaluation is an explicit user-run step; package validation is CPU/lightweight only.
