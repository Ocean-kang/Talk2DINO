# ECSD v3 Metric Definitions

Let the selected caption set for one image be `S = {c1,...,c5}`. Let `z_i` be the L2-normalized CLIP ViT-B/16 text embedding of `c_i`.

## CISD

There are ten unordered pairs:

`CISD(S) = 1 - (1/10) * sum_{i<j} cosine(z_i, z_j)`.

This is computed from the final caption JSON strings. It is not copied from the greedy selector's intermediate diversity score.

## Weighted fact coverage

Let `F_raw` be the union of normalized content tokens in the five Raw Captions. Let `freq(t)` be the number of Raw Captions containing token `t`, and `N=5`.

`w(t) = 1 + log((N+1)/(freq(t)+1))`.

Let `F_sel` be the union of content tokens in the selected five captions:

`Coverage = sum_{t in F_sel intersect F_raw} w(t) / sum_{t in F_raw} w(t)`.

This is a lexical source-evidence proxy, not an object detector or verified semantic recall.

## Role coverage

The fixed rule classifier can assign Entity, Attribute, Action, Relation, and Context. Unknown is excluded:

`RoleCoverage = number of distinct known roles present / 5`.

## Weighted Jaccard redundancy

For caption fact sets `A` and `B`:

`J_w(A,B) = sum_{t in A intersect B} w(t) / sum_{t in A union B} w(t)`.

Redundancy is the mean of `J_w` across the ten unordered caption pairs.

## Raw exact-match rate

A selected caption is an exact match when its lowercase alphanumeric token sequence, preserving stopwords and token order, equals one of the five Raw Caption token sequences. The rate is the count divided by five.

## Candidate type distribution

The five candidate types are read directly from the selected Trace records and aggregated over all images. No type is inferred from caption wording.

## Selection consistency

For each pair of methods, compare the sets of five selected `candidate_id` values per image. Report the number/rate of identical sets and the average intersection size (0-5). Candidate IDs are deterministic per image and canonical surface text.

## Limitations

CISD is text embedding dispersion. Lexical fact coverage and role coverage are proxies. None of these metrics establishes image grounding or human correctness. Use the manual audit for human judgments.
