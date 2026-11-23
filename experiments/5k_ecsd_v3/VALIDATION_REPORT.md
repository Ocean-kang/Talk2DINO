# ECSD v3 Package Validation Report

## Checks executed in the package-building environment

The package was validated after implementation using synthetic inputs and local unit tests.

- Python compilation of all `experiments/5k_ecsd_v3` modules.
- 12 unit tests for strict Clause subspans, duplicate source-span merging, deterministic IDs, shared-anchor invariance, deterministic selection, Candidate-ID-based manual selection, annotation migration conflict handling, final-string metric recomputation, invalid-JSON safe overwrite, and JSON/Trace integrity validation.
- `bash -n` for all v3 shell scripts.
- Synthetic two-image output/trace validator test.
- Synthetic candidate-pool generation and source-span inspection.

## Not executed

- The full 5,000-image COCO candidate build was not run in this package-building environment.
- The Talk2DINO checkout and `experiments/5k_ecsd/clip_utils.py` were not part of the uploaded ZIPs. CLIP loading/encoding on the experiment server was not run.
- No CLIP weights were downloaded.
- No DINO feature extraction, GPU training, or OVSS segmentation evaluation was run.
- Manual labels were not created or inferred.

## Source audit of the uploaded v2 result ZIP

The uploaded result archive was inspected directly. It contains 5,000 candidate-pool records and 84,198 candidate entries. Under lowercase alphanumeric surface normalization preserving stopwords, 30,034 entries are repeated after their first occurrence, leaving 54,164 distinct candidate texts. There are 22,904 Clause entries whose normalized text equals an Original candidate text in the same image. The uploaded manual CSV has 1,651 candidate rows; checking its `selection_status` against B4 trace candidate IDs finds 351 false-positive Selected rows and zero missed selected IDs. The uploaded generation log also ends with a JSONDecodeError while reading an existing B1 output.

The four uploaded trace files each contain 5,000 images and five selected records per image. Anchor IDs agree B2/B3/B4 pairwise, but B1 differs from each of those methods on 20 images. This is why v3 computes the anchor once and passes that same candidate ID to every method.

## Interpretation

These checks establish package integrity and algorithm-level behavior on synthetic fixtures. They do not establish real COCO caption quality, CLIP metric values, or downstream OVSS performance. Those require running the v3 A/B scripts in the user's Talk2DINO environment.

## Synthetic end-to-end smoke run

A separate synthetic five-image dataset was passed through the v3 candidate-pool CLI and B1-B4 selection orchestration using deterministic test embeddings (not CLIP). It produced 76 unique candidates, rejected 24 full-caption Clause fragments, generated four five-caption-per-image output sets and traces, and passed the full source/pool/trace/JSON/anchor validator with zero anchor mismatches. The B4 manual-audit exporter marked exactly five selected candidate IDs for each of the five synthetic images. The complete metrics CLI was also exercised against a pre-populated synthetic embedding store: it produced a 5-row per-image table, a 7-arm summary (Raw/Rich/Rich-Diversity/B1-B4), six pairwise consistency rows, and an anchor-invariance PASS report. These synthetic vectors are only a pipeline test, not CLIP results.
