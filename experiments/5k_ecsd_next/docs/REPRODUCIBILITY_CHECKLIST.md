# Reproducibility Checklist

## Before Stage A/B
- [ ] Use the intended 5,000 image `5k_rich_diversity.json`.
- [ ] Confirm unique `image_id` values and five non-empty Raw Captions/image.
- [ ] Confirm existing CLIP ViT-B/16 cache. No weight download is allowed.
- [ ] Confirm original `experiments/5k_ecsd/` remains unchanged.
- [ ] Save `configs/default.json` used for the run.

## Stage A
- [ ] Legacy selected JSON uses the same source image IDs.
- [ ] Candidate pool has source spans for every candidate.
- [ ] Candidate-pool and selected-five statistics are both reported.
- [ ] 100-image audit sample uses fixed seed 42.
- [ ] Automatic heuristic results are not described as human labels.
- [ ] Manual annotation CSV is preserved after labels are filled.

## Stage B
- [ ] B1-B4 use the exact same candidate pool.
- [ ] Same CLIP ViT-B/16 model and cache.
- [ ] Same candidate filter.
- [ ] Same deterministic anchor.
- [ ] Same inverse-frequency fact weights.
- [ ] Exactly five captions/image for each method.
- [ ] Stable candidate-ID tie-break is recorded.
- [ ] Score weights and filter thresholds are saved in metadata.
- [ ] CISD and semantic coverage are reported separately.

## Stage C
- [ ] `5k_ECSD_v2.json` contains Raw, Rich, Rich-Diversity, ECSD v2.
- [ ] DINO feature IDs exactly match the 5,000 caption IDs.
- [ ] Train/validation split is fixed at 4,500/500.
- [ ] Same `val_ids.json` across all arms.
- [ ] `--seed` patch applied and verified.
- [ ] Training seeds are 123, 456, 789.
- [ ] Checkpoint names are unique and not overwritten.
- [ ] All four eval configs explicitly set `pamr: false`.
- [ ] Same evaluation configuration across all 12 checkpoints.
- [ ] Mean and population standard deviation are reported.
- [ ] No statistical-significance claim is made from three seeds.

## Reporting
- [ ] Include final train/validation loss.
- [ ] Include all four dataset mIoUs.
- [ ] Include CISD, semantic coverage, category coverage, and redundancy.
- [ ] Include manual audit outcomes when available.
- [ ] Separate confirmed results from missing/incomplete runs.
