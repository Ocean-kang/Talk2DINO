# Experiment Design

## Hypothesis

At full MSCOCO14 scale, image-level CoCu concept augmentation may improve the semantic coverage available to Talk2DINO. Local supervision may provide an additional alignment signal. Cross-image RCM should help only if the sampled regions truly correspond to the same concept; therefore RCM must be compared against a source-image-matched shuffled control.

No arm is assumed to be best.

## Controlled factors

- Same raw Train2014 and Val2014 IDs.
- Same five original captions per image.
- Same DINOv2 ViT-B/14-reg features and CLIP ViT-B/16 text features.
- Same projector config, seed initialization, optimizer, scheduler, batch size, epochs and validation.
- Same OVSS benchmark configs, inference settings, class prompts and PAMR mode.
- Same K-shot and negative count for local arms.

## Arms

1. Raw: original five captions, Global InfoNCE.
2. CoCu: five original captions + up to three CoCu concept captions, Global InfoNCE.
3. CoCu-Same: CoCu plus K verified same-image regions.
4. CoCu-RCM: CoCu plus K same-concept regions from K distinct train images.
5. CoCu-RCM-Shuffled: same anchors, source images and negative regions as RCM, but positive regions are verified for another concept.

## RCM/Shuffled pairing rule

For each anchor concept, candidate source images must contain:
- at least one verified region for the anchor concept,
- at least one verified region that is not verified for the anchor concept.

Select K distinct source images. RCM selects one target-concept region per source image. Shuffled selects one non-target-concept region from those same source images. Both arms share the exact same negative region IDs. If the paired source-image set cannot be constructed, the anchor is skipped in both arms.

This rule intentionally prioritizes a clean mechanism test over maximizing the number of eligible anchors. Report the resulting coverage.

## Full-scale implementation

- Concept Archive: sharded JSON, explicit empty records, exact Train2014 ID coverage.
- Region Gallery: sharded `[N,12,768]` CPU features, 12 attention-derived bboxes per image, cached-vs-recomputed DINO feature check.
- CLIP Verification: one CLIP model per run, cached unique concept text embeddings, one image's 12 regions plus full-image view encoded at a time.
- Sampling: concept inverted index, deterministic per-anchor RNG, JSONL shard outputs.
- Training: original `DinoClipDataset`, `ProjectionLayer`, `ContrastiveLoss`, exact 5K local loss, lazy region shard loading, epoch-boundary resume.
- Evaluation: original Talk2DINO OVSS evaluator and configs.

## Falsification

The RCM hypothesis is not supported if:
- RCM does not outperform CoCu or CoCu-Same consistently,
- RCM does not outperform the paired Shuffled control,
- improvements disappear across seeds,
- local loss/gradient scale is negligible or unstable,
- RCM coverage is too small or concentrated in a few frequent concepts.

## Required reporting

Always report:
- concept coverage and frequency distribution,
- verified region ratio and CLIP score diagnostics,
- Same/RCM/Shuffled eligible anchor coverage,
- per-frequency-bucket coverage,
- local/global losses and gradient norms,
- all five arms' mIoU per seed,
- paired RCM minus Shuffled differences.

## 5K Seed-123 reference supplied for this migration

| Arm | VOC20 | PC59 | COCO-Stuff | ADE20K |
|---|---:|---:|---:|---:|
| Raw | 80.90 | 32.81 | 23.31 | 16.88 |
| CoCu | 83.34 | 35.97 | 24.43 | 18.72 |
| CoCu-Same | 84.44 | 35.86 | 24.27 | 18.68 |
| CoCu-RCM | 83.83 | 35.44 | 24.16 | 18.65 |
| CoCu-RCM-Shuffled | 84.10 | 35.47 | 24.28 | 18.65 |
| Rich-Diversity | 84.67 | 36.55 | 25.25 | 19.00 |

The 5K observation is not that RCM already works: CoCu clearly exceeds Raw, CoCu-Same has a VOC20 advantage, but RCM does not stably exceed CoCu and Shuffled is not worse than RCM. Rich-Diversity is still the strongest supplied 5K reference. The Full experiment therefore tests whether the larger concept/region gallery changes the cross-image matching mechanism, not whether more data automatically improves scores.
