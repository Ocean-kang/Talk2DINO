# Implementation Report

## Added code

All additions are confined to `experiments/full_mscoco14_cocu_rcm/`.

- `data/prepare_full_data.py`: validates raw full PTH files and stages hardlinks/copies.
- `data/generate_full_cocu_matches.py`: wraps existing Local-CoCu `mine('D')` in a chunked resumable output journal.
- `data/build_full_concept_archive.py`: exact Train2014 coverage, lexical novelty recheck, sharded concept archive.
- `data/build_augmented_train.py`: guarded wrapper around original `local_cocu_novel/build_train.py`.
- `data/extract_region_gallery.py`: sharded 12-head features and attention bboxes; validates cached DINO features.
- `data/verify_regions.py`: batched CLIP region verification and unique concept text cache.
- `data/build_sampling_index.py`: deterministic Same/RCM/paired-Shuffled sampling.
- `training/rcm_dataset.py`: lazy shard-loading local dataset wrapper.
- `training/train_full_cocu_rcm.py`: original Talk2DINO model/loss plus 5K local loss and epoch-boundary resume.
- `diagnostics/`: Full result collector and report builder.
- `scripts/`: isolated preprocess, train, multi-GPU arm-parallel, evaluation and smoke-test wrappers.

## Reused code

The package imports the actual repository implementations for:
- Talk2DINO PTH Dataset, Projector, Global Loss and seed/scheduler utilities.
- 5K `multi_positive_infonce` via a direct file import when the 5K package is present; an identical local fallback copy exists only so isolated package tests can run.
- Local-CoCu `Vision`, `Teacher`, `mine('D')`, `bbox_from_map`.
- Local-CoCu Novel `build_train.py`.
- Original OVSS configs and evaluator.

## Full-scale adaptations

1. Chunked concept discovery: one `mine('D')` iterator per run keeps Vision loaded while committing each image chunk atomically.
2. Region feature storage: `[images, 12, 768]` CPU tensor shards and an image-to-shard index.
3. Lazy training access: only the region feature shards needed by a DataLoader worker are loaded; no full gallery is copied to GPU.
4. Verification cache: unique concept text features are computed once and persisted.
5. Paired shuffled sampling: RCM/Shuffled share anchor keys, source image IDs and negative IDs; sampling checkpoints every anchor shard.
6. Resume: concept discovery, region extraction, verification and sampling resume by completed shard; training resumes at epoch boundaries.

## Original files modified

None. The ZIP contains only the new experiment directory. It is intended to be extracted at repository root.

## Important compatibility note

The audited 5K `run_preprocess.sh` references a `data/` folder absent from the current GitHub `main` tree. The new Full package implements these data stages independently rather than importing missing scripts. The existing 5K training Dataset and trainer remain intact.

## Final code review corrections

- Corrected a missing argparse attribute in the full Local-CoCu discovery wrapper. The checkpoint cadence is controlled by `--chunk-size`; no undefined `save_every` attribute is read.
- Tightened raw PTH validation to require Train `disentangled_self_attn` exactly `[12,768]` and Val `avg_self_attn_out` exactly `[768]` or `[1,768]`, matching the actual `dino_extraction_v2.py` output contract.
- Removed an unused accumulation of all Region Gallery metadata rows to avoid unnecessary host-memory growth.
- Local/Global gradient norms are now measured on the same first active local-supervision batch, making the reported ratio comparable.

## Known limitations

- The full Local-CoCu D discovery requires complete Train2014 `[12,768]` head features. By default these are reused directly from `raw_train.pth`; a separate full head cache is optional. Discovery remains GPU-expensive because DINO attention regions are evaluated per image.
- DINO/CLIP model caches and COCO data must already exist locally.
- Region Gallery extraction currently uses one DINO forward per concept-bearing image and is not multi-GPU distributed; it is shard-resumable and can be launched on a selected device.
- CLIP verification processes one image's 12 crops at a time. This is memory-safe but may not maximize throughput.
- Full training uses the original Talk2DINO PTH Dataset, which materializes annotation-level records in CPU memory. It avoids GPU-resident full feature galleries but can require substantial host RAM.
- Preprocessing accepts an explicit CUDA device per process; `run_train_parallel.sh` distributes independent Arms across GPUs while keeping each Arm's original single-GPU batch semantics. OVSS evaluation uses the repository's distributed launcher.
- No actual full COCO preprocessing, training, or OVSS evaluation was run in the artifact sandbox.
