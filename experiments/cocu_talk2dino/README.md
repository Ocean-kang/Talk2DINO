# CoCu-Talk2DINO side-car experiment

This directory is intentionally isolated from the original Talk2DINO source. It reuses
`src.model`, `src.loss`, `src.train_util`, the repository CLIP package, and the existing
DINO/text feature files.

## Files

- `build_cocu.py`: `init-pth` workaround for this fork + CoCu curation builder.
- `cocu_dataset.py`: samples original/curated text while reusing base image features.
- `train_cocu.py`: thin training entry point that calls the original `do_train`.
- `self_check.py`: dependency-light loader check.

Run all commands from the Talk2DINO repository root. See the accompanying ChatGPT answer
for the exact environment, data symlink, smoke test, full build, baseline, and ablations.
