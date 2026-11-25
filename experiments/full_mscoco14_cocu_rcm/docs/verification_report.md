# Verification Report

## Checks executed in the artifact sandbox

| Check | Result | Scope |
|---|---|---|
| Python syntax compilation | PASS | All 27 package Python files |
| Shell syntax (`bash -n`) | PASS | All 6 package shell scripts |
| Unit tests | PASS | 11 synthetic/unit tests |
| Package smoke wrapper | PASS | `scripts/run_smoke_test.sh` executed in an isolated Talk2DINO-shaped skeleton checkout |
| CLI help smoke | PASS | 8 data, archive, build, sampling, verification, diagnostics and evaluation-manifest entrypoints |
| Synthetic sampling | PASS | Deterministic negatives, distinct source images, paired RCM/Shuffled and sampling resume |
| Full archive guard | PASS | Intentional one-image smoke population accepted only with explicit override; default full count rejects subset data |
| Local loss gradient | PASS | Finite loss, backward gradient and differentiable empty-anchor zero |
| Synthetic Dataset smoke | PASS | Lazy region shard loading, feature-index lookup, positive/negative masks, manifest and arm checks |
| CLI namespace contract | PASS | Full concept discovery script: all `a.*` attributes are declared by argparse |
| ZIP integrity | PASS | Final archive CRC and `testzip()` check |

## Not executed

The artifact sandbox does not contain the Talk2DINO checkout, COCO2014 data, DINOv2/CLIP checkpoints, or the user's 4090D server environment. Therefore these are **not claimed as passed**:

- Full Train/Val PTH audit against the user's real files.
- Full Local-CoCu D discovery.
- Full Region Gallery extraction and cached-vs-recomputed DINO consistency.
- Full CLIP region verification.
- Full K-shot index coverage on all Train2014 concepts.
- Real Talk2DINO forward/backward and epoch-resume on GPU.
- Five-arm full MSCOCO14 training.
- VOC20 / PC59 / COCO-Stuff / ADE20K evaluation.

Run `scripts/run_smoke_test.sh` in the Talk2DINO checkout after extraction. Then run each preprocessing stage separately before the Seed-123 pilot.
