#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../..
python -m unittest discover -s experiments/role_aware_concept_composition/tests -v
python -m experiments.role_aware_concept_composition.build_ra_c2c \
  --base-train outputs/coco2014_b14/train.pth \
  --concept-source f5=outputs/f5_ponytail/run1/audit.jsonl \
  --gallery experiments/local_cocu/MSCOCO14_train_gallery.json \
  --output-dir outputs/role_aware_concept_composition/smoke_plan \
  --limit 32 --plan-only --missing-policy keep_raw
