#!/usr/bin/env bash
set -euo pipefail
# Optional convenience script. Run from repository root.
# GPU assignments can be edited if you want to parallelize manually.
bash experiments/CaptionsConcept_Completion/A_raw_caption_anchored/run.sh build clip 0
bash experiments/CaptionsConcept_Completion/A_raw_caption_anchored/run.sh build local 0
bash experiments/CaptionsConcept_Completion/B_filtered_semantic_residual/run.sh build clip 0
bash experiments/CaptionsConcept_Completion/B_filtered_semantic_residual/run.sh build local 0
bash experiments/CaptionsConcept_Completion/D_concept_set_supplementary/run.sh build clip 0
bash experiments/CaptionsConcept_Completion/D_concept_set_supplementary/run.sh build local 0
