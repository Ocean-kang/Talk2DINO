#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
bash experiments/ecsd_scale_sensitivity/scripts/00_check_environment.sh
bash experiments/ecsd_scale_sensitivity/scripts/01_build_raw.sh
bash experiments/ecsd_scale_sensitivity/scripts/02_generate_ecsd_b4.sh
bash experiments/ecsd_scale_sensitivity/scripts/02b_cisd_full.sh
bash experiments/ecsd_scale_sensitivity/scripts/03_prepare_full_features.sh
bash experiments/ecsd_scale_sensitivity/scripts/04_build_nested_subsets.sh
bash experiments/ecsd_scale_sensitivity/scripts/08_collect_caption_metrics.sh
bash experiments/ecsd_scale_sensitivity/scripts/05_train.sh
bash experiments/ecsd_scale_sensitivity/scripts/06_seg_eval.sh
bash experiments/ecsd_scale_sensitivity/scripts/07_collect_results.sh
