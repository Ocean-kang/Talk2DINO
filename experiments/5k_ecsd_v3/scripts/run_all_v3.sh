#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

bash experiments/5k_ecsd_v3/scripts/run_experiment_a_v3.sh
bash experiments/5k_ecsd_v3/scripts/run_experiment_b_v3.sh
