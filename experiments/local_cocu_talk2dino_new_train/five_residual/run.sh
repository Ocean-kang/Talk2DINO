#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../run.sh" "${1:?Scheme A-H}" 5 "${2:-all}" "${3:-0}" "${@:4}"
