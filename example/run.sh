#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/solver"
    exit 1
fi

SOLVER="$(realpath "$1")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$ROOT/results/free_triaxial_top" "$ROOT/results/rolling_asymmetric_sphere"

(
    cd "$ROOT/results/free_triaxial_top"
    "$SOLVER" "$ROOT/inputs/free_triaxial_top.in"
)
cp "$ROOT/results/free_triaxial_top/output.csv" "$ROOT/data/free_triaxial_top.csv"

(
    cd "$ROOT/results/rolling_asymmetric_sphere"
    "$SOLVER" "$ROOT/inputs/rolling_asymmetric_sphere.in"
)
cp "$ROOT/results/rolling_asymmetric_sphere/output.csv" "$ROOT/data/rolling_asymmetric_sphere.csv"

echo "Created:"
echo "  $ROOT/data/free_triaxial_top.csv"
echo "  $ROOT/data/rolling_asymmetric_sphere.csv"
