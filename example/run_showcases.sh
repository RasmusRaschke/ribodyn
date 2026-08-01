#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/solver"
    exit 1
fi

SOLVER="$(realpath "$1")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$ROOT/results/orbit" "$ROOT/results/dipole" "$ROOT/results/rotatingEM" "$ROOT/data"

(
    cd "$ROOT/results/orbit"
    "$SOLVER" "$ROOT/inputs/orbit.in"
)
cp "$ROOT/results/orbit/output.csv" "$ROOT/data/orbit.csv"

(
    cd "$ROOT/results/dipole"
    "$SOLVER" "$ROOT/inputs/dipole.in"
)
cp "$ROOT/results/dipole/output.csv" "$ROOT/data/dipole.csv"

(
    cd "$ROOT/results/rotatingEM"
    "$SOLVER" "$ROOT/inputs/rotating_field.in"
)
cp "$ROOT/results/rotatingEM/output.csv" "$ROOT/data/rotatingEM.csv"

echo "Created:"
echo "  $ROOT/data/orbit.csv"
echo "  $ROOT/data/dipole.csv"
echo "  $ROOT/data/rotatingEM.csv"
