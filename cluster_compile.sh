#! /usr/bin/env bash
set -euo pipefail

module load clang
module load gcc/13.1
module load petsc/3.21.1-real-4amd
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
CMAKE=/software/compiler/cmake/3.26.3_dir/bin/cmake
CLANGXX="$(command -v clang++)"
EIGEN=/sw/ubuntu22/4amd/petsc/3.21.1-real_dir/include/eigen3

if [[ -z "$CLANGXX" ]]; then
    echo "Error: clang++ not found." >&2
    exit 1
fi

if [[ ! -x "$CMAKE" ]]; then
    echo "Error: CMake not found." >&2
    exit 1
fi

if [[ ! -f "$EIGEN/Eigen/Dense" ]]; then
    echo "Error: Eigen headers not found." >&2
    exit 1
fi

"$CMAKE" \
    -S "$ROOT_DIR" \
    -B "$BUILD_DIR" \
    -DCMAKE_CXX_COMPILER="$CLANGXX" \
    -DRIBODYN_EIGEN_INCLUDE_DIR="$EIGEN" \
    -DCMAKE_BUILD_TYPE=Release

"$CMAKE" \
    --build "$BUILD_DIR" \
    --verbose \
    -j1
