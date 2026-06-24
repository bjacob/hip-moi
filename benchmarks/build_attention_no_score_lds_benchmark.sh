#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${VENV:-${REPO_ROOT}/.venv}"

if [[ -n "${ROCM_ROOT:-}" ]]; then
  if [[ ! -x "${ROCM_ROOT}/bin/hipcc" ]]; then
    echo "error: expected hipcc under ROCM_ROOT=${ROCM_ROOT}" >&2
    exit 1
  fi
elif [[ -x "${VENV}/bin/rocm-sdk" && -x "${VENV}/bin/hipcc" ]]; then
  ROCM_ROOT="$("${VENV}/bin/rocm-sdk" path --root)"
elif [[ -x "/home/benoit/workspace/TheRock-build/dist/rocm/bin/hipcc" ]]; then
  ROCM_ROOT="/home/benoit/workspace/TheRock-build/dist/rocm"
else
  echo "error: expected TheRock ROCm SDK in ${VENV} or ROCM_ROOT=/path/to/rocm" \
    >&2
  exit 1
fi

export PATH="${ROCM_ROOT}/bin:${VENV}/bin:${PATH}"
export ROCM_PATH="${ROCM_ROOT}"
export ROCM_HOME="${ROCM_ROOT}"
export HIP_PATH="${ROCM_ROOT}"
export LD_LIBRARY_PATH="${ROCM_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export HIP_PLATFORM=amd

HIPCC="${HIPCC:-${ROCM_ROOT}/bin/hipcc}"
OFFLOAD_ARCH="${OFFLOAD_ARCH:-gfx1201}"
BUILD_DIR="${SCRIPT_DIR}/build"
OUT="${BUILD_DIR}/attention_no_score_lds_benchmark"
HIP_MOI_ROOT="${HIP_MOI_ROOT:-${REPO_ROOT}}"
if [[ ! -d "${HIP_MOI_ROOT}/include/hip_moi" ]]; then
  echo "error: expected hip-moi headers under HIP_MOI_ROOT=${HIP_MOI_ROOT}" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"

"${HIPCC}" \
  --offload-arch="${OFFLOAD_ARCH}" \
  -std=c++20 \
  -O3 \
  -ffast-math \
  -Wall \
  -Wextra \
  -Wno-unused-command-line-argument \
  -I "${HIP_MOI_ROOT}/include" \
  "${SCRIPT_DIR}/attention_no_score_lds_benchmark.hip" \
  -o "${OUT}"

HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" "${OUT}" "$@"
