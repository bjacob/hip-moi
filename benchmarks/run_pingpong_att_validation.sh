#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-private-pass-through}"
if [[ "${MODE}" != "private-pass-through" && "${MODE}" != "private-sampled" ]]; then
  echo "usage: $0 [private-pass-through|private-sampled]" >&2
  exit 1
fi

if [[ -n "${ROCPROFV3:-}" ]]; then
  if [[ ! -x "${ROCPROFV3}" ]]; then
    echo "error: ROCPROFV3=${ROCPROFV3} is not executable" >&2
    exit 1
  fi
elif [[ -x "/home/benoit/workspace/TheRock-build/profiler/rocprofiler-sdk/dist/bin/rocprofv3" ]]; then
  ROCPROFV3="/home/benoit/workspace/TheRock-build/profiler/rocprofiler-sdk/dist/bin/rocprofv3"
else
  ROCPROFV3="rocprofv3"
fi

ATT_LIBRARY_PATH="${ATT_LIBRARY_PATH:-$(cd "$(dirname "${ROCPROFV3}")/.." && pwd)/lib}"
OUT_DIR="${OUT_DIR:-$(mktemp -d "/tmp/hip-moi-pingpong-att-${MODE}.XXXXXX")}"
ATT_TARGET_CU="${ATT_TARGET_CU:-0}"
ATT_SIMD_SELECT="${ATT_SIMD_SELECT:-0x0}"
ATT_SHADER_ENGINE_MASK="${ATT_SHADER_ENGINE_MASK:-0xffffffff}"
ATT_SERIALIZE_ALL="${ATT_SERIALIZE_ALL:-true}"
EXPECTED_LDS_PRIORITY_SIGNATURE="${EXPECTED_LDS_PRIORITY_SIGNATURE:-}"

PROBE="$(HIP_MOI_BUILD_ONLY=1 "${SCRIPT_DIR}/build_pingpong_att_probe.sh")"

echo "probe=${PROBE}"
echo "mode=${MODE}"
echo "att_output=${OUT_DIR}"
echo "att_serialize_all=${ATT_SERIALIZE_ALL}"
if [[ -n "${EXPECTED_LDS_PRIORITY_SIGNATURE}" ]]; then
  echo "expected_lds_priority_signature=${EXPECTED_LDS_PRIORITY_SIGNATURE}"
fi

HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
  "${ROCPROFV3}" \
    --att \
    --att-library-path "${ATT_LIBRARY_PATH}" \
    --att-target-cu "${ATT_TARGET_CU}" \
    --att-simd-select "${ATT_SIMD_SELECT}" \
    --att-shader-engine-mask "${ATT_SHADER_ENGINE_MASK}" \
    --att-serialize-all "${ATT_SERIALIZE_ALL}" \
    -d "${OUT_DIR}" \
    -- "${PROBE}" "${MODE}"

VALIDATOR_ARGS=("${OUT_DIR}")
if [[ -n "${EXPECTED_LDS_PRIORITY_SIGNATURE}" ]]; then
  VALIDATOR_ARGS+=("--expected-lds-priority-signature" "${EXPECTED_LDS_PRIORITY_SIGNATURE}")
fi

"${REPO_ROOT}/benchmarks/validate_pingpong_att.py" "${VALIDATOR_ARGS[@]}"
