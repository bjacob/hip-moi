#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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

OFFLOAD_ARCH="${OFFLOAD_ARCH:-gfx1201}"
HIP_MOI_BUILD_DIR="${HIP_MOI_BUILD_DIR:-${REPO_ROOT}-build}"
BUILD_DIR="${SCRIPT_DIR}/build/codegen"
OBJCOPY="${ROCM_ROOT}/llvm/bin/llvm-objcopy"
BUNDLER="${ROCM_ROOT}/llvm/bin/clang-offload-bundler"
READOBJ="${ROCM_ROOT}/llvm/bin/llvm-readobj"
OBJDUMP="${ROCM_ROOT}/llvm/bin/llvm-objdump"

if [[ ! -x "${OBJCOPY}" || ! -x "${BUNDLER}" || ! -x "${READOBJ}" || ! -x "${OBJDUMP}" ]]; then
  echo "error: expected LLVM object tools under ${ROCM_ROOT}/llvm/bin" >&2
  exit 1
fi

PRIVATE_EXE="${HIP_MOI_BUILD_DIR}/tests/hip_moi_instrumented_rdna4_pingpong_private_lds_test"
COOPERATIVE_EXE="${HIP_MOI_BUILD_DIR}/tests/hip_moi_instrumented_rdna4_pingpong_cooperative_lds_test"
if [[ ! -x "${PRIVATE_EXE}" || ! -x "${COOPERATIVE_EXE}" ]]; then
  echo "error: build the RDNA4 ping-pong instrumented tests first" >&2
  echo "       expected ${PRIVATE_EXE}" >&2
  echo "       expected ${COOPERATIVE_EXE}" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"

extract_device_object() {
  local name="$1"
  local exe="$2"
  local fatbin="${BUILD_DIR}/${name}.fatbin"
  local device="${BUILD_DIR}/${name}.device.o"

  "${OBJCOPY}" --dump-section .hip_fatbin="${fatbin}" "${exe}"
  "${BUNDLER}" \
    --unbundle \
    --type=o \
    --targets=hipv4-amdgcn-amd-amdhsa--"${OFFLOAD_ARCH}" \
    --input="${fatbin}" \
    --output="${device}"
  echo "${device}"
}

print_metadata() {
  local device="$1"
  "${READOBJ}" --notes "${device}" | awk '
    function row_name(mangled) {
      if(mangled ~ /private_pingpong_kernel/ && mangled ~ /ExactContextPolicy/) {
        return "private exact-shadow";
      }
      if(mangled ~ /private_pingpong_kernel/ && mangled ~ /FastContextPolicy/) {
        return "private publish-only";
      }
      if(mangled ~ /cooperative_pingpong_kernel/ && mangled ~ /ExactContextPolicy/ &&
         mangled ~ /ELb1/) {
        return "cooperative exact-shadow synchronized";
      }
      if(mangled ~ /cooperative_pingpong_kernel/ && mangled ~ /FastContextPolicy/ &&
         mangled ~ /ELb1/) {
        return "cooperative publish-only synchronized";
      }
      if(mangled ~ /cooperative_pingpong_kernel/ && mangled ~ /ExactContextPolicy/ &&
         mangled ~ /ELb0/) {
        return "cooperative exact-shadow unsynchronized";
      }
      return "";
    }
    /\.name:/ {
      name = row_name($0);
    }
    /\.group_segment_fixed_size:/ { group = $2; }
    /\.private_segment_fixed_size:/ { private = $2; }
    /\.sgpr_count:/ { sgpr = $2; }
    /\.sgpr_spill_count:/ { sgpr_spill = $2; }
    /\.vgpr_count:/ { vgpr = $2; }
    /\.vgpr_spill_count:/ {
      vgpr_spill = $2;
      if(name != "") {
        printf("%-44s lds=%-6s sgpr=%-4s vgpr=%-4s sgpr_spill=%-4s vgpr_spill=%-4s private=%s\n",
               name, group, sgpr, vgpr, sgpr_spill, vgpr_spill, private);
      }
    }
  '
}

print_instruction_counts() {
  local device="$1"
  "${OBJDUMP}" -d -C --mcpu="${OFFLOAD_ARCH}" "${device}" | awk '
    function classify(header) {
      if(header ~ /private_pingpong_kernel/ && header ~ /ExactContextPolicy/) {
        return "private exact-shadow";
      }
      if(header ~ /private_pingpong_kernel/ && header ~ /FastContextPolicy/) {
        return "private publish-only";
      }
      if(header ~ /cooperative_pingpong_kernel/ && header ~ /ExactContextPolicy, true/) {
        return "cooperative exact-shadow synchronized";
      }
      if(header ~ /cooperative_pingpong_kernel/ && header ~ /FastContextPolicy, true/) {
        return "cooperative publish-only synchronized";
      }
      if(header ~ /cooperative_pingpong_kernel/ && header ~ /ExactContextPolicy, false/) {
        return "cooperative exact-shadow unsynchronized";
      }
      return "";
    }
    function print_counts() {
      if(name != "") {
        printf("%-44s instr=%-5d setprio1=%-2d setprio0=%-2d wmma=%-2d barrier=%-3d flat_load_b128=%-3d flat_store_b128=%-3d scratch=%-4d swappc=%-3d\n",
               name, instr, setprio1, setprio0, wmma, barrier, flat_load_b128,
               flat_store_b128, scratch, swappc);
      }
    }
    /^0[0-9a-f]+ </ {
      print_counts();
      name = classify($0);
      instr = setprio1 = setprio0 = wmma = barrier = flat_load_b128 = 0;
      flat_store_b128 = scratch = swappc = 0;
      next;
    }
    name != "" && /^[[:space:]]+[a-z]/ {
      instr++;
      if($1 == "s_setprio" && $2 == "1") setprio1++;
      if($1 == "s_setprio" && $2 == "0") setprio0++;
      if($1 ~ /^v_wmma/) wmma++;
      if($1 ~ /^s_barrier/) barrier++;
      if($1 == "flat_load_b128") flat_load_b128++;
      if($1 == "flat_store_b128") flat_store_b128++;
      if($1 ~ /^scratch_/) scratch++;
      if($1 == "s_swappc_b64") swappc++;
    }
    END {
      print_counts();
    }
  '
}

print_object_marker_counts() {
  local label="$1"
  local device="$2"
  "${OBJDUMP}" -d -C --mcpu="${OFFLOAD_ARCH}" "${device}" | awk -v label="${label}" '
    /^[[:space:]]+[a-z]/ {
      if($1 == "s_setprio" && $2 == "1") setprio1++;
      if($1 == "s_setprio" && $2 == "0") setprio0++;
      if($1 ~ /^v_wmma/) wmma++;
      if($1 ~ /^s_barrier/) barrier++;
      if($1 == "s_swappc_b64") swappc++;
    }
    END {
      printf("%-24s setprio1=%-3d setprio0=%-3d wmma=%-3d barrier=%-3d swappc=%-3d\n",
             label, setprio1, setprio0, wmma, barrier, swappc);
    }
  '
}

private_device="$(extract_device_object private_pingpong "${PRIVATE_EXE}")"
cooperative_device="$(extract_device_object cooperative_pingpong "${COOPERATIVE_EXE}")"

echo "===== ping-pong metadata ====="
print_metadata "${private_device}"
print_metadata "${cooperative_device}"

echo "===== ping-pong instruction counts ====="
print_instruction_counts "${private_device}"
print_instruction_counts "${cooperative_device}"

echo "===== object-level marker counts ====="
print_object_marker_counts "private test object" "${private_device}"
print_object_marker_counts "cooperative test object" "${cooperative_device}"
