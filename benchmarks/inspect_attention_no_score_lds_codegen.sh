#!/usr/bin/env bash
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

HIPCC="${HIPCC:-${ROCM_ROOT}/bin/hipcc}"
OFFLOAD_ARCH="${OFFLOAD_ARCH:-gfx1201}"
HIP_MOI_ROOT="${HIP_MOI_ROOT:-${REPO_ROOT}}"
BUILD_DIR="${SCRIPT_DIR}/build/codegen"
OBJCOPY="${ROCM_ROOT}/llvm/bin/llvm-objcopy"
BUNDLER="${ROCM_ROOT}/llvm/bin/clang-offload-bundler"
READOBJ="${ROCM_ROOT}/llvm/bin/llvm-readobj"
OBJDUMP="${ROCM_ROOT}/llvm/bin/llvm-objdump"

if [[ ! -x "${OBJCOPY}" || ! -x "${BUNDLER}" || ! -x "${READOBJ}" || ! -x "${OBJDUMP}" ]]; then
  echo "error: expected LLVM object tools under ${ROCM_ROOT}/llvm/bin" >&2
  exit 1
fi
if [[ ! -d "${HIP_MOI_ROOT}/include/hip_moi" ]]; then
  echo "error: expected hip-moi headers under HIP_MOI_ROOT=${HIP_MOI_ROOT}" >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  set -- d16 d128
fi

mkdir -p "${BUILD_DIR}"

for shape in "$@"; do
  case "${shape}" in
  d16)
    source="${SCRIPT_DIR}/014_rdna4_wmma_no_score_lds_attention_benchmark.hip"
    kernel_regex="attention_no_score_lds_kernel<"
    ;;
  d128)
    source="${SCRIPT_DIR}/015_rdna4_d128_no_score_lds_attention_benchmark.hip"
    kernel_regex="attention_no_score_lds_kernel<"
    ;;
  *)
    echo "error: expected shape d16 or d128, got ${shape}" >&2
    exit 1
    ;;
  esac

  exe="${BUILD_DIR}/attention_no_score_lds_${shape}"
  fatbin="${BUILD_DIR}/attention_no_score_lds_${shape}.fatbin"
  device="${BUILD_DIR}/attention_no_score_lds_${shape}.device.o"

  "${HIPCC}" \
    --offload-arch="${OFFLOAD_ARCH}" \
    -std=c++20 \
    -O3 \
    -ffast-math \
    -Wall \
    -Wextra \
    -Wno-unused-command-line-argument \
    -I "${HIP_MOI_ROOT}/include" \
    "${source}" \
    -o "${exe}"

  "${OBJCOPY}" --dump-section .hip_fatbin="${fatbin}" "${exe}"
  "${BUNDLER}" \
    --unbundle \
    --type=o \
    --targets=hipv4-amdgcn-amd-amdhsa--"${OFFLOAD_ARCH}" \
    --input="${fatbin}" \
    --output="${device}"

  echo "===== no-score ${shape} metadata ====="
  "${READOBJ}" --notes "${device}" | awk '
    function row_name(mangled) {
      if(mangled ~ /I17PassThroughPolicy/) return "pass-through";
      if(mangled ~ /I22JakubSampledLoomPolicy/) return "Jakub-Sampled-Loom";
      if(mangled ~ /I20ContextSampledPolicy/) return "context + sampled_watchpoint";
      if(mangled ~ /I30SampledWatchpointContextPolicy/) return "sampled_watchpoint_context";
      return mangled;
    }
    /\.name:/ {
      name = row_name($0);
    }
    /\.private_segment_fixed_size:/ { private = $2; }
    /\.sgpr_count:/ { sgpr = $2; }
    /\.sgpr_spill_count:/ { sgpr_spill = $2; }
    /\.vgpr_count:/ { vgpr = $2; }
    /\.vgpr_spill_count:/ {
      vgpr_spill = $2;
      if(name != "") {
        printf("%-32s sgpr=%-4s vgpr=%-4s sgpr_spill=%-4s vgpr_spill=%-4s private=%s\n",
               name, sgpr, vgpr, sgpr_spill, vgpr_spill, private);
      }
    }
  '

  echo "===== no-score ${shape} instruction counts ====="
  "${OBJDUMP}" -d -C --mcpu="${OFFLOAD_ARCH}" "${device}" | awk -v kernel_regex="${kernel_regex}" '
    function print_counts() {
      printf("%-32s instr=%-5d flat_atomic=%-3d global_atomic=%-3d flat_load=%-3d flat_store=%-3d global_mem=%-3d ds=%-3d scratch=%-3d branch=%-4d\n",
             name, instr, flat_atomic, global_atomic, flat_load, flat_store, global_mem, ds,
             scratch, branch);
    }
    /^0[0-9a-f]+ </ {
      if(name != "") print_counts();
      name = "";
      if(index($0, kernel_regex) != 0) {
        name = $0;
        sub(/^.*attention_no_score_lds_kernel</, "", name);
        sub(/Policy>.*$/, "", name);
        if(name == "PassThrough") name = "pass-through";
        else if(name == "JakubSampledLoom") name = "Jakub-Sampled-Loom";
        else if(name == "ContextSampled") name = "context + sampled_watchpoint";
        else if(name == "SampledWatchpointContext") name = "sampled_watchpoint_context";
        instr = flat_atomic = global_atomic = flat_load = flat_store = global_mem = ds = scratch = branch = 0;
      }
      next;
    }
    name != "" && /^[[:space:]]+[a-z]/ {
      instr++;
      if($1 ~ /^flat_atomic/) flat_atomic++;
      if($1 ~ /^global_atomic/) global_atomic++;
      if($1 ~ /^flat_load/) flat_load++;
      if($1 ~ /^flat_store/) flat_store++;
      if($1 ~ /^global_/) global_mem++;
      if($1 ~ /^ds_/) ds++;
      if($1 ~ /^scratch_/) scratch++;
      if($1 ~ /^[sv]_cbranch/ || $1 ~ /^s_branch/) branch++;
    }
    END {
      if(name != "") print_counts();
    }
  '
done
