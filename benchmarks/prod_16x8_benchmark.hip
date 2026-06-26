// Copyright 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Focused RDNA4 WMMA instrumentation benchmark vendored into hip-moi from the
// focused sanitizer-strategy/rdna4_matmul extraction, itself derived from
// sanitizer-strategy/rdna4_matmul/rdna4_matmul.hip.
//
// This file keeps the production fp16 WMMA row family from the large harness:
//   fp16 WMMA, 8 waves per workgroup, 16x8 WMMA tiles, KGroup=2,
//   hip-matmul-style pipelined LDS staging, M=N=K configurable at runtime.
//
// The extraction deliberately preserves the same kernel shape, packed WMMA data
// layout, manual LDS helper sites, Jakub-Sampled-Loom policy, and hip-moi
// exact-shadow/sampled-watchpoint policies used by the large harness. It drops
// correctness tests, FP8, RFC, same-wave policies, racy variants, and the
// general variant table. Future focused benchmarks can be produced by repeating
// this extraction pattern from rdna4_matmul.hip with a different row family.

#include <hip/hip_runtime.h>

#include "hip_moi/hip_moi.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using half_t = _Float16;
using f16x8 = half_t __attribute__((ext_vector_type(8)));
using f32x8 = float __attribute__((ext_vector_type(8)));

static constexpr int kWmmaM = 16;
static constexpr int kWmmaN = 16;
static constexpr int kWmmaK = 16;
static constexpr int kWaveSize = 32;
#ifndef HIP_MOI_BENCH_STORAGE_BYTES
#define HIP_MOI_BENCH_STORAGE_BYTES (128 * 1024)
#endif
static constexpr int kWaves = 8;
static constexpr int kThreads = kWaves * kWaveSize;
static constexpr int kMTiles = 16;
static constexpr int kNTiles = 8;
static constexpr int kKGroup = 2;
static constexpr int kWaveTileM = 4;
static constexpr int kWaveTileN = 4;
static constexpr double kFp16TargetTflops = 191.0;
static_assert(kWaves > 0);
static_assert((kMTiles / kWaveTileM) * (kNTiles / kWaveTileN) == kWaves);

static constexpr uint32_t kTsanShadowAccessRead = 1;
static constexpr uint32_t kTsanShadowAccessWrite = 2;
static constexpr uint32_t kTsanSampledGranuleShift = 2;
static constexpr uint32_t kTsanSampledGranuleBytes =
    1u << kTsanSampledGranuleShift;
static constexpr uint32_t kTsanSampledValidShift = 0;
static constexpr uint32_t kTsanSampledConsumedShift = 1;
static constexpr uint32_t kTsanSampledKindShift = 2;
static constexpr uint32_t kTsanSampledKindBits = 3;
static constexpr uint32_t kTsanSampledWorkitemShift = 5;
static constexpr uint32_t kTsanSampledWorkitemBits = 10;
static constexpr uint32_t kTsanSampledEpochShift = 15;
static constexpr uint32_t kTsanSampledEpochBits = 10;
static constexpr uint32_t kTsanSampledGenerationShift = 25;
static constexpr uint32_t kTsanSampledGenerationBits = 20;
static constexpr uint32_t kTsanSampledAddrShift = 45;
static constexpr uint32_t kTsanSampledAddrBits = 14;
static constexpr uint32_t kTsanSampledCountShift = 59;
static constexpr uint32_t kTsanSampledCountBits = 5;
static constexpr uint64_t kTsanSampledValid = 1ull << kTsanSampledValidShift;
static constexpr uint64_t kTsanSampledConsumed = 1ull
                                                 << kTsanSampledConsumedShift;
static constexpr uint64_t kTsanSampledKindMask =
    ((1ull << kTsanSampledKindBits) - 1) << kTsanSampledKindShift;
static constexpr uint64_t kTsanSampledWorkitemMask =
    ((1ull << kTsanSampledWorkitemBits) - 1) << kTsanSampledWorkitemShift;
static constexpr uint64_t kTsanSampledEpochMask =
    ((1ull << kTsanSampledEpochBits) - 1) << kTsanSampledEpochShift;
static constexpr uint64_t kTsanSampledGenerationMask =
    ((1ull << kTsanSampledGenerationBits) - 1) << kTsanSampledGenerationShift;
static constexpr uint64_t kTsanSampledEpochGenerationMask =
    kTsanSampledEpochMask | kTsanSampledGenerationMask;
static constexpr uint64_t kTsanSampledAddrMask =
    ((1ull << kTsanSampledAddrBits) - 1) << kTsanSampledAddrShift;
static constexpr uint64_t kTsanSampledCountMask =
    ((1ull << kTsanSampledCountBits) - 1) << kTsanSampledCountShift;
static constexpr uint32_t kTsanSampledMaxAddr =
    (1u << kTsanSampledAddrBits) - 1;
static constexpr uint32_t kTsanSampledMaxCount = 1u << kTsanSampledCountBits;

static void hip_check_impl(hipError_t err, const char *expr, const char *file,
                           int line) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP error %s from `%s` at %s:%d\n",
                 hipGetErrorString(err), expr, file, line);
    std::exit(EXIT_FAILURE);
  }
}

#define HIP_CHECK(expr) hip_check_impl((expr), #expr, __FILE__, __LINE__)

static int get_int_env(const char *name, int default_value) {
  const char *value = std::getenv(name);
  return value ? std::stoi(value) : default_value;
}

static int get_int_env_alias(const char *name, const char *legacy_name,
                             int default_value) {
  const char *value = std::getenv(name);
  if (value) {
    return std::stoi(value);
  }
  if (legacy_name) {
    value = std::getenv(legacy_name);
    if (value) {
      return std::stoi(value);
    }
  }
  return default_value;
}

static double get_double_env(const char *name, double default_value) {
  const char *value = std::getenv(name);
  return value ? std::strtod(value, nullptr) : default_value;
}

struct DisplayLatency {
  double value;
  const char *unit;
};

static DisplayLatency display_latency(double ms) {
  if (ms < 1.0) {
    return DisplayLatency{/*value=*/ms * 1000.0, /*unit=*/"µs"};
  }
  return DisplayLatency{/*value=*/ms, /*unit=*/"ms"};
}

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

__device__ hip_moi::context::storage_ref *g_hip_moi_context_storages = nullptr;
__device__ uint64_t g_hip_moi_context_storage_count = 0;

__device__ __forceinline__ hip_moi::context
make_hip_moi_context(uint32_t generation) {
  int thread_count = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);
  uint64_t workgroup_id = static_cast<uint64_t>(blockIdx.x) +
                          static_cast<uint64_t>(gridDim.x) *
                              (static_cast<uint64_t>(blockIdx.y) +
                               static_cast<uint64_t>(gridDim.y) *
                                   static_cast<uint64_t>(blockIdx.z));
  hip_moi::context::config cfg{
      /*thread_count=*/thread_count,
      /*threads_per_subgroup=*/kWaveSize,
      /*subgroup_count=*/(thread_count + kWaveSize - 1) / kWaveSize,
  };
  hip_moi::context::storage_ref storage{};
  if (g_hip_moi_context_storages &&
      workgroup_id < g_hip_moi_context_storage_count) {
    storage = g_hip_moi_context_storages[workgroup_id];
  }
  storage.generation = generation;
  return hip_moi::context(storage, cfg);
}

__device__ __forceinline__ hip_moi::sampled_watchpoint_context
make_hip_moi_sampled_watchpoint_context(uint32_t generation) {
  uint64_t workgroup_id = static_cast<uint64_t>(blockIdx.x) +
                          static_cast<uint64_t>(gridDim.x) *
                              (static_cast<uint64_t>(blockIdx.y) +
                               static_cast<uint64_t>(gridDim.y) *
                                   static_cast<uint64_t>(blockIdx.z));
  hip_moi::sampled_watchpoint_context::config cfg{
      /*threads_per_subgroup=*/kWaveSize,
  };
  hip_moi::context::storage_ref storage{};
  if (g_hip_moi_context_storages &&
      workgroup_id < g_hip_moi_context_storage_count) {
    storage = g_hip_moi_context_storages[workgroup_id];
  }
  storage.generation = generation;
  return hip_moi::make_sampled_watchpoint_context(storage, cfg);
}

__device__ __forceinline__ f32x8 zero_f32x8() {
  f32x8 value;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    value[i] = 0.0f;
  }
  return value;
}

namespace tsan {

enum class PolicyKind {
  PassThrough,
  JakubSampledLoom,
  HipMoiExactShadow,
  HipMoiSampledWatchpoint,
};

struct PassThroughPolicy {
  static constexpr bool enabled = false;
  static constexpr PolicyKind kind = PolicyKind::PassThrough;
};

// Benchmark-local HIP reproduction of Jakub's sampled Loom-like policy.
// It is not upstream HRX/Loom; use it only as the local comparison row.
struct JakubSampledLoomPolicy {
  static constexpr bool enabled = true;
  static constexpr PolicyKind kind = PolicyKind::JakubSampledLoom;
};

struct HipMoiExactShadowPolicy {
  static constexpr bool enabled = true;
  static constexpr PolicyKind kind = PolicyKind::HipMoiExactShadow;
  static constexpr bool use_static_sampled_policy = false;
};

struct HipMoiSampledWatchpointPolicy {
  static constexpr bool enabled = true;
  static constexpr PolicyKind kind = PolicyKind::HipMoiSampledWatchpoint;
  static constexpr bool use_static_sampled_policy = false;
};

struct HipMoiSampledWatchpointStaticPublishOnlyPolicy {
  static constexpr bool enabled = true;
  static constexpr PolicyKind kind = PolicyKind::HipMoiSampledWatchpoint;
  static constexpr bool use_static_sampled_policy = true;
  using sampled_policy = hip_moi::sampled_watchpoint_policy<
      /*SampleSkip=*/32,
      /*ProbeCount=*/1,
      /*DelayIters=*/32,
      /*ReportConflicts=*/false,
      /*StaticWatchpointCapacity=*/1>;
};

template <typename Policy> struct IsHipMoiPolicy {
  static constexpr bool value =
      Policy::kind == PolicyKind::HipMoiExactShadow ||
      Policy::kind == PolicyKind::HipMoiSampledWatchpoint;
};

template <typename Policy> struct HipMoiContextArg {};

template <> struct HipMoiContextArg<HipMoiExactShadowPolicy> {
  hip_moi::context ctx;
};

template <> struct HipMoiContextArg<HipMoiSampledWatchpointPolicy> {
  hip_moi::context ctx;
};

template <>
struct HipMoiContextArg<HipMoiSampledWatchpointStaticPublishOnlyPolicy> {
  hip_moi::sampled_watchpoint_context ctx;
};

template <typename Policy>
__device__ __forceinline__ HipMoiContextArg<Policy>
make_hip_moi_context_arg(uint32_t generation) {
  if constexpr (IsHipMoiPolicy<Policy>::value) {
    if constexpr (Policy::use_static_sampled_policy) {
      return HipMoiContextArg<Policy>{
          make_hip_moi_sampled_watchpoint_context(generation)};
    } else {
      return HipMoiContextArg<Policy>{make_hip_moi_context(generation)};
    }
  } else {
    return {};
  }
}

struct Runtime {
  uint64_t *shadow;
  unsigned long long *report_count;
  uint32_t workgroup_stride_entries;
  uint32_t generation;
  uint32_t sample_skip;
  uint32_t delay_iters;
  uint32_t probe_count;
};

__device__ __forceinline__ uint32_t flat_workitem_id() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

__device__ __forceinline__ uint32_t flat_workgroup_id() {
  return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}

template <typename Policy>
__device__ __forceinline__ uint64_t *workgroup_shadow_base(Runtime runtime) {
  if constexpr (!Policy::enabled) {
    return nullptr;
  } else {
    return runtime.shadow + static_cast<uint64_t>(flat_workgroup_id()) *
                                runtime.workgroup_stride_entries;
  }
}

template <typename Policy>
__device__ __forceinline__ void
prepare_runtime(Runtime &runtime,
                HipMoiContextArg<Policy> &hip_moi_context_arg) {
  if constexpr (IsHipMoiPolicy<Policy>::value) {
    (void)runtime;
    hip_moi_context_arg.ctx.init_workgroup();
  } else {
    (void)runtime;
    (void)hip_moi_context_arg;
  }
}

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__device__ __forceinline__ bool conflicting_access(uint32_t current_kind,
                                                   uint32_t prior_kind) {
  return current_kind == kTsanShadowAccessRead
             ? prior_kind != kTsanShadowAccessRead
             : true;
}

__device__ __forceinline__ void delay(uint32_t iterations) {
  for (uint32_t i = 0; i < iterations; ++i) {
    asm volatile("s_nop 0" ::: "memory");
  }
}

__device__ __forceinline__ uint64_t
pack_sampled_watchpoint(Runtime runtime, uint32_t start, uint32_t count,
                        uint32_t access_kind, uint64_t epoch) {
  return kTsanSampledValid |
         ((static_cast<uint64_t>(access_kind) &
           ((1ull << kTsanSampledKindBits) - 1))
          << kTsanSampledKindShift) |
         ((static_cast<uint64_t>(flat_workitem_id()) &
           ((1ull << kTsanSampledWorkitemBits) - 1))
          << kTsanSampledWorkitemShift) |
         ((epoch & ((1ull << kTsanSampledEpochBits) - 1))
          << kTsanSampledEpochShift) |
         ((static_cast<uint64_t>(runtime.generation) &
           ((1ull << kTsanSampledGenerationBits) - 1))
          << kTsanSampledGenerationShift) |
         ((static_cast<uint64_t>(start) & ((1ull << kTsanSampledAddrBits) - 1))
          << kTsanSampledAddrShift) |
         ((static_cast<uint64_t>(count - 1) &
           ((1ull << kTsanSampledCountBits) - 1))
          << kTsanSampledCountShift);
}

__device__ __forceinline__ uint32_t sampled_kind(uint64_t entry) {
  return static_cast<uint32_t>((entry & kTsanSampledKindMask) >>
                               kTsanSampledKindShift);
}

__device__ __forceinline__ uint32_t sampled_workitem(uint64_t entry) {
  return static_cast<uint32_t>((entry & kTsanSampledWorkitemMask) >>
                               kTsanSampledWorkitemShift);
}

__device__ __forceinline__ uint32_t sampled_start(uint64_t entry) {
  return static_cast<uint32_t>((entry & kTsanSampledAddrMask) >>
                               kTsanSampledAddrShift);
}

__device__ __forceinline__ uint32_t sampled_count(uint64_t entry) {
  return static_cast<uint32_t>((entry & kTsanSampledCountMask) >>
                               kTsanSampledCountShift) +
         1;
}

__device__ __forceinline__ bool sampled_range_overlaps(uint32_t lhs_start,
                                                       uint32_t lhs_count,
                                                       uint32_t rhs_start,
                                                       uint32_t rhs_count) {
  return lhs_start < rhs_start + rhs_count && rhs_start < lhs_start + lhs_count;
}

__device__ __forceinline__ uint32_t lds_cell_start(uint32_t byte_offset) {
  return byte_offset >> kTsanSampledGranuleShift;
}

__device__ __forceinline__ uint32_t wave_selection_seed(Runtime runtime,
                                                        uint32_t wave,
                                                        uint32_t site_id) {
  uint32_t seed = runtime.generation * 0x9e3779b9U;
  seed ^= (flat_workgroup_id() + 1) * 0x85ebca6bU;
  seed ^= (wave + 1) * 0xc2b2ae35U;
  seed ^= site_id * 0x27d4eb2dU;
  return mix32(seed);
}

__device__ __forceinline__ uint32_t sampled_watchpoint_count(Runtime runtime) {
  return runtime.workgroup_stride_entries - 1;
}

__device__ __forceinline__ uint32_t sampled_watchpoint_mask(Runtime runtime) {
  return sampled_watchpoint_count(runtime) - 1;
}

__device__ __forceinline__ bool
sampled_conflicts(uint64_t prior, uint64_t current, uint32_t start,
                  uint32_t count, uint32_t access_kind) {
  if ((prior & kTsanSampledValid) == 0 || (prior & kTsanSampledConsumed) != 0) {
    return false;
  }
  if (((prior ^ current) & kTsanSampledEpochGenerationMask) != 0) {
    return false;
  }
  if (sampled_workitem(prior) == flat_workitem_id()) {
    return false;
  }
  if (!conflicting_access(access_kind, sampled_kind(prior))) {
    return false;
  }
  return sampled_range_overlaps(start, count, sampled_start(prior),
                                sampled_count(prior));
}

__device__ __attribute__((noinline)) void
sampled_scan_reports(Runtime runtime, uint64_t *base, uint64_t current,
                     uint32_t start, uint32_t count, uint32_t access_kind,
                     uint32_t site_id, uint32_t publish_index,
                     uint32_t selection_seed) {
  uint32_t watchpoints = sampled_watchpoint_count(runtime);
  uint32_t probe_count =
      runtime.probe_count == 0 ? watchpoints : runtime.probe_count;
  if (probe_count > watchpoints) {
    probe_count = watchpoints;
  }
  uint32_t probe_step =
      (mix32(selection_seed ^ start ^ (site_id * 0x94d049bbU)) | 1u) &
      sampled_watchpoint_mask(runtime);
#pragma unroll 1
  for (uint32_t probe = 0; probe < probe_count; ++probe) {
    uint32_t index = 1 + ((publish_index - 1 + probe * probe_step) &
                          sampled_watchpoint_mask(runtime));
    uint64_t *entry64 = base + index;
    uint64_t prior = *reinterpret_cast<volatile uint64_t *>(entry64);
    if (sampled_conflicts(prior, current, start, count, access_kind)) {
      uint64_t consumed = prior | kTsanSampledConsumed;
      auto *entry = reinterpret_cast<unsigned long long *>(entry64);
      if (atomicCAS(entry, static_cast<unsigned long long>(prior),
                    static_cast<unsigned long long>(consumed)) == prior &&
          runtime.report_count) {
        atomicAdd(runtime.report_count, 1ull);
      }
    }
  }
}

__device__ __forceinline__ bool
sampled_should_publish(Runtime runtime, uint32_t seed, uint32_t lane) {
  uint32_t skip = runtime.sample_skip == 0 ? 1 : runtime.sample_skip;
  if (skip > 1) {
    bool power_of_two = (skip & (skip - 1)) == 0;
    uint32_t remainder = power_of_two ? (seed & (skip - 1)) : (seed % skip);
    if (remainder != 0) {
      return false;
    }
  }
  return lane == ((seed >> 16) & (kWaveSize - 1));
}

__device__ __forceinline__ uint32_t sampled_publish_index(Runtime runtime,
                                                          uint32_t site_id,
                                                          uint64_t epoch,
                                                          uint32_t wave,
                                                          uint32_t start) {
  uint32_t seed = site_id * 0x9e3779b9U;
  seed ^= static_cast<uint32_t>(epoch) * 0x85ebca6bU;
  seed ^= runtime.generation * 0xc2b2ae35U;
  seed ^= wave * 0x27d4eb2dU;
  seed ^= start * 0x165667b1U;
  return 1 + (mix32(seed) & sampled_watchpoint_mask(runtime));
}

__device__ __forceinline__ void
observe_sampled_range(Runtime runtime, uint32_t start, uint32_t count,
                      uint32_t access_kind, uint32_t site_id, uint32_t wave,
                      uint32_t lane, uint32_t selection_seed) {
  if (runtime.workgroup_stride_entries <= 1 || count == 0 ||
      start > kTsanSampledMaxAddr) {
    return;
  }
  uint32_t max_count_for_addr = kTsanSampledMaxAddr - start + 1;
  if (count > max_count_for_addr) {
    count = max_count_for_addr;
  }

  if (!sampled_should_publish(runtime, selection_seed, lane)) {
    return;
  }
  uint64_t *base = workgroup_shadow_base<JakubSampledLoomPolicy>(runtime);
  uint64_t epoch = base[0] & ((1ull << kTsanSampledEpochBits) - 1);
  uint64_t current =
      pack_sampled_watchpoint(runtime, start, count, access_kind, epoch);
  uint32_t publish_index =
      sampled_publish_index(runtime, site_id, epoch, wave, start);

  if (runtime.report_count) {
    sampled_scan_reports(runtime, base, current, start, count, access_kind,
                         site_id, publish_index, selection_seed);
  }

  auto *entry = reinterpret_cast<unsigned long long *>(base + publish_index);
  atomicExch(entry, static_cast<unsigned long long>(current));
  delay(runtime.delay_iters);
}

template <uint32_t AccessSize>
__device__ __forceinline__ void observe_jakub_sampled_loom_access(
    Runtime runtime, uint32_t byte_offset, uint32_t access_kind,
    uint32_t site_id, uint32_t wave, uint32_t lane, uint32_t selection_seed) {
  uint32_t start = lds_cell_start(byte_offset);
  uint32_t end = (byte_offset + AccessSize + kTsanSampledGranuleBytes - 1) >>
                 kTsanSampledGranuleShift;
  while (start < end) {
    uint32_t chunk =
        end - start < kTsanSampledMaxCount ? end - start : kTsanSampledMaxCount;
    observe_sampled_range(runtime, start, chunk, access_kind, site_id, wave,
                          lane, selection_seed);
    start += chunk;
  }
}

template <typename Policy, uint32_t AccessSize>
__device__ __forceinline__ void observe(Runtime runtime, uint32_t byte_offset,
                                        uint32_t access_kind,
                                        uint32_t site_id) {
  if constexpr (!Policy::enabled) {
    return;
  } else if constexpr (Policy::kind == PolicyKind::JakubSampledLoom) {
    uint32_t flat = flat_workitem_id();
    uint32_t wave = flat / kWaveSize;
    uint32_t lane = flat & (kWaveSize - 1);
    uint32_t seed = wave_selection_seed(runtime, wave, site_id);
    observe_jakub_sampled_loom_access<AccessSize>(runtime, byte_offset, access_kind,
                                            site_id, wave, lane, seed);
  }
}

template <typename Policy, typename T>
__device__ __forceinline__ void
ds_write(T *ptr, const T &value, Runtime runtime, uint32_t byte_offset,
         uint32_t site_id, HipMoiContextArg<Policy> &hip_moi_context_arg) {
  if constexpr (IsHipMoiPolicy<Policy>::value) {
    (void)runtime;
    if constexpr (Policy::kind == PolicyKind::HipMoiSampledWatchpoint) {
      if constexpr (Policy::use_static_sampled_policy) {
        hip_moi_context_arg.ctx
            .template lds_store_at<typename Policy::sampled_policy>(
                ptr, value, byte_offset,
                hip_moi::site_id(static_cast<uint64_t>(site_id)));
      } else {
        hip_moi_context_arg.ctx
            .template lds_store_at<hip_moi::backend_kind::sampled_watchpoint>(
                ptr, value, byte_offset,
                hip_moi::site_id(static_cast<uint64_t>(site_id)));
      }
    } else {
      hip_moi_context_arg.ctx
          .template lds_store_at<hip_moi::backend_kind::exact_shadow>(
              ptr, value, byte_offset,
              hip_moi::site_id(static_cast<uint64_t>(site_id)));
    }
  } else {
    observe<Policy, static_cast<uint32_t>(sizeof(T))>(
        runtime, byte_offset, kTsanShadowAccessWrite, site_id);
    *ptr = value;
  }
}

template <typename Policy, typename T>
__device__ __forceinline__ T
ds_read(const T *ptr, Runtime runtime, uint32_t byte_offset, uint32_t site_id,
        HipMoiContextArg<Policy> &hip_moi_context_arg) {
  if constexpr (IsHipMoiPolicy<Policy>::value) {
    (void)runtime;
    if constexpr (Policy::kind == PolicyKind::HipMoiSampledWatchpoint) {
      if constexpr (Policy::use_static_sampled_policy) {
        return hip_moi_context_arg.ctx
            .template lds_load_at<typename Policy::sampled_policy>(
                ptr, byte_offset,
                hip_moi::site_id(static_cast<uint64_t>(site_id)));
      } else {
        return hip_moi_context_arg.ctx
            .template lds_load_at<hip_moi::backend_kind::sampled_watchpoint>(
                ptr, byte_offset,
                hip_moi::site_id(static_cast<uint64_t>(site_id)));
      }
    } else {
      return hip_moi_context_arg.ctx
          .template lds_load_at<hip_moi::backend_kind::exact_shadow>(
              ptr, byte_offset,
              hip_moi::site_id(static_cast<uint64_t>(site_id)));
    }
  } else {
    observe<Policy, static_cast<uint32_t>(sizeof(T))>(
        runtime, byte_offset, kTsanShadowAccessRead, site_id);
    return *ptr;
  }
}

template <typename Policy>
__device__ __forceinline__ void
syncthreads(Runtime runtime, HipMoiContextArg<Policy> &hip_moi_context_arg) {
  if constexpr (!Policy::enabled) {
    (void)hip_moi_context_arg;
    __syncthreads();
  } else if constexpr (IsHipMoiPolicy<Policy>::value) {
    (void)runtime;
    hip_moi_context_arg.ctx.syncthreads();
  } else if constexpr (Policy::kind == PolicyKind::JakubSampledLoom) {
    (void)hip_moi_context_arg;
    __syncthreads();
    if (flat_workitem_id() == 0) {
      atomicAdd(reinterpret_cast<unsigned long long *>(
                    workgroup_shadow_base<Policy>(runtime)),
                1ull);
    }
    __syncthreads();
  }
}

} // namespace tsan

template <typename TSanPolicy>
__global__ __launch_bounds__(kThreads) void prod_16x8_kernel(
    const half_t *__restrict__ A_scalar, const half_t *__restrict__ B_scalar,
    float *__restrict__ C_scalar, int M, int N, int K,
    tsan::Runtime tsan_runtime) {
  constexpr int kWaveGridM = kMTiles / kWaveTileM;
  constexpr int kTilesPerWave = kWaveTileM;
  constexpr int kNTilesPerWave = kWaveTileN;
  constexpr int kThreadLoadStride = kThreads;
  constexpr int kAThreadLoads =
      (kMTiles * kWaveSize + kThreadLoadStride - 1) / kThreadLoadStride;
  constexpr int kBThreadLoads =
      (kNTiles * kWaveSize + kThreadLoadStride - 1) / kThreadLoadStride;

  const auto *A = reinterpret_cast<const f16x8 *>(A_scalar);
  const auto *B = reinterpret_cast<const f16x8 *>(B_scalar);
  auto *C = reinterpret_cast<f32x8 *>(C_scalar);

  __shared__ f16x8 a_s[kKGroup][kMTiles][kWaveSize];
  __shared__ f16x8 b_s[kKGroup][kNTiles][kWaveSize];
  auto hip_moi_context_arg =
      tsan::make_hip_moi_context_arg<TSanPolicy>(tsan_runtime.generation);
  tsan::prepare_runtime<TSanPolicy>(tsan_runtime, hip_moi_context_arg);

  int tid = threadIdx.x;
  int wave = tid / kWaveSize;
  int lane = tid & (kWaveSize - 1);
  int wave_m = wave % kWaveGridM;
  int wave_n = wave / kWaveGridM;
  int m_outer = blockIdx.y;
  int n_outer = blockIdx.x;
  int n_outer_count = N / (kNTiles * kWmmaN);
  int k_outer_count = K / kWmmaK;
  (void)M;

  f32x8 acc[kTilesPerWave][kNTilesPerWave];
#pragma unroll
  for (int t = 0; t < kTilesPerWave; ++t) {
#pragma unroll
    for (int n = 0; n < kNTilesPerWave; ++n) {
      acc[t][n] = zero_f32x8();
    }
  }

  auto a_byte_offset = [](int k_group, int tile, int lane) {
    return static_cast<uint32_t>(
        (((k_group * (kMTiles + kNTiles) + tile) * kWaveSize + lane) *
         sizeof(f16x8)));
  };
  auto b_byte_offset = [](int k_group, int tile, int lane) {
    return static_cast<uint32_t>(
        (((k_group * (kMTiles + kNTiles) + kMTiles + tile) * kWaveSize + lane) *
         sizeof(f16x8)));
  };

  struct StageRegs {
    f16x8 a[kKGroup][kAThreadLoads];
    f16x8 b[kKGroup][kBThreadLoads];
  };
  struct ComputeRegs {
    f16x8 a[kKGroup][kTilesPerWave];
    f16x8 b[kKGroup][kNTilesPerWave];
  };

  auto group_size = [&](int k_group_outer) {
    int remaining = k_outer_count - k_group_outer * kKGroup;
    return remaining < kKGroup ? remaining : kKGroup;
  };

  auto global_to_regs = [&](int k_group_outer, StageRegs &regs) {
#pragma unroll
    for (int g = 0; g < kKGroup; ++g) {
      int k_outer = k_group_outer * kKGroup + g;
      if (k_outer >= k_outer_count) {
        continue;
      }
      const int a_base =
          (m_outer * k_outer_count + k_outer) * kMTiles * kWaveSize;
      const int b_base =
          (n_outer * k_outer_count + k_outer) * kNTiles * kWaveSize;
#pragma unroll
      for (int copy = 0; copy < kAThreadLoads; ++copy) {
        int idx = tid + copy * kThreadLoadStride;
        if (idx < kMTiles * kWaveSize) {
          regs.a[g][copy] = A[a_base + idx];
        }
      }
#pragma unroll
      for (int copy = 0; copy < kBThreadLoads; ++copy) {
        int idx = tid + copy * kThreadLoadStride;
        if (idx < kNTiles * kWaveSize) {
          regs.b[g][copy] = B[b_base + idx];
        }
      }
    }
  };

  auto regs_to_shared = [&](int k_group_outer, const StageRegs &regs) {
#pragma unroll
    for (int g = 0; g < kKGroup; ++g) {
      int k_outer = k_group_outer * kKGroup + g;
      if (k_outer >= k_outer_count) {
        continue;
      }
#pragma unroll
      for (int copy = 0; copy < kAThreadLoads; ++copy) {
        int idx = tid + copy * kThreadLoadStride;
        if (idx < kMTiles * kWaveSize) {
          int tile = idx / kWaveSize;
          int local_lane = idx & (kWaveSize - 1);
          tsan::ds_write<TSanPolicy>(
              &a_s[g][tile][local_lane], regs.a[g][copy], tsan_runtime,
              a_byte_offset(g, tile, local_lane), 1, hip_moi_context_arg);
        }
      }
#pragma unroll
      for (int copy = 0; copy < kBThreadLoads; ++copy) {
        int idx = tid + copy * kThreadLoadStride;
        if (idx < kNTiles * kWaveSize) {
          int tile = idx / kWaveSize;
          int local_lane = idx & (kWaveSize - 1);
          tsan::ds_write<TSanPolicy>(
              &b_s[g][tile][local_lane], regs.b[g][copy], tsan_runtime,
              b_byte_offset(g, tile, local_lane), 2, hip_moi_context_arg);
        }
      }
    }
  };

  auto shared_to_compute_regs = [&](int valid_groups, ComputeRegs &regs) {
#pragma unroll
    for (int g = 0; g < kKGroup; ++g) {
      if (g >= valid_groups) {
        continue;
      }
#pragma unroll
      for (int n = 0; n < kNTilesPerWave; ++n) {
        int n_tile = wave_n * kWaveTileN + n;
        regs.b[g][n] = tsan::ds_read<TSanPolicy>(
            &b_s[g][n_tile][lane], tsan_runtime, b_byte_offset(g, n_tile, lane),
            3, hip_moi_context_arg);
      }
#pragma unroll
      for (int t = 0; t < kTilesPerWave; ++t) {
        int m_tile = wave_m * kWaveTileM + t;
        regs.a[g][t] = tsan::ds_read<TSanPolicy>(
            &a_s[g][m_tile][lane], tsan_runtime, a_byte_offset(g, m_tile, lane),
            4, hip_moi_context_arg);
      }
    }
  };

  auto compute_from_regs = [&](int valid_groups, const ComputeRegs &regs) {
#pragma unroll
    for (int g = 0; g < kKGroup; ++g) {
      if (g >= valid_groups) {
        continue;
      }
#pragma unroll
      for (int t = 0; t < kTilesPerWave; ++t) {
#pragma unroll
        for (int n = 0; n < kNTilesPerWave; ++n) {
          acc[t][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
              regs.a[g][t], regs.b[g][n], acc[t][n]);
        }
      }
    }
  };

  auto sync = [&]() {
    tsan::syncthreads<TSanPolicy>(tsan_runtime, hip_moi_context_arg);
  };

  int k_group_count = (k_outer_count + kKGroup - 1) / kKGroup;
  StageRegs pending_regs;
  ComputeRegs compute_regs;
  global_to_regs(0, pending_regs);
  regs_to_shared(0, pending_regs);
  if (k_group_count >= 2) {
    StageRegs next_regs;
    global_to_regs(1, next_regs);
    sync();
    shared_to_compute_regs(group_size(0), compute_regs);
    for (int k_group_outer = 0; k_group_outer < k_group_count - 2;
         ++k_group_outer) {
      sync();
      regs_to_shared(k_group_outer + 1, next_regs);
      global_to_regs(k_group_outer + 2, next_regs);
      compute_from_regs(group_size(k_group_outer), compute_regs);
      sync();
      shared_to_compute_regs(group_size(k_group_outer + 1), compute_regs);
    }
    int penultimate = k_group_count - 2;
    sync();
    regs_to_shared(penultimate + 1, next_regs);
    compute_from_regs(group_size(penultimate), compute_regs);
    sync();
    shared_to_compute_regs(group_size(penultimate + 1), compute_regs);
    compute_from_regs(group_size(penultimate + 1), compute_regs);
  } else {
    sync();
    shared_to_compute_regs(group_size(0), compute_regs);
    compute_from_regs(group_size(0), compute_regs);
  }

  int c_tile_base =
      (m_outer * n_outer_count + n_outer) * kMTiles * kNTiles * kWaveSize;
#pragma unroll
  for (int t = 0; t < kTilesPerWave; ++t) {
#pragma unroll
    for (int n = 0; n < kNTilesPerWave; ++n) {
      int m_tile = wave_m * kWaveTileM + t;
      int n_tile = wave_n * kWaveTileN + n;
      C[c_tile_base + (m_tile * kNTiles + n_tile) * kWaveSize + lane] =
          acc[t][n];
    }
  }
}

template <typename T> static T *gpu_alloc(std::size_t count) {
  T *ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, count * sizeof(T)));
  return ptr;
}

enum class Mode {
  PassThrough,
  JakubSampledLoom,
  HipMoiExactShadow,
  HipMoiSampledWatchpoint,
};

static bool is_hip_moi_mode(Mode mode) {
  return mode == Mode::HipMoiExactShadow ||
         mode == Mode::HipMoiSampledWatchpoint;
}

struct Allocation {
  uint64_t *shadow = nullptr;
  unsigned long long *report_count = nullptr;
  std::vector<hip_moi::host_context *> hip_moi_contexts;
  hip_moi::context::storage_ref *hip_moi_context_storage_refs = nullptr;
  uint32_t sampled_watchpoints = 1;
  uint32_t workgroup_stride_entries = 0;
  uint32_t generation = 1;
  uint32_t sample_skip = 32;
  uint32_t delay_iters = 32;
  uint32_t probe_count = 1;
  bool sampled_reports = false;
};

static uint64_t workgroup_count(int M, int N) {
  return static_cast<uint64_t>(ceil_div(N, kNTiles * kWmmaN)) *
         static_cast<uint64_t>(ceil_div(M, kMTiles * kWmmaM));
}

struct SampledKnobs {
  uint32_t watchpoints;
  uint32_t sample_skip;
  uint32_t delay_iters;
  uint32_t probe_count;
  bool reports;
};

static SampledKnobs sampled_knobs_from_env() {
  int watchpoints =
      get_int_env_alias("SAMPLED_WATCHPOINTS", "TSAN_SAMPLED_WATCHPOINTS", 1);
  int sample_skip = get_int_env_alias("SAMPLED_SKIP", "TSAN_SAMPLED_SKIP", 32);
  int delay_iters =
      get_int_env_alias("SAMPLED_DELAY", "TSAN_SAMPLED_DELAY", 32);
  int probe_count =
      get_int_env_alias("SAMPLED_PROBES", "TSAN_SAMPLED_PROBES", 1);
  bool reports = get_int_env("SAMPLED_REPORTS", 0) != 0;

  if (watchpoints <= 0 || (watchpoints & (watchpoints - 1)) != 0) {
    std::fprintf(
        stderr, "error: SAMPLED_WATCHPOINTS must be a positive power of two\n");
    std::exit(EXIT_FAILURE);
  }
  if (sample_skip < 0 || delay_iters < 0 || probe_count < 0) {
    std::fprintf(stderr,
                 "error: SAMPLED_SKIP, SAMPLED_DELAY, and SAMPLED_PROBES must "
                 "be nonnegative\n");
    std::exit(EXIT_FAILURE);
  }

  return SampledKnobs{
      static_cast<uint32_t>(watchpoints),
      static_cast<uint32_t>(sample_skip),
      static_cast<uint32_t>(delay_iters),
      static_cast<uint32_t>(probe_count),
      reports,
  };
}

static bool uses_default_static_hip_moi_sampled_policy(SampledKnobs knobs) {
  return knobs.watchpoints == 1 && knobs.sample_skip == 32 &&
         knobs.probe_count == 1 && knobs.delay_iters == 32 && !knobs.reports;
}

static Allocation make_allocation(Mode mode, int M, int N) {
  Allocation allocation;
  SampledKnobs sampled_knobs = sampled_knobs_from_env();
  uint64_t workgroups = workgroup_count(M, N);
  allocation.sampled_watchpoints = sampled_knobs.watchpoints;
  allocation.sample_skip = sampled_knobs.sample_skip;
  allocation.delay_iters = sampled_knobs.delay_iters;
  allocation.probe_count = sampled_knobs.probe_count;
  allocation.sampled_reports = sampled_knobs.reports;

  if (mode == Mode::JakubSampledLoom) {
    allocation.workgroup_stride_entries =
        1 + static_cast<uint32_t>(allocation.sampled_watchpoints);
    HIP_CHECK(hipMalloc(&allocation.shadow,
                        workgroups * allocation.workgroup_stride_entries *
                            sizeof(uint64_t)));
    HIP_CHECK(hipMemset(allocation.shadow, 0,
                        workgroups * allocation.workgroup_stride_entries *
                            sizeof(uint64_t)));
    if (allocation.sampled_reports) {
      HIP_CHECK(hipMalloc(&allocation.report_count,
                          workgroups * sizeof(*allocation.report_count)));
      HIP_CHECK(hipMemset(allocation.report_count, 0,
                          workgroups * sizeof(*allocation.report_count)));
    }
  } else if (is_hip_moi_mode(mode)) {
    hip_moi::host_context_options options;
    options.storage_bytes = static_cast<std::size_t>(
        get_int_env("HIP_MOI_STORAGE_BYTES", HIP_MOI_BENCH_STORAGE_BYTES));
    options.diagnostic_capacity = get_int_env("HIP_MOI_DIAGNOSTICS", 1024);
    options.subgroup_capacity = get_int_env("HIP_MOI_SUBGROUPS", kWaves);
    options.sampled_watchpoint_capacity =
        static_cast<int>(allocation.sampled_watchpoints);
    options.sampled_watchpoint_sample_skip = allocation.sample_skip;
    options.sampled_watchpoint_probe_count = allocation.probe_count;
    options.sampled_watchpoint_delay_iters = allocation.delay_iters;
    options.sampled_watchpoint_reports = allocation.sampled_reports;
    options.backend = mode == Mode::HipMoiSampledWatchpoint
                          ? hip_moi::backend_kind::sampled_watchpoint
                          : hip_moi::backend_kind::exact_shadow;
    options.destructor_reports = false;
    options.destructor_aborts = false;
    options.diagnostic_stream = nullptr;

    std::vector<hip_moi::context::storage_ref> storage_refs;
    storage_refs.reserve(workgroups);
    for (uint64_t i = 0; i < workgroups; ++i) {
      auto *context = new hip_moi::host_context(options);
      context->disable_destructor_check();
      allocation.hip_moi_contexts.push_back(context);
      storage_refs.push_back(context->launch_ref());
    }
    HIP_CHECK(hipMalloc(&allocation.hip_moi_context_storage_refs,
                        storage_refs.size() * sizeof(storage_refs[0])));
    HIP_CHECK(hipMemcpy(
        allocation.hip_moi_context_storage_refs, storage_refs.data(),
        storage_refs.size() * sizeof(storage_refs[0]), hipMemcpyHostToDevice));
    auto *device_storage_refs = allocation.hip_moi_context_storage_refs;
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_hip_moi_context_storages),
                                &device_storage_refs,
                                sizeof(device_storage_refs)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_hip_moi_context_storage_count),
                                &workgroups, sizeof(workgroups)));
  }
  return allocation;
}

static tsan::Runtime next_runtime(Allocation &allocation) {
  tsan::Runtime runtime{
      allocation.shadow,
      allocation.report_count,
      allocation.workgroup_stride_entries,
      allocation.generation,
      allocation.sample_skip,
      allocation.delay_iters,
      allocation.probe_count,
  };
  ++allocation.generation;
  return runtime;
}

static void free_allocation(Allocation &allocation) {
  if (allocation.hip_moi_context_storage_refs) {
    HIP_CHECK(hipFree(allocation.hip_moi_context_storage_refs));
  }
  for (auto *context : allocation.hip_moi_contexts) {
    delete context;
  }
  if (allocation.shadow) {
    HIP_CHECK(hipFree(allocation.shadow));
  }
  if (allocation.report_count) {
    HIP_CHECK(hipFree(allocation.report_count));
  }
  allocation = {};
}

template <typename Policy>
static void launch_kernel(half_t *a, half_t *b, float *c, tsan::Runtime runtime,
                          int M, int N, int K, hipStream_t stream) {
  dim3 block(kThreads);
  dim3 grid(ceil_div(N, kNTiles * kWmmaN), ceil_div(M, kMTiles * kWmmaM));
  prod_16x8_kernel<Policy>
      <<<grid, block, 0, stream>>>(a, b, c, M, N, K, runtime);
}

template <typename Policy>
static double run_one(const char *name, Mode mode, half_t *a, half_t *b,
                      float *c, int M, int N, int K, double min_ms,
                      double warmup_ms) {
  Allocation allocation = make_allocation(mode, M, N);
  hipStream_t stream = nullptr;
  for (int i = 0; i < 5; ++i) {
    launch_kernel<Policy>(a, b, c, next_runtime(allocation), M, N, K, stream);
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  int calibration_iterations =
      std::max(1, get_int_env("BENCH_CALIBRATION_ITERS", 10));
  int max_iterations = std::max(1, get_int_env("MAX_ITERS", 100000));

  auto run_timed_iterations = [&](int iteration_count) {
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < iteration_count; ++i) {
      launch_kernel<Policy>(a, b, c, next_runtime(allocation), M, N, K, stream);
    }
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipGetLastError());
    return elapsed_ms;
  };

  auto target_iterations = [&](double target_ms, double estimate_ms) {
    if (target_ms <= 0.0) {
      return 0;
    }
    int iterations = std::max(
        1,
        static_cast<int>(std::ceil(target_ms / std::max(0.001, estimate_ms))));
    return std::min(iterations, max_iterations);
  };

  auto run_until_target = [&](double target_ms, int *iteration_count) {
    if (target_ms <= 0.0 || *iteration_count <= 0) {
      *iteration_count = 0;
      return 0.0f;
    }

    float elapsed_ms = 0.0f;
    for (;;) {
      elapsed_ms = run_timed_iterations(*iteration_count);
      if (elapsed_ms >= target_ms || *iteration_count == max_iterations) {
        return elapsed_ms;
      }

      double scale = target_ms / std::max(0.001f, elapsed_ms);
      int next_iterations = static_cast<int>(
          std::ceil(static_cast<double>(*iteration_count) * scale));
      *iteration_count = std::min(
          max_iterations, std::max(*iteration_count + 1, next_iterations));
    }
  };

  float calibration_ms = run_timed_iterations(calibration_iterations);
  float one_ms = calibration_ms / calibration_iterations;

  int warmup_iterations = target_iterations(warmup_ms, one_ms);
  float warmup_elapsed_ms = run_until_target(warmup_ms, &warmup_iterations);

  double measurement_estimate_ms =
      warmup_iterations > 0 && warmup_elapsed_ms > 0.0f
          ? static_cast<double>(warmup_elapsed_ms) / warmup_iterations
          : one_ms;
  int iterations = target_iterations(min_ms, measurement_estimate_ms);
  float elapsed_ms = run_until_target(min_ms, &iterations);

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  uint64_t reports = 0;
  if (mode == Mode::JakubSampledLoom && allocation.report_count) {
    HIP_CHECK(hipMemcpy(&reports, allocation.report_count, sizeof(reports),
                        hipMemcpyDeviceToHost));
    if (reports != 0) {
      std::printf("%-58s benchmark_tsan_reports=%llu\n", name,
                  static_cast<unsigned long long>(reports));
    }
  } else if (is_hip_moi_mode(mode)) {
    for (auto *context : allocation.hip_moi_contexts) {
      int count = context->consume_diagnostics(nullptr);
      if (count < 0) {
        std::fprintf(stderr, "error: failed to read hip-moi diagnostics\n");
        std::exit(EXIT_FAILURE);
      }
      reports += static_cast<uint64_t>(count);
    }
    if (reports != 0) {
      std::printf("%-58s benchmark_tsan_reports=%llu\n", name,
                  static_cast<unsigned long long>(reports));
    }
  }
  free_allocation(allocation);

  double ms = elapsed_ms / iterations;
  double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
               static_cast<double>(K);
  double achieved = ops / (ms * 1.0e9);
  DisplayLatency latency = display_latency(ms);
  std::printf("%-58s %#9.3g %s  %8.2f TFLOP/s  %5.1f%% of 191 TFLOP/s"
              "  total=%8.3f ms  iters=%d  warmup=%8.3f ms"
              "  warmup_iters=%d\n",
              name, latency.value, latency.unit, achieved,
              100.0 * achieved / kFp16TargetTflops, elapsed_ms, iterations,
              warmup_elapsed_ms, warmup_iterations);
  return ms;
}

static void print_device() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::printf("device %d: %s, gcnArch=%s, CUs=%d\n", device, prop.name,
              prop.gcnArchName, prop.multiProcessorCount);
}

int main() {
  int bench_m = get_int_env("BENCH_M", 4096);
  int bench_n = get_int_env("BENCH_N", 4096);
  int bench_k = get_int_env("BENCH_K", 4096);
  double min_ms = get_double_env("MIN_MS", 100.0);
  double warmup_ms = get_double_env("WARMUP_MS", min_ms);
  if (bench_m <= 0 || bench_m % (kMTiles * kWmmaM) != 0 || bench_n <= 0 ||
      bench_n % (kNTiles * kWmmaN) != 0 || bench_k <= 0 ||
      bench_k % kWmmaK != 0) {
    std::fprintf(stderr, "error: BENCH_M/N/K must be positive multiples of the "
                         "production tile sizes\n");
    return EXIT_FAILURE;
  }
  if (min_ms <= 0.0) {
    std::fprintf(stderr, "error: MIN_MS must be positive\n");
    return EXIT_FAILURE;
  }
  if (warmup_ms < 0.0) {
    std::fprintf(stderr, "error: WARMUP_MS must be nonnegative\n");
    return EXIT_FAILURE;
  }

  int m_outer_count = bench_m / (kMTiles * kWmmaM);
  int n_outer_count = bench_n / (kNTiles * kWmmaN);
  int k_outer_count = bench_k / kWmmaK;
  std::size_t a_vectors = static_cast<std::size_t>(m_outer_count) *
                          k_outer_count * kMTiles * kWaveSize;
  std::size_t b_vectors = static_cast<std::size_t>(n_outer_count) *
                          k_outer_count * kNTiles * kWaveSize;
  std::size_t c_scalars = static_cast<std::size_t>(bench_m) * bench_n;
  half_t *a = gpu_alloc<half_t>(a_vectors * 8);
  half_t *b = gpu_alloc<half_t>(b_vectors * 8);
  float *c = gpu_alloc<float>(c_scalars);
  HIP_CHECK(hipMemset(a, 0, a_vectors * sizeof(f16x8)));
  HIP_CHECK(hipMemset(b, 0, b_vectors * sizeof(f16x8)));
  HIP_CHECK(hipMemset(c, 0, c_scalars * sizeof(float)));

  print_device();
  std::printf(
      "bench shape: M=%d N=%d K=%d waves=%d min_ms=%.1f warmup_ms=%.1f\n",
      bench_m, bench_n, bench_k, kWaves, min_ms, warmup_ms);
  SampledKnobs sampled_knobs = sampled_knobs_from_env();
  std::printf("sampled knobs: watchpoints=%u skip=%u probes=%u delay=%u "
              "reports=%s\n",
              sampled_knobs.watchpoints, sampled_knobs.sample_skip,
              sampled_knobs.probe_count, sampled_knobs.delay_iters,
              sampled_knobs.reports ? "on" : "off");
  bool use_static_hip_moi_sampled =
      uses_default_static_hip_moi_sampled_policy(sampled_knobs);
  if (use_static_hip_moi_sampled) {
    std::printf("hip-moi sampled_watchpoint_context policy: static default "
                "publish-only\n");
  }
  std::string prefix = "fp16_wmma_tiled_w" + std::to_string(kWaves) + "_" +
                       std::to_string(kMTiles) + "x" + std::to_string(kNTiles);
  std::string sampled_mode_suffix =
      sampled_knobs.reports ? "_reporting" : "_publish_only";
  run_one<tsan::PassThroughPolicy>((prefix + "_pass_through").c_str(), Mode::PassThrough, a, b, c,
                            bench_m, bench_n, bench_k, min_ms, warmup_ms);
  run_one<tsan::JakubSampledLoomPolicy>(
      (prefix + "_jakub_sampled_loom" + sampled_mode_suffix).c_str(),
      Mode::JakubSampledLoom, a, b, c, bench_m, bench_n, bench_k, min_ms, warmup_ms);
  run_one<tsan::HipMoiSampledWatchpointPolicy>(
      (prefix + "_hip_moi_context_sampled_watchpoint" + sampled_mode_suffix)
          .c_str(),
      Mode::HipMoiSampledWatchpoint, a, b, c, bench_m, bench_n, bench_k, min_ms,
      warmup_ms);
  if (use_static_hip_moi_sampled) {
    run_one<tsan::HipMoiSampledWatchpointStaticPublishOnlyPolicy>(
        (prefix + "_hip_moi_sampled_watchpoint_context" + sampled_mode_suffix)
            .c_str(),
        Mode::HipMoiSampledWatchpoint, a, b, c, bench_m, bench_n, bench_k,
        min_ms, warmup_ms);
  }

  HIP_CHECK(hipFree(a));
  HIP_CHECK(hipFree(b));
  HIP_CHECK(hipFree(c));
  return 0;
}
