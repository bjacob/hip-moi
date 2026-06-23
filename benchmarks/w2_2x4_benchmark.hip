// Copyright 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tiny RDNA4 WMMA instrumentation benchmark vendored into hip-moi from the
// focused sanitizer-strategy/rdna4_matmul extraction, itself derived from
// sanitizer-strategy/rdna4_matmul/rdna4_matmul.hip.
//
// This file keeps one benchmark subset from the large harness by default:
//   fp16 WMMA, 2 waves per workgroup, 2x4 WMMA tiles, M=32 N=64 K=16.
// The build script can also compile nearby 4-wave and 8-wave shapes by passing
// preprocessor definitions for the shape constants below.
//
// The extraction deliberately preserves the same kernel shape, packed WMMA data
// layout, manual LDS helper sites, sampled-Loom policy, and hip-moi
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
#ifndef HIP_MOI_BENCH_WAVES
#define HIP_MOI_BENCH_WAVES 2
#endif
#ifndef HIP_MOI_BENCH_MTILES
#define HIP_MOI_BENCH_MTILES 2
#endif
#ifndef HIP_MOI_BENCH_NTILES
#define HIP_MOI_BENCH_NTILES 4
#endif
#ifndef HIP_MOI_BENCH_KGROUP
#define HIP_MOI_BENCH_KGROUP 1
#endif
#ifndef HIP_MOI_BENCH_WAVE_TILE_M
#define HIP_MOI_BENCH_WAVE_TILE_M HIP_MOI_BENCH_MTILES
#endif
#ifndef HIP_MOI_BENCH_WAVE_TILE_N
#define HIP_MOI_BENCH_WAVE_TILE_N (HIP_MOI_BENCH_NTILES / HIP_MOI_BENCH_WAVES)
#endif
#ifndef HIP_MOI_BENCH_STORAGE_BYTES
#define HIP_MOI_BENCH_STORAGE_BYTES (128 * 1024)
#endif
static constexpr int kWaves = HIP_MOI_BENCH_WAVES;
static constexpr int kThreads = kWaves * kWaveSize;
static constexpr int kMTiles = HIP_MOI_BENCH_MTILES;
static constexpr int kNTiles = HIP_MOI_BENCH_NTILES;
static constexpr int kKGroup = HIP_MOI_BENCH_KGROUP;
static constexpr int kWaveTileM = HIP_MOI_BENCH_WAVE_TILE_M;
static constexpr int kWaveTileN = HIP_MOI_BENCH_WAVE_TILE_N;
static constexpr int kDefaultM = kMTiles * kWmmaM;
static constexpr int kDefaultN = kNTiles * kWmmaN;
static constexpr int kDefaultK = kWmmaK;
static constexpr double kFp16TargetTflops = 191.0;
static_assert(kWaves > 0);
static_assert(kMTiles > 0 && kNTiles > 0);
static_assert(kWaveTileM > 0 && kWaveTileN > 0);
static_assert(kMTiles % kWaveTileM == 0);
static_assert(kNTiles % kWaveTileN == 0);
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
  Noop,
  SampledLoom,
  HipMoiExactShadow,
  HipMoiSampledWatchpoint,
};

struct NoopPolicy {
  static constexpr bool enabled = false;
  static constexpr PolicyKind kind = PolicyKind::Noop;
};

struct SampledLoomPolicy {
  static constexpr bool enabled = true;
  static constexpr PolicyKind kind = PolicyKind::SampledLoom;
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
  uint64_t *base = workgroup_shadow_base<SampledLoomPolicy>(runtime);
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
__device__ __forceinline__ void observe_sampled_loom_access(
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
  } else if constexpr (Policy::kind == PolicyKind::SampledLoom) {
    uint32_t flat = flat_workitem_id();
    uint32_t wave = flat / kWaveSize;
    uint32_t lane = flat & (kWaveSize - 1);
    uint32_t seed = wave_selection_seed(runtime, wave, site_id);
    observe_sampled_loom_access<AccessSize>(runtime, byte_offset, access_kind,
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
  } else if constexpr (Policy::kind == PolicyKind::SampledLoom) {
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
__global__ __launch_bounds__(kThreads) void w2_2x4_kernel(
    const half_t *__restrict__ A_scalar, const half_t *__restrict__ B_scalar,
    float *__restrict__ C_scalar, tsan::Runtime tsan_runtime) {
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

  f32x8 acc[kTilesPerWave][kNTilesPerWave];
#pragma unroll
  for (int t = 0; t < kTilesPerWave; ++t) {
#pragma unroll
    for (int n = 0; n < kNTilesPerWave; ++n) {
      acc[t][n] = zero_f32x8();
    }
  }

  auto a_byte_offset = [](int tile, int lane) {
    return static_cast<uint32_t>((tile * kWaveSize + lane) * sizeof(f16x8));
  };
  auto b_byte_offset = [](int tile, int lane) {
    return static_cast<uint32_t>(((kMTiles + tile) * kWaveSize + lane) *
                                 sizeof(f16x8));
  };

  struct StageRegs {
    f16x8 a[kAThreadLoads];
    f16x8 b[kBThreadLoads];
  };
  struct ComputeRegs {
    f16x8 a[kTilesPerWave];
    f16x8 b[kNTilesPerWave];
  };

  StageRegs stage_regs;
#pragma unroll
  for (int copy = 0; copy < kAThreadLoads; ++copy) {
    int idx = tid + copy * kThreadLoadStride;
    if (idx < kMTiles * kWaveSize) {
      stage_regs.a[copy] = A[idx];
    }
  }
#pragma unroll
  for (int copy = 0; copy < kBThreadLoads; ++copy) {
    int idx = tid + copy * kThreadLoadStride;
    if (idx < kNTiles * kWaveSize) {
      stage_regs.b[copy] = B[idx];
    }
  }

#pragma unroll
  for (int copy = 0; copy < kAThreadLoads; ++copy) {
    int idx = tid + copy * kThreadLoadStride;
    if (idx < kMTiles * kWaveSize) {
      int tile = idx / kWaveSize;
      int local_lane = idx & (kWaveSize - 1);
      tsan::ds_write<TSanPolicy>(&a_s[0][tile][local_lane], stage_regs.a[copy],
                                 tsan_runtime, a_byte_offset(tile, local_lane),
                                 1, hip_moi_context_arg);
    }
  }
#pragma unroll
  for (int copy = 0; copy < kBThreadLoads; ++copy) {
    int idx = tid + copy * kThreadLoadStride;
    if (idx < kNTiles * kWaveSize) {
      int tile = idx / kWaveSize;
      int local_lane = idx & (kWaveSize - 1);
      tsan::ds_write<TSanPolicy>(&b_s[0][tile][local_lane], stage_regs.b[copy],
                                 tsan_runtime, b_byte_offset(tile, local_lane),
                                 2, hip_moi_context_arg);
    }
  }

  tsan::syncthreads<TSanPolicy>(tsan_runtime, hip_moi_context_arg);

  ComputeRegs compute_regs;
#pragma unroll
  for (int n = 0; n < kNTilesPerWave; ++n) {
    int n_tile = wave_n * kWaveTileN + n;
    compute_regs.b[n] = tsan::ds_read<TSanPolicy>(
        &b_s[0][n_tile][lane], tsan_runtime, b_byte_offset(n_tile, lane), 3,
        hip_moi_context_arg);
  }
#pragma unroll
  for (int t = 0; t < kTilesPerWave; ++t) {
    int m_tile = wave_m * kWaveTileM + t;
    compute_regs.a[t] = tsan::ds_read<TSanPolicy>(
        &a_s[0][m_tile][lane], tsan_runtime, a_byte_offset(m_tile, lane), 4,
        hip_moi_context_arg);
  }

#pragma unroll
  for (int t = 0; t < kTilesPerWave; ++t) {
#pragma unroll
    for (int n = 0; n < kNTilesPerWave; ++n) {
      acc[t][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
          compute_regs.a[t], compute_regs.b[n], acc[t][n]);
    }
  }

  constexpr int kNOuterCount = 1;
  int c_tile_base =
      (blockIdx.y * kNOuterCount + blockIdx.x) * kMTiles * kNTiles * kWaveSize;
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
  Noop,
  SampledLoom,
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

static uint64_t workgroup_count() { return 1; }

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

static Allocation make_allocation(Mode mode) {
  Allocation allocation;
  SampledKnobs sampled_knobs = sampled_knobs_from_env();
  allocation.sampled_watchpoints = sampled_knobs.watchpoints;
  allocation.sample_skip = sampled_knobs.sample_skip;
  allocation.delay_iters = sampled_knobs.delay_iters;
  allocation.probe_count = sampled_knobs.probe_count;
  allocation.sampled_reports = sampled_knobs.reports;

  if (mode == Mode::SampledLoom) {
    allocation.workgroup_stride_entries =
        1 + static_cast<uint32_t>(allocation.sampled_watchpoints);
    HIP_CHECK(
        hipMalloc(&allocation.shadow, workgroup_count() *
                                          allocation.workgroup_stride_entries *
                                          sizeof(uint64_t)));
    HIP_CHECK(
        hipMemset(allocation.shadow, 0,
                  workgroup_count() * allocation.workgroup_stride_entries *
                      sizeof(uint64_t)));
    if (allocation.sampled_reports) {
      HIP_CHECK(
          hipMalloc(&allocation.report_count,
                    workgroup_count() * sizeof(*allocation.report_count)));
      HIP_CHECK(
          hipMemset(allocation.report_count, 0,
                    workgroup_count() * sizeof(*allocation.report_count)));
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
    storage_refs.reserve(workgroup_count());
    for (uint64_t i = 0; i < workgroup_count(); ++i) {
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
    uint64_t count = workgroup_count();
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_hip_moi_context_storages),
                                &device_storage_refs,
                                sizeof(device_storage_refs)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_hip_moi_context_storage_count),
                                &count, sizeof(count)));
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
                          hipStream_t stream) {
  dim3 block(kThreads);
  dim3 grid(1, 1);
  w2_2x4_kernel<Policy><<<grid, block, 0, stream>>>(a, b, c, runtime);
}

template <typename Policy>
static double run_one(const char *name, Mode mode, half_t *a, half_t *b,
                      float *c, double min_ms, double warmup_ms) {
  Allocation allocation = make_allocation(mode);
  hipStream_t stream = nullptr;
  for (int i = 0; i < 5; ++i) {
    launch_kernel<Policy>(a, b, c, next_runtime(allocation), stream);
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
      launch_kernel<Policy>(a, b, c, next_runtime(allocation), stream);
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
  if (mode == Mode::SampledLoom && allocation.report_count) {
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
  double ops = 2.0 * static_cast<double>(kDefaultM) *
               static_cast<double>(kDefaultN) * static_cast<double>(kDefaultK);
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
  double min_ms = get_double_env("MIN_MS", 100.0);
  double warmup_ms = get_double_env("WARMUP_MS", min_ms);
  if (min_ms <= 0.0) {
    std::fprintf(stderr, "error: MIN_MS must be positive\n");
    return EXIT_FAILURE;
  }
  if (warmup_ms < 0.0) {
    std::fprintf(stderr, "error: WARMUP_MS must be nonnegative\n");
    return EXIT_FAILURE;
  }

  constexpr std::size_t a_vectors = kMTiles * kWaveSize;
  constexpr std::size_t b_vectors = kNTiles * kWaveSize;
  constexpr std::size_t c_scalars =
      static_cast<std::size_t>(kDefaultM) * kDefaultN;
  half_t *a = gpu_alloc<half_t>(a_vectors * 8);
  half_t *b = gpu_alloc<half_t>(b_vectors * 8);
  float *c = gpu_alloc<float>(c_scalars);
  HIP_CHECK(hipMemset(a, 0, a_vectors * sizeof(f16x8)));
  HIP_CHECK(hipMemset(b, 0, b_vectors * sizeof(f16x8)));
  HIP_CHECK(hipMemset(c, 0, c_scalars * sizeof(float)));

  print_device();
  std::printf(
      "bench shape: M=%d N=%d K=%d waves=%d min_ms=%.1f warmup_ms=%.1f\n",
      kDefaultM, kDefaultN, kDefaultK, kWaves, min_ms, warmup_ms);
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
  run_one<tsan::NoopPolicy>((prefix + "_noop").c_str(), Mode::Noop, a, b, c,
                            min_ms, warmup_ms);
  run_one<tsan::SampledLoomPolicy>(
      (prefix + "_sampled_loom" + sampled_mode_suffix).c_str(),
      Mode::SampledLoom, a, b, c, min_ms, warmup_ms);
  run_one<tsan::HipMoiExactShadowPolicy>(
      (prefix + "_hip_moi_exact_shadow").c_str(), Mode::HipMoiExactShadow, a, b,
      c, min_ms, warmup_ms);
  run_one<tsan::HipMoiSampledWatchpointPolicy>(
      (prefix + "_hip_moi_context_sampled_watchpoint" + sampled_mode_suffix)
          .c_str(),
      Mode::HipMoiSampledWatchpoint, a, b, c, min_ms, warmup_ms);
  if (use_static_hip_moi_sampled) {
    run_one<tsan::HipMoiSampledWatchpointStaticPublishOnlyPolicy>(
        (prefix + "_hip_moi_sampled_watchpoint_context" + sampled_mode_suffix)
            .c_str(),
        Mode::HipMoiSampledWatchpoint, a, b, c, min_ms, warmup_ms);
  }

  HIP_CHECK(hipFree(a));
  HIP_CHECK(hipFree(b));
  HIP_CHECK(hipFree(c));
  return 0;
}
