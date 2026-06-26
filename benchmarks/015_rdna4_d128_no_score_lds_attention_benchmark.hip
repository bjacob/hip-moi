// Copyright 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RDNA4 D128/V128 WMMA no-score/weight-LDS attention benchmark for hip-moi.
//
// This grows tests/instrumented/015_rdna4_d128_no_score_lds_attention_test.hip
// into a benchmark-sized workload. Each workgroup computes a 32-query block
// with two subgroups. K/V fragments are staged through LDS and instrumented,
// while the QK-to-PV handoff stays in registers with ds_bpermute instead of
// materializing dense score or weight scratch in LDS.

#include <hip/hip_runtime.h>

#include "hip_moi/hip_moi.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using half_t = _Float16;
using f16x8  = half_t __attribute__((ext_vector_type(8)));
using f32x8  = float __attribute__((ext_vector_type(8)));

static constexpr int kWaveSize           = 32;
static constexpr int kSubgroupCount      = 2;
static constexpr int kThreads            = kWaveSize * kSubgroupCount;
static constexpr int kThreadsPerSubgroup = kWaveSize;
static constexpr int kQueryTile          = 16;
static constexpr int kQueriesPerBlock    = kQueryTile * kSubgroupCount;
static constexpr int kKeyTile            = 16;
static constexpr int kFragmentDim        = 16;
static constexpr int kFragmentElements   = 8;

#ifndef HIP_MOI_BENCH_HEAD_DIM
#define HIP_MOI_BENCH_HEAD_DIM 128
#endif

#ifndef HIP_MOI_BENCH_VALUE_DIM
#define HIP_MOI_BENCH_VALUE_DIM HIP_MOI_BENCH_HEAD_DIM
#endif

static constexpr int kHeadDim        = HIP_MOI_BENCH_HEAD_DIM;
static constexpr int kValueDim       = HIP_MOI_BENCH_VALUE_DIM;
static constexpr int kHeadFragments  = kHeadDim / kFragmentDim;
static constexpr int kValueFragments = kValueDim / kFragmentDim;

#if HIP_MOI_BENCH_HEAD_DIM == 16
static constexpr float kQkScale = 0.125f;
#elif HIP_MOI_BENCH_HEAD_DIM == 128
static constexpr float kQkScale = 0.08838834764831845f;
#else
#error "015_rdna4_d128_no_score_lds_attention_benchmark.hip currently supports head_dim 16 or 128"
#endif

static_assert(kHeadDim == kHeadFragments * kFragmentDim);
static_assert(kValueDim == kValueFragments * kFragmentDim);

static constexpr uint32_t kKSharedByteOffset = 0;
static constexpr uint32_t kVSharedByteOffset = kWaveSize * static_cast<uint32_t>(sizeof(f16x8));

static constexpr hip_moi::site_id kKStoreSite(0xf01);
static constexpr hip_moi::site_id kVStoreSite(0xf02);
static constexpr hip_moi::site_id kKLoadSite(0xf03);
static constexpr hip_moi::site_id kVLoadSite(0xf04);

#ifndef HIP_MOI_BENCH_STORAGE_BYTES
#define HIP_MOI_BENCH_STORAGE_BYTES (128 * 1024)
#endif

using StaticFastPolicy = hip_moi::sampled_watchpoint_policy<
    /*SampleSkip=*/32,
    /*ProbeCount=*/1,
    /*DelayIters=*/32,
    /*ReportConflicts=*/false,
    /*StaticWatchpointCapacity=*/1>;

static constexpr uint32_t kTsanShadowAccessRead       = 1;
static constexpr uint32_t kTsanShadowAccessWrite      = 2;
static constexpr uint32_t kTsanSampledGranuleShift    = 2;
static constexpr uint32_t kTsanSampledGranuleBytes    = 1u << kTsanSampledGranuleShift;
static constexpr uint32_t kTsanSampledValidShift      = 0;
static constexpr uint32_t kTsanSampledConsumedShift   = 1;
static constexpr uint32_t kTsanSampledKindShift       = 2;
static constexpr uint32_t kTsanSampledKindBits        = 3;
static constexpr uint32_t kTsanSampledWorkitemShift   = 5;
static constexpr uint32_t kTsanSampledWorkitemBits    = 10;
static constexpr uint32_t kTsanSampledEpochShift      = 15;
static constexpr uint32_t kTsanSampledEpochBits       = 10;
static constexpr uint32_t kTsanSampledGenerationShift = 25;
static constexpr uint32_t kTsanSampledGenerationBits  = 20;
static constexpr uint32_t kTsanSampledAddrShift       = 45;
static constexpr uint32_t kTsanSampledAddrBits        = 14;
static constexpr uint32_t kTsanSampledCountShift      = 59;
static constexpr uint32_t kTsanSampledCountBits       = 5;
static constexpr uint64_t kTsanSampledValid           = 1ull << kTsanSampledValidShift;
static constexpr uint64_t kTsanSampledConsumed        = 1ull << kTsanSampledConsumedShift;
static constexpr uint64_t kTsanSampledKindMask        = ((1ull << kTsanSampledKindBits) - 1)
                                                 << kTsanSampledKindShift;
static constexpr uint64_t kTsanSampledWorkitemMask = ((1ull << kTsanSampledWorkitemBits) - 1)
                                                     << kTsanSampledWorkitemShift;
static constexpr uint64_t kTsanSampledEpochMask = ((1ull << kTsanSampledEpochBits) - 1)
                                                  << kTsanSampledEpochShift;
static constexpr uint64_t kTsanSampledGenerationMask = ((1ull << kTsanSampledGenerationBits) - 1)
                                                       << kTsanSampledGenerationShift;
static constexpr uint64_t kTsanSampledEpochGenerationMask
    = kTsanSampledEpochMask | kTsanSampledGenerationMask;
static constexpr uint64_t kTsanSampledAddrMask = ((1ull << kTsanSampledAddrBits) - 1)
                                                 << kTsanSampledAddrShift;
static constexpr uint64_t kTsanSampledCountMask = ((1ull << kTsanSampledCountBits) - 1)
                                                  << kTsanSampledCountShift;
static constexpr uint32_t kTsanSampledMaxAddr  = (1u << kTsanSampledAddrBits) - 1;
static constexpr uint32_t kTsanSampledMaxCount = 1u << kTsanSampledCountBits;

static void hip_check_impl(hipError_t err, const char* expr, const char* file, int line)
{
    if(err != hipSuccess)
    {
        std::fprintf(
            stderr, "HIP error %s from `%s` at %s:%d\n", hipGetErrorString(err), expr, file, line);
        std::exit(EXIT_FAILURE);
    }
}

#define HIP_CHECK(expr) hip_check_impl((expr), #expr, __FILE__, __LINE__)

static int get_int_env(const char* name, int default_value)
{
    const char* value = std::getenv(name);
    return value ? std::stoi(value) : default_value;
}

static int get_int_env_alias(const char* name, const char* legacy_name, int default_value)
{
    const char* value = std::getenv(name);
    if(value)
    {
        return std::stoi(value);
    }
    if(legacy_name)
    {
        value = std::getenv(legacy_name);
        if(value)
        {
            return std::stoi(value);
        }
    }
    return default_value;
}

static double get_double_env(const char* name, double default_value)
{
    const char* value = std::getenv(name);
    return value ? std::strtod(value, nullptr) : default_value;
}

struct DisplayLatency
{
    double      value;
    const char* unit;
};

static DisplayLatency display_latency(double ms)
{
    if(ms < 1.0)
    {
        return DisplayLatency{/*value=*/ms * 1000.0, /*unit=*/"µs"};
    }
    return DisplayLatency{/*value=*/ms, /*unit=*/"ms"};
}

__host__ __device__ static int rdna4_wmma_f16_k(int lane, int idx)
{
    int reg  = idx >> 1;
    int half = idx & 1;
    return half | ((reg & 1) << 1) | ((lane >> 4) << 2) | ((reg >> 1) << 3);
}

__host__ __device__ static int rdna4_wmma_acc_m(int lane, int elem)
{
    return elem + 8 * (lane >> 4);
}

__host__ __device__ static uint32_t k_fragment_offset(int lane)
{
    return kKSharedByteOffset + static_cast<uint32_t>(lane * sizeof(f16x8));
}

__host__ __device__ static uint32_t v_fragment_offset(int lane)
{
    return kVSharedByteOffset + static_cast<uint32_t>(lane * sizeof(f16x8));
}

__device__ __forceinline__ static f32x8 zero_f32x8()
{
    f32x8 value;
#pragma unroll
    for(int i = 0; i < kFragmentElements; ++i)
    {
        value[i] = 0.0f;
    }
    return value;
}

__device__ __forceinline__ static float read_float_lane(float value, int source_lane)
{
    uint32_t bits      = __builtin_bit_cast(uint32_t, value);
    uint32_t lane_bits = __builtin_amdgcn_ds_bpermute(source_lane * sizeof(uint32_t), bits);
    return __builtin_bit_cast(float, lane_bits);
}

__device__ __forceinline__ static float
    read_accumulator_element_from_lane(f32x8 acc, int elem, int source_lane)
{
    float value0 = read_float_lane(acc[0], source_lane);
    float value1 = read_float_lane(acc[1], source_lane);
    float value2 = read_float_lane(acc[2], source_lane);
    float value3 = read_float_lane(acc[3], source_lane);
    float value4 = read_float_lane(acc[4], source_lane);
    float value5 = read_float_lane(acc[5], source_lane);
    float value6 = read_float_lane(acc[6], source_lane);
    float value7 = read_float_lane(acc[7], source_lane);
    switch(elem)
    {
    case 0:
        return value0;
    case 1:
        return value1;
    case 2:
        return value2;
    case 3:
        return value3;
    case 4:
        return value4;
    case 5:
        return value5;
    case 6:
        return value6;
    default:
        return value7;
    }
}

__device__ __forceinline__ static f16x8 qk_accumulator_to_pv_fragment(f32x8 qk_acc, int lane)
{
    f16x8 p_fragment;
    int   query_row       = lane & 15;
    int   source_acc_elem = query_row & 7;
    int   source_lane_row = query_row >= 8 ? 16 : 0;
#pragma unroll
    for(int elem = 0; elem < kFragmentElements; ++elem)
    {
        int   local_key   = rdna4_wmma_f16_k(lane, elem);
        int   source_lane = source_lane_row + local_key;
        float qk_value   = read_accumulator_element_from_lane(qk_acc, source_acc_elem, source_lane);
        p_fragment[elem] = static_cast<half_t>(qk_value * kQkScale);
    }
    return p_fragment;
}

struct Runtime
{
    uint64_t*                      shadow;
    unsigned long long*            report_count;
    uint32_t                       workgroup_stride_entries;
    hip_moi::context::storage_ref* storage_refs;
    uint64_t                       storage_count;
    uint32_t                       generation;
    uint32_t                       sample_skip;
    uint32_t                       delay_iters;
    uint32_t                       probe_count;
};

__device__ __forceinline__ static uint32_t flat_workitem_id()
{
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

__device__ __forceinline__ static uint64_t flat_workgroup_id()
{
    return static_cast<uint64_t>(blockIdx.x)
           + static_cast<uint64_t>(gridDim.x)
                 * (static_cast<uint64_t>(blockIdx.y)
                    + static_cast<uint64_t>(gridDim.y) * static_cast<uint64_t>(blockIdx.z));
}

__device__ __forceinline__ static hip_moi::context::storage_ref workgroup_storage(Runtime runtime)
{
    hip_moi::context::storage_ref storage{};
    uint64_t                      workgroup_id = flat_workgroup_id();
    if(runtime.storage_refs && workgroup_id < runtime.storage_count)
    {
        storage = runtime.storage_refs[workgroup_id];
    }
    storage.generation = runtime.generation;
    return storage;
}

__device__ __forceinline__ static uint64_t* workgroup_shadow_base(Runtime runtime)
{
    return runtime.shadow + flat_workgroup_id() * runtime.workgroup_stride_entries;
}

__device__ __forceinline__ static uint32_t mix32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ static bool conflicting_access(uint32_t current_kind,
                                                          uint32_t prior_kind)
{
    return current_kind == kTsanShadowAccessRead ? prior_kind != kTsanShadowAccessRead : true;
}

__device__ __forceinline__ static void delay(uint32_t iterations)
{
    for(uint32_t i = 0; i < iterations; ++i)
    {
        asm volatile("s_nop 0" ::: "memory");
    }
}

__device__ __forceinline__ static uint32_t lds_cell_start(uint32_t byte_offset)
{
    return byte_offset >> kTsanSampledGranuleShift;
}

__device__ __forceinline__ static uint32_t sampled_watchpoint_count(Runtime runtime)
{
    return runtime.workgroup_stride_entries == 0 ? 0 : runtime.workgroup_stride_entries - 1;
}

__device__ __forceinline__ static uint32_t sampled_watchpoint_mask(Runtime runtime)
{
    return sampled_watchpoint_count(runtime) - 1;
}

__device__ __forceinline__ static uint32_t
    wave_selection_seed(Runtime runtime, uint32_t wave, uint32_t site_id)
{
    uint32_t seed = runtime.generation * 0x9e3779b9U;
    seed ^= static_cast<uint32_t>(flat_workgroup_id() + 1) * 0x85ebca6bU;
    seed ^= (wave + 1) * 0xc2b2ae35U;
    seed ^= site_id * 0x27d4eb2dU;
    return mix32(seed);
}

__device__ __forceinline__ static uint64_t pack_sampled_watchpoint(
    Runtime runtime, uint32_t start, uint32_t count, uint32_t access_kind, uint64_t epoch)
{
    return kTsanSampledValid
           | ((static_cast<uint64_t>(access_kind) & ((1ull << kTsanSampledKindBits) - 1))
              << kTsanSampledKindShift)
           | ((static_cast<uint64_t>(flat_workitem_id()) & ((1ull << kTsanSampledWorkitemBits) - 1))
              << kTsanSampledWorkitemShift)
           | ((epoch & ((1ull << kTsanSampledEpochBits) - 1)) << kTsanSampledEpochShift)
           | ((static_cast<uint64_t>(runtime.generation)
               & ((1ull << kTsanSampledGenerationBits) - 1))
              << kTsanSampledGenerationShift)
           | ((static_cast<uint64_t>(start) & ((1ull << kTsanSampledAddrBits) - 1))
              << kTsanSampledAddrShift)
           | ((static_cast<uint64_t>(count - 1) & ((1ull << kTsanSampledCountBits) - 1))
              << kTsanSampledCountShift);
}

__device__ __forceinline__ static uint32_t sampled_kind(uint64_t entry)
{
    return static_cast<uint32_t>((entry & kTsanSampledKindMask) >> kTsanSampledKindShift);
}

__device__ __forceinline__ static uint32_t sampled_workitem(uint64_t entry)
{
    return static_cast<uint32_t>((entry & kTsanSampledWorkitemMask) >> kTsanSampledWorkitemShift);
}

__device__ __forceinline__ static uint32_t sampled_start(uint64_t entry)
{
    return static_cast<uint32_t>((entry & kTsanSampledAddrMask) >> kTsanSampledAddrShift);
}

__device__ __forceinline__ static uint32_t sampled_count(uint64_t entry)
{
    return static_cast<uint32_t>((entry & kTsanSampledCountMask) >> kTsanSampledCountShift) + 1;
}

__device__ __forceinline__ static bool sampled_range_overlaps(uint32_t lhs_start,
                                                              uint32_t lhs_count,
                                                              uint32_t rhs_start,
                                                              uint32_t rhs_count)
{
    return lhs_start < rhs_start + rhs_count && rhs_start < lhs_start + lhs_count;
}

__device__ __forceinline__ static bool sampled_conflicts(
    uint64_t prior, uint64_t current, uint32_t start, uint32_t count, uint32_t access_kind)
{
    if((prior & kTsanSampledValid) == 0 || (prior & kTsanSampledConsumed) != 0)
    {
        return false;
    }
    if(((prior ^ current) & kTsanSampledEpochGenerationMask) != 0)
    {
        return false;
    }
    if(sampled_workitem(prior) == flat_workitem_id())
    {
        return false;
    }
    if(!conflicting_access(access_kind, sampled_kind(prior)))
    {
        return false;
    }
    return sampled_range_overlaps(start, count, sampled_start(prior), sampled_count(prior));
}

__device__ __attribute__((noinline)) static void sampled_scan_reports(Runtime   runtime,
                                                                      uint64_t* base,
                                                                      uint64_t  current,
                                                                      uint32_t  start,
                                                                      uint32_t  count,
                                                                      uint32_t  access_kind,
                                                                      uint32_t  site_id,
                                                                      uint32_t  publish_index,
                                                                      uint32_t  selection_seed)
{
    uint32_t watchpoints = sampled_watchpoint_count(runtime);
    uint32_t probe_count = runtime.probe_count == 0 ? watchpoints : runtime.probe_count;
    if(probe_count > watchpoints)
    {
        probe_count = watchpoints;
    }
    uint32_t probe_step = (mix32(selection_seed ^ start ^ (site_id * 0x94d049bbU)) | 1u)
                          & sampled_watchpoint_mask(runtime);
#pragma unroll 1
    for(uint32_t probe = 0; probe < probe_count; ++probe)
    {
        uint32_t index
            = 1 + ((publish_index - 1 + probe * probe_step) & sampled_watchpoint_mask(runtime));
        uint64_t* entry64 = base + index;
        uint64_t  prior   = *reinterpret_cast<volatile uint64_t*>(entry64);
        if(sampled_conflicts(prior, current, start, count, access_kind))
        {
            uint64_t consumed = prior | kTsanSampledConsumed;
            auto*    entry    = reinterpret_cast<unsigned long long*>(entry64);
            if(atomicCAS(entry,
                         static_cast<unsigned long long>(prior),
                         static_cast<unsigned long long>(consumed))
                   == prior
               && runtime.report_count)
            {
                atomicAdd(runtime.report_count, 1ull);
            }
        }
    }
}

__device__ __forceinline__ static bool
    sampled_should_publish(Runtime runtime, uint32_t seed, uint32_t lane)
{
    uint32_t skip = runtime.sample_skip == 0 ? 1 : runtime.sample_skip;
    if(skip > 1)
    {
        bool     power_of_two = (skip & (skip - 1)) == 0;
        uint32_t remainder    = power_of_two ? (seed & (skip - 1)) : (seed % skip);
        if(remainder != 0)
        {
            return false;
        }
    }
    return lane == ((seed >> 16) & (kWaveSize - 1));
}

__device__ __forceinline__ static uint32_t sampled_publish_index(
    Runtime runtime, uint32_t site_id, uint64_t epoch, uint32_t wave, uint32_t start)
{
    uint32_t seed = site_id * 0x9e3779b9U;
    seed ^= static_cast<uint32_t>(epoch) * 0x85ebca6bU;
    seed ^= runtime.generation * 0xc2b2ae35U;
    seed ^= wave * 0x27d4eb2dU;
    seed ^= start * 0x165667b1U;
    return 1 + (mix32(seed) & sampled_watchpoint_mask(runtime));
}

__device__ __forceinline__ static void observe_sampled_range(Runtime  runtime,
                                                             uint32_t start,
                                                             uint32_t count,
                                                             uint32_t access_kind,
                                                             uint32_t site_id,
                                                             uint32_t wave,
                                                             uint32_t lane,
                                                             uint32_t selection_seed)
{
    if(!runtime.shadow || runtime.workgroup_stride_entries <= 1 || count == 0
       || start > kTsanSampledMaxAddr)
    {
        return;
    }
    uint32_t max_count_for_addr = kTsanSampledMaxAddr - start + 1;
    if(count > max_count_for_addr)
    {
        count = max_count_for_addr;
    }

    if(!sampled_should_publish(runtime, selection_seed, lane))
    {
        return;
    }
    uint64_t* base          = workgroup_shadow_base(runtime);
    uint64_t  epoch         = base[0] & ((1ull << kTsanSampledEpochBits) - 1);
    uint64_t  current       = pack_sampled_watchpoint(runtime, start, count, access_kind, epoch);
    uint32_t  publish_index = sampled_publish_index(runtime, site_id, epoch, wave, start);

    if(runtime.report_count)
    {
        sampled_scan_reports(runtime,
                             base,
                             current,
                             start,
                             count,
                             access_kind,
                             site_id,
                             publish_index,
                             selection_seed);
    }

    auto* entry = reinterpret_cast<unsigned long long*>(base + publish_index);
    atomicExch(entry, static_cast<unsigned long long>(current));
    delay(runtime.delay_iters);
}

template <uint32_t AccessSize>
__device__ __forceinline__ static void observe_jakub_sampled_loom_access(Runtime  runtime,
                                                                   uint32_t byte_offset,
                                                                   uint32_t access_kind,
                                                                   uint32_t site_id,
                                                                   uint32_t wave,
                                                                   uint32_t lane,
                                                                   uint32_t selection_seed)
{
    uint32_t start = lds_cell_start(byte_offset);
    uint32_t end
        = (byte_offset + AccessSize + kTsanSampledGranuleBytes - 1) >> kTsanSampledGranuleShift;
    while(start < end)
    {
        uint32_t chunk = end - start < kTsanSampledMaxCount ? end - start : kTsanSampledMaxCount;
        observe_sampled_range(
            runtime, start, chunk, access_kind, site_id, wave, lane, selection_seed);
        start += chunk;
    }
}

struct PassThroughPolicy
{
    struct context_type
    {
    };

    __device__ static context_type make_context(Runtime)
    {
        return {};
    }

    __device__ static void init_workgroup(context_type&) {}

    __device__ static void syncthreads(context_type&)
    {
        __syncthreads();
    }

    template <typename T>
    __device__ static void store(context_type&, T* ptr, T value, uint32_t, hip_moi::site_id)
    {
        *ptr = value;
    }

    template <typename T>
    __device__ static T load(context_type&, const T* ptr, uint32_t, hip_moi::site_id)
    {
        return *ptr;
    }
};

// Benchmark-local HIP reproduction of Jakub's sampled Loom-like policy.
// It is not upstream HRX/Loom; use it only as the local comparison row.
struct JakubSampledLoomPolicy
{
    struct context_type
    {
        Runtime runtime;
    };

    __device__ static context_type make_context(Runtime runtime)
    {
        return context_type{runtime};
    }

    __device__ static void init_workgroup(context_type&) {}

    __device__ static void syncthreads(context_type& ctx)
    {
        __syncthreads();
        if(flat_workitem_id() == 0 && ctx.runtime.shadow && ctx.runtime.workgroup_stride_entries)
        {
            atomicAdd(reinterpret_cast<unsigned long long*>(workgroup_shadow_base(ctx.runtime)),
                      1ull);
        }
        __syncthreads();
    }

    template <typename T>
    __device__ static void
        store(context_type& ctx, T* ptr, T value, uint32_t offset, hip_moi::site_id site)
    {
        uint32_t flat      = flat_workitem_id();
        uint32_t wave      = flat / kWaveSize;
        uint32_t lane      = flat & (kWaveSize - 1);
        uint32_t site_bits = static_cast<uint32_t>(site.value() ^ (site.value() >> 32));
        uint32_t seed      = wave_selection_seed(ctx.runtime, wave, site_bits);
        observe_jakub_sampled_loom_access<static_cast<uint32_t>(sizeof(T))>(
            ctx.runtime, offset, kTsanShadowAccessWrite, site_bits, wave, lane, seed);
        *ptr = value;
    }

    template <typename T>
    __device__ static T
        load(context_type& ctx, const T* ptr, uint32_t offset, hip_moi::site_id site)
    {
        uint32_t flat      = flat_workitem_id();
        uint32_t wave      = flat / kWaveSize;
        uint32_t lane      = flat & (kWaveSize - 1);
        uint32_t site_bits = static_cast<uint32_t>(site.value() ^ (site.value() >> 32));
        uint32_t seed      = wave_selection_seed(ctx.runtime, wave, site_bits);
        observe_jakub_sampled_loom_access<static_cast<uint32_t>(sizeof(T))>(
            ctx.runtime, offset, kTsanShadowAccessRead, site_bits, wave, lane, seed);
        return *ptr;
    }
};

struct ContextSampledPolicy
{
    using context_type = hip_moi::context;

    __device__ static context_type make_context(Runtime runtime)
    {
        int thread_count = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);
        hip_moi::context::config cfg{
            /*thread_count=*/thread_count,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/(thread_count + kThreadsPerSubgroup - 1) / kThreadsPerSubgroup,
        };
        return hip_moi::context(workgroup_storage(runtime), cfg);
    }

    __device__ static void init_workgroup(context_type& ctx)
    {
        ctx.init_workgroup();
    }

    __device__ static void syncthreads(context_type& ctx)
    {
        ctx.syncthreads();
    }

    template <typename T>
    __device__ static void
        store(context_type& ctx, T* ptr, T value, uint32_t offset, hip_moi::site_id site)
    {
        ctx.template lds_store_at<hip_moi::backend_kind::sampled_watchpoint>(
            ptr, value, offset, site);
    }

    template <typename T>
    __device__ static T
        load(context_type& ctx, const T* ptr, uint32_t offset, hip_moi::site_id site)
    {
        return ctx.template lds_load_at<hip_moi::backend_kind::sampled_watchpoint>(
            ptr, offset, site);
    }
};

struct SampledWatchpointContextPolicy
{
    using context_type = hip_moi::sampled_watchpoint_context;

    __device__ static context_type make_context(Runtime runtime)
    {
        hip_moi::sampled_watchpoint_context::config cfg{
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
        };
        return hip_moi::make_sampled_watchpoint_context(workgroup_storage(runtime), cfg);
    }

    __device__ static void init_workgroup(context_type& ctx)
    {
        ctx.init_workgroup();
    }

    __device__ static void syncthreads(context_type& ctx)
    {
        ctx.syncthreads();
    }

    template <typename T>
    __device__ static void
        store(context_type& ctx, T* ptr, T value, uint32_t offset, hip_moi::site_id site)
    {
        ctx.template lds_store_at<StaticFastPolicy>(ptr, value, offset, site);
    }

    template <typename T>
    __device__ static T
        load(context_type& ctx, const T* ptr, uint32_t offset, hip_moi::site_id site)
    {
        return ctx.template lds_load_at<StaticFastPolicy>(ptr, offset, site);
    }
};

template <typename Policy>
__global__ __launch_bounds__(kThreads) void attention_no_score_lds_kernel(
    const f16x8* __restrict__ q_fragments,
    const f16x8* __restrict__ k_fragments,
    const f16x8* __restrict__ v_fragments,
    float* __restrict__ out,
    int     sequence_length,
    Runtime runtime)
{
    __shared__ f16x8 k_shared[kWaveSize];
    __shared__ f16x8 v_shared[kWaveSize];

    typename Policy::context_type ctx = Policy::make_context(runtime);
    Policy::init_workgroup(ctx);

    int tid            = static_cast<int>(threadIdx.x);
    int subgroup       = tid / kWaveSize;
    int lane           = tid & (kWaveSize - 1);
    int query_block    = static_cast<int>(blockIdx.x);
    int key_tile_count = sequence_length / kKeyTile;

    f32x8 out_acc[kValueFragments];
#pragma unroll
    for(int value_fragment = 0; value_fragment < kValueFragments; ++value_fragment)
    {
        out_acc[value_fragment] = zero_f32x8();
    }

    for(int key_tile = 0; key_tile < key_tile_count; ++key_tile)
    {
        f32x8 qk_acc = zero_f32x8();

#pragma unroll
        for(int head_fragment = 0; head_fragment < kHeadFragments; ++head_fragment)
        {
            if(subgroup == 0)
            {
                std::size_t k_index
                    = (static_cast<std::size_t>(key_tile) * kHeadFragments + head_fragment)
                          * kWaveSize
                      + lane;
                Policy::store(ctx,
                              &k_shared[lane],
                              k_fragments[k_index],
                              k_fragment_offset(lane),
                              kKStoreSite);
            }

            Policy::syncthreads(ctx);

            std::size_t q_index
                = ((static_cast<std::size_t>(query_block) * kSubgroupCount + subgroup)
                       * kHeadFragments
                   + head_fragment)
                      * kWaveSize
                  + lane;
            f16x8 q_fragment = q_fragments[q_index];
            f16x8 k_fragment
                = Policy::load(ctx, &k_shared[lane], k_fragment_offset(lane), kKLoadSite);
            qk_acc
                = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(q_fragment, k_fragment, qk_acc);

            Policy::syncthreads(ctx);
        }

        f16x8 p_fragment = qk_accumulator_to_pv_fragment(qk_acc, lane);

#pragma unroll
        for(int value_fragment = 0; value_fragment < kValueFragments; ++value_fragment)
        {
            if(subgroup == 0)
            {
                std::size_t v_index
                    = (static_cast<std::size_t>(key_tile) * kValueFragments + value_fragment)
                          * kWaveSize
                      + lane;
                Policy::store(ctx,
                              &v_shared[lane],
                              v_fragments[v_index],
                              v_fragment_offset(lane),
                              kVStoreSite);
            }

            Policy::syncthreads(ctx);

            f16x8 v_fragment
                = Policy::load(ctx, &v_shared[lane], v_fragment_offset(lane), kVLoadSite);
            f32x8 pv_acc = zero_f32x8();
            pv_acc
                = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(p_fragment, v_fragment, pv_acc);

#pragma unroll
            for(int elem = 0; elem < kFragmentElements; ++elem)
            {
                out_acc[value_fragment][elem] += pv_acc[elem];
            }

            Policy::syncthreads(ctx);
        }
    }

    int value_column_in_fragment = lane & 15;
#pragma unroll
    for(int value_fragment = 0; value_fragment < kValueFragments; ++value_fragment)
    {
#pragma unroll
        for(int elem = 0; elem < kFragmentElements; ++elem)
        {
            int query_row    = rdna4_wmma_acc_m(lane, elem);
            int global_query = query_block * kQueriesPerBlock + subgroup * kQueryTile + query_row;
            int value_column = value_fragment * kFragmentDim + value_column_in_fragment;
            out[global_query * kValueDim + value_column] = out_acc[value_fragment][elem];
        }
    }
}

template <typename T>
static T* gpu_alloc(std::size_t count)
{
    T* ptr = nullptr;
    HIP_CHECK(hipMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

enum class Mode
{
    PassThrough,
    JakubSampledLoom,
    HipMoiContextSampledWatchpoint,
    HipMoiSampledWatchpointContext,
};

static bool is_hip_moi_mode(Mode mode)
{
    return mode == Mode::HipMoiContextSampledWatchpoint
           || mode == Mode::HipMoiSampledWatchpointContext;
}

struct SampledKnobs
{
    uint32_t watchpoints;
    uint32_t sample_skip;
    uint32_t delay_iters;
    uint32_t probe_count;
    bool     reports;
};

static SampledKnobs sampled_knobs_from_env()
{
    int  watchpoints = get_int_env_alias("SAMPLED_WATCHPOINTS", "TSAN_SAMPLED_WATCHPOINTS", 1);
    int  sample_skip = get_int_env_alias("SAMPLED_SKIP", "TSAN_SAMPLED_SKIP", 32);
    int  delay_iters = get_int_env_alias("SAMPLED_DELAY", "TSAN_SAMPLED_DELAY", 32);
    int  probe_count = get_int_env_alias("SAMPLED_PROBES", "TSAN_SAMPLED_PROBES", 1);
    bool reports     = get_int_env("SAMPLED_REPORTS", 0) != 0;

    if(watchpoints <= 0 || (watchpoints & (watchpoints - 1)) != 0)
    {
        std::fprintf(stderr, "error: SAMPLED_WATCHPOINTS must be a positive power of two\n");
        std::exit(EXIT_FAILURE);
    }
    if(sample_skip < 0 || delay_iters < 0 || probe_count < 0)
    {
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

static bool uses_default_static_sampled_policy(SampledKnobs knobs)
{
    return knobs.watchpoints == 1 && knobs.sample_skip == 32 && knobs.probe_count == 1
           && knobs.delay_iters == 32 && !knobs.reports;
}

struct Allocation
{
    uint64_t*                           shadow       = nullptr;
    unsigned long long*                 report_count = nullptr;
    std::vector<hip_moi::host_context*> contexts;
    hip_moi::context::storage_ref*      device_storage_refs      = nullptr;
    uint64_t                            storage_count            = 0;
    uint32_t                            generation               = 1;
    uint32_t                            sampled_watchpoints      = 1;
    uint32_t                            workgroup_stride_entries = 0;
    uint32_t                            sample_skip              = 32;
    uint32_t                            delay_iters              = 32;
    uint32_t                            probe_count              = 1;
    bool                                sampled_reports          = false;
};

static Allocation make_allocation(Mode mode, int sequence_length, SampledKnobs sampled_knobs)
{
    Allocation allocation;
    uint64_t   workgroups          = static_cast<uint64_t>(sequence_length / kQueriesPerBlock);
    allocation.storage_count       = workgroups;
    allocation.sampled_watchpoints = sampled_knobs.watchpoints;
    allocation.sample_skip         = sampled_knobs.sample_skip;
    allocation.delay_iters         = sampled_knobs.delay_iters;
    allocation.probe_count         = sampled_knobs.probe_count;
    allocation.sampled_reports     = sampled_knobs.reports;

    if(mode == Mode::JakubSampledLoom)
    {
        allocation.workgroup_stride_entries
            = 1 + static_cast<uint32_t>(allocation.sampled_watchpoints);
        HIP_CHECK(hipMalloc(&allocation.shadow,
                            workgroups * allocation.workgroup_stride_entries * sizeof(uint64_t)));
        HIP_CHECK(hipMemset(allocation.shadow,
                            0,
                            workgroups * allocation.workgroup_stride_entries * sizeof(uint64_t)));
        if(allocation.sampled_reports)
        {
            HIP_CHECK(
                hipMalloc(&allocation.report_count, workgroups * sizeof(*allocation.report_count)));
            HIP_CHECK(hipMemset(
                allocation.report_count, 0, workgroups * sizeof(*allocation.report_count)));
        }
        return allocation;
    }

    if(!is_hip_moi_mode(mode))
    {
        return allocation;
    }

    hip_moi::host_context_options options;
    options.storage_bytes = static_cast<std::size_t>(
        get_int_env("HIP_MOI_STORAGE_BYTES", HIP_MOI_BENCH_STORAGE_BYTES));
    options.diagnostic_capacity            = get_int_env("HIP_MOI_DIAGNOSTICS", 1024);
    options.subgroup_capacity              = get_int_env("HIP_MOI_SUBGROUPS", kSubgroupCount);
    options.sampled_watchpoint_capacity    = static_cast<int>(sampled_knobs.watchpoints);
    options.sampled_watchpoint_sample_skip = sampled_knobs.sample_skip;
    options.sampled_watchpoint_probe_count = sampled_knobs.probe_count;
    options.sampled_watchpoint_delay_iters = sampled_knobs.delay_iters;
    options.sampled_watchpoint_reports     = sampled_knobs.reports;
    options.backend                        = hip_moi::backend_kind::sampled_watchpoint;
    options.destructor_reports             = false;
    options.destructor_aborts              = false;
    options.diagnostic_stream              = nullptr;

    std::vector<hip_moi::context::storage_ref> storage_refs;
    storage_refs.reserve(workgroups);
    for(uint64_t i = 0; i < workgroups; ++i)
    {
        auto* context = new hip_moi::host_context(options);
        context->disable_destructor_check();
        allocation.contexts.push_back(context);
        storage_refs.push_back(context->launch_ref());
    }

    HIP_CHECK(
        hipMalloc(&allocation.device_storage_refs, storage_refs.size() * sizeof(storage_refs[0])));
    HIP_CHECK(hipMemcpy(allocation.device_storage_refs,
                        storage_refs.data(),
                        storage_refs.size() * sizeof(storage_refs[0]),
                        hipMemcpyHostToDevice));
    return allocation;
}

static Runtime next_runtime(Allocation& allocation)
{
    Runtime runtime{
        allocation.shadow,
        allocation.report_count,
        allocation.workgroup_stride_entries,
        allocation.device_storage_refs,
        allocation.storage_count,
        allocation.generation,
        allocation.sample_skip,
        allocation.delay_iters,
        allocation.probe_count,
    };
    ++allocation.generation;
    return runtime;
}

static void free_allocation(Allocation& allocation)
{
    if(allocation.device_storage_refs)
    {
        HIP_CHECK(hipFree(allocation.device_storage_refs));
    }
    for(auto* context : allocation.contexts)
    {
        delete context;
    }
    if(allocation.shadow)
    {
        HIP_CHECK(hipFree(allocation.shadow));
    }
    if(allocation.report_count)
    {
        HIP_CHECK(hipFree(allocation.report_count));
    }
    allocation = {};
}

template <typename Policy>
static void launch_kernel(const f16x8* q,
                          const f16x8* k,
                          const f16x8* v,
                          float*       out,
                          int          sequence_length,
                          Runtime      runtime,
                          hipStream_t  stream)
{
    dim3 block(kThreads);
    dim3 grid(sequence_length / kQueriesPerBlock);
    attention_no_score_lds_kernel<Policy>
        <<<grid, block, 0, stream>>>(q, k, v, out, sequence_length, runtime);
}

template <typename Policy>
static double run_one(const char*  name,
                      Mode         mode,
                      const f16x8* q,
                      const f16x8* k,
                      const f16x8* v,
                      float*       out,
                      int          sequence_length,
                      SampledKnobs sampled_knobs,
                      double       min_ms,
                      double       warmup_ms)
{
    Allocation  allocation = make_allocation(mode, sequence_length, sampled_knobs);
    hipStream_t stream     = nullptr;
    for(int i = 0; i < 5; ++i)
    {
        launch_kernel<Policy>(q, k, v, out, sequence_length, next_runtime(allocation), stream);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    int calibration_iterations = std::max(1, get_int_env("BENCH_CALIBRATION_ITERS", 10));
    int max_iterations         = std::max(1, get_int_env("MAX_ITERS", 100000));

    auto run_timed_iterations = [&](int iteration_count) {
        HIP_CHECK(hipEventRecord(start, stream));
        for(int i = 0; i < iteration_count; ++i)
        {
            launch_kernel<Policy>(q, k, v, out, sequence_length, next_runtime(allocation), stream);
        }
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        HIP_CHECK(hipGetLastError());
        return elapsed_ms;
    };

    auto target_iterations = [&](double target_ms, double estimate_ms) {
        if(target_ms <= 0.0)
        {
            return 0;
        }
        int iterations
            = std::max(1, static_cast<int>(std::ceil(target_ms / std::max(0.001, estimate_ms))));
        return std::min(iterations, max_iterations);
    };

    auto run_until_target = [&](double target_ms, int* iteration_count) {
        if(target_ms <= 0.0 || *iteration_count <= 0)
        {
            *iteration_count = 0;
            return 0.0f;
        }

        float elapsed_ms = 0.0f;
        for(;;)
        {
            elapsed_ms = run_timed_iterations(*iteration_count);
            if(elapsed_ms >= target_ms || *iteration_count == max_iterations)
            {
                return elapsed_ms;
            }

            double scale = target_ms / std::max(0.001f, elapsed_ms);
            int    next_iterations
                = static_cast<int>(std::ceil(static_cast<double>(*iteration_count) * scale));
            *iteration_count
                = std::min(max_iterations, std::max(*iteration_count + 1, next_iterations));
        }
    };

    float calibration_ms = run_timed_iterations(calibration_iterations);
    float one_ms         = calibration_ms / calibration_iterations;

    int   warmup_iterations = target_iterations(warmup_ms, one_ms);
    float warmup_elapsed_ms = run_until_target(warmup_ms, &warmup_iterations);

    double measurement_estimate_ms
        = warmup_iterations > 0 && warmup_elapsed_ms > 0.0f
              ? static_cast<double>(warmup_elapsed_ms) / warmup_iterations
              : one_ms;
    int   iterations = target_iterations(min_ms, measurement_estimate_ms);
    float elapsed_ms = run_until_target(min_ms, &iterations);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    uint64_t reports = 0;
    if(mode == Mode::JakubSampledLoom && allocation.report_count)
    {
        HIP_CHECK(
            hipMemcpy(&reports, allocation.report_count, sizeof(reports), hipMemcpyDeviceToHost));
        if(reports != 0)
        {
            std::printf("%-68s benchmark_tsan_reports=%llu\n",
                        name,
                        static_cast<unsigned long long>(reports));
        }
    }
    else if(is_hip_moi_mode(mode))
    {
        for(auto* context : allocation.contexts)
        {
            int count = context->consume_diagnostics(nullptr);
            if(count < 0)
            {
                std::fprintf(stderr, "error: failed to read hip-moi diagnostics\n");
                std::exit(EXIT_FAILURE);
            }
            reports += static_cast<uint64_t>(count);
        }
        if(reports != 0)
        {
            std::printf("%-68s benchmark_tsan_reports=%llu\n",
                        name,
                        static_cast<unsigned long long>(reports));
        }
    }
    free_allocation(allocation);

    double ms  = elapsed_ms / iterations;
    double ops = 2.0 * static_cast<double>(sequence_length) * static_cast<double>(sequence_length)
                 * static_cast<double>(kHeadDim + kValueDim);
    double         achieved = ops / (ms * 1.0e9);
    DisplayLatency latency  = display_latency(ms);
    std::printf("%-68s %#9.3g %s  %8.3f effective TFLOP/s"
                "  total=%8.3f ms  iters=%d  warmup=%8.3f ms"
                "  warmup_iters=%d\n",
                name,
                latency.value,
                latency.unit,
                achieved,
                elapsed_ms,
                iterations,
                warmup_elapsed_ms,
                warmup_iterations);
    return ms;
}

static void print_device()
{
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    std::printf("device %d: %s, gcnArch=%s, CUs=%d\n",
                device,
                prop.name,
                prop.gcnArchName,
                prop.multiProcessorCount);
}

int main()
{
    int    sequence_length = get_int_env("BENCH_SEQ", 12288);
    double min_ms          = get_double_env("MIN_MS", 100.0);
    double warmup_ms       = get_double_env("WARMUP_MS", min_ms);
    if(sequence_length <= 0 || sequence_length % kQueriesPerBlock != 0
       || sequence_length % kKeyTile != 0)
    {
        std::fprintf(
            stderr, "error: BENCH_SEQ must be a positive multiple of %d\n", kQueriesPerBlock);
        return EXIT_FAILURE;
    }
    if(min_ms <= 0.0)
    {
        std::fprintf(stderr, "error: MIN_MS must be positive\n");
        return EXIT_FAILURE;
    }
    if(warmup_ms < 0.0)
    {
        std::fprintf(stderr, "error: WARMUP_MS must be nonnegative\n");
        return EXIT_FAILURE;
    }

    int         query_blocks = sequence_length / kQueriesPerBlock;
    int         key_tiles    = sequence_length / kKeyTile;
    std::size_t q_vectors
        = static_cast<std::size_t>(query_blocks) * kSubgroupCount * kHeadFragments * kWaveSize;
    std::size_t k_vectors   = static_cast<std::size_t>(key_tiles) * kHeadFragments * kWaveSize;
    std::size_t v_vectors   = static_cast<std::size_t>(key_tiles) * kValueFragments * kWaveSize;
    std::size_t out_scalars = static_cast<std::size_t>(sequence_length) * kValueDim;
    f16x8*      q           = gpu_alloc<f16x8>(q_vectors);
    f16x8*      k           = gpu_alloc<f16x8>(k_vectors);
    f16x8*      v           = gpu_alloc<f16x8>(v_vectors);
    float*      out         = gpu_alloc<float>(out_scalars);
    HIP_CHECK(hipMemset(q, 0, q_vectors * sizeof(f16x8)));
    HIP_CHECK(hipMemset(k, 0, k_vectors * sizeof(f16x8)));
    HIP_CHECK(hipMemset(v, 0, v_vectors * sizeof(f16x8)));
    HIP_CHECK(hipMemset(out, 0, out_scalars * sizeof(float)));

    print_device();
    std::printf("attention no-score-LDS shape: seq=%d head_dim=%d value_dim=%d "
                "head_fragments=%d value_fragments=%d query_blocks=%d "
                "waves_per_workgroup=%d min_ms=%.1f warmup_ms=%.1f\n",
                sequence_length,
                kHeadDim,
                kValueDim,
                kHeadFragments,
                kValueFragments,
                query_blocks,
                kSubgroupCount,
                min_ms,
                warmup_ms);
    std::printf("attention no-score-LDS instrumentation sites: K/V fragment staging only\n");

    SampledKnobs sampled_knobs = sampled_knobs_from_env();
    std::printf("sampled knobs: watchpoints=%u skip=%u probes=%u delay=%u "
                "reports=%s\n",
                sampled_knobs.watchpoints,
                sampled_knobs.sample_skip,
                sampled_knobs.probe_count,
                sampled_knobs.delay_iters,
                sampled_knobs.reports ? "on" : "off");

    bool use_static_fast = uses_default_static_sampled_policy(sampled_knobs);
    if(use_static_fast)
    {
        std::printf("hip-moi sampled_watchpoint_context policy: static default "
                    "publish-only\n");
    }

    std::string prefix = "attention_no_score_lds";
    if(kHeadDim != 16 || kValueDim != 16)
    {
        prefix += "_d" + std::to_string(kHeadDim) + "_v" + std::to_string(kValueDim);
    }
    prefix += "_seq" + std::to_string(sequence_length);
    std::string sampled_mode_suffix = sampled_knobs.reports ? "_reporting" : "_publish_only";
    run_one<PassThroughPolicy>((prefix + "_pass_through").c_str(),
                        Mode::PassThrough,
                        q,
                        k,
                        v,
                        out,
                        sequence_length,
                        sampled_knobs,
                        min_ms,
                        warmup_ms);
    run_one<JakubSampledLoomPolicy>((prefix + "_jakub_sampled_loom" + sampled_mode_suffix).c_str(),
                               Mode::JakubSampledLoom,
                               q,
                               k,
                               v,
                               out,
                               sequence_length,
                               sampled_knobs,
                               min_ms,
                               warmup_ms);
    run_one<ContextSampledPolicy>(
        (prefix + "_hip_moi_context_sampled_watchpoint" + sampled_mode_suffix).c_str(),
        Mode::HipMoiContextSampledWatchpoint,
        q,
        k,
        v,
        out,
        sequence_length,
        sampled_knobs,
        min_ms,
        warmup_ms);
    if(use_static_fast)
    {
        run_one<SampledWatchpointContextPolicy>(
            (prefix + "_hip_moi_sampled_watchpoint_context" + sampled_mode_suffix).c_str(),
            Mode::HipMoiSampledWatchpointContext,
            q,
            k,
            v,
            out,
            sequence_length,
            sampled_knobs,
            min_ms,
            warmup_ms);
    }

    HIP_CHECK(hipFree(q));
    HIP_CHECK(hipFree(k));
    HIP_CHECK(hipFree(v));
    HIP_CHECK(hipFree(out));
    return 0;
}
