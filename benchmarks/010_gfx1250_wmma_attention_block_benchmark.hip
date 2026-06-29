// Copyright 2026
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// gfx1250 WMMA attention-block benchmark for hip-moi.
//
// This grows tests/instrumented/010_gfx1250_wmma_attention_block_test.hip into a
// benchmark-sized workload. Each workgroup computes a 32-query block with two
// subgroups. K/V fragments and scalar softmax scratch are staged through LDS,
// QK and PV are computed with gfx1250 WMMA, and every LDS access is wrapped
// when comparing pass-through execution, the local sampled baseline, and
// hip-moi sampled watchpoint paths.

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
using f16x16 = half_t __attribute__((ext_vector_type(16)));
using f32x8  = float __attribute__((ext_vector_type(8)));

static constexpr int kWaveSize            = 32;
static constexpr int kSubgroupCount       = 2;
static constexpr int kThreads             = kWaveSize * kSubgroupCount;
static constexpr int kThreadsPerSubgroup  = kWaveSize;
static constexpr int kQueryTile           = 16;
static constexpr int kQueriesPerBlock     = kQueryTile * kSubgroupCount;
static constexpr int kKeyTile             = 16;
static constexpr int kHeadDim             = 32;
static constexpr int kValueDim            = 16;
static constexpr int kOperandElements     = 16;
static constexpr int kAccumulatorElements = 8;

static constexpr uint32_t kKSharedByteOffset = 0;
static constexpr uint32_t kVSharedByteOffset = kWaveSize * static_cast<uint32_t>(sizeof(f16x16));
static constexpr uint32_t kScoresSharedByteOffset
    = kVSharedByteOffset + kWaveSize * static_cast<uint32_t>(sizeof(f16x16));
static constexpr uint32_t kWeightsSharedByteOffset
    = kScoresSharedByteOffset
      + kSubgroupCount * kQueryTile * kKeyTile * static_cast<uint32_t>(sizeof(float));
static constexpr uint32_t kRowOldScaleSharedByteOffset
    = kWeightsSharedByteOffset
      + kSubgroupCount * kQueryTile * kKeyTile * static_cast<uint32_t>(sizeof(half_t));
static constexpr uint32_t kRowSumSharedByteOffset
    = kRowOldScaleSharedByteOffset
      + kSubgroupCount * kQueryTile * static_cast<uint32_t>(sizeof(float));

static constexpr uint32_t kAttentionSiteKv      = 1u << 0;
static constexpr uint32_t kAttentionSiteScores  = 1u << 1;
static constexpr uint32_t kAttentionSiteWeights = 1u << 2;
static constexpr uint32_t kAttentionSiteRows    = 1u << 3;
static constexpr uint32_t kAttentionSiteAll
    = kAttentionSiteKv | kAttentionSiteScores | kAttentionSiteWeights | kAttentionSiteRows;

#ifndef HIP_MOI_ATTENTION_SITE_MASK
#define HIP_MOI_ATTENTION_SITE_MASK 15
#endif

static constexpr uint32_t kAttentionSiteMask = HIP_MOI_ATTENTION_SITE_MASK;

static constexpr hip_moi::site_id kKStoreSite(0xb01);
static constexpr hip_moi::site_id kVStoreSite(0xb02);
static constexpr hip_moi::site_id kKLoadSite(0xb03);
static constexpr hip_moi::site_id kVLoadSite(0xb04);
static constexpr hip_moi::site_id kScoreStoreSite(0xb05);
static constexpr hip_moi::site_id kScoreLoadMaxSite(0xb06);
static constexpr hip_moi::site_id kScoreLoadWeightSite(0xb07);
static constexpr hip_moi::site_id kWeightStoreSite(0xb08);
static constexpr hip_moi::site_id kWeightLoadSite(0xb09);
static constexpr hip_moi::site_id kRowOldScaleStoreSite(0xb0a);
static constexpr hip_moi::site_id kRowOldScaleLoadSite(0xb0b);
static constexpr hip_moi::site_id kRowSumStoreSite(0xb0c);
static constexpr hip_moi::site_id kRowSumLoadSite(0xb0d);

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

__host__ __device__ static int gfx1250_wmma_f16_k(int lane, int idx)
{
    int half = idx & 1;
    int reg  = idx >> 1;
    return half | ((reg & 1) << 1) | (((lane >> 4) & 1) << 2) | (((reg >> 1) & 1) << 3)
           | (((reg >> 2) & 1) << 4);
}

__host__ __device__ static int gfx1250_wmma_acc_m(int lane, int elem)
{
    return elem + 8 * (lane >> 4);
}

__host__ __device__ static uint32_t k_fragment_offset(int lane)
{
    return kKSharedByteOffset + static_cast<uint32_t>(lane * sizeof(f16x16));
}

__host__ __device__ static uint32_t v_fragment_offset(int lane)
{
    return kVSharedByteOffset + static_cast<uint32_t>(lane * sizeof(f16x16));
}

__host__ __device__ static uint32_t score_offset(int subgroup, int query_row, int key)
{
    return kScoresSharedByteOffset
           + static_cast<uint32_t>(((subgroup * kQueryTile + query_row) * kKeyTile + key)
                                   * sizeof(float));
}

__host__ __device__ static uint32_t weight_offset(int subgroup, int query_row, int key)
{
    return kWeightsSharedByteOffset
           + static_cast<uint32_t>(((subgroup * kQueryTile + query_row) * kKeyTile + key)
                                   * sizeof(half_t));
}

__host__ __device__ static uint32_t row_old_scale_offset(int subgroup, int query_row)
{
    return kRowOldScaleSharedByteOffset
           + static_cast<uint32_t>((subgroup * kQueryTile + query_row) * sizeof(float));
}

__host__ __device__ static uint32_t row_sum_offset(int subgroup, int query_row)
{
    return kRowSumSharedByteOffset
           + static_cast<uint32_t>((subgroup * kQueryTile + query_row) * sizeof(float));
}

template <uint32_t SiteClass, typename Policy, typename T>
__device__ static void attention_store(
    typename Policy::context_type& ctx, T* ptr, T value, uint32_t offset, hip_moi::site_id site)
{
    if constexpr((kAttentionSiteMask & SiteClass) != 0)
    {
        Policy::store(ctx, ptr, value, offset, site);
    }
    else
    {
        *ptr = value;
    }
}

template <uint32_t SiteClass, typename Policy, typename T>
__device__ static T attention_load(typename Policy::context_type& ctx,
                                   const T*                       ptr,
                                   uint32_t                       offset,
                                   hip_moi::site_id               site)
{
    if constexpr((kAttentionSiteMask & SiteClass) != 0)
    {
        return Policy::load(ctx, ptr, offset, site);
    }
    else
    {
        return *ptr;
    }
}

__device__ __forceinline__ static f32x8 zero_f32x8()
{
    f32x8 value;
#pragma unroll
    for(int i = 0; i < kAccumulatorElements; ++i)
    {
        value[i] = 0.0f;
    }
    return value;
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
__global__
    __launch_bounds__(kThreads) void attention_block_kernel(const f16x16* __restrict__ q_fragments,
                                                            const f16x16* __restrict__ k_fragments,
                                                            const f16x16* __restrict__ v_fragments,
                                                            float* __restrict__ out,
                                                            int     sequence_length,
                                                            Runtime runtime)
{
    __shared__ f16x16 k_shared[kWaveSize];
    __shared__ f16x16 v_shared[kWaveSize];
    __shared__ float  scores[kSubgroupCount][kQueryTile][kKeyTile];
    __shared__ half_t weights[kSubgroupCount][kQueryTile][kKeyTile];
    __shared__ float  row_old_scale[kSubgroupCount][kQueryTile];
    __shared__ float  row_sum[kSubgroupCount][kQueryTile];

    typename Policy::context_type ctx = Policy::make_context(runtime);
    Policy::init_workgroup(ctx);

    int tid            = static_cast<int>(threadIdx.x);
    int subgroup       = tid / kWaveSize;
    int lane           = tid & (kWaveSize - 1);
    int query_block    = static_cast<int>(blockIdx.x);
    int key_tile_count = sequence_length / kKeyTile;

    f16x16 q_fragment = q_fragments[(query_block * kSubgroupCount + subgroup) * kWaveSize + lane];
    f32x8  out_acc    = zero_f32x8();

    float running_max[kQueryTile];
    float running_sum[kQueryTile];
#pragma unroll
    for(int row = 0; row < kQueryTile; ++row)
    {
        running_max[row] = -3.402823466e38f;
        running_sum[row] = 0.0f;
    }

    for(int key_tile = 0; key_tile < key_tile_count; ++key_tile)
    {
        if(subgroup == 0)
        {
            attention_store<kAttentionSiteKv, Policy>(ctx,
                                                      &k_shared[lane],
                                                      k_fragments[key_tile * kWaveSize + lane],
                                                      k_fragment_offset(lane),
                                                      kKStoreSite);
            attention_store<kAttentionSiteKv, Policy>(ctx,
                                                      &v_shared[lane],
                                                      v_fragments[key_tile * kWaveSize + lane],
                                                      v_fragment_offset(lane),
                                                      kVStoreSite);
        }

        Policy::syncthreads(ctx);

        f16x16 k_fragment = attention_load<kAttentionSiteKv, Policy>(
            ctx, &k_shared[lane], k_fragment_offset(lane), kKLoadSite);
        f32x8 qk_acc = zero_f32x8();
        qk_acc       = __builtin_amdgcn_wmma_f32_16x16x32_f16(
            false, q_fragment, false, k_fragment, static_cast<short>(0), qk_acc, false, false);

        int key_column = lane & 15;
#pragma unroll
        for(int elem = 0; elem < kAccumulatorElements; ++elem)
        {
            int query_row = gfx1250_wmma_acc_m(lane, elem);
            attention_store<kAttentionSiteScores, Policy>(
                ctx,
                &scores[subgroup][query_row][key_column],
                qk_acc[elem] * 0.25f,
                score_offset(subgroup, query_row, key_column),
                kScoreStoreSite);
        }

        Policy::syncthreads(ctx);

        if(lane == 0)
        {
            for(int query_row = 0; query_row < kQueryTile; ++query_row)
            {
                float tile_max = -3.402823466e38f;
                for(int key = 0; key < kKeyTile; ++key)
                {
                    float score = attention_load<kAttentionSiteScores, Policy>(
                        ctx,
                        &scores[subgroup][query_row][key],
                        score_offset(subgroup, query_row, key),
                        kScoreLoadMaxSite);
                    tile_max = fmaxf(tile_max, score);
                }

                float new_max   = fmaxf(running_max[query_row], tile_max);
                float old_scale = running_sum[query_row] == 0.0f
                                      ? 0.0f
                                      : expf(running_max[query_row] - new_max);
                attention_store<kAttentionSiteRows, Policy>(
                    ctx,
                    &row_old_scale[subgroup][query_row],
                    old_scale,
                    row_old_scale_offset(subgroup, query_row),
                    kRowOldScaleStoreSite);

                float tile_sum = 0.0f;
                for(int key = 0; key < kKeyTile; ++key)
                {
                    float score = attention_load<kAttentionSiteScores, Policy>(
                        ctx,
                        &scores[subgroup][query_row][key],
                        score_offset(subgroup, query_row, key),
                        kScoreLoadWeightSite);
                    float  weight      = expf(score - new_max);
                    half_t weight_half = static_cast<half_t>(weight);
                    attention_store<kAttentionSiteWeights, Policy>(
                        ctx,
                        &weights[subgroup][query_row][key],
                        weight_half,
                        weight_offset(subgroup, query_row, key),
                        kWeightStoreSite);
                    tile_sum += static_cast<float>(weight_half);
                }

                running_sum[query_row] = running_sum[query_row] * old_scale + tile_sum;
                running_max[query_row] = new_max;
                attention_store<kAttentionSiteRows, Policy>(ctx,
                                                            &row_sum[subgroup][query_row],
                                                            running_sum[query_row],
                                                            row_sum_offset(subgroup, query_row),
                                                            kRowSumStoreSite);
            }
        }

        Policy::syncthreads(ctx);

#pragma unroll
        for(int elem = 0; elem < kAccumulatorElements; ++elem)
        {
            int   query_row = gfx1250_wmma_acc_m(lane, elem);
            float old_scale = attention_load<kAttentionSiteRows, Policy>(
                ctx,
                &row_old_scale[subgroup][query_row],
                row_old_scale_offset(subgroup, query_row),
                kRowOldScaleLoadSite);
            out_acc[elem] *= old_scale;
        }

        f16x16 p_fragment;
#pragma unroll
        for(int elem = 0; elem < kOperandElements; ++elem)
        {
            int query_row    = lane & 15;
            int key          = gfx1250_wmma_f16_k(lane, elem);
            p_fragment[elem] = key < kKeyTile ? attention_load<kAttentionSiteWeights, Policy>(
                                                    ctx,
                                                    &weights[subgroup][query_row][key],
                                                    weight_offset(subgroup, query_row, key),
                                                    kWeightLoadSite)
                                              : static_cast<half_t>(0.0f);
        }

        f16x16 v_fragment = attention_load<kAttentionSiteKv, Policy>(
            ctx, &v_shared[lane], v_fragment_offset(lane), kVLoadSite);
        f32x8 pv_acc = zero_f32x8();
        pv_acc       = __builtin_amdgcn_wmma_f32_16x16x32_f16(
            false, p_fragment, false, v_fragment, static_cast<short>(0), pv_acc, false, false);

#pragma unroll
        for(int elem = 0; elem < kAccumulatorElements; ++elem)
        {
            out_acc[elem] += pv_acc[elem];
        }

        Policy::syncthreads(ctx);
    }

    int value_column = lane & 15;
#pragma unroll
    for(int elem = 0; elem < kAccumulatorElements; ++elem)
    {
        int   query_row    = gfx1250_wmma_acc_m(lane, elem);
        int   global_query = query_block * kQueriesPerBlock + subgroup * kQueryTile + query_row;
        float normalizer
            = attention_load<kAttentionSiteRows, Policy>(ctx,
                                                         &row_sum[subgroup][query_row],
                                                         row_sum_offset(subgroup, query_row),
                                                         kRowSumLoadSite);
        out[global_query * kValueDim + value_column] = out_acc[elem] / normalizer;
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
static void launch_kernel(const f16x16* q,
                          const f16x16* k,
                          const f16x16* v,
                          float*        out,
                          int           sequence_length,
                          Runtime       runtime,
                          hipStream_t   stream)
{
    dim3 block(kThreads);
    dim3 grid(sequence_length / kQueriesPerBlock);
    attention_block_kernel<Policy>
        <<<grid, block, 0, stream>>>(q, k, v, out, sequence_length, runtime);
}

template <typename Policy>
static double run_one(const char*   name,
                      Mode          mode,
                      const f16x16* q,
                      const f16x16* k,
                      const f16x16* v,
                      float*        out,
                      int           sequence_length,
                      SampledKnobs  sampled_knobs,
                      double        min_ms,
                      double        warmup_ms)
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
    std::printf(
        "device %d: gcnArch=%s, CUs=%d\n", device, prop.gcnArchName, prop.multiProcessorCount);
}

static void print_site_mask()
{
    std::printf("attention LDS instrumentation sites: mask=0x%x kv=%s scores=%s "
                "weights=%s rows=%s\n",
                kAttentionSiteMask,
                (kAttentionSiteMask & kAttentionSiteKv) ? "on" : "off",
                (kAttentionSiteMask & kAttentionSiteScores) ? "on" : "off",
                (kAttentionSiteMask & kAttentionSiteWeights) ? "on" : "off",
                (kAttentionSiteMask & kAttentionSiteRows) ? "on" : "off");
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
    std::size_t q_vectors    = static_cast<std::size_t>(query_blocks) * kSubgroupCount * kWaveSize;
    std::size_t kv_vectors   = static_cast<std::size_t>(key_tiles) * kWaveSize;
    std::size_t output_scalars = static_cast<std::size_t>(sequence_length) * kValueDim;
    f16x16*     q              = gpu_alloc<f16x16>(q_vectors);
    f16x16*     k              = gpu_alloc<f16x16>(kv_vectors);
    f16x16*     v              = gpu_alloc<f16x16>(kv_vectors);
    float*      out            = gpu_alloc<float>(output_scalars);
    HIP_CHECK(hipMemset(q, 0, q_vectors * sizeof(f16x16)));
    HIP_CHECK(hipMemset(k, 0, kv_vectors * sizeof(f16x16)));
    HIP_CHECK(hipMemset(v, 0, kv_vectors * sizeof(f16x16)));
    HIP_CHECK(hipMemset(out, 0, output_scalars * sizeof(float)));

    print_device();
    std::printf("attention shape: seq=%d head_dim=%d value_dim=%d "
                "query_blocks=%d waves_per_workgroup=%d min_ms=%.1f "
                "warmup_ms=%.1f\n",
                sequence_length,
                kHeadDim,
                kValueDim,
                query_blocks,
                kSubgroupCount,
                min_ms,
                warmup_ms);
    print_site_mask();

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

    std::string prefix = "gfx1250_attention_wmma_seq" + std::to_string(sequence_length);
    if(kAttentionSiteMask != kAttentionSiteAll)
    {
        char mask_suffix[32];
        std::snprintf(mask_suffix, sizeof(mask_suffix), "_site_mask_0x%x", kAttentionSiteMask);
        prefix += mask_suffix;
    }
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
