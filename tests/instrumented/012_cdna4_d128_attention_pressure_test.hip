// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/012_cdna4_d128_attention_pressure_test.hip
//
// CDNA4-only D128/V128 attention pressure correctness rungs. These are still
// CTest-sized kernels, but they exercise the two source-mined LDS pressure
// shapes that should become benchmarks next: full K/V double-buffering for a
// 16-key tile, and a wider 32-key double-buffer pressure tile.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{
    using half_t  = _Float16;
    using f16x4_t = half_t __attribute__((ext_vector_type(4)));
    using f32x4_t = float __attribute__((ext_vector_type(4)));

    constexpr int kWaveSize              = 64;
    constexpr int kSubgroupCount         = 2;
    constexpr int kThreadCount           = kWaveSize * kSubgroupCount;
    constexpr int kThreadsPerSubgroup    = kWaveSize;
    constexpr int kQueryTile             = 16;
    constexpr int kQueryCount            = kQueryTile * kSubgroupCount;
    constexpr int kKeyCount              = 64;
    constexpr int kHeadDim               = 128;
    constexpr int kValueDim              = 128;
    constexpr int kFragmentDim           = 16;
    constexpr int kFragmentElements      = 4;
    constexpr int kHeadFragments         = kHeadDim / kFragmentDim;
    constexpr int kValueFragments        = kValueDim / kFragmentDim;
    constexpr int kKeySubtileCount       = kKeyCount / kFragmentDim;
    constexpr int kDoubleBufferCount     = 2;
    constexpr int kDiagnosticCapacity    = 16;
    constexpr int kSampledWatchpoints    = 256;
    constexpr int kRepresentativeQHeads  = 64;
    constexpr int kRepresentativeKvHeads = 8;
    constexpr int kRepresentativeGqa     = kRepresentativeQHeads / kRepresentativeKvHeads;

    static_assert(kRepresentativeGqa == 8);
    static_assert(kHeadDim == kHeadFragments * kFragmentDim);
    static_assert(kValueDim == kValueFragments * kFragmentDim);
    static_assert(kKeyCount % kFragmentDim == 0);

    constexpr std::size_t kQFragmentCount = kSubgroupCount * kHeadFragments * kWaveSize;
    constexpr std::size_t kKFragmentCount = kKeySubtileCount * kHeadFragments * kWaveSize;
    constexpr std::size_t kVFragmentCount = kKeySubtileCount * kValueFragments * kWaveSize;
    constexpr std::size_t kOutputCount    = kQueryCount * kValueDim;

    constexpr std::size_t kQBytes      = kQFragmentCount * sizeof(f16x4_t);
    constexpr std::size_t kKBytes      = kKFragmentCount * sizeof(f16x4_t);
    constexpr std::size_t kVBytes      = kVFragmentCount * sizeof(f16x4_t);
    constexpr std::size_t kOutputBytes = kOutputCount * sizeof(float);

    struct FullKvDoubleBuffered16Key
    {
        static constexpr int kKeyTile     = 16;
        static constexpr int kKeySubtiles = kKeyTile / kFragmentDim;
        static constexpr int kKeyTiles    = kKeyCount / kKeyTile;

        static constexpr uint32_t kKSharedByteOffset = 0;
        static constexpr uint32_t kKSharedBytes
            = kDoubleBufferCount * kHeadFragments * kKeySubtiles * kWaveSize * sizeof(f16x4_t);
        static constexpr uint32_t kVSharedByteOffset = kKSharedByteOffset + kKSharedBytes;
        static constexpr uint32_t kVSharedBytes
            = kDoubleBufferCount * kValueFragments * kKeySubtiles * kWaveSize * sizeof(f16x4_t);
        static constexpr uint32_t kScoresSharedByteOffset = kVSharedByteOffset + kVSharedBytes;
        static constexpr uint32_t kScoresSharedBytes
            = kSubgroupCount * kQueryTile * kKeyTile * sizeof(float);
        static constexpr uint32_t kWeightsSharedByteOffset
            = kScoresSharedByteOffset + kScoresSharedBytes;
        static constexpr uint32_t kWeightsSharedBytes
            = kSubgroupCount * kQueryTile * kKeyTile * sizeof(half_t);
        static constexpr uint32_t kRowOldScaleSharedByteOffset
            = kWeightsSharedByteOffset + kWeightsSharedBytes;
        static constexpr uint32_t kRowOldScaleSharedBytes
            = kSubgroupCount * kQueryTile * sizeof(float);
        static constexpr uint32_t kRowSumSharedByteOffset
            = kRowOldScaleSharedByteOffset + kRowOldScaleSharedBytes;
        static constexpr uint32_t kRowSumSharedBytes = kSubgroupCount * kQueryTile * sizeof(float);
        static constexpr int      kSharedBytes
            = static_cast<int>(kRowSumSharedByteOffset + kRowSumSharedBytes);
        static constexpr int kExactShadowEntries
            = (kSharedBytes + static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes) - 1)
              / static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes);
    };

    struct DoubleBuffered32KeyPressure
    {
        static constexpr int kKeyTile     = 32;
        static constexpr int kKeySubtiles = kKeyTile / kFragmentDim;
        static constexpr int kKeyTiles    = kKeyCount / kKeyTile;

        static constexpr uint32_t kKSharedByteOffset = 0;
        static constexpr uint32_t kKSharedBytes
            = kDoubleBufferCount * kHeadFragments * kKeySubtiles * kWaveSize * sizeof(f16x4_t);
        static constexpr uint32_t kVSharedByteOffset = kKSharedByteOffset + kKSharedBytes;
        static constexpr uint32_t kVSharedBytes
            = kDoubleBufferCount * kValueFragments * kKeySubtiles * kWaveSize * sizeof(f16x4_t);
        static constexpr uint32_t kScoresSharedByteOffset = kVSharedByteOffset + kVSharedBytes;
        static constexpr uint32_t kScoresSharedBytes
            = kSubgroupCount * kQueryTile * kKeyTile * sizeof(float);
        static constexpr uint32_t kWeightsSharedByteOffset
            = kScoresSharedByteOffset + kScoresSharedBytes;
        static constexpr uint32_t kWeightsSharedBytes
            = kSubgroupCount * kQueryTile * kKeyTile * sizeof(half_t);
        static constexpr uint32_t kRowOldScaleSharedByteOffset
            = kWeightsSharedByteOffset + kWeightsSharedBytes;
        static constexpr uint32_t kRowOldScaleSharedBytes
            = kSubgroupCount * kQueryTile * sizeof(float);
        static constexpr uint32_t kRowSumSharedByteOffset
            = kRowOldScaleSharedByteOffset + kRowOldScaleSharedBytes;
        static constexpr uint32_t kRowSumSharedBytes = kSubgroupCount * kQueryTile * sizeof(float);
        static constexpr int      kSharedBytes
            = static_cast<int>(kRowSumSharedByteOffset + kRowSumSharedBytes);
        static constexpr int kExactShadowEntries
            = (kSharedBytes + static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes) - 1)
              / static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes);
    };

    static_assert(FullKvDoubleBuffered16Key::kSharedBytes == 19712);
    static_assert(DoubleBuffered32KeyPressure::kSharedBytes == 39168);
    static_assert(FullKvDoubleBuffered16Key::kKeySubtiles == 1);
    static_assert(DoubleBuffered32KeyPressure::kKeySubtiles == 2);

    constexpr hip_moi::site_id kKStoreSite(0xd01);
    constexpr hip_moi::site_id kVStoreSite(0xd02);
    constexpr hip_moi::site_id kKLoadSite(0xd03);
    constexpr hip_moi::site_id kVLoadSite(0xd04);
    constexpr hip_moi::site_id kScoreStoreSite(0xd05);
    constexpr hip_moi::site_id kScoreLoadMaxSite(0xd06);
    constexpr hip_moi::site_id kScoreLoadWeightSite(0xd07);
    constexpr hip_moi::site_id kWeightStoreSite(0xd08);
    constexpr hip_moi::site_id kWeightLoadSite(0xd09);
    constexpr hip_moi::site_id kRowOldScaleStoreSite(0xd0a);
    constexpr hip_moi::site_id kRowOldScaleLoadSite(0xd0b);
    constexpr hip_moi::site_id kRowSumStoreSite(0xd0c);
    constexpr hip_moi::site_id kRowSumLoadSite(0xd0d);

    using FastPublishOnlyPolicy = hip_moi::sampled_watchpoint_policy<
        /*SampleSkip=*/1,
        /*ProbeCount=*/1,
        /*DelayIters=*/0,
        /*ReportConflicts=*/false>;

    __host__ __device__ float input_value(int row, int col, int seed)
    {
        int value = (row * 19 + col * 7 + seed * 29 + ((row + 3) ^ (col * 5))) % 17;
        return static_cast<float>(value - 8) * 0.03125f;
    }

    __host__ __device__ int cdna4_mfma_f16_k(int lane, int elem)
    {
        return elem + 4 * (lane >> 4);
    }

    __host__ __device__ int cdna4_mfma_acc_m(int lane, int elem)
    {
        return elem + 4 * (lane >> 4);
    }

    template <typename Candidate>
    __host__ __device__ uint32_t
        k_fragment_offset(int buffer, int head_fragment, int key_subtile, int lane)
    {
        int index
            = ((buffer * kHeadFragments + head_fragment) * Candidate::kKeySubtiles + key_subtile)
                  * kWaveSize
              + lane;
        return Candidate::kKSharedByteOffset
               + static_cast<uint32_t>(index * static_cast<int>(sizeof(f16x4_t)));
    }

    template <typename Candidate>
    __host__ __device__ uint32_t
        v_fragment_offset(int buffer, int value_fragment, int key_subtile, int lane)
    {
        int index
            = ((buffer * kValueFragments + value_fragment) * Candidate::kKeySubtiles + key_subtile)
                  * kWaveSize
              + lane;
        return Candidate::kVSharedByteOffset
               + static_cast<uint32_t>(index * static_cast<int>(sizeof(f16x4_t)));
    }

    template <typename Candidate>
    __host__ __device__ uint32_t score_offset(int subgroup, int query_row, int key)
    {
        return Candidate::kScoresSharedByteOffset
               + static_cast<uint32_t>(
                   ((subgroup * kQueryTile + query_row) * Candidate::kKeyTile + key)
                   * sizeof(float));
    }

    template <typename Candidate>
    __host__ __device__ uint32_t weight_offset(int subgroup, int query_row, int key)
    {
        return Candidate::kWeightsSharedByteOffset
               + static_cast<uint32_t>(
                   ((subgroup * kQueryTile + query_row) * Candidate::kKeyTile + key)
                   * sizeof(half_t));
    }

    template <typename Candidate>
    __host__ __device__ uint32_t row_old_scale_offset(int subgroup, int query_row)
    {
        return Candidate::kRowOldScaleSharedByteOffset
               + static_cast<uint32_t>((subgroup * kQueryTile + query_row) * sizeof(float));
    }

    template <typename Candidate>
    __host__ __device__ uint32_t row_sum_offset(int subgroup, int query_row)
    {
        return Candidate::kRowSumSharedByteOffset
               + static_cast<uint32_t>((subgroup * kQueryTile + query_row) * sizeof(float));
    }

    __device__ f32x4_t zero_f32x4()
    {
        f32x4_t value;
#pragma unroll
        for(int i = 0; i < kFragmentElements; ++i)
        {
            value[i] = 0.0f;
        }
        return value;
    }

    struct ExactContextPolicy
    {
        using context_type = hip_moi::context;

        __device__ static context_type make_context(hip_moi::context::storage_ref storage)
        {
            hip_moi::context::config cfg{
                /*thread_count=*/static_cast<int>(blockDim.x),
                /*threads_per_subgroup=*/kThreadsPerSubgroup,
                /*subgroup_count=*/kSubgroupCount,
            };
            return hip_moi::context(storage, cfg);
        }

        template <typename T>
        __device__ static void
            store(context_type& ctx, T* ptr, T value, uint32_t offset, hip_moi::site_id site)
        {
            ctx.template lds_store_at<hip_moi::backend_kind::exact_shadow>(
                ptr, value, offset, site);
        }

        template <typename T>
        __device__ static T
            load(context_type& ctx, const T* ptr, uint32_t offset, hip_moi::site_id site)
        {
            return ctx.template lds_load_at<hip_moi::backend_kind::exact_shadow>(ptr, offset, site);
        }
    };

    struct FastContextPolicy
    {
        using context_type = hip_moi::sampled_watchpoint_context;

        __device__ static context_type make_context(hip_moi::context::storage_ref storage)
        {
            hip_moi::sampled_watchpoint_context::config cfg{
                /*threads_per_subgroup=*/kThreadsPerSubgroup,
            };
            return hip_moi::make_sampled_watchpoint_context(storage, cfg);
        }

        template <typename T>
        __device__ static void
            store(context_type& ctx, T* ptr, T value, uint32_t offset, hip_moi::site_id site)
        {
            ctx.template lds_store_at<FastPublishOnlyPolicy>(ptr, value, offset, site);
        }

        template <typename T>
        __device__ static T
            load(context_type& ctx, const T* ptr, uint32_t offset, hip_moi::site_id site)
        {
            return ctx.template lds_load_at<FastPublishOnlyPolicy>(ptr, offset, site);
        }
    };

    template <typename Candidate, typename InstrumentationPolicy>
    __global__ void attention_pressure_kernel(const f16x4_t* __restrict__ q_fragments,
                                              const f16x4_t* __restrict__ k_fragments,
                                              const f16x4_t* __restrict__ v_fragments,
                                              float* __restrict__ out,
                                              hip_moi::context::storage_ref storage)
    {
        __shared__ f16x4_t
            k_shared[kDoubleBufferCount][kHeadFragments][Candidate::kKeySubtiles][kWaveSize];
        __shared__ f16x4_t
            v_shared[kDoubleBufferCount][kValueFragments][Candidate::kKeySubtiles][kWaveSize];
        __shared__ float  scores[kSubgroupCount][kQueryTile][Candidate::kKeyTile];
        __shared__ half_t weights[kSubgroupCount][kQueryTile][Candidate::kKeyTile];
        __shared__ float  row_old_scale[kSubgroupCount][kQueryTile];
        __shared__ float  row_sum[kSubgroupCount][kQueryTile];

        typename InstrumentationPolicy::context_type ctx
            = InstrumentationPolicy::make_context(storage);
        ctx.init_workgroup();

        int tid      = static_cast<int>(threadIdx.x);
        int subgroup = tid / kWaveSize;
        int lane     = tid & (kWaveSize - 1);

        f32x4_t out_acc[kValueFragments];
#pragma unroll
        for(int value_fragment = 0; value_fragment < kValueFragments; ++value_fragment)
        {
            out_acc[value_fragment] = zero_f32x4();
        }

        float running_max[kQueryTile];
        float running_sum[kQueryTile];
#pragma unroll
        for(int row = 0; row < kQueryTile; ++row)
        {
            running_max[row] = -3.402823466e38f;
            running_sum[row] = 0.0f;
        }

        for(int key_tile = 0; key_tile < Candidate::kKeyTiles; ++key_tile)
        {
            int buffer = key_tile & (kDoubleBufferCount - 1);

            constexpr int kStagedKElements = kHeadFragments * Candidate::kKeySubtiles * kWaveSize;
            for(int linear = tid; linear < kStagedKElements; linear += kThreadCount)
            {
                int         stage_lane         = linear % kWaveSize;
                int         stage_rest         = linear / kWaveSize;
                int         key_subtile        = stage_rest % Candidate::kKeySubtiles;
                int         head_fragment      = stage_rest / Candidate::kKeySubtiles;
                int         global_key_subtile = key_tile * Candidate::kKeySubtiles + key_subtile;
                std::size_t k_index = (static_cast<std::size_t>(global_key_subtile) * kHeadFragments
                                       + head_fragment)
                                          * kWaveSize
                                      + stage_lane;
                InstrumentationPolicy::store(
                    ctx,
                    &k_shared[buffer][head_fragment][key_subtile][stage_lane],
                    k_fragments[k_index],
                    k_fragment_offset<Candidate>(buffer, head_fragment, key_subtile, stage_lane),
                    kKStoreSite);
            }

            constexpr int kStagedVElements = kValueFragments * Candidate::kKeySubtiles * kWaveSize;
            for(int linear = tid; linear < kStagedVElements; linear += kThreadCount)
            {
                int         stage_lane         = linear % kWaveSize;
                int         stage_rest         = linear / kWaveSize;
                int         key_subtile        = stage_rest % Candidate::kKeySubtiles;
                int         value_fragment     = stage_rest / Candidate::kKeySubtiles;
                int         global_key_subtile = key_tile * Candidate::kKeySubtiles + key_subtile;
                std::size_t v_index
                    = (static_cast<std::size_t>(global_key_subtile) * kValueFragments
                       + value_fragment)
                          * kWaveSize
                      + stage_lane;
                InstrumentationPolicy::store(
                    ctx,
                    &v_shared[buffer][value_fragment][key_subtile][stage_lane],
                    v_fragments[v_index],
                    v_fragment_offset<Candidate>(buffer, value_fragment, key_subtile, stage_lane),
                    kVStoreSite);
            }

            ctx.syncthreads();

#pragma unroll
            for(int key_subtile = 0; key_subtile < Candidate::kKeySubtiles; ++key_subtile)
            {
                f32x4_t qk_acc = zero_f32x4();

#pragma unroll
                for(int head_fragment = 0; head_fragment < kHeadFragments; ++head_fragment)
                {
                    std::size_t q_index
                        = (static_cast<std::size_t>(subgroup) * kHeadFragments + head_fragment)
                              * kWaveSize
                          + lane;
                    f16x4_t q_fragment = q_fragments[q_index];
                    f16x4_t k_fragment = InstrumentationPolicy::load(
                        ctx,
                        &k_shared[buffer][head_fragment][key_subtile][lane],
                        k_fragment_offset<Candidate>(buffer, head_fragment, key_subtile, lane),
                        kKLoadSite);
                    qk_acc = __builtin_amdgcn_mfma_f32_16x16x16f16(
                        q_fragment, k_fragment, qk_acc, 0, 0, 0);
                }

                int key_column = key_subtile * kFragmentDim + (lane & 15);
#pragma unroll
                for(int elem = 0; elem < kFragmentElements; ++elem)
                {
                    int query_row = cdna4_mfma_acc_m(lane, elem);
                    InstrumentationPolicy::store(
                        ctx,
                        &scores[subgroup][query_row][key_column],
                        qk_acc[elem] * 0.08838834764831845f,
                        score_offset<Candidate>(subgroup, query_row, key_column),
                        kScoreStoreSite);
                }
            }

            ctx.syncthreads();

            if(lane == 0)
            {
                for(int query_row = 0; query_row < kQueryTile; ++query_row)
                {
                    float tile_max = -3.402823466e38f;
                    for(int key = 0; key < Candidate::kKeyTile; ++key)
                    {
                        float score = InstrumentationPolicy::load(
                            ctx,
                            &scores[subgroup][query_row][key],
                            score_offset<Candidate>(subgroup, query_row, key),
                            kScoreLoadMaxSite);
                        tile_max = fmaxf(tile_max, score);
                    }

                    float new_max   = fmaxf(running_max[query_row], tile_max);
                    float old_scale = running_sum[query_row] == 0.0f
                                          ? 0.0f
                                          : expf(running_max[query_row] - new_max);
                    InstrumentationPolicy::store(
                        ctx,
                        &row_old_scale[subgroup][query_row],
                        old_scale,
                        row_old_scale_offset<Candidate>(subgroup, query_row),
                        kRowOldScaleStoreSite);

                    float tile_sum = 0.0f;
                    for(int key = 0; key < Candidate::kKeyTile; ++key)
                    {
                        float score = InstrumentationPolicy::load(
                            ctx,
                            &scores[subgroup][query_row][key],
                            score_offset<Candidate>(subgroup, query_row, key),
                            kScoreLoadWeightSite);
                        float  weight      = expf(score - new_max);
                        half_t weight_half = static_cast<half_t>(weight);
                        InstrumentationPolicy::store(
                            ctx,
                            &weights[subgroup][query_row][key],
                            weight_half,
                            weight_offset<Candidate>(subgroup, query_row, key),
                            kWeightStoreSite);
                        tile_sum += static_cast<float>(weight_half);
                    }

                    running_sum[query_row] = running_sum[query_row] * old_scale + tile_sum;
                    running_max[query_row] = new_max;
                    InstrumentationPolicy::store(ctx,
                                                 &row_sum[subgroup][query_row],
                                                 running_sum[query_row],
                                                 row_sum_offset<Candidate>(subgroup, query_row),
                                                 kRowSumStoreSite);
                }
            }

            ctx.syncthreads();

#pragma unroll
            for(int value_fragment = 0; value_fragment < kValueFragments; ++value_fragment)
            {
#pragma unroll
                for(int elem = 0; elem < kFragmentElements; ++elem)
                {
                    int   query_row = cdna4_mfma_acc_m(lane, elem);
                    float old_scale = InstrumentationPolicy::load(
                        ctx,
                        &row_old_scale[subgroup][query_row],
                        row_old_scale_offset<Candidate>(subgroup, query_row),
                        kRowOldScaleLoadSite);
                    out_acc[value_fragment][elem] *= old_scale;
                }
            }

#pragma unroll
            for(int key_subtile = 0; key_subtile < Candidate::kKeySubtiles; ++key_subtile)
            {
                f16x4_t p_fragment;
#pragma unroll
                for(int elem = 0; elem < kFragmentElements; ++elem)
                {
                    int query_row    = lane & 15;
                    int key          = key_subtile * kFragmentDim + cdna4_mfma_f16_k(lane, elem);
                    p_fragment[elem] = InstrumentationPolicy::load(
                        ctx,
                        &weights[subgroup][query_row][key],
                        weight_offset<Candidate>(subgroup, query_row, key),
                        kWeightLoadSite);
                }

#pragma unroll
                for(int value_fragment = 0; value_fragment < kValueFragments; ++value_fragment)
                {
                    f16x4_t v_fragment = InstrumentationPolicy::load(
                        ctx,
                        &v_shared[buffer][value_fragment][key_subtile][lane],
                        v_fragment_offset<Candidate>(buffer, value_fragment, key_subtile, lane),
                        kVLoadSite);
                    f32x4_t pv_acc = zero_f32x4();
                    pv_acc         = __builtin_amdgcn_mfma_f32_16x16x16f16(
                        p_fragment, v_fragment, pv_acc, 0, 0, 0);

#pragma unroll
                    for(int elem = 0; elem < kFragmentElements; ++elem)
                    {
                        out_acc[value_fragment][elem] += pv_acc[elem];
                    }
                }
            }

            ctx.syncthreads();
        }

        int value_column_in_fragment = lane & 15;
#pragma unroll
        for(int value_fragment = 0; value_fragment < kValueFragments; ++value_fragment)
        {
#pragma unroll
            for(int elem = 0; elem < kFragmentElements; ++elem)
            {
                int   query_row    = cdna4_mfma_acc_m(lane, elem);
                int   global_query = subgroup * kQueryTile + query_row;
                int   value_column = value_fragment * kFragmentDim + value_column_in_fragment;
                float normalizer
                    = InstrumentationPolicy::load(ctx,
                                                  &row_sum[subgroup][query_row],
                                                  row_sum_offset<Candidate>(subgroup, query_row),
                                                  kRowSumLoadSite);
                out[global_query * kValueDim + value_column]
                    = out_acc[value_fragment][elem] / normalizer;
            }
        }
    }

    float quantized_input_value(int row, int col, int seed)
    {
        return static_cast<float>(static_cast<half_t>(input_value(row, col, seed)));
    }

    void fill_fragments(std::vector<f16x4_t>* q_fragments,
                        std::vector<f16x4_t>* k_fragments,
                        std::vector<f16x4_t>* v_fragments)
    {
        for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
        {
            int query_base = subgroup * kQueryTile;
            for(int head_fragment = 0; head_fragment < kHeadFragments; ++head_fragment)
            {
                for(int lane = 0; lane < kWaveSize; ++lane)
                {
                    int query_row = query_base + (lane & 15);
                    for(int elem = 0; elem < kFragmentElements; ++elem)
                    {
                        int dim = head_fragment * kFragmentDim + cdna4_mfma_f16_k(lane, elem);
                        (*q_fragments)[(subgroup * kHeadFragments + head_fragment) * kWaveSize
                                       + lane][elem]
                            = static_cast<half_t>(input_value(query_row, dim, /*seed=*/1));
                    }
                }
            }
        }

        for(int key_subtile = 0; key_subtile < kKeySubtileCount; ++key_subtile)
        {
            int key_base = key_subtile * kFragmentDim;
            for(int head_fragment = 0; head_fragment < kHeadFragments; ++head_fragment)
            {
                for(int lane = 0; lane < kWaveSize; ++lane)
                {
                    int key = key_base + (lane & 15);
                    for(int elem = 0; elem < kFragmentElements; ++elem)
                    {
                        int dim = head_fragment * kFragmentDim + cdna4_mfma_f16_k(lane, elem);
                        (*k_fragments)[(key_subtile * kHeadFragments + head_fragment) * kWaveSize
                                       + lane][elem]
                            = static_cast<half_t>(input_value(key, dim, /*seed=*/2));
                    }
                }
            }

            for(int value_fragment = 0; value_fragment < kValueFragments; ++value_fragment)
            {
                for(int lane = 0; lane < kWaveSize; ++lane)
                {
                    int value_column = value_fragment * kFragmentDim + (lane & 15);
                    for(int elem = 0; elem < kFragmentElements; ++elem)
                    {
                        int key = key_base + cdna4_mfma_f16_k(lane, elem);
                        (*v_fragments)[(key_subtile * kValueFragments + value_fragment) * kWaveSize
                                       + lane][elem]
                            = static_cast<half_t>(input_value(key, value_column, /*seed=*/3));
                    }
                }
            }
        }
    }

    template <typename Candidate>
    std::vector<float> host_attention_reference()
    {
        std::vector<float> out(kOutputCount, 0.0f);
        for(int query = 0; query < kQueryCount; ++query)
        {
            float running_max    = -std::numeric_limits<float>::infinity();
            float running_sum    = 0.0f;
            float acc[kValueDim] = {};
            float scale          = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

            for(int key_base = 0; key_base < kKeyCount; key_base += Candidate::kKeyTile)
            {
                float scores[Candidate::kKeyTile];
                float tile_max = -std::numeric_limits<float>::infinity();
                for(int key_offset = 0; key_offset < Candidate::kKeyTile; ++key_offset)
                {
                    int   key   = key_base + key_offset;
                    float score = 0.0f;
                    for(int dim = 0; dim < kHeadDim; ++dim)
                    {
                        float q = quantized_input_value(query, dim, /*seed=*/1);
                        float k = quantized_input_value(key, dim, /*seed=*/2);
                        score += q * k;
                    }
                    score *= scale;
                    scores[key_offset] = score;
                    tile_max           = std::max(tile_max, score);
                }

                float new_max   = running_sum == 0.0f ? tile_max : std::max(running_max, tile_max);
                float old_scale = running_sum == 0.0f ? 0.0f : std::exp(running_max - new_max);
                for(int value_col = 0; value_col < kValueDim; ++value_col)
                {
                    acc[value_col] *= old_scale;
                }

                float tile_sum = 0.0f;
                for(int key_offset = 0; key_offset < Candidate::kKeyTile; ++key_offset)
                {
                    int    key         = key_base + key_offset;
                    float  weight      = std::exp(scores[key_offset] - new_max);
                    half_t weight_half = static_cast<half_t>(weight);
                    float  weight_used = static_cast<float>(weight_half);
                    tile_sum += weight_used;
                    for(int value_col = 0; value_col < kValueDim; ++value_col)
                    {
                        float value = quantized_input_value(key, value_col, /*seed=*/3);
                        acc[value_col] += weight_used * value;
                    }
                }

                running_sum = running_sum * old_scale + tile_sum;
                running_max = new_max;
            }

            for(int value_col = 0; value_col < kValueDim; ++value_col)
            {
                out[query * kValueDim + value_col] = acc[value_col] / running_sum;
            }
        }
        return out;
    }

    template <typename Candidate, typename InstrumentationPolicy>
    void launch_attention(const f16x4_t*                device_q,
                          const f16x4_t*                device_k,
                          const f16x4_t*                device_v,
                          float*                        device_out,
                          hip_moi::context::storage_ref storage)
    {
        hipLaunchKernelGGL((attention_pressure_kernel<Candidate, InstrumentationPolicy>),
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_q,
                           device_k,
                           device_v,
                           device_out,
                           storage);
    }

    struct device_attention_buffers
    {
        f16x4_t* q   = nullptr;
        f16x4_t* k   = nullptr;
        f16x4_t* v   = nullptr;
        float*   out = nullptr;

        ~device_attention_buffers()
        {
            if(q)
            {
                (void)hipFree(q);
            }
            if(k)
            {
                (void)hipFree(k);
            }
            if(v)
            {
                (void)hipFree(v);
            }
            if(out)
            {
                (void)hipFree(out);
            }
        }
    };

    void allocate_and_copy_inputs(const std::vector<f16x4_t>& q,
                                  const std::vector<f16x4_t>& k,
                                  const std::vector<f16x4_t>& v,
                                  device_attention_buffers*   buffers)
    {
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->q, kQBytes));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->k, kKBytes));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->v, kVBytes));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->out, kOutputBytes));

        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(buffers->q, q.data(), kQBytes, hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(buffers->k, k.data(), kKBytes, hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(buffers->v, v.data(), kVBytes, hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(hipMemset(buffers->out, 0, kOutputBytes));
    }

    void copy_output(const device_attention_buffers& buffers, std::vector<float>* out)
    {
        out->assign(kOutputCount, 0.0f);
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(out->data(), buffers.out, kOutputBytes, hipMemcpyDeviceToHost));
    }

    void expect_matches_reference(const std::vector<float>& actual,
                                  const std::vector<float>& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for(std::size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_NEAR(actual[i], expected[i], 2.0e-2f) << "mismatch at element " << i;
        }
    }

    void make_host_fragments(std::vector<f16x4_t>* q,
                             std::vector<f16x4_t>* k,
                             std::vector<f16x4_t>* v)
    {
        q->resize(kQFragmentCount);
        k->resize(kKFragmentCount);
        v->resize(kVFragmentCount);
        fill_fragments(q, k, v);
    }

    template <typename Candidate>
    void run_exact_attention_pressure()
    {
        std::vector<f16x4_t> q;
        std::vector<f16x4_t> k;
        std::vector<f16x4_t> v;
        make_host_fragments(&q, &k, &v);
        std::vector<float> expected = host_attention_reference<Candidate>();

        device_attention_buffers buffers;
        ASSERT_NO_FATAL_FAILURE(allocate_and_copy_inputs(q, k, v, &buffers));

        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_exact_shadow(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*exact_shadow_entry_capacity=*/Candidate::kExactShadowEntries));

        launch_attention<Candidate, ExactContextPolicy>(
            buffers.q, buffers.k, buffers.v, buffers.out, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());
        ASSERT_NO_FATAL_FAILURE(hip_moi::test::expect_diagnostic_count(storage, 0));

        std::vector<float> actual;
        ASSERT_NO_FATAL_FAILURE(copy_output(buffers, &actual));
        expect_matches_reference(actual, expected);
    }

    template <typename Candidate>
    void run_sampled_fast_attention_pressure()
    {
        std::vector<f16x4_t> q;
        std::vector<f16x4_t> k;
        std::vector<f16x4_t> v;
        make_host_fragments(&q, &k, &v);
        std::vector<float> expected = host_attention_reference<Candidate>();

        device_attention_buffers buffers;
        ASSERT_NO_FATAL_FAILURE(allocate_and_copy_inputs(q, k, v, &buffers));

        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_sampled_watchpoints(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*sampled_watchpoint_capacity=*/kSampledWatchpoints));

        launch_attention<Candidate, FastContextPolicy>(
            buffers.q, buffers.k, buffers.v, buffers.out, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        std::vector<float> actual;
        ASSERT_NO_FATAL_FAILURE(copy_output(buffers, &actual));
        expect_matches_reference(actual, expected);
    }

    TEST(HipMoiCdna4D128AttentionPressure, FullKvDoubleBufferedExactContextMatchesHostReference)
    {
        run_exact_attention_pressure<FullKvDoubleBuffered16Key>();
    }

    TEST(HipMoiCdna4D128AttentionPressure, FullKvDoubleBufferedSampledFastMatchesHostReference)
    {
        run_sampled_fast_attention_pressure<FullKvDoubleBuffered16Key>();
    }

    TEST(HipMoiCdna4D128AttentionPressure, WideKeyDoubleBufferedExactContextMatchesHostReference)
    {
        run_exact_attention_pressure<DoubleBuffered32KeyPressure>();
    }

    TEST(HipMoiCdna4D128AttentionPressure, WideKeyDoubleBufferedSampledFastMatchesHostReference)
    {
        run_sampled_fast_attention_pressure<DoubleBuffered32KeyPressure>();
    }
} // namespace
