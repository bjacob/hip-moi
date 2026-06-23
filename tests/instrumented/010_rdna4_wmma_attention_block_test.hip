// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/010_rdna4_wmma_attention_block_test.hip
//
// RDNA4-only WMMA-heavy attention-shaped correctness tests. This grows the
// scalar attention test into the layout we want for a later benchmark: two
// subgroups compute a 32-query attention block, K/V are staged through LDS, QK
// and PV both use RDNA4 WMMA, and the results are compared with a host reference
// that mirrors the FP16 softmax-tile conversion used by the second WMMA.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace
{
    using half_t  = _Float16;
    using f16x8_t = half_t __attribute__((ext_vector_type(8)));
    using f32x8_t = float __attribute__((ext_vector_type(8)));

    constexpr int kWaveSize           = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kThreadCount        = kWaveSize * kSubgroupCount;
    constexpr int kThreadsPerSubgroup = kWaveSize;
    constexpr int kQueryTile          = 16;
    constexpr int kSeqLen             = kQueryTile * kSubgroupCount;
    constexpr int kKeyTile            = 16;
    constexpr int kHeadDim            = 16;
    constexpr int kValueDim           = 16;
    constexpr int kFragmentElements   = 8;
    constexpr int kDiagnosticCapacity = 16;
    constexpr int kSampledWatchpoints = 64;

    static_assert(kHeadDim == 16);
    static_assert(kValueDim == 16);
    static_assert(kSeqLen % kKeyTile == 0);

    constexpr int kKeyTileCount = kSeqLen / kKeyTile;

    constexpr std::size_t kQFragmentCount = kSubgroupCount * kWaveSize;
    constexpr std::size_t kKFragmentCount = kKeyTileCount * kWaveSize;
    constexpr std::size_t kVFragmentCount = kKeyTileCount * kWaveSize;
    constexpr std::size_t kOutputCount    = kSeqLen * kValueDim;

    constexpr std::size_t kQBytes      = kQFragmentCount * sizeof(f16x8_t);
    constexpr std::size_t kKBytes      = kKFragmentCount * sizeof(f16x8_t);
    constexpr std::size_t kVBytes      = kVFragmentCount * sizeof(f16x8_t);
    constexpr std::size_t kOutputBytes = kOutputCount * sizeof(float);

    constexpr uint32_t kKSharedByteOffset = 0;
    constexpr uint32_t kVSharedByteOffset = kWaveSize * sizeof(f16x8_t);
    constexpr int kSharedBytes = static_cast<int>(kVSharedByteOffset + kWaveSize * sizeof(f16x8_t));
    constexpr int kExactShadowEntries
        = (kSharedBytes + static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes) - 1)
          / static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes);

    constexpr hip_moi::site_id kKStoreSite(0xa01);
    constexpr hip_moi::site_id kVStoreSite(0xa02);
    constexpr hip_moi::site_id kKLoadSite(0xa03);
    constexpr hip_moi::site_id kVLoadSite(0xa04);

    using FastPublishOnlyPolicy = hip_moi::sampled_watchpoint_policy<
        /*SampleSkip=*/1,
        /*ProbeCount=*/1,
        /*DelayIters=*/0,
        /*ReportConflicts=*/false>;

    __host__ __device__ float input_value(int row, int col, int seed)
    {
        int value = (row * 17 + col * 11 + seed * 23 + (row ^ (col * 3))) % 13;
        return static_cast<float>(value - 6) * 0.125f;
    }

    __host__ __device__ int rdna4_wmma_f16_k(int lane, int idx)
    {
        int reg  = idx >> 1;
        int half = idx & 1;
        return half | ((reg & 1) << 1) | ((lane >> 4) << 2) | ((reg >> 1) << 3);
    }

    __host__ __device__ int rdna4_wmma_acc_m(int lane, int elem)
    {
        return elem + 8 * (lane >> 4);
    }

    __host__ __device__ uint32_t k_fragment_offset(int lane)
    {
        return kKSharedByteOffset + static_cast<uint32_t>(lane * sizeof(f16x8_t));
    }

    __host__ __device__ uint32_t v_fragment_offset(int lane)
    {
        return kVSharedByteOffset + static_cast<uint32_t>(lane * sizeof(f16x8_t));
    }

    __device__ f32x8_t zero_f32x8()
    {
        f32x8_t value;
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

    template <typename InstrumentationPolicy>
    __global__ void wmma_attention_block_kernel(const f16x8_t* __restrict__ q_fragments,
                                                const f16x8_t* __restrict__ k_fragments,
                                                const f16x8_t* __restrict__ v_fragments,
                                                float* __restrict__ out,
                                                hip_moi::context::storage_ref storage)
    {
        __shared__ f16x8_t k_shared[kWaveSize];
        __shared__ f16x8_t v_shared[kWaveSize];
        __shared__ float   scores[kSubgroupCount][kQueryTile][kKeyTile];
        __shared__ half_t  weights[kSubgroupCount][kQueryTile][kKeyTile];
        __shared__ float   row_old_scale[kSubgroupCount][kQueryTile];
        __shared__ float   row_sum[kSubgroupCount][kQueryTile];

        typename InstrumentationPolicy::context_type ctx
            = InstrumentationPolicy::make_context(storage);
        ctx.init_workgroup();

        int tid      = static_cast<int>(threadIdx.x);
        int subgroup = tid / kWaveSize;
        int lane     = tid & (kWaveSize - 1);

        f16x8_t q_fragment = q_fragments[subgroup * kWaveSize + lane];
        f32x8_t out_acc    = zero_f32x8();

        float running_max[kQueryTile];
        float running_sum[kQueryTile];
#pragma unroll
        for(int row = 0; row < kQueryTile; ++row)
        {
            running_max[row] = -3.402823466e38f;
            running_sum[row] = 0.0f;
        }

        for(int key_tile = 0; key_tile < kKeyTileCount; ++key_tile)
        {
            if(subgroup == 0)
            {
                InstrumentationPolicy::store(ctx,
                                             &k_shared[lane],
                                             k_fragments[key_tile * kWaveSize + lane],
                                             k_fragment_offset(lane),
                                             kKStoreSite);
                InstrumentationPolicy::store(ctx,
                                             &v_shared[lane],
                                             v_fragments[key_tile * kWaveSize + lane],
                                             v_fragment_offset(lane),
                                             kVStoreSite);
            }

            ctx.syncthreads();

            f16x8_t k_fragment = InstrumentationPolicy::load(
                ctx, &k_shared[lane], k_fragment_offset(lane), kKLoadSite);
            f32x8_t qk_acc = zero_f32x8();
            qk_acc
                = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(q_fragment, k_fragment, qk_acc);

            int key_column = lane & 15;
#pragma unroll
            for(int elem = 0; elem < kFragmentElements; ++elem)
            {
                int query_row                           = rdna4_wmma_acc_m(lane, elem);
                scores[subgroup][query_row][key_column] = qk_acc[elem] * 0.25f;
            }

            ctx.syncthreads();

            if(lane == 0)
            {
                for(int query_row = 0; query_row < kQueryTile; ++query_row)
                {
                    float tile_max = -3.402823466e38f;
                    for(int key = 0; key < kKeyTile; ++key)
                    {
                        tile_max = fmaxf(tile_max, scores[subgroup][query_row][key]);
                    }

                    float new_max                      = fmaxf(running_max[query_row], tile_max);
                    float old_scale                    = running_sum[query_row] == 0.0f
                                                             ? 0.0f
                                                             : expf(running_max[query_row] - new_max);
                    row_old_scale[subgroup][query_row] = old_scale;

                    float tile_sum = 0.0f;
                    for(int key = 0; key < kKeyTile; ++key)
                    {
                        float  weight      = expf(scores[subgroup][query_row][key] - new_max);
                        half_t weight_half = static_cast<half_t>(weight);
                        weights[subgroup][query_row][key] = weight_half;
                        tile_sum += static_cast<float>(weight_half);
                    }

                    running_sum[query_row]       = running_sum[query_row] * old_scale + tile_sum;
                    running_max[query_row]       = new_max;
                    row_sum[subgroup][query_row] = running_sum[query_row];
                }
            }

            ctx.syncthreads();

#pragma unroll
            for(int elem = 0; elem < kFragmentElements; ++elem)
            {
                int query_row = rdna4_wmma_acc_m(lane, elem);
                out_acc[elem] *= row_old_scale[subgroup][query_row];
            }

            f16x8_t p_fragment;
#pragma unroll
            for(int elem = 0; elem < kFragmentElements; ++elem)
            {
                int query_row    = lane & 15;
                int key          = rdna4_wmma_f16_k(lane, elem);
                p_fragment[elem] = weights[subgroup][query_row][key];
            }

            f16x8_t v_fragment = InstrumentationPolicy::load(
                ctx, &v_shared[lane], v_fragment_offset(lane), kVLoadSite);
            f32x8_t pv_acc = zero_f32x8();
            pv_acc
                = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(p_fragment, v_fragment, pv_acc);

#pragma unroll
            for(int elem = 0; elem < kFragmentElements; ++elem)
            {
                out_acc[elem] += pv_acc[elem];
            }

            ctx.syncthreads();
        }

        int value_column = lane & 15;
#pragma unroll
        for(int elem = 0; elem < kFragmentElements; ++elem)
        {
            int query_row    = rdna4_wmma_acc_m(lane, elem);
            int global_query = subgroup * kQueryTile + query_row;
            out[global_query * kValueDim + value_column]
                = out_acc[elem] / row_sum[subgroup][query_row];
        }
    }

    float quantized_input_value(int row, int col, int seed)
    {
        return static_cast<float>(static_cast<half_t>(input_value(row, col, seed)));
    }

    void fill_fragments(std::vector<f16x8_t>* q_fragments,
                        std::vector<f16x8_t>* k_fragments,
                        std::vector<f16x8_t>* v_fragments)
    {
        for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
        {
            int query_base = subgroup * kQueryTile;
            for(int lane = 0; lane < kWaveSize; ++lane)
            {
                int query_row = query_base + (lane & 15);
                for(int elem = 0; elem < kFragmentElements; ++elem)
                {
                    int dim = rdna4_wmma_f16_k(lane, elem);
                    (*q_fragments)[subgroup * kWaveSize + lane][elem]
                        = static_cast<half_t>(input_value(query_row, dim, /*seed=*/1));
                }
            }
        }

        for(int key_tile = 0; key_tile < kKeyTileCount; ++key_tile)
        {
            int key_base = key_tile * kKeyTile;
            for(int lane = 0; lane < kWaveSize; ++lane)
            {
                int key = key_base + (lane & 15);
                int n   = lane & 15;
                for(int elem = 0; elem < kFragmentElements; ++elem)
                {
                    int dim = rdna4_wmma_f16_k(lane, elem);
                    (*k_fragments)[key_tile * kWaveSize + lane][elem]
                        = static_cast<half_t>(input_value(key, dim, /*seed=*/2));
                    (*v_fragments)[key_tile * kWaveSize + lane][elem]
                        = static_cast<half_t>(input_value(key_base + dim, n, /*seed=*/3));
                }
            }
        }
    }

    std::vector<float> host_attention_reference()
    {
        std::vector<float> out(kOutputCount, 0.0f);
        for(int query = 0; query < kSeqLen; ++query)
        {
            float running_max    = -std::numeric_limits<float>::infinity();
            float running_sum    = 0.0f;
            float acc[kValueDim] = {};

            for(int key_base = 0; key_base < kSeqLen; key_base += kKeyTile)
            {
                float scores[kKeyTile];
                float tile_max = -std::numeric_limits<float>::infinity();
                for(int key_offset = 0; key_offset < kKeyTile; ++key_offset)
                {
                    int   key   = key_base + key_offset;
                    float score = 0.0f;
                    for(int dim = 0; dim < kHeadDim; ++dim)
                    {
                        float q = quantized_input_value(query, dim, /*seed=*/1);
                        float k = quantized_input_value(key, dim, /*seed=*/2);
                        score += q * k;
                    }
                    score *= 0.25f;
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
                for(int key_offset = 0; key_offset < kKeyTile; ++key_offset)
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

    template <typename InstrumentationPolicy>
    void launch_attention(const f16x8_t*                device_q,
                          const f16x8_t*                device_k,
                          const f16x8_t*                device_v,
                          float*                        device_out,
                          hip_moi::context::storage_ref storage)
    {
        hipLaunchKernelGGL((wmma_attention_block_kernel<InstrumentationPolicy>),
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
        f16x8_t* q   = nullptr;
        f16x8_t* k   = nullptr;
        f16x8_t* v   = nullptr;
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

    void allocate_and_copy_inputs(const std::vector<f16x8_t>& q,
                                  const std::vector<f16x8_t>& k,
                                  const std::vector<f16x8_t>& v,
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
            EXPECT_NEAR(actual[i], expected[i], 3.0e-3f) << "mismatch at element " << i;
        }
    }

    void make_host_fragments(std::vector<f16x8_t>* q,
                             std::vector<f16x8_t>* k,
                             std::vector<f16x8_t>* v)
    {
        q->resize(kQFragmentCount);
        k->resize(kKFragmentCount);
        v->resize(kVFragmentCount);
        fill_fragments(q, k, v);
    }

    TEST(HipMoiRdna4WmmaAttentionBlock, ExactContextMatchesHostReference)
    {
        std::vector<f16x8_t> q;
        std::vector<f16x8_t> k;
        std::vector<f16x8_t> v;
        make_host_fragments(&q, &k, &v);
        std::vector<float> expected = host_attention_reference();

        device_attention_buffers buffers;
        ASSERT_NO_FATAL_FAILURE(allocate_and_copy_inputs(q, k, v, &buffers));

        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_exact_shadow(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*exact_shadow_entry_capacity=*/kExactShadowEntries));

        launch_attention<ExactContextPolicy>(
            buffers.q, buffers.k, buffers.v, buffers.out, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());
        ASSERT_NO_FATAL_FAILURE(hip_moi::test::expect_diagnostic_count(storage, 0));

        std::vector<float> actual;
        ASSERT_NO_FATAL_FAILURE(copy_output(buffers, &actual));
        expect_matches_reference(actual, expected);
    }

    TEST(HipMoiRdna4WmmaAttentionBlock, SampledFastContextMatchesHostReference)
    {
        std::vector<f16x8_t> q;
        std::vector<f16x8_t> k;
        std::vector<f16x8_t> v;
        make_host_fragments(&q, &k, &v);
        std::vector<float> expected = host_attention_reference();

        device_attention_buffers buffers;
        ASSERT_NO_FATAL_FAILURE(allocate_and_copy_inputs(q, k, v, &buffers));

        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_sampled_watchpoints(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*sampled_watchpoint_capacity=*/kSampledWatchpoints));

        launch_attention<FastContextPolicy>(
            buffers.q, buffers.k, buffers.v, buffers.out, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        std::vector<float> actual;
        ASSERT_NO_FATAL_FAILURE(copy_output(buffers, &actual));
        expect_matches_reference(actual, expected);
    }
} // namespace
