// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/009_attention_block_test.hip
//
// Attention-shaped correctness tests for hip-moi instrumentation. This is not a
// performance benchmark: it keeps scalar math so the Q/K/V tiled LDS dataflow is
// easy to audit before the benchmark version grows into a WMMA-heavy kernel.
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
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kSeqLen             = 16;
    constexpr int kHeadDim            = 16;
    constexpr int kValueDim           = 16;
    constexpr int kTileKeys           = 8;
    constexpr int kDiagnosticCapacity = 16;
    constexpr int kSampledWatchpoints = 64;

    constexpr std::size_t kQBytes      = kSeqLen * kHeadDim * sizeof(float);
    constexpr std::size_t kKBytes      = kSeqLen * kHeadDim * sizeof(float);
    constexpr std::size_t kVBytes      = kSeqLen * kValueDim * sizeof(float);
    constexpr std::size_t kOutputBytes = kSeqLen * kValueDim * sizeof(float);

    constexpr uint32_t kKTileByteOffset = 0;
    constexpr uint32_t kVTileByteOffset = kTileKeys * kHeadDim * sizeof(float);
    constexpr int      kSharedBytes
        = static_cast<int>(kVTileByteOffset + kTileKeys * kValueDim * sizeof(float));
    constexpr int kExactShadowEntries
        = (kSharedBytes + static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes) - 1)
          / static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes);

    constexpr hip_moi::site_id kKStoreSite(0x901);
    constexpr hip_moi::site_id kVStoreSite(0x902);
    constexpr hip_moi::site_id kKLoadSite(0x903);
    constexpr hip_moi::site_id kVLoadSite(0x904);

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

    __host__ __device__ uint32_t k_tile_offset(int tile_key, int dim)
    {
        return kKTileByteOffset
               + static_cast<uint32_t>((tile_key * kHeadDim + dim) * sizeof(float));
    }

    __host__ __device__ uint32_t v_tile_offset(int tile_key, int dim)
    {
        return kVTileByteOffset
               + static_cast<uint32_t>((tile_key * kValueDim + dim) * sizeof(float));
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
    __global__ void attention_block_kernel(const float* __restrict__ q,
                                           const float* __restrict__ k,
                                           const float* __restrict__ v,
                                           float* __restrict__ out,
                                           hip_moi::context::storage_ref storage)
    {
        __shared__ float k_tile[kTileKeys][kHeadDim];
        __shared__ float v_tile[kTileKeys][kValueDim];

        typename InstrumentationPolicy::context_type ctx
            = InstrumentationPolicy::make_context(storage);
        ctx.init_workgroup();

        int             thread = static_cast<int>(threadIdx.x);
        constexpr float kScale = 0.25f;

        for(int query = 0; query < kSeqLen; ++query)
        {
            float running_max    = -3.402823466e38f;
            float running_sum    = 0.0f;
            float acc[kValueDim] = {};

            for(int key_base = 0; key_base < kSeqLen; key_base += kTileKeys)
            {
                for(int index = thread; index < kTileKeys * kHeadDim; index += kThreadCount)
                {
                    int   tile_key = index / kHeadDim;
                    int   dim      = index - tile_key * kHeadDim;
                    int   key      = key_base + tile_key;
                    float value    = k[key * kHeadDim + dim];
                    InstrumentationPolicy::store(ctx,
                                                 &k_tile[tile_key][dim],
                                                 value,
                                                 k_tile_offset(tile_key, dim),
                                                 kKStoreSite);
                }

                for(int index = thread; index < kTileKeys * kValueDim; index += kThreadCount)
                {
                    int   tile_key = index / kValueDim;
                    int   dim      = index - tile_key * kValueDim;
                    int   key      = key_base + tile_key;
                    float value    = v[key * kValueDim + dim];
                    InstrumentationPolicy::store(ctx,
                                                 &v_tile[tile_key][dim],
                                                 value,
                                                 v_tile_offset(tile_key, dim),
                                                 kVStoreSite);
                }

                ctx.syncthreads();

                if(thread == 0)
                {
                    float scores[kTileKeys];
                    float tile_max = -3.402823466e38f;
                    for(int tile_key = 0; tile_key < kTileKeys; ++tile_key)
                    {
                        float score = 0.0f;
                        for(int dim = 0; dim < kHeadDim; ++dim)
                        {
                            float k_value
                                = InstrumentationPolicy::load(ctx,
                                                              &k_tile[tile_key][dim],
                                                              k_tile_offset(tile_key, dim),
                                                              kKLoadSite);
                            score += q[query * kHeadDim + dim] * k_value;
                        }
                        score *= kScale;
                        scores[tile_key] = score;
                        tile_max         = fmaxf(tile_max, score);
                    }

                    float new_max   = fmaxf(running_max, tile_max);
                    float old_scale = running_sum == 0.0f ? 0.0f : expf(running_max - new_max);
                    for(int dim = 0; dim < kValueDim; ++dim)
                    {
                        acc[dim] *= old_scale;
                    }

                    float tile_sum = 0.0f;
                    for(int tile_key = 0; tile_key < kTileKeys; ++tile_key)
                    {
                        float weight = expf(scores[tile_key] - new_max);
                        tile_sum += weight;
                        for(int dim = 0; dim < kValueDim; ++dim)
                        {
                            float v_value
                                = InstrumentationPolicy::load(ctx,
                                                              &v_tile[tile_key][dim],
                                                              v_tile_offset(tile_key, dim),
                                                              kVLoadSite);
                            acc[dim] += weight * v_value;
                        }
                    }

                    running_sum = running_sum * old_scale + tile_sum;
                    running_max = new_max;
                }

                ctx.syncthreads();
            }

            if(thread == 0)
            {
                for(int dim = 0; dim < kValueDim; ++dim)
                {
                    out[query * kValueDim + dim] = acc[dim] / running_sum;
                }
            }
        }
    }

    void fill_inputs(std::vector<float>* q, std::vector<float>* k, std::vector<float>* v)
    {
        for(int row = 0; row < kSeqLen; ++row)
        {
            for(int dim = 0; dim < kHeadDim; ++dim)
            {
                (*q)[row * kHeadDim + dim] = input_value(row, dim, /*seed=*/1);
                (*k)[row * kHeadDim + dim] = input_value(row, dim, /*seed=*/2);
            }
            for(int dim = 0; dim < kValueDim; ++dim)
            {
                (*v)[row * kValueDim + dim] = input_value(row, dim, /*seed=*/3);
            }
        }
    }

    std::vector<float> host_attention_reference(const std::vector<float>& q,
                                                const std::vector<float>& k,
                                                const std::vector<float>& v)
    {
        std::vector<float> out(kSeqLen * kValueDim, 0.0f);
        constexpr float    kScale = 0.25f;
        for(int query = 0; query < kSeqLen; ++query)
        {
            float scores[kSeqLen];
            float max_score = -std::numeric_limits<float>::infinity();
            for(int key = 0; key < kSeqLen; ++key)
            {
                float score = 0.0f;
                for(int dim = 0; dim < kHeadDim; ++dim)
                {
                    score += q[query * kHeadDim + dim] * k[key * kHeadDim + dim];
                }
                score *= kScale;
                scores[key] = score;
                max_score   = std::max(max_score, score);
            }

            float denom = 0.0f;
            for(int key = 0; key < kSeqLen; ++key)
            {
                float weight = std::exp(scores[key] - max_score);
                denom += weight;
                for(int dim = 0; dim < kValueDim; ++dim)
                {
                    out[query * kValueDim + dim] += weight * v[key * kValueDim + dim];
                }
            }

            for(int dim = 0; dim < kValueDim; ++dim)
            {
                out[query * kValueDim + dim] /= denom;
            }
        }
        return out;
    }

    template <typename InstrumentationPolicy>
    void launch_attention(const float*                  device_q,
                          const float*                  device_k,
                          const float*                  device_v,
                          float*                        device_out,
                          hip_moi::context::storage_ref storage)
    {
        hipLaunchKernelGGL((attention_block_kernel<InstrumentationPolicy>),
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

    void expect_matches_reference(const std::vector<float>& actual,
                                  const std::vector<float>& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for(std::size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_NEAR(actual[i], expected[i], 2.0e-4f) << "mismatch at element " << i;
        }
    }

    struct device_attention_buffers
    {
        float* q   = nullptr;
        float* k   = nullptr;
        float* v   = nullptr;
        float* out = nullptr;

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

    void allocate_and_copy_inputs(const std::vector<float>& q,
                                  const std::vector<float>& k,
                                  const std::vector<float>& v,
                                  device_attention_buffers* buffers)
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
        out->assign(kSeqLen * kValueDim, 0.0f);
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(out->data(), buffers.out, kOutputBytes, hipMemcpyDeviceToHost));
    }

    TEST(HipMoiAttentionBlock, ExactContextMatchesHostReference)
    {
        std::vector<float> q(kSeqLen * kHeadDim);
        std::vector<float> k(kSeqLen * kHeadDim);
        std::vector<float> v(kSeqLen * kValueDim);
        fill_inputs(&q, &k, &v);
        std::vector<float> expected = host_attention_reference(q, k, v);

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

    TEST(HipMoiAttentionBlock, SampledFastContextMatchesHostReference)
    {
        std::vector<float> q(kSeqLen * kHeadDim);
        std::vector<float> k(kSeqLen * kHeadDim);
        std::vector<float> v(kSeqLen * kValueDim);
        fill_inputs(&q, &k, &v);
        std::vector<float> expected = host_attention_reference(q, k, v);

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
