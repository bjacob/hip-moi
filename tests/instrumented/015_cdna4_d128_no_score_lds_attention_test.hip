// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/015_cdna4_d128_no_score_lds_attention_test.hip
//
// CDNA4-only D128/V128 attention-shaped correctness test using the QK-to-PV
// register handoff from 013. This is the D128 counterpart to 014: K/V fragments
// are staged through instrumented LDS, but dense score and weight LDS scratch
// are deliberately absent.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{
    using half_t  = _Float16;
    using f16x4_t = half_t __attribute__((ext_vector_type(4)));
    using f32x4_t = float __attribute__((ext_vector_type(4)));

    constexpr int   kWaveSize           = 64;
    constexpr int   kSubgroupCount      = 2;
    constexpr int   kThreadCount        = kWaveSize * kSubgroupCount;
    constexpr int   kThreadsPerSubgroup = kWaveSize;
    constexpr int   kQueryTile          = 16;
    constexpr int   kQueryCount         = kSubgroupCount * kQueryTile;
    constexpr int   kKeyTile            = 16;
    constexpr int   kKeyTileCount       = 2;
    constexpr int   kKeyCount           = kKeyTile * kKeyTileCount;
    constexpr int   kHeadDim            = 128;
    constexpr int   kValueDim           = 128;
    constexpr int   kFragmentDim        = 16;
    constexpr int   kFragmentElements   = 4;
    constexpr int   kHeadFragments      = kHeadDim / kFragmentDim;
    constexpr int   kValueFragments     = kValueDim / kFragmentDim;
    constexpr int   kDiagnosticCapacity = 16;
    constexpr int   kSampledWatchpoints = 128;
    constexpr float kQkScale            = 0.08838834764831845f;

    // The outer shape mirrors the D128/V128 production signal used by the
    // benchmark ladder. The CTest problem stays intentionally tiny.
    constexpr int kRepresentativeQueryHeads = 64;
    constexpr int kRepresentativeKvHeads    = 8;
    constexpr int kRepresentativeGqaRatio   = kRepresentativeQueryHeads / kRepresentativeKvHeads;

    static_assert(kRepresentativeGqaRatio == 8);
    static_assert(kHeadDim == kHeadFragments * kFragmentDim);
    static_assert(kValueDim == kValueFragments * kFragmentDim);

    constexpr std::size_t kQFragmentCount
        = kSubgroupCount * kHeadFragments * static_cast<std::size_t>(kWaveSize);
    constexpr std::size_t kKFragmentCount
        = kKeyTileCount * kHeadFragments * static_cast<std::size_t>(kWaveSize);
    constexpr std::size_t kVFragmentCount
        = kKeyTileCount * kValueFragments * static_cast<std::size_t>(kWaveSize);
    constexpr std::size_t kOutputCount = kQueryCount * kValueDim;

    constexpr std::size_t kQBytes      = kQFragmentCount * sizeof(f16x4_t);
    constexpr std::size_t kKBytes      = kKFragmentCount * sizeof(f16x4_t);
    constexpr std::size_t kVBytes      = kVFragmentCount * sizeof(f16x4_t);
    constexpr std::size_t kOutputBytes = kOutputCount * sizeof(float);

    constexpr uint32_t kKSharedByteOffset = 0;
    constexpr uint32_t kVSharedByteOffset = kWaveSize * sizeof(f16x4_t);
    constexpr int kSharedBytes = static_cast<int>(kVSharedByteOffset + kWaveSize * sizeof(f16x4_t));
    constexpr int kExactShadowEntries
        = (kSharedBytes + static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes) - 1)
          / static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes);

    constexpr hip_moi::site_id kKStoreSite(0xf11);
    constexpr hip_moi::site_id kVStoreSite(0xf12);
    constexpr hip_moi::site_id kKLoadSite(0xf13);
    constexpr hip_moi::site_id kVLoadSite(0xf14);

    using FastPublishOnlyPolicy = hip_moi::sampled_watchpoint_policy<
        /*SampleSkip=*/1,
        /*ProbeCount=*/1,
        /*DelayIters=*/0,
        /*ReportConflicts=*/false>;

    __host__ __device__ float input_value(int row, int col, int seed)
    {
        int value = (row * 17 + col * 11 + seed * 23 + (row ^ (col * 3))) % 13;
        return static_cast<float>(value - 6) * 0.0625f;
    }

    __host__ __device__ int cdna4_mfma_f16_k(int lane, int elem)
    {
        return elem + 4 * (lane >> 4);
    }

    __host__ __device__ int cdna4_mfma_acc_m(int lane, int elem)
    {
        return elem + 4 * (lane >> 4);
    }

    __host__ __device__ uint32_t k_fragment_offset(int lane)
    {
        return kKSharedByteOffset + static_cast<uint32_t>(lane * sizeof(f16x4_t));
    }

    __host__ __device__ uint32_t v_fragment_offset(int lane)
    {
        return kVSharedByteOffset + static_cast<uint32_t>(lane * sizeof(f16x4_t));
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

    __device__ float read_float_lane(float value, int source_lane)
    {
        uint32_t bits      = __builtin_bit_cast(uint32_t, value);
        uint32_t lane_bits = __builtin_amdgcn_ds_bpermute(source_lane * sizeof(uint32_t), bits);
        return __builtin_bit_cast(float, lane_bits);
    }

    __device__ float read_accumulator_element_from_lane(f32x4_t acc, int elem, int source_lane)
    {
        float value0 = read_float_lane(acc[0], source_lane);
        float value1 = read_float_lane(acc[1], source_lane);
        float value2 = read_float_lane(acc[2], source_lane);
        float value3 = read_float_lane(acc[3], source_lane);
        switch(elem)
        {
        case 0:
            return value0;
        case 1:
            return value1;
        case 2:
            return value2;
        default:
            return value3;
        }
    }

    __device__ f16x4_t qk_accumulator_to_pv_fragment(f32x4_t qk_acc, int lane)
    {
        f16x4_t p_fragment;
        int     query_row       = lane & 15;
        int     source_acc_elem = query_row & 3;
        int     source_lane_row = (query_row >> 2) * 16;
#pragma unroll
        for(int elem = 0; elem < kFragmentElements; ++elem)
        {
            int   local_key   = cdna4_mfma_f16_k(lane, elem);
            int   source_lane = source_lane_row + local_key;
            float qk_value
                = read_accumulator_element_from_lane(qk_acc, source_acc_elem, source_lane);
            p_fragment[elem] = static_cast<half_t>(qk_value * kQkScale);
        }
        return p_fragment;
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
    __global__ void d128_no_score_lds_attention_kernel(const f16x4_t* __restrict__ q_fragments,
                                                       const f16x4_t* __restrict__ k_fragments,
                                                       const f16x4_t* __restrict__ v_fragments,
                                                       float* __restrict__ out,
                                                       hip_moi::context::storage_ref storage)
    {
        __shared__ f16x4_t k_shared[kWaveSize];
        __shared__ f16x4_t v_shared[kWaveSize];

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

        for(int key_tile = 0; key_tile < kKeyTileCount; ++key_tile)
        {
            f32x4_t qk_acc = zero_f32x4();

#pragma unroll
            for(int head_fragment = 0; head_fragment < kHeadFragments; ++head_fragment)
            {
                if(subgroup == 0)
                {
                    std::size_t k_index
                        = (static_cast<std::size_t>(key_tile) * kHeadFragments + head_fragment)
                              * kWaveSize
                          + lane;
                    InstrumentationPolicy::store(ctx,
                                                 &k_shared[lane],
                                                 k_fragments[k_index],
                                                 k_fragment_offset(lane),
                                                 kKStoreSite);
                }

                ctx.syncthreads();

                std::size_t q_index
                    = (static_cast<std::size_t>(subgroup) * kHeadFragments + head_fragment)
                          * kWaveSize
                      + lane;
                f16x4_t q_fragment = q_fragments[q_index];
                f16x4_t k_fragment = InstrumentationPolicy::load(
                    ctx, &k_shared[lane], k_fragment_offset(lane), kKLoadSite);
                qk_acc = __builtin_amdgcn_mfma_f32_16x16x16f16(
                    q_fragment, k_fragment, qk_acc, 0, 0, 0);

                ctx.syncthreads();
            }

            f16x4_t p_fragment = qk_accumulator_to_pv_fragment(qk_acc, lane);

#pragma unroll
            for(int value_fragment = 0; value_fragment < kValueFragments; ++value_fragment)
            {
                if(subgroup == 0)
                {
                    std::size_t v_index
                        = (static_cast<std::size_t>(key_tile) * kValueFragments + value_fragment)
                              * kWaveSize
                          + lane;
                    InstrumentationPolicy::store(ctx,
                                                 &v_shared[lane],
                                                 v_fragments[v_index],
                                                 v_fragment_offset(lane),
                                                 kVStoreSite);
                }

                ctx.syncthreads();

                f16x4_t v_fragment = InstrumentationPolicy::load(
                    ctx, &v_shared[lane], v_fragment_offset(lane), kVLoadSite);
                f32x4_t pv_acc = zero_f32x4();
                pv_acc         = __builtin_amdgcn_mfma_f32_16x16x16f16(
                    p_fragment, v_fragment, pv_acc, 0, 0, 0);

#pragma unroll
                for(int elem = 0; elem < kFragmentElements; ++elem)
                {
                    out_acc[value_fragment][elem] += pv_acc[elem];
                }

                ctx.syncthreads();
            }
        }

        int value_column_in_fragment = lane & 15;
#pragma unroll
        for(int value_fragment = 0; value_fragment < kValueFragments; ++value_fragment)
        {
#pragma unroll
            for(int elem = 0; elem < kFragmentElements; ++elem)
            {
                int query_row    = cdna4_mfma_acc_m(lane, elem);
                int query        = subgroup * kQueryTile + query_row;
                int value_column = value_fragment * kFragmentDim + value_column_in_fragment;
                out[query * kValueDim + value_column] = out_acc[value_fragment][elem];
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

        for(int key_tile = 0; key_tile < kKeyTileCount; ++key_tile)
        {
            int key_base = key_tile * kKeyTile;
            for(int head_fragment = 0; head_fragment < kHeadFragments; ++head_fragment)
            {
                for(int lane = 0; lane < kWaveSize; ++lane)
                {
                    int key = key_base + (lane & 15);
                    for(int elem = 0; elem < kFragmentElements; ++elem)
                    {
                        int dim = head_fragment * kFragmentDim + cdna4_mfma_f16_k(lane, elem);
                        (*k_fragments)[(key_tile * kHeadFragments + head_fragment) * kWaveSize
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
                        (*v_fragments)[(key_tile * kValueFragments + value_fragment) * kWaveSize
                                       + lane][elem]
                            = static_cast<half_t>(input_value(key, value_column, /*seed=*/3));
                    }
                }
            }
        }
    }

    std::vector<float> host_reference()
    {
        std::vector<float> out(kOutputCount, 0.0f);
        for(int query = 0; query < kQueryCount; ++query)
        {
            for(int value_column = 0; value_column < kValueDim; ++value_column)
            {
                float acc = 0.0f;
                for(int key = 0; key < kKeyCount; ++key)
                {
                    float qk = 0.0f;
                    for(int dim = 0; dim < kHeadDim; ++dim)
                    {
                        float q = quantized_input_value(query, dim, /*seed=*/1);
                        float k = quantized_input_value(key, dim, /*seed=*/2);
                        qk += q * k;
                    }
                    half_t p     = static_cast<half_t>(qk * kQkScale);
                    float  value = quantized_input_value(key, value_column, /*seed=*/3);
                    acc += static_cast<float>(p) * value;
                }
                out[query * kValueDim + value_column] = acc;
            }
        }
        return out;
    }

    template <typename InstrumentationPolicy>
    void launch_attention(const f16x4_t*                device_q,
                          const f16x4_t*                device_k,
                          const f16x4_t*                device_v,
                          float*                        device_out,
                          hip_moi::context::storage_ref storage)
    {
        hipLaunchKernelGGL((d128_no_score_lds_attention_kernel<InstrumentationPolicy>),
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

    struct device_buffers
    {
        f16x4_t* q   = nullptr;
        f16x4_t* k   = nullptr;
        f16x4_t* v   = nullptr;
        float*   out = nullptr;

        ~device_buffers()
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
                                  device_buffers*             buffers)
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

    void copy_output(const device_buffers& buffers, std::vector<float>* out)
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
            EXPECT_NEAR(actual[i], expected[i], 1.0e-2f) << "mismatch at element " << i;
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

    template <typename InstrumentationPolicy>
    void run_attention_test(const std::vector<float>&              expected,
                            hip_moi::test::device_context_storage* storage)
    {
        std::vector<f16x4_t> q;
        std::vector<f16x4_t> k;
        std::vector<f16x4_t> v;
        make_host_fragments(&q, &k, &v);

        device_buffers buffers;
        ASSERT_NO_FATAL_FAILURE(allocate_and_copy_inputs(q, k, v, &buffers));

        launch_attention<InstrumentationPolicy>(
            buffers.q, buffers.k, buffers.v, buffers.out, storage->ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        std::vector<float> actual;
        ASSERT_NO_FATAL_FAILURE(copy_output(buffers, &actual));
        expect_matches_reference(actual, expected);
    }

    TEST(HipMoiCdna4D128NoScoreLdsAttention, ExactContextMatchesHostReference)
    {
        std::vector<float> expected = host_reference();

        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_exact_shadow(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*exact_shadow_entry_capacity=*/kExactShadowEntries));

        ASSERT_NO_FATAL_FAILURE(run_attention_test<ExactContextPolicy>(expected, &storage));
        ASSERT_NO_FATAL_FAILURE(hip_moi::test::expect_diagnostic_count(storage, 0));
    }

    TEST(HipMoiCdna4D128NoScoreLdsAttention, SampledFastContextMatchesHostReference)
    {
        std::vector<float> expected = host_reference();

        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_sampled_watchpoints(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*sampled_watchpoint_capacity=*/kSampledWatchpoints));

        ASSERT_NO_FATAL_FAILURE(run_attention_test<FastContextPolicy>(expected, &storage));
    }
} // namespace
