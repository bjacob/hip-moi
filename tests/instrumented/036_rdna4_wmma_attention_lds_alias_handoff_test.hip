// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/036_rdna4_wmma_attention_lds_alias_handoff_test.hip
//
// RDNA4-only WMMA-shaped version of the attention LDS alias handoff. Subgroup 0
// stages a tail-K fragment tile in LDS, subgroup 1 consumes it through WMMA,
// and subgroup 0 then writes the next logical tile into the same LDS slot. The
// aliased handoff needs a full-workgroup barrier between the consume and
// prefetch phases.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstddef>
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
    constexpr int kTileM              = 16;
    constexpr int kTileN              = 16;
    constexpr int kKTile              = 16;
    constexpr int kFragmentElements   = 8;
    constexpr int kDiagnosticCapacity = 32;
    constexpr int kSampledWatchpoints = 32;

    constexpr std::size_t kQFragmentCount = kWaveSize;
    constexpr std::size_t kKFragmentCount = kWaveSize;
    constexpr std::size_t kVFragmentCount = kWaveSize;
    constexpr std::size_t kOutputCount    = kTileM * kTileN;

    constexpr std::size_t kQBytes      = kQFragmentCount * sizeof(f16x8_t);
    constexpr std::size_t kKBytes      = kKFragmentCount * sizeof(f16x8_t);
    constexpr std::size_t kVBytes      = kVFragmentCount * sizeof(f16x8_t);
    constexpr std::size_t kOutputBytes = kOutputCount * sizeof(float);

    constexpr int kSharedBytes = kWaveSize * static_cast<int>(sizeof(f16x8_t));
    constexpr int kExactShadowEntries
        = (kSharedBytes + static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes) - 1)
          / static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes);

    constexpr hip_moi::site_id kTailKStoreSite(0x3601);
    constexpr hip_moi::site_id kTailKLoadSite(0x3602);
    constexpr hip_moi::site_id kNextVStoreSite(0x3603);

    using FastPublishOnlyPolicy = hip_moi::sampled_watchpoint_policy<
        /*SampleSkip=*/1,
        /*ProbeCount=*/1,
        /*DelayIters=*/0,
        /*ReportConflicts=*/false>;

    __host__ __device__ float input_value(int row, int col, int seed)
    {
        int value = (row * 7 + col * 5 + seed * 13 + (row ^ (col * 3))) % 9;
        return static_cast<float>(value - 4) * 0.25f;
    }

    __host__ __device__ int rdna4_wmma_f16_k(int lane, int elem)
    {
        int reg  = elem >> 1;
        int half = elem & 1;
        return half | ((reg & 1) << 1) | ((lane >> 4) << 2) | ((reg >> 1) << 3);
    }

    __host__ __device__ int rdna4_wmma_acc_m(int lane, int elem)
    {
        return elem + 8 * (lane >> 4);
    }

    __host__ __device__ uint32_t fragment_offset(int lane)
    {
        return static_cast<uint32_t>(lane * sizeof(f16x8_t));
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
                /*thread_count=*/kThreadCount,
                /*threads_per_subgroup=*/kThreadsPerSubgroup,
                /*subgroup_count=*/kSubgroupCount,
            };
            return hip_moi::context(storage, cfg);
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
            ctx.template lds_store_at<FastPublishOnlyPolicy>(ptr, value, offset, site);
        }

        template <typename T>
        __device__ static T
            load(context_type& ctx, const T* ptr, uint32_t offset, hip_moi::site_id site)
        {
            return ctx.template lds_load_at<FastPublishOnlyPolicy>(ptr, offset, site);
        }
    };

    template <typename InstrumentationPolicy, bool InsertHandoffBarrier>
    __global__ void wmma_attention_alias_handoff_kernel(const f16x8_t* __restrict__ q_fragments,
                                                        const f16x8_t* __restrict__ k_fragments,
                                                        const f16x8_t* __restrict__ v_fragments,
                                                        float* __restrict__ out,
                                                        hip_moi::context::storage_ref storage)
    {
        __shared__ f16x8_t aliased_slot[kWaveSize];

        typename InstrumentationPolicy::context_type ctx
            = InstrumentationPolicy::make_context(storage);
        InstrumentationPolicy::init_workgroup(ctx);

        int tid      = static_cast<int>(threadIdx.x);
        int subgroup = tid / kWaveSize;
        int lane     = tid & (kWaveSize - 1);

        if(subgroup == 0)
        {
            InstrumentationPolicy::store(ctx,
                                         &aliased_slot[lane],
                                         k_fragments[lane],
                                         fragment_offset(lane),
                                         kTailKStoreSite);
        }
        InstrumentationPolicy::syncthreads(ctx);

        f32x8_t acc = zero_f32x8();
        if(subgroup == 1)
        {
            f16x8_t tail_k = InstrumentationPolicy::load(
                ctx, &aliased_slot[lane], fragment_offset(lane), kTailKLoadSite);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(q_fragments[lane], tail_k, acc);
        }

        if constexpr(InsertHandoffBarrier)
        {
            InstrumentationPolicy::syncthreads(ctx);
        }

        if(subgroup == 0)
        {
            InstrumentationPolicy::store(ctx,
                                         &aliased_slot[lane],
                                         v_fragments[lane],
                                         fragment_offset(lane),
                                         kNextVStoreSite);
        }
        InstrumentationPolicy::syncthreads(ctx);

        if(subgroup == 1)
        {
            int col = lane & 15;
#pragma unroll
            for(int elem = 0; elem < kFragmentElements; ++elem)
            {
                int row                 = rdna4_wmma_acc_m(lane, elem);
                out[row * kTileN + col] = acc[elem];
            }
        }
    }

    void fill_fragments(std::vector<f16x8_t>* q, std::vector<f16x8_t>* k, std::vector<f16x8_t>* v)
    {
        q->resize(kQFragmentCount);
        k->resize(kKFragmentCount);
        v->resize(kVFragmentCount);
        for(int lane = 0; lane < kWaveSize; ++lane)
        {
            int row = lane & 15;
            int col = lane & 15;
            for(int elem = 0; elem < kFragmentElements; ++elem)
            {
                int k_index      = rdna4_wmma_f16_k(lane, elem);
                (*q)[lane][elem] = static_cast<half_t>(input_value(row, k_index, /*seed=*/1));
                (*k)[lane][elem] = static_cast<half_t>(input_value(k_index, col, /*seed=*/2));
                (*v)[lane][elem] = static_cast<half_t>(input_value(k_index, col, /*seed=*/3));
            }
        }
    }

    float quantized_input_value(int row, int col, int seed)
    {
        return static_cast<float>(static_cast<half_t>(input_value(row, col, seed)));
    }

    std::vector<float> host_reference()
    {
        std::vector<float> expected(kOutputCount, 0.0f);
        for(int m = 0; m < kTileM; ++m)
        {
            for(int n = 0; n < kTileN; ++n)
            {
                float acc = 0.0f;
                for(int k = 0; k < kKTile; ++k)
                {
                    acc += quantized_input_value(m, k, /*seed=*/1)
                           * quantized_input_value(k, n, /*seed=*/2);
                }
                expected[m * kTileN + n] = acc;
            }
        }
        return expected;
    }

    struct device_buffers
    {
        f16x8_t* q   = nullptr;
        f16x8_t* k   = nullptr;
        f16x8_t* v   = nullptr;
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

    void allocate_and_copy_inputs(const std::vector<f16x8_t>& q,
                                  const std::vector<f16x8_t>& k,
                                  const std::vector<f16x8_t>& v,
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
            EXPECT_NEAR(actual[i], expected[i], 1.0e-4f) << "mismatch at element " << i;
        }
    }

    void expect_diagnostic_count_gt_zero(const hip_moi::test::device_context_storage& storage)
    {
        int count = -1;
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(
            &count, storage.diagnostic_count_device(), sizeof(count), hipMemcpyDeviceToHost));
        EXPECT_GT(count, 0);
    }

    template <typename InstrumentationPolicy, bool InsertHandoffBarrier>
    void launch_alias_handoff(const std::vector<float>*              expected,
                              hip_moi::context::storage_ref          storage_ref,
                              hip_moi::test::device_context_storage* storage)
    {
        std::vector<f16x8_t> q;
        std::vector<f16x8_t> k;
        std::vector<f16x8_t> v;
        fill_fragments(&q, &k, &v);

        device_buffers buffers;
        ASSERT_NO_FATAL_FAILURE(allocate_and_copy_inputs(q, k, v, &buffers));

        hipLaunchKernelGGL(
            (wmma_attention_alias_handoff_kernel<InstrumentationPolicy, InsertHandoffBarrier>),
            dim3(1),
            dim3(kThreadCount),
            0,
            0,
            buffers.q,
            buffers.k,
            buffers.v,
            buffers.out,
            storage_ref);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        if(expected)
        {
            std::vector<float> actual;
            ASSERT_NO_FATAL_FAILURE(copy_output(buffers, &actual));
            expect_matches_reference(actual, *expected);
        }

        if(storage && InsertHandoffBarrier)
        {
            ASSERT_NO_FATAL_FAILURE(hip_moi::test::expect_diagnostic_count(*storage, 0));
        }
    }
} // namespace

TEST(HipMoiRdna4WmmaAttentionLdsAliasHandoff, SynchronizedExactContextMatchesHostReference)
{
    std::vector<float> expected = host_reference();

    hip_moi::test::device_context_storage storage;
    ASSERT_NO_FATAL_FAILURE(storage.allocate_with_exact_shadow(
        /*diagnostic_capacity=*/kDiagnosticCapacity,
        /*subgroup_capacity=*/kSubgroupCount,
        /*exact_shadow_entry_capacity=*/kExactShadowEntries));

    ASSERT_NO_FATAL_FAILURE(
        (launch_alias_handoff<ExactContextPolicy, /*InsertHandoffBarrier=*/true>(
            &expected, storage.ref(), &storage)));
}

TEST(HipMoiRdna4WmmaAttentionLdsAliasHandoff, SynchronizedSampledFastContextMatchesHostReference)
{
    std::vector<float> expected = host_reference();

    hip_moi::test::device_context_storage storage;
    ASSERT_NO_FATAL_FAILURE(storage.allocate_with_sampled_watchpoints(
        /*diagnostic_capacity=*/kDiagnosticCapacity,
        /*subgroup_capacity=*/kSubgroupCount,
        /*sampled_watchpoint_capacity=*/kSampledWatchpoints));

    ASSERT_NO_FATAL_FAILURE((launch_alias_handoff<FastContextPolicy, /*InsertHandoffBarrier=*/true>(
        &expected, storage.ref(), nullptr)));
}

TEST(HipMoiRdna4WmmaAttentionLdsAliasHandoff, UnsynchronizedExactContextReportsAliasRace)
{
    hip_moi::test::device_context_storage storage;
    ASSERT_NO_FATAL_FAILURE(storage.allocate_with_exact_shadow(
        /*diagnostic_capacity=*/kDiagnosticCapacity,
        /*subgroup_capacity=*/kSubgroupCount,
        /*exact_shadow_entry_capacity=*/kExactShadowEntries));

    ASSERT_NO_FATAL_FAILURE(
        (launch_alias_handoff<ExactContextPolicy, /*InsertHandoffBarrier=*/false>(
            nullptr, storage.ref(), &storage)));

    ASSERT_NO_FATAL_FAILURE(expect_diagnostic_count_gt_zero(storage));
}
