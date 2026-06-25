// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/016_rdna4_pingpong_private_lds_test.hip
//
// RDNA4-only ping-pong-shaped WMMA test where each subgroup stages A and B
// fragments in its own LDS slots. This isolates setprio/sched_barrier codegen
// from cooperative LDS sharing: the LDS traffic is real and fully
// instrumented, but no subgroup consumes another subgroup's LDS data.
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
    using f16x8_t = half_t __attribute__((ext_vector_type(8)));
    using f32x8_t = float __attribute__((ext_vector_type(8)));

    constexpr int kWaveSize           = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kThreadCount        = kWaveSize * kSubgroupCount;
    constexpr int kThreadsPerSubgroup = kWaveSize;
    constexpr int kTileM              = 16;
    constexpr int kTileN              = 16;
    constexpr int kKTile              = 16;
    constexpr int kKTileCount         = 4;
    constexpr int kBufferCount        = 2;
    constexpr int kFragmentElements   = 8;
    constexpr int kDiagnosticCapacity = 16;
    constexpr int kSampledWatchpoints = 64;

    constexpr std::size_t kAFragmentCount
        = kSubgroupCount * kKTileCount * static_cast<std::size_t>(kWaveSize);
    constexpr std::size_t kBFragmentCount
        = kSubgroupCount * kKTileCount * static_cast<std::size_t>(kWaveSize);
    constexpr std::size_t kOutputCount = kSubgroupCount * kTileM * kTileN;

    constexpr std::size_t kABytes      = kAFragmentCount * sizeof(f16x8_t);
    constexpr std::size_t kBBytes      = kBFragmentCount * sizeof(f16x8_t);
    constexpr std::size_t kOutputBytes = kOutputCount * sizeof(float);

    constexpr uint32_t kBSharedByteOffset
        = kBufferCount * kSubgroupCount * kWaveSize * sizeof(f16x8_t);
    constexpr int kSharedBytes
        = static_cast<int>(kBSharedByteOffset
                           + kBufferCount * kSubgroupCount * kWaveSize * sizeof(f16x8_t));
    constexpr int kExactShadowEntries
        = (kSharedBytes + static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes) - 1)
          / static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes);

    constexpr hip_moi::site_id kAStoreSite(0x1001);
    constexpr hip_moi::site_id kBStoreSite(0x1002);
    constexpr hip_moi::site_id kALoadSite(0x1003);
    constexpr hip_moi::site_id kBLoadSite(0x1004);

    using FastPublishOnlyPolicy = hip_moi::sampled_watchpoint_policy<
        /*SampleSkip=*/1,
        /*ProbeCount=*/1,
        /*DelayIters=*/0,
        /*ReportConflicts=*/false>;

    __host__ __device__ float input_value(int row, int col, int seed)
    {
        int value = (row * 7 + col * 5 + seed * 13 + (row ^ (col * 3))) % 9;
        return static_cast<float>(value - 4) * 0.125f;
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

    __host__ __device__ std::size_t fragment_index(int subgroup, int k_tile, int lane)
    {
        return (static_cast<std::size_t>(k_tile) * kSubgroupCount + subgroup) * kWaveSize + lane;
    }

    __host__ __device__ uint32_t a_shared_offset(int buffer, int subgroup, int lane)
    {
        return static_cast<uint32_t>(
            ((buffer * kSubgroupCount + subgroup) * kWaveSize + lane) * sizeof(f16x8_t));
    }

    __host__ __device__ uint32_t b_shared_offset(int buffer, int subgroup, int lane)
    {
        return kBSharedByteOffset
               + static_cast<uint32_t>(
                   ((buffer * kSubgroupCount + subgroup) * kWaveSize + lane) * sizeof(f16x8_t));
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

    __device__ __forceinline__ void set_pingpong_priority(int subgroup, int k_tile)
    {
        if(((subgroup + k_tile) & 1) == 0)
        {
            __builtin_amdgcn_s_setprio(1);
        }
        else
        {
            __builtin_amdgcn_s_setprio(0);
        }
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
    __global__ void private_pingpong_kernel(const f16x8_t* __restrict__ a_fragments,
                                            const f16x8_t* __restrict__ b_fragments,
                                            float* __restrict__ out,
                                            hip_moi::context::storage_ref storage)
    {
        __shared__ f16x8_t a_shared[kBufferCount][kSubgroupCount][kWaveSize];
        __shared__ f16x8_t b_shared[kBufferCount][kSubgroupCount][kWaveSize];

        typename InstrumentationPolicy::context_type ctx
            = InstrumentationPolicy::make_context(storage);
        ctx.init_workgroup();

        int tid      = static_cast<int>(threadIdx.x);
        int subgroup = tid / kWaveSize;
        int lane     = tid & (kWaveSize - 1);

        f32x8_t acc = zero_f32x8();
#pragma unroll
        for(int k_tile = 0; k_tile < kKTileCount; ++k_tile)
        {
            int         buffer = k_tile & (kBufferCount - 1);
            std::size_t index  = fragment_index(subgroup, k_tile, lane);

            set_pingpong_priority(subgroup, k_tile);
            InstrumentationPolicy::store(ctx,
                                         &a_shared[buffer][subgroup][lane],
                                         a_fragments[index],
                                         a_shared_offset(buffer, subgroup, lane),
                                         kAStoreSite);
            InstrumentationPolicy::store(ctx,
                                         &b_shared[buffer][subgroup][lane],
                                         b_fragments[index],
                                         b_shared_offset(buffer, subgroup, lane),
                                         kBStoreSite);

            __builtin_amdgcn_sched_barrier(0);

            f16x8_t a_fragment = InstrumentationPolicy::load(ctx,
                                                             &a_shared[buffer][subgroup][lane],
                                                             a_shared_offset(buffer, subgroup, lane),
                                                             kALoadSite);
            f16x8_t b_fragment = InstrumentationPolicy::load(ctx,
                                                             &b_shared[buffer][subgroup][lane],
                                                             b_shared_offset(buffer, subgroup, lane),
                                                             kBLoadSite);

            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(0);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                a_fragment, b_fragment, acc);
        }

        __builtin_amdgcn_s_setprio(0);

        int col = lane & 15;
#pragma unroll
        for(int elem = 0; elem < kFragmentElements; ++elem)
        {
            int row = subgroup * kTileM + rdna4_wmma_acc_m(lane, elem);
            out[row * kTileN + col] = acc[elem];
        }
    }

    void fill_fragments(std::vector<f16x8_t>* a_fragments, std::vector<f16x8_t>* b_fragments)
    {
        a_fragments->resize(kAFragmentCount);
        b_fragments->resize(kBFragmentCount);
        for(int k_tile = 0; k_tile < kKTileCount; ++k_tile)
        {
            for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
            {
                for(int lane = 0; lane < kWaveSize; ++lane)
                {
                    int row = subgroup * kTileM + (lane & 15);
                    int col = lane & 15;
                    for(int elem = 0; elem < kFragmentElements; ++elem)
                    {
                        int k = k_tile * kKTile + rdna4_wmma_f16_k(lane, elem);
                        (*a_fragments)[fragment_index(subgroup, k_tile, lane)][elem]
                            = static_cast<half_t>(input_value(row, k, /*seed=*/1));
                        (*b_fragments)[fragment_index(subgroup, k_tile, lane)][elem]
                            = static_cast<half_t>(input_value(k, col, /*seed=*/2 + subgroup));
                    }
                }
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
        for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
        {
            for(int m = 0; m < kTileM; ++m)
            {
                int row = subgroup * kTileM + m;
                for(int n = 0; n < kTileN; ++n)
                {
                    float acc = 0.0f;
                    for(int k = 0; k < kKTileCount * kKTile; ++k)
                    {
                        float a = quantized_input_value(row, k, /*seed=*/1);
                        float b = quantized_input_value(k, n, /*seed=*/2 + subgroup);
                        acc += a * b;
                    }
                    expected[row * kTileN + n] = acc;
                }
            }
        }
        return expected;
    }

    struct device_buffers
    {
        f16x8_t* a   = nullptr;
        f16x8_t* b   = nullptr;
        float*   out = nullptr;

        ~device_buffers()
        {
            if(a)
            {
                (void)hipFree(a);
            }
            if(b)
            {
                (void)hipFree(b);
            }
            if(out)
            {
                (void)hipFree(out);
            }
        }
    };

    void allocate_and_copy_inputs(const std::vector<f16x8_t>& a,
                                  const std::vector<f16x8_t>& b,
                                  device_buffers*             buffers)
    {
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->a, kABytes));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->b, kBBytes));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->out, kOutputBytes));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(buffers->a, a.data(), kABytes, hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(buffers->b, b.data(), kBBytes, hipMemcpyHostToDevice));
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

    template <typename InstrumentationPolicy>
    void run_private_pingpong_test(const std::vector<float>&              expected,
                                   hip_moi::context::storage_ref         storage_ref,
                                   hip_moi::test::device_context_storage* storage)
    {
        std::vector<f16x8_t> a_fragments;
        std::vector<f16x8_t> b_fragments;
        fill_fragments(&a_fragments, &b_fragments);

        device_buffers buffers;
        ASSERT_NO_FATAL_FAILURE(allocate_and_copy_inputs(a_fragments, b_fragments, &buffers));

        hipLaunchKernelGGL((private_pingpong_kernel<InstrumentationPolicy>),
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           buffers.a,
                           buffers.b,
                           buffers.out,
                           storage_ref);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        std::vector<float> actual;
        ASSERT_NO_FATAL_FAILURE(copy_output(buffers, &actual));
        expect_matches_reference(actual, expected);

        if(storage)
        {
            ASSERT_NO_FATAL_FAILURE(hip_moi::test::expect_diagnostic_count(*storage, 0));
        }
    }

    TEST(HipMoiRdna4PingpongPrivateLds, ExactContextMatchesHostReference)
    {
        std::vector<float> expected = host_reference();

        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_exact_shadow(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*exact_shadow_entry_capacity=*/kExactShadowEntries));

        ASSERT_NO_FATAL_FAILURE(
            run_private_pingpong_test<ExactContextPolicy>(expected, storage.ref(), &storage));
    }

    TEST(HipMoiRdna4PingpongPrivateLds, SampledFastContextMatchesHostReference)
    {
        std::vector<float> expected = host_reference();

        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_sampled_watchpoints(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*sampled_watchpoint_capacity=*/kSampledWatchpoints));

        ASSERT_NO_FATAL_FAILURE(
            run_private_pingpong_test<FastContextPolicy>(expected, storage.ref(), nullptr));
    }
} // namespace
