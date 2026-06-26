// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/033_atomic_fence_extended_test.hip
//
// Extended fence-plus-atomic tests: release/acquire fences paired with relaxed
// exchange and compare-exchange operations can order an instrumented LDS payload
// handoff. This is the same source-level synchronization family as
// fence-plus-relaxed-store/load and fence-plus-relaxed-RMW.
//
#include "test_support.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kPayloadValue       = 409;
    constexpr int kInitialOutput      = -1;
    constexpr int kInitialFlag        = 0;
    constexpr int kPublishedFlag      = 1;
    constexpr int kConsumedFlag       = 2;
    constexpr int kLocked             = 1;
    constexpr int kUnlocked           = 0;

    hip_moi::host_context_options atomic_fence_extended_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes               = 64 * 1024;
        options.exact_shadow_entry_capacity = 16;
        options.sampled_watchpoint_capacity = 0;
        options.diagnostic_capacity         = 8;
        options.subgroup_capacity           = kSubgroupCount;
        return options;
    }

    __device__ hip_moi::context
               make_atomic_fence_extended_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    template <bool UseFences>
    __global__ void atomic_fence_exchange_handoff_kernel(hip_moi::context::storage_ref storage,
                                                         int*                          flag,
                                                         int*                          out)
    {
        __shared__ int payload;

        hip_moi::context ctx = make_atomic_fence_extended_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&payload,
                             kPayloadValue,
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
            if constexpr(UseFences)
            {
                ctx.atomic_fence(hip_moi::atomic_memory_order::release,
                                 hip_moi::atomic_memory_scope::agent,
                                 HIP_MOI_SITE_ID());
            }
            (void)ctx.atomic_exchange(flag,
                                      kPublishedFlag,
                                      hip_moi::atomic_memory_order::relaxed,
                                      hip_moi::atomic_memory_scope::agent,
                                      HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            while(__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
                  != kPublishedFlag)
            {
            }
            int old_value = ctx.atomic_exchange(flag,
                                                kConsumedFlag,
                                                hip_moi::atomic_memory_order::relaxed,
                                                hip_moi::atomic_memory_scope::agent,
                                                HIP_MOI_SITE_ID());
            if constexpr(UseFences)
            {
                ctx.atomic_fence(hip_moi::atomic_memory_order::acquire,
                                 hip_moi::atomic_memory_scope::agent,
                                 HIP_MOI_SITE_ID());
            }
            if(old_value == kPublishedFlag)
            {
                out[0] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            }
        }
    }

    template <bool UseFences>
    __global__ void atomic_fence_successful_cas_handoff_kernel(
        hip_moi::context::storage_ref storage, int* lock, int* out)
    {
        __shared__ int payload;

        hip_moi::context ctx = make_atomic_fence_extended_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&payload,
                             kPayloadValue,
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
            if constexpr(UseFences)
            {
                ctx.atomic_fence(hip_moi::atomic_memory_order::release,
                                 hip_moi::atomic_memory_scope::agent,
                                 HIP_MOI_SITE_ID());
            }
            ctx.atomic_store(lock,
                             kUnlocked,
                             hip_moi::atomic_memory_order::relaxed,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            int expected = kUnlocked;
            while(!ctx.atomic_compare_exchange_strong(lock,
                                                      &expected,
                                                      kLocked,
                                                      hip_moi::atomic_memory_order::relaxed,
                                                      hip_moi::atomic_memory_order::relaxed,
                                                      hip_moi::atomic_memory_scope::agent,
                                                      HIP_MOI_SITE_ID()))
            {
                expected = kUnlocked;
            }
            if constexpr(UseFences)
            {
                ctx.atomic_fence(hip_moi::atomic_memory_order::acquire,
                                 hip_moi::atomic_memory_scope::agent,
                                 HIP_MOI_SITE_ID());
            }
            out[0] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
        }
    }

    template <bool UseFences>
    __global__ void atomic_fence_failed_cas_handoff_kernel(hip_moi::context::storage_ref storage,
                                                           int*                          flag,
                                                           int*                          out)
    {
        __shared__ int payload;

        hip_moi::context ctx = make_atomic_fence_extended_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&payload,
                             kPayloadValue,
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
            if constexpr(UseFences)
            {
                ctx.atomic_fence(hip_moi::atomic_memory_order::release,
                                 hip_moi::atomic_memory_scope::agent,
                                 HIP_MOI_SITE_ID());
            }
            ctx.atomic_store(flag,
                             kPublishedFlag,
                             hip_moi::atomic_memory_order::relaxed,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            while(__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
                  != kPublishedFlag)
            {
            }
            int  expected = kInitialFlag;
            bool exchanged
                = ctx.atomic_compare_exchange_strong(flag,
                                                     &expected,
                                                     kConsumedFlag,
                                                     hip_moi::atomic_memory_order::relaxed,
                                                     hip_moi::atomic_memory_order::relaxed,
                                                     hip_moi::atomic_memory_scope::agent,
                                                     HIP_MOI_SITE_ID());
            if constexpr(UseFences)
            {
                ctx.atomic_fence(hip_moi::atomic_memory_order::acquire,
                                 hip_moi::atomic_memory_scope::agent,
                                 HIP_MOI_SITE_ID());
            }
            if(!exchanged && expected == kPublishedFlag)
            {
                out[0] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            }
        }
    }

    __global__ void
        atomic_seq_cst_handoff_kernel(hip_moi::context::storage_ref storage, int* flag, int* out)
    {
        __shared__ int payload;

        hip_moi::context ctx = make_atomic_fence_extended_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&payload,
                             kPayloadValue,
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
            ctx.atomic_store(flag,
                             kPublishedFlag,
                             hip_moi::atomic_memory_order::seq_cst,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            while(ctx.atomic_load(flag,
                                  hip_moi::atomic_memory_order::seq_cst,
                                  hip_moi::atomic_memory_scope::agent,
                                  HIP_MOI_SITE_ID())
                  != kPublishedFlag)
            {
            }
            out[0] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
        }
    }

    void reset_flag_and_output(int* device_flag, int flag_value, int* device_output)
    {
        int output = kInitialOutput;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_flag, &flag_value, sizeof(flag_value), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &output, sizeof(output), hipMemcpyHostToDevice));
    }

    template <typename Launch>
    void run_ordered_case(Launch launch, int initial_flag)
    {
        hip_moi::host_context moi(atomic_fence_extended_options());
        int*                  device_flag   = nullptr;
        int*                  device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flag, sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
        reset_flag_and_output(device_flag, initial_flag, device_output);

        launch(moi.launch_ref(), device_flag, device_output);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output = 0;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
        EXPECT_EQ(output, kPayloadValue);
        EXPECT_TRUE(moi.check(nullptr));

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_flag));
    }

    template <typename Launch>
    void run_unordered_case(Launch launch, int initial_flag)
    {
        hip_moi::host_context moi(atomic_fence_extended_options());
        int*                  device_flag   = nullptr;
        int*                  device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flag, sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
        reset_flag_and_output(device_flag, initial_flag, device_output);

        launch(moi.launch_ref(), device_flag, device_output);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output = 0;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
        EXPECT_EQ(output, kPayloadValue);
        EXPECT_FALSE(moi.check(nullptr));
        EXPECT_GT(moi.last_diagnostic_count(), 0);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_flag));
    }
} // namespace

TEST(HipMoiAtomicFenceExtended, ReleaseAcquireFencesOrderRelaxedExchangePayload)
{
    run_ordered_case(
        [](hip_moi::context::storage_ref storage, int* flag, int* output) {
            hipLaunchKernelGGL((atomic_fence_exchange_handoff_kernel</*UseFences=*/true>),
                               dim3(1),
                               dim3(kThreadCount),
                               0,
                               0,
                               storage,
                               flag,
                               output);
        },
        kInitialFlag);
}

TEST(HipMoiAtomicFenceExtended, RelaxedExchangeWithoutFencesDoesNotOrderPayload)
{
    run_unordered_case(
        [](hip_moi::context::storage_ref storage, int* flag, int* output) {
            hipLaunchKernelGGL((atomic_fence_exchange_handoff_kernel</*UseFences=*/false>),
                               dim3(1),
                               dim3(kThreadCount),
                               0,
                               0,
                               storage,
                               flag,
                               output);
        },
        kInitialFlag);
}

TEST(HipMoiAtomicFenceExtended, ReleaseAcquireFencesOrderSuccessfulRelaxedCasPayload)
{
    run_ordered_case(
        [](hip_moi::context::storage_ref storage, int* lock, int* output) {
            hipLaunchKernelGGL((atomic_fence_successful_cas_handoff_kernel</*UseFences=*/true>),
                               dim3(1),
                               dim3(kThreadCount),
                               0,
                               0,
                               storage,
                               lock,
                               output);
        },
        kLocked);
}

TEST(HipMoiAtomicFenceExtended, SuccessfulRelaxedCasWithoutFencesDoesNotOrderPayload)
{
    run_unordered_case(
        [](hip_moi::context::storage_ref storage, int* lock, int* output) {
            hipLaunchKernelGGL((atomic_fence_successful_cas_handoff_kernel</*UseFences=*/false>),
                               dim3(1),
                               dim3(kThreadCount),
                               0,
                               0,
                               storage,
                               lock,
                               output);
        },
        kLocked);
}

TEST(HipMoiAtomicFenceExtended, ReleaseAcquireFencesOrderFailedRelaxedCasPayload)
{
    run_ordered_case(
        [](hip_moi::context::storage_ref storage, int* flag, int* output) {
            hipLaunchKernelGGL((atomic_fence_failed_cas_handoff_kernel</*UseFences=*/true>),
                               dim3(1),
                               dim3(kThreadCount),
                               0,
                               0,
                               storage,
                               flag,
                               output);
        },
        kInitialFlag);
}

TEST(HipMoiAtomicFenceExtended, FailedRelaxedCasWithoutFencesDoesNotOrderPayload)
{
    run_unordered_case(
        [](hip_moi::context::storage_ref storage, int* flag, int* output) {
            hipLaunchKernelGGL((atomic_fence_failed_cas_handoff_kernel</*UseFences=*/false>),
                               dim3(1),
                               dim3(kThreadCount),
                               0,
                               0,
                               storage,
                               flag,
                               output);
        },
        kInitialFlag);
}

TEST(HipMoiAtomicFenceExtended, SeqCstStoreLoadOrdersPayload)
{
    run_ordered_case(
        [](hip_moi::context::storage_ref storage, int* flag, int* output) {
            hipLaunchKernelGGL(atomic_seq_cst_handoff_kernel,
                               dim3(1),
                               dim3(kThreadCount),
                               0,
                               0,
                               storage,
                               flag,
                               output);
        },
        kInitialFlag);
}
