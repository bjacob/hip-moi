// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/030_atomic_exchange_compare_exchange_test.hip
//
// Extended source-level atomics tests: exchange and compare-exchange are RMW
// operations that can carry release/acquire synchronization for LDS payload
// diagnostics.
//
#include "test_support.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kPayloadValue       = 101;
    constexpr int kInitialOutput      = -1;
    constexpr int kInitialFlag        = 0;
    constexpr int kPublishedFlag      = 1;
    constexpr int kConsumedFlag       = 2;
    constexpr int kLocked             = 1;
    constexpr int kUnlocked           = 0;

    hip_moi::host_context_options atomic_exchange_compare_exchange_options()
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
        make_atomic_exchange_compare_exchange_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    template <hip_moi::atomic_memory_order ProducerOrder,
              hip_moi::atomic_memory_order ConsumerOrder>
    __global__ void atomic_exchange_handoff_kernel(hip_moi::context::storage_ref storage,
                                                   int*                          flag,
                                                   int*                          out)
    {
        __shared__ int payload;

        hip_moi::context ctx = make_atomic_exchange_compare_exchange_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&payload,
                             kPayloadValue,
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
            (void)ctx.atomic_exchange(flag,
                                      kPublishedFlag,
                                      ProducerOrder,
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
                                                ConsumerOrder,
                                                hip_moi::atomic_memory_scope::agent,
                                                HIP_MOI_SITE_ID());
            if(old_value == kPublishedFlag)
            {
                out[0] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            }
        }
    }

    template <hip_moi::atomic_memory_order UnlockOrder,
              hip_moi::atomic_memory_order LockSuccessOrder>
    __global__ void atomic_compare_exchange_lock_handoff_kernel(
        hip_moi::context::storage_ref storage,
        int*                          lock,
        int*                          out)
    {
        __shared__ int payload;

        hip_moi::context ctx = make_atomic_exchange_compare_exchange_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&payload,
                             kPayloadValue,
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
            ctx.atomic_store(lock,
                             kUnlocked,
                             UnlockOrder,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            int expected = kUnlocked;
            while(!ctx.atomic_compare_exchange_strong(lock,
                                                      &expected,
                                                      kLocked,
                                                      LockSuccessOrder,
                                                      hip_moi::atomic_memory_order::relaxed,
                                                      hip_moi::atomic_memory_scope::agent,
                                                      HIP_MOI_SITE_ID()))
            {
                expected = kUnlocked;
            }
            out[0] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
        }
    }

    template <hip_moi::atomic_memory_order FailureOrder>
    __global__ void failed_compare_exchange_handoff_kernel(
        hip_moi::context::storage_ref storage,
        int*                          flag,
        int*                          out)
    {
        __shared__ int payload;

        hip_moi::context ctx = make_atomic_exchange_compare_exchange_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&payload,
                             kPayloadValue,
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
            ctx.atomic_store(flag,
                             kPublishedFlag,
                             hip_moi::atomic_memory_order::release,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            while(__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
                  != kPublishedFlag)
            {
            }
            int expected  = kInitialFlag;
            bool exchanged = ctx.atomic_compare_exchange_strong(flag,
                                                                &expected,
                                                                kConsumedFlag,
                                                                hip_moi::atomic_memory_order::relaxed,
                                                                FailureOrder,
                                                                hip_moi::atomic_memory_scope::agent,
                                                                HIP_MOI_SITE_ID());
            if(!exchanged && expected == kPublishedFlag)
            {
                out[0] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            }
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
} // namespace

TEST(HipMoiAtomicExchangeCompareExchange, ReleaseAcquireExchangeOrdersPayload)
{
    hip_moi::host_context moi(atomic_exchange_compare_exchange_options());
    int*                  device_flag   = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flag, sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_flag_and_output(device_flag, kInitialFlag, device_output);

    hipLaunchKernelGGL((atomic_exchange_handoff_kernel<hip_moi::atomic_memory_order::release,
                                                       hip_moi::atomic_memory_order::acquire>),
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_flag,
                       device_output);
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

TEST(HipMoiAtomicExchangeCompareExchange, RelaxedExchangeDoesNotOrderPayload)
{
    hip_moi::host_context moi(atomic_exchange_compare_exchange_options());
    int*                  device_flag   = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flag, sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_flag_and_output(device_flag, kInitialFlag, device_output);

    hipLaunchKernelGGL((atomic_exchange_handoff_kernel<hip_moi::atomic_memory_order::relaxed,
                                                       hip_moi::atomic_memory_order::relaxed>),
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_flag,
                       device_output);
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

TEST(HipMoiAtomicExchangeCompareExchange, SuccessfulAcquireCompareExchangeOrdersPayload)
{
    hip_moi::host_context moi(atomic_exchange_compare_exchange_options());
    int*                  device_lock   = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_lock, sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_flag_and_output(device_lock, kLocked, device_output);

    hipLaunchKernelGGL(
        (atomic_compare_exchange_lock_handoff_kernel<hip_moi::atomic_memory_order::release,
                                                     hip_moi::atomic_memory_order::acquire>),
        dim3(1),
        dim3(kThreadCount),
        0,
        0,
        moi.launch_ref(),
        device_lock,
        device_output);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

    int output = 0;
    HIP_MOI_TEST_HIP_ASSERT(
        hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
    EXPECT_EQ(output, kPayloadValue);
    EXPECT_TRUE(moi.check(nullptr));

    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_lock));
}

TEST(HipMoiAtomicExchangeCompareExchange, RelaxedCompareExchangeLockDoesNotOrderPayload)
{
    hip_moi::host_context moi(atomic_exchange_compare_exchange_options());
    int*                  device_lock   = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_lock, sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_flag_and_output(device_lock, kLocked, device_output);

    hipLaunchKernelGGL(
        (atomic_compare_exchange_lock_handoff_kernel<hip_moi::atomic_memory_order::relaxed,
                                                     hip_moi::atomic_memory_order::relaxed>),
        dim3(1),
        dim3(kThreadCount),
        0,
        0,
        moi.launch_ref(),
        device_lock,
        device_output);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

    int output = 0;
    HIP_MOI_TEST_HIP_ASSERT(
        hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
    EXPECT_EQ(output, kPayloadValue);
    EXPECT_FALSE(moi.check(nullptr));
    EXPECT_GT(moi.last_diagnostic_count(), 0);

    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_lock));
}

TEST(HipMoiAtomicExchangeCompareExchange, FailedAcquireCompareExchangeOrdersPayload)
{
    hip_moi::host_context moi(atomic_exchange_compare_exchange_options());
    int*                  device_flag   = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flag, sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_flag_and_output(device_flag, kInitialFlag, device_output);

    hipLaunchKernelGGL(
        (failed_compare_exchange_handoff_kernel<hip_moi::atomic_memory_order::acquire>),
        dim3(1),
        dim3(kThreadCount),
        0,
        0,
        moi.launch_ref(),
        device_flag,
        device_output);
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

TEST(HipMoiAtomicExchangeCompareExchange, FailedRelaxedCompareExchangeDoesNotOrderPayload)
{
    hip_moi::host_context moi(atomic_exchange_compare_exchange_options());
    int*                  device_flag   = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flag, sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_flag_and_output(device_flag, kInitialFlag, device_output);

    hipLaunchKernelGGL(
        (failed_compare_exchange_handoff_kernel<hip_moi::atomic_memory_order::relaxed>),
        dim3(1),
        dim3(kThreadCount),
        0,
        0,
        moi.launch_ref(),
        device_flag,
        device_output);
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
