// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/021_atomic_happens_before_test.hip
//
// Stage 4 atomics tests: release/acquire atomic synchronization can order an
// instrumented LDS payload handoff. Broken ordering still diagnoses.
//
#include "test_support.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kPayloadValue       = 37;
    constexpr int kInitialOutput      = -1;
    constexpr int kFlagReady          = 0;
    constexpr int kFlagWaiting        = 1;

    hip_moi::host_context_options atomic_happens_before_options()
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
               make_atomic_happens_before_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    template <hip_moi::atomic_memory_order StoreOrder>
    __global__ void atomic_flag_orders_lds_payload_kernel(hip_moi::context::storage_ref storage,
                                                          int*                          flag,
                                                          int*                          out)
    {
        __shared__ int payload;

        hip_moi::context ctx = make_atomic_happens_before_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&payload,
                             kPayloadValue,
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
            ctx.atomic_store(flag,
                             kFlagReady,
                             StoreOrder,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            while(ctx.atomic_load(flag,
                                  hip_moi::atomic_memory_order::acquire,
                                  hip_moi::atomic_memory_scope::agent,
                                  HIP_MOI_SITE_ID())
                  != kFlagReady)
            {
            }
            out[0] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
        }
    }

    __global__ void wrong_flag_index_does_not_order_lds_payload_kernel(
        hip_moi::context::storage_ref storage, int* flags, int* out)
    {
        __shared__ int payload;

        hip_moi::context ctx = make_atomic_happens_before_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&payload,
                             kPayloadValue,
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
            ctx.atomic_store(&flags[0],
                             kFlagReady,
                             hip_moi::atomic_memory_order::release,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
            ctx.atomic_store(&flags[1],
                             kFlagReady,
                             hip_moi::atomic_memory_order::relaxed,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            while(ctx.atomic_load(&flags[1],
                                  hip_moi::atomic_memory_order::acquire,
                                  hip_moi::atomic_memory_scope::agent,
                                  HIP_MOI_SITE_ID())
                  != kFlagReady)
            {
            }
            out[0] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
        }
    }

    void reset_flag_and_output(int* device_flag, int* device_output)
    {
        int flag   = kFlagWaiting;
        int output = kInitialOutput;
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(device_flag, &flag, sizeof(flag), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &output, sizeof(output), hipMemcpyHostToDevice));
    }

    void reset_flags_and_output(int* device_flags, int* device_output)
    {
        int flags[2] = {kFlagWaiting, kFlagWaiting};
        int output   = kInitialOutput;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_flags, flags, sizeof(flags), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &output, sizeof(output), hipMemcpyHostToDevice));
    }

} // namespace

TEST(HipMoiAtomicHappensBefore, ReleaseAcquireOrdersInstrumentedLdsPayload)
{
    hip_moi::host_context moi(atomic_happens_before_options());
    int*                  device_flag   = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flag, sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_flag_and_output(device_flag, device_output);

    hipLaunchKernelGGL(
        (atomic_flag_orders_lds_payload_kernel<hip_moi::atomic_memory_order::release>),
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

TEST(HipMoiAtomicHappensBefore, RelaxedStoreDoesNotOrderInstrumentedLdsPayload)
{
    hip_moi::host_context moi(atomic_happens_before_options());
    int*                  device_flag   = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flag, sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_flag_and_output(device_flag, device_output);

    hipLaunchKernelGGL(
        (atomic_flag_orders_lds_payload_kernel<hip_moi::atomic_memory_order::relaxed>),
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

TEST(HipMoiAtomicHappensBefore, ReleasingDifferentFlagDoesNotOrderInstrumentedLdsPayload)
{
    hip_moi::host_context moi(atomic_happens_before_options());
    int*                  device_flags  = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flags, 2 * sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_flags_and_output(device_flags, device_output);

    hipLaunchKernelGGL(wrong_flag_index_does_not_order_lds_payload_kernel,
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_flags,
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
    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_flags));
}
