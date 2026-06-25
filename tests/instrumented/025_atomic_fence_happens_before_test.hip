// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/025_atomic_fence_happens_before_test.hip
//
// Stage 8 atomics tests: a release fence sequenced before a relaxed atomic
// publication, paired with an acquire fence sequenced after a relaxed atomic
// observation, can order an instrumented LDS payload handoff.
//
#include "test_support.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kPayloadValue       = 83;
    constexpr int kInitialOutput      = -1;
    constexpr int kFlagReady          = 0;
    constexpr int kFlagWaiting        = 1;

    hip_moi::host_context_options atomic_fence_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes               = 64 * 1024;
        options.exact_shadow_entry_capacity = 16;
        options.sampled_watchpoint_capacity = 0;
        options.diagnostic_capacity         = 8;
        options.subgroup_capacity           = kSubgroupCount;
        return options;
    }

    __device__ hip_moi::context make_atomic_fence_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    template <bool UseFences>
    __global__ void atomic_fence_orders_lds_payload_kernel(hip_moi::context::storage_ref storage,
                                                           int*                          flag,
                                                           int*                          out)
    {
        __shared__ int payload;

        hip_moi::context ctx = make_atomic_fence_context(storage);
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
                             kFlagReady,
                             hip_moi::atomic_memory_order::relaxed,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            while(ctx.atomic_load(flag,
                                  hip_moi::atomic_memory_order::relaxed,
                                  hip_moi::atomic_memory_scope::agent,
                                  HIP_MOI_SITE_ID())
                  != kFlagReady)
            {
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

    void reset_flag_and_output(int* device_flag, int* device_output)
    {
        int flag   = kFlagWaiting;
        int output = kInitialOutput;
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(device_flag, &flag, sizeof(flag), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &output, sizeof(output), hipMemcpyHostToDevice));
    }
} // namespace

TEST(HipMoiAtomicFenceHappensBefore, ReleaseAcquireFencesOrderRelaxedFlagPayload)
{
    hip_moi::host_context moi(atomic_fence_options());
    int*                  device_flag   = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flag, sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_flag_and_output(device_flag, device_output);

    hipLaunchKernelGGL((atomic_fence_orders_lds_payload_kernel</*UseFences=*/true>),
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

TEST(HipMoiAtomicFenceHappensBefore, RelaxedFlagWithoutFencesDoesNotOrderPayload)
{
    hip_moi::host_context moi(atomic_fence_options());
    int*                  device_flag   = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flag, sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_flag_and_output(device_flag, device_output);

    hipLaunchKernelGGL((atomic_fence_orders_lds_payload_kernel</*UseFences=*/false>),
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
