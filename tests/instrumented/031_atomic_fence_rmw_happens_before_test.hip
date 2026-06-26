// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/031_atomic_fence_rmw_happens_before_test.hip
//
// Extended fence-plus-atomic tests: raw release/acquire fences can matter when
// they are paired with relaxed RMW atomics on the same synchronization object.
// Fence-only synchronization is still intentionally out of scope.
//
#include "test_support.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kFirstPayload       = 107;
    constexpr int kSecondPayload      = 211;
    constexpr int kInitialOutput      = -1;

    hip_moi::host_context_options atomic_fence_rmw_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes               = 64 * 1024;
        options.exact_shadow_entry_capacity = 16;
        options.sampled_watchpoint_capacity = 0;
        options.diagnostic_capacity         = 8;
        options.subgroup_capacity           = kSubgroupCount;
        return options;
    }

    __device__ hip_moi::context make_atomic_fence_rmw_context(
        hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    template <bool UseFences>
    __global__ void atomic_fence_rmw_orders_lds_payload_kernel(
        hip_moi::context::storage_ref storage,
        int*                          counter,
        int*                          out)
    {
        __shared__ int payloads[2];

        hip_moi::context ctx = make_atomic_fence_rmw_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&payloads[0],
                             kFirstPayload,
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
            if constexpr(UseFences)
            {
                ctx.atomic_fence(hip_moi::atomic_memory_order::release,
                                 hip_moi::atomic_memory_scope::agent,
                                 HIP_MOI_SITE_ID());
            }
            (void)ctx.atomic_fetch_add(counter,
                                       1,
                                       hip_moi::atomic_memory_order::relaxed,
                                       hip_moi::atomic_memory_scope::agent,
                                       HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            while(__hip_atomic_load(counter, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 1)
            {
            }
            ctx.lds_store_at(&payloads[1],
                             kSecondPayload,
                             /*lds_byte_offset=*/sizeof(int),
                             HIP_MOI_SITE_ID());
            int old_value = ctx.atomic_fetch_add(counter,
                                                 1,
                                                 hip_moi::atomic_memory_order::relaxed,
                                                 hip_moi::atomic_memory_scope::agent,
                                                 HIP_MOI_SITE_ID());
            if constexpr(UseFences)
            {
                ctx.atomic_fence(hip_moi::atomic_memory_order::acquire,
                                 hip_moi::atomic_memory_scope::agent,
                                 HIP_MOI_SITE_ID());
            }
            if(old_value == 1)
            {
                int first = ctx.lds_load_at(&payloads[0],
                                            /*lds_byte_offset=*/0,
                                            HIP_MOI_SITE_ID());
                int second = ctx.lds_load_at(&payloads[1],
                                             /*lds_byte_offset=*/sizeof(int),
                                             HIP_MOI_SITE_ID());
                out[0] = first + second;
            }
        }
    }

    void reset_counter_and_output(int* device_counter, int* device_output)
    {
        int counter = 0;
        int output  = kInitialOutput;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_counter, &counter, sizeof(counter), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &output, sizeof(output), hipMemcpyHostToDevice));
    }
} // namespace

TEST(HipMoiAtomicFenceRmwHappensBefore, ReleaseAcquireFencesOrderRelaxedRmwPayload)
{
    hip_moi::host_context moi(atomic_fence_rmw_options());
    int*                  device_counter = nullptr;
    int*                  device_output  = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_counter, sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_counter_and_output(device_counter, device_output);

    hipLaunchKernelGGL((atomic_fence_rmw_orders_lds_payload_kernel</*UseFences=*/true>),
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_counter,
                       device_output);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

    int output = 0;
    HIP_MOI_TEST_HIP_ASSERT(
        hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
    EXPECT_EQ(output, kFirstPayload + kSecondPayload);
    EXPECT_TRUE(moi.check(nullptr));

    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_counter));
}

TEST(HipMoiAtomicFenceRmwHappensBefore, RelaxedRmwWithoutFencesDoesNotOrderPayload)
{
    hip_moi::host_context moi(atomic_fence_rmw_options());
    int*                  device_counter = nullptr;
    int*                  device_output  = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_counter, sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_counter_and_output(device_counter, device_output);

    hipLaunchKernelGGL((atomic_fence_rmw_orders_lds_payload_kernel</*UseFences=*/false>),
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_counter,
                       device_output);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

    int output = 0;
    HIP_MOI_TEST_HIP_ASSERT(
        hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
    EXPECT_EQ(output, kFirstPayload + kSecondPayload);
    EXPECT_FALSE(moi.check(nullptr));
    EXPECT_GT(moi.last_diagnostic_count(), 0);

    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_counter));
}
