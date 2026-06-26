// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/032_workgroup_barrier_fence_test.hip
//
// Workgroup fence/barrier tests: the lower-level spelling
// release_fence/barrier/acquire_fence is an instrumented full-workgroup epoch
// boundary, while fences without a barrier do not advance the epoch.
//
#include "test_support.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kPayloadValue       = 313;

    hip_moi::host_context_options workgroup_barrier_fence_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes               = 64 * 1024;
        options.exact_shadow_entry_capacity = 16;
        options.sampled_watchpoint_capacity = 0;
        options.diagnostic_capacity         = 8;
        options.subgroup_capacity           = kSubgroupCount;
        return options;
    }

    __device__ hip_moi::context make_workgroup_barrier_fence_context(
        hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    template <bool UseBarrier>
    __global__ void workgroup_barrier_fence_handoff_kernel(hip_moi::context::storage_ref storage,
                                                           int*                          out)
    {
        __shared__ int payload;

        hip_moi::context ctx = make_workgroup_barrier_fence_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&payload,
                             kPayloadValue,
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
        }

        ctx.release_fence(hip_moi::atomic_memory_scope::workgroup, HIP_MOI_SITE_ID());
        if constexpr(UseBarrier)
        {
            ctx.barrier(HIP_MOI_SITE_ID());
        }
        ctx.acquire_fence(hip_moi::atomic_memory_scope::workgroup, HIP_MOI_SITE_ID());

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            out[0] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
        }
    }
} // namespace

TEST(HipMoiWorkgroupBarrierFence, ReleaseBarrierAcquireFenceOrdersPayload)
{
    hip_moi::host_context moi(workgroup_barrier_fence_options());
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));

    hipLaunchKernelGGL((workgroup_barrier_fence_handoff_kernel</*UseBarrier=*/true>),
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_output);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

    int output = 0;
    HIP_MOI_TEST_HIP_ASSERT(
        hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
    EXPECT_EQ(output, kPayloadValue);
    EXPECT_TRUE(moi.check(nullptr));

    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
}

TEST(HipMoiWorkgroupBarrierFence, FencesWithoutBarrierDoNotOrderPayload)
{
    hip_moi::host_context moi(workgroup_barrier_fence_options());
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));

    hipLaunchKernelGGL((workgroup_barrier_fence_handoff_kernel</*UseBarrier=*/false>),
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_output);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

    EXPECT_FALSE(moi.check(nullptr));
    EXPECT_GT(moi.last_diagnostic_count(), 0);

    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
}
