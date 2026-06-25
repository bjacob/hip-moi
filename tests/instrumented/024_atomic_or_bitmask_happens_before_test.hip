// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/024_atomic_or_bitmask_happens_before_test.hip
//
// Stage 6 atomics tests: bitmask RMW atomics can both publish a subgroup's
// arrival bit and acquire a sibling bit observed in the returned old value.
// This models the small old-value-dependent control-flow core of a Stream-K
// tree-style handoff.
//
#include "test_support.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kFirstBit           = 1;
    constexpr int kSecondBit          = 2;
    constexpr int kFirstPayload       = 41;
    constexpr int kSecondPayload      = 73;
    constexpr int kInitialOutput      = -1;

    hip_moi::host_context_options atomic_or_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes               = 64 * 1024;
        options.exact_shadow_entry_capacity = 16;
        options.sampled_watchpoint_capacity = 0;
        options.diagnostic_capacity         = 8;
        options.subgroup_capacity           = kSubgroupCount;
        return options;
    }

    __device__ hip_moi::context make_atomic_or_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    template <hip_moi::atomic_memory_order FirstOrder,
              hip_moi::atomic_memory_order SecondOrder>
    __global__ void atomic_or_bitmask_kernel(hip_moi::context::storage_ref storage,
                                             int*                          bitmask,
                                             int*                          out)
    {
        __shared__ int payloads[2];

        hip_moi::context ctx = make_atomic_or_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&payloads[0],
                             kFirstPayload,
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
            (void)ctx.atomic_fetch_or(bitmask,
                                      kFirstBit,
                                      FirstOrder,
                                      hip_moi::atomic_memory_scope::agent,
                                      HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            while((__hip_atomic_load(bitmask, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
                   & kFirstBit)
                  == 0)
            {
            }
            ctx.lds_store_at(&payloads[1],
                             kSecondPayload,
                             /*lds_byte_offset=*/sizeof(int),
                             HIP_MOI_SITE_ID());
            int old_mask = ctx.atomic_fetch_or(bitmask,
                                               kSecondBit,
                                               SecondOrder,
                                               hip_moi::atomic_memory_scope::agent,
                                               HIP_MOI_SITE_ID());
            if((old_mask & kFirstBit) != 0)
            {
                int first = ctx.lds_load_at(&payloads[0],
                                            /*lds_byte_offset=*/0,
                                            HIP_MOI_SITE_ID());
                int second = ctx.lds_load_at(&payloads[1],
                                             /*lds_byte_offset=*/sizeof(int),
                                             HIP_MOI_SITE_ID());
                out[0] = first + second + old_mask;
            }
        }
    }

    void reset_bitmask_and_output(int* device_bitmask, int* device_output)
    {
        int bitmask = 0;
        int output  = kInitialOutput;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_bitmask, &bitmask, sizeof(bitmask), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &output, sizeof(output), hipMemcpyHostToDevice));
    }
} // namespace

TEST(HipMoiAtomicOrBitmaskHappensBefore, AcqRelFetchOrOrdersOldValueSelectedPayload)
{
    hip_moi::host_context moi(atomic_or_options());
    int*                  device_bitmask = nullptr;
    int*                  device_output  = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_bitmask, sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_bitmask_and_output(device_bitmask, device_output);

    hipLaunchKernelGGL((atomic_or_bitmask_kernel<hip_moi::atomic_memory_order::release,
                                                 hip_moi::atomic_memory_order::acq_rel>),
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_bitmask,
                       device_output);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

    int output = 0;
    HIP_MOI_TEST_HIP_ASSERT(
        hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
    EXPECT_EQ(output, kFirstPayload + kSecondPayload + kFirstBit);
    EXPECT_TRUE(moi.check(nullptr));

    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_bitmask));
}

TEST(HipMoiAtomicOrBitmaskHappensBefore, RelaxedFetchOrDoesNotOrderOldValueSelectedPayload)
{
    hip_moi::host_context moi(atomic_or_options());
    int*                  device_bitmask = nullptr;
    int*                  device_output  = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_bitmask, sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_bitmask_and_output(device_bitmask, device_output);

    hipLaunchKernelGGL((atomic_or_bitmask_kernel<hip_moi::atomic_memory_order::relaxed,
                                                 hip_moi::atomic_memory_order::relaxed>),
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_bitmask,
                       device_output);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

    int output = 0;
    HIP_MOI_TEST_HIP_ASSERT(
        hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
    EXPECT_EQ(output, kFirstPayload + kSecondPayload + kFirstBit);
    EXPECT_FALSE(moi.check(nullptr));

    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_bitmask));
}
