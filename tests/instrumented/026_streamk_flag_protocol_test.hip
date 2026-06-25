// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/026_streamk_flag_protocol_test.hip
//
// Stage 9 atomics test: a RocJITsu hip-stream-k-style owner/helper flag
// protocol, distilled to one owner subgroup that loops over two helper flags
// and folds the helpers' LDS partial payloads.
//
#include "test_support.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 3;
    constexpr int kThreadCount        = kThreadsPerSubgroup * kSubgroupCount;
    constexpr int kHelperCount        = 2;
    constexpr int kFlagReady          = 0;
    constexpr int kFlagWaiting        = 1;
    constexpr int kInitialOutput      = -1;

    __host__ __device__ int owner_partial_value()
    {
        return 19;
    }

    __host__ __device__ int helper_partial_value(int helper)
    {
        return 31 + helper * 17;
    }

    __host__ __device__ int expected_streamk_sum()
    {
        int sum = owner_partial_value();
        for(int helper = 0; helper < kHelperCount; ++helper)
        {
            sum += helper_partial_value(helper);
        }
        return sum;
    }

    hip_moi::host_context_options streamk_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes               = 64 * 1024;
        options.exact_shadow_entry_capacity = 16;
        options.sampled_watchpoint_capacity = 0;
        options.diagnostic_capacity         = 8;
        options.subgroup_capacity           = kSubgroupCount;
        return options;
    }

    __device__ hip_moi::context make_streamk_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    template <bool BreakSecondHelper>
    __global__ void streamk_flag_fixup_kernel(hip_moi::context::storage_ref storage,
                                              int*                          flags,
                                              int*                          out)
    {
        __shared__ int partials[1 + kHelperCount];

        hip_moi::context ctx = make_streamk_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&partials[0],
                             owner_partial_value(),
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
        }

        for(int helper = 0; helper < kHelperCount; ++helper)
        {
            int helper_tid = (helper + 1) * kThreadsPerSubgroup;
            if(ctx.thread_id() == static_cast<uint32_t>(helper_tid))
            {
                uint32_t partial_offset = static_cast<uint32_t>((helper + 1) * sizeof(int));
                ctx.lds_store_at(&partials[helper + 1],
                                 helper_partial_value(helper),
                                 partial_offset,
                                 HIP_MOI_SITE_ID());
                hip_moi::atomic_memory_order order
                    = (BreakSecondHelper && helper == 1) ? hip_moi::atomic_memory_order::relaxed
                                                         : hip_moi::atomic_memory_order::release;
                ctx.atomic_store(&flags[helper],
                                 kFlagReady,
                                 order,
                                 hip_moi::atomic_memory_scope::agent,
                                 HIP_MOI_SITE_ID());
            }
        }

        if(ctx.thread_id() == 0)
        {
            int sum = ctx.lds_load_at(&partials[0], /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            for(int helper = 0; helper < kHelperCount; ++helper)
            {
                while(ctx.atomic_load(&flags[helper],
                                      hip_moi::atomic_memory_order::acquire,
                                      hip_moi::atomic_memory_scope::agent,
                                      HIP_MOI_SITE_ID())
                      != kFlagReady)
                {
                }
                uint32_t partial_offset = static_cast<uint32_t>((helper + 1) * sizeof(int));
                sum += ctx.lds_load_at(&partials[helper + 1],
                                       partial_offset,
                                       HIP_MOI_SITE_ID());
            }
            out[0] = sum;
        }
    }

    void reset_flags_and_output(int* device_flags, int* device_output)
    {
        int flags[kHelperCount] = {kFlagWaiting, kFlagWaiting};
        int output              = kInitialOutput;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_flags, flags, sizeof(flags), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &output, sizeof(output), hipMemcpyHostToDevice));
    }
} // namespace

TEST(HipMoiStreamKFlagProtocol, ReleaseAcquireHelperFlagsOrderStreamKFixupPayloads)
{
    hip_moi::host_context moi(streamk_options());
    int*                  device_flags  = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flags, kHelperCount * sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_flags_and_output(device_flags, device_output);

    hipLaunchKernelGGL((streamk_flag_fixup_kernel</*BreakSecondHelper=*/false>),
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
    EXPECT_EQ(output, expected_streamk_sum());
    EXPECT_TRUE(moi.check(nullptr));

    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_flags));
}

TEST(HipMoiStreamKFlagProtocol, RelaxedSecondHelperFlagDoesNotOrderStreamKFixupPayload)
{
    hip_moi::host_context moi(streamk_options());
    int*                  device_flags  = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flags, kHelperCount * sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
    reset_flags_and_output(device_flags, device_output);

    hipLaunchKernelGGL((streamk_flag_fixup_kernel</*BreakSecondHelper=*/true>),
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
    EXPECT_EQ(output, expected_streamk_sum());
    EXPECT_FALSE(moi.check(nullptr));
    EXPECT_GT(moi.last_diagnostic_count(), 0);

    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_flags));
}
