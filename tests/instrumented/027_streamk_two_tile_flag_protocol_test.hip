// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/027_streamk_two_tile_flag_protocol_test.hip
//
// Stage 9 atomics test: a two-tile Stream-K-style flag protocol. Two owner
// subgroups each wait on one helper flag and fold one helper LDS partial.
//
#include "test_support.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kTileCount          = 2;
    constexpr int kSubgroupCount      = 2 * kTileCount;
    constexpr int kThreadCount        = kThreadsPerSubgroup * kSubgroupCount;
    constexpr int kFlagReady          = 0;
    constexpr int kFlagWaiting        = 1;
    constexpr int kInitialOutput      = -1;

    __host__ __device__ int owner_partial_value(int tile)
    {
        return 23 + tile * 101;
    }

    __host__ __device__ int helper_partial_value(int tile)
    {
        return 47 + tile * 103;
    }

    __host__ __device__ int expected_tile_sum(int tile)
    {
        return owner_partial_value(tile) + helper_partial_value(tile);
    }

    hip_moi::host_context_options streamk_two_tile_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes               = 64 * 1024;
        options.exact_shadow_entry_capacity = 16;
        options.sampled_watchpoint_capacity = 0;
        options.diagnostic_capacity         = 8;
        options.subgroup_capacity           = kSubgroupCount;
        return options;
    }

    __device__ hip_moi::context make_streamk_two_tile_context(
        hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    template <bool BreakSecondTile>
    __global__ void streamk_two_tile_flag_fixup_kernel(hip_moi::context::storage_ref storage,
                                                       int*                          flags,
                                                       int*                          out)
    {
        __shared__ int partials[kSubgroupCount];

        hip_moi::context ctx = make_streamk_two_tile_context(storage);
        ctx.init_workgroup();

        for(int tile = 0; tile < kTileCount; ++tile)
        {
            int owner_tid  = (2 * tile) * kThreadsPerSubgroup;
            int helper_tid = (2 * tile + 1) * kThreadsPerSubgroup;

            if(ctx.thread_id() == static_cast<uint32_t>(owner_tid))
            {
                uint32_t owner_offset = static_cast<uint32_t>((2 * tile) * sizeof(int));
                ctx.lds_store_at(&partials[2 * tile],
                                 owner_partial_value(tile),
                                 owner_offset,
                                 HIP_MOI_SITE_ID());
            }

            if(ctx.thread_id() == static_cast<uint32_t>(helper_tid))
            {
                uint32_t helper_offset = static_cast<uint32_t>((2 * tile + 1) * sizeof(int));
                ctx.lds_store_at(&partials[2 * tile + 1],
                                 helper_partial_value(tile),
                                 helper_offset,
                                 HIP_MOI_SITE_ID());
                hip_moi::atomic_memory_order order
                    = (BreakSecondTile && tile == 1) ? hip_moi::atomic_memory_order::relaxed
                                                     : hip_moi::atomic_memory_order::release;
                ctx.atomic_store(&flags[tile],
                                 kFlagReady,
                                 order,
                                 hip_moi::atomic_memory_scope::agent,
                                 HIP_MOI_SITE_ID());
            }
        }

        for(int tile = 0; tile < kTileCount; ++tile)
        {
            int owner_tid = (2 * tile) * kThreadsPerSubgroup;
            if(ctx.thread_id() == static_cast<uint32_t>(owner_tid))
            {
                uint32_t owner_offset = static_cast<uint32_t>((2 * tile) * sizeof(int));
                int      sum
                    = ctx.lds_load_at(&partials[2 * tile], owner_offset, HIP_MOI_SITE_ID());
                while(ctx.atomic_load(&flags[tile],
                                      hip_moi::atomic_memory_order::acquire,
                                      hip_moi::atomic_memory_scope::agent,
                                      HIP_MOI_SITE_ID())
                      != kFlagReady)
                {
                }
                uint32_t helper_offset = static_cast<uint32_t>((2 * tile + 1) * sizeof(int));
                sum += ctx.lds_load_at(&partials[2 * tile + 1],
                                       helper_offset,
                                       HIP_MOI_SITE_ID());
                out[tile] = sum;
            }
        }
    }

    void reset_flags_and_output(int* device_flags, int* device_output)
    {
        int flags[kTileCount]   = {kFlagWaiting, kFlagWaiting};
        int outputs[kTileCount] = {kInitialOutput, kInitialOutput};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_flags, flags, sizeof(flags), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, outputs, sizeof(outputs), hipMemcpyHostToDevice));
    }

    void expect_outputs(int* device_output)
    {
        int outputs[kTileCount] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(outputs, device_output, sizeof(outputs), hipMemcpyDeviceToHost));
        for(int tile = 0; tile < kTileCount; ++tile)
        {
            EXPECT_EQ(outputs[tile], expected_tile_sum(tile));
        }
    }
} // namespace

TEST(HipMoiStreamKTwoTileFlagProtocol, ReleaseAcquireHelperFlagsOrderBothTileFixups)
{
    hip_moi::host_context moi(streamk_two_tile_options());
    int*                  device_flags  = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flags, kTileCount * sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kTileCount * sizeof(int)));
    reset_flags_and_output(device_flags, device_output);

    hipLaunchKernelGGL((streamk_two_tile_flag_fixup_kernel</*BreakSecondTile=*/false>),
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_flags,
                       device_output);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

    expect_outputs(device_output);
    EXPECT_TRUE(moi.check(nullptr));

    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_flags));
}

TEST(HipMoiStreamKTwoTileFlagProtocol, RelaxedSecondTileHelperFlagDiagnosesSecondFixup)
{
    hip_moi::host_context moi(streamk_two_tile_options());
    int*                  device_flags  = nullptr;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flags, kTileCount * sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kTileCount * sizeof(int)));
    reset_flags_and_output(device_flags, device_output);

    hipLaunchKernelGGL((streamk_two_tile_flag_fixup_kernel</*BreakSecondTile=*/true>),
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_flags,
                       device_output);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

    expect_outputs(device_output);
    EXPECT_FALSE(moi.check(nullptr));
    EXPECT_GT(moi.last_diagnostic_count(), 0);

    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_flags));
}
