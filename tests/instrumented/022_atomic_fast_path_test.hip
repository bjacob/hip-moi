// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/022_atomic_fast_path_test.hip
//
// Stage 5 atomics tests: optimization guardrails for the release/acquire
// metadata path.
//
#include "test_support.hpp"

#include <gtest/gtest.h>

namespace
{
    bool is_power_of_two(int value)
    {
        return value > 0 && (value & (value - 1)) == 0;
    }
} // namespace

TEST(HipMoiAtomicFastPath, AutoAtomicObjectCapacityIsPowerOfTwo)
{
    hip_moi::host_context default_context;
    EXPECT_TRUE(is_power_of_two(default_context.atomic_object_capacity()));

    hip_moi::host_context_options small_options;
    small_options.storage_bytes               = 64 * 1024;
    small_options.exact_shadow_entry_capacity = 16;
    small_options.sampled_watchpoint_capacity = 0;
    small_options.diagnostic_capacity         = 8;
    small_options.subgroup_capacity           = 2;
    hip_moi::host_context small_context(small_options);
    EXPECT_TRUE(is_power_of_two(small_context.atomic_object_capacity()));
}

namespace
{
    __global__ void partial_atomic_cache_kernel(hip_moi::context::storage_ref storage,
                                                int*                          bitmask,
                                                uint64_t                      cache_mask,
                                                bool                          third_release)
    {
        hip_moi::context ctx(storage,
                             {/*thread_count=*/static_cast<int>(blockDim.x),
                              /*threads_per_subgroup=*/warpSize,
                              /*subgroup_count=*/4});
        ctx.init_workgroup();
        const uint32_t subgroup = ctx.subgroup_id();
        if(ctx.lane_in_subgroup() == 0 && subgroup < 3)
        {
            (void)ctx.atomic_fetch_or(bitmask,
                                      1 << subgroup,
                                      subgroup == 2 && !third_release
                                          ? hip_moi::atomic_memory_order::relaxed
                                          : hip_moi::atomic_memory_order::release,
                                      hip_moi::atomic_memory_scope::agent);
        }
        // Stabilize the metadata for this lookup regression. This deliberately
        // does not advance the context's logical synchronization epoch.
        __syncthreads();
        if(ctx.lane_in_subgroup() == 0 && subgroup == 3)
        {
            const uintptr_t address = reinterpret_cast<uintptr_t>(bitmask);
            const uint32_t  seed
                = static_cast<uint32_t>(address >> 2) ^ static_cast<uint32_t>(address >> 34);
            const uint32_t index = hip_moi::detail::mix32(seed)
                                   & (static_cast<uint32_t>(storage.atomic_object_capacity) - 1u);
            auto*          cache = reinterpret_cast<hip_moi::context::atomic_address_cache_record*>(
                storage.atomic_objects + storage.atomic_object_capacity);
            cache[index].address       = address;
            cache[index].producer_mask = cache_mask;
            cache[index].generation    = storage.generation;
            (void)ctx.atomic_fetch_or(bitmask,
                                      8,
                                      hip_moi::atomic_memory_order::acquire,
                                      hip_moi::atomic_memory_scope::agent);
        }
    }

    void check_partial_atomic_cache(uint64_t cache_mask, bool third_release)
    {
        hip_moi::host_context_options options;
        options.storage_bytes               = 64 * 1024;
        options.exact_shadow_entry_capacity = 1;
        options.sampled_watchpoint_capacity = 0;
        options.diagnostic_capacity         = 8;
        options.subgroup_capacity           = 4;
        hip_moi::host_context moi(options);
        auto                  storage   = moi.launch_ref();
        int                   wave_size = 0;
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceGetAttribute(&wave_size, hipDeviceAttributeWarpSize, 0));
        int* bitmask = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&bitmask, sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMemset(bitmask, 0, sizeof(int)));
        partial_atomic_cache_kernel<<<1, 4 * wave_size>>>(
            storage, bitmask, cache_mask, third_release);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());
        uint32_t tokens[16]{};
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(
            tokens, storage.acquired_subgroup_epoch_tokens, sizeof(tokens), hipMemcpyDeviceToHost));
        int result = 0;
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&result, bitmask, sizeof(result), hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(bitmask));
        EXPECT_EQ(result, 15);
        EXPECT_EQ(tokens[12], 1u);
        EXPECT_EQ(tokens[13], 1u);
        EXPECT_EQ(tokens[14], third_release ? 1u : 0u);
        EXPECT_EQ(tokens[15], 0u);
        EXPECT_TRUE(moi.check(nullptr));
    }
} // namespace

TEST(HipMoiAtomicFastPath, PartialAddressCacheDoesNotHideReleaseRecord)
{
    check_partial_atomic_cache(/*cache_mask=*/3, /*third_release=*/true);
}

TEST(HipMoiAtomicFastPath, EmptyAddressCacheDoesNotHideReleaseRecords)
{
    check_partial_atomic_cache(/*cache_mask=*/0, /*third_release=*/true);
}

TEST(HipMoiAtomicFastPath, AddressCacheCannotInventReleaseRecord)
{
    check_partial_atomic_cache(/*cache_mask=*/7, /*third_release=*/false);
}
