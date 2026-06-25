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
