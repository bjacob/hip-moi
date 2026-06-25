// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/020_atomic_metadata_test.hip
//
// Stage 3 atomics tests: release-style atomic operations populate the bounded
// atomic-object metadata table. The table is not used for happens-before
// suppression until Stage 4.
//
#include "test_support.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <string>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;

    __device__ hip_moi::context make_atomic_metadata_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kThreadCount / kThreadsPerSubgroup,
        };
        return hip_moi::context(storage, cfg);
    }

    __global__ void publish_many_atomic_objects_kernel(hip_moi::context::storage_ref storage,
                                                       int*                          flags,
                                                       int                           flag_count)
    {
        hip_moi::context ctx = make_atomic_metadata_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            for(int i = 0; i < flag_count; ++i)
            {
                ctx.atomic_store(&flags[i],
                                 i + 1,
                                 hip_moi::atomic_memory_order::release,
                                 hip_moi::atomic_memory_scope::agent,
                                 HIP_MOI_SITE_ID());
            }
        }
    }

    hip_moi::host_context_options atomic_metadata_test_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes               = 64 * 1024;
        options.exact_shadow_entry_capacity = 1;
        options.sampled_watchpoint_capacity = 0;
        options.diagnostic_capacity         = 8;
        options.subgroup_capacity           = kThreadCount / kThreadsPerSubgroup;
        return options;
    }

    void expect_contains(const std::string& text, const char* needle)
    {
        EXPECT_NE(text.find(needle), std::string::npos) << "missing diagnostic text: " << needle;
    }
} // namespace

TEST(HipMoiAtomicMetadata, SaturationReportsMetadataFullWithAtomicSite)
{
    hip_moi::host_context moi(atomic_metadata_test_options());
    int                   flag_count = moi.atomic_object_capacity() + 1;
    ASSERT_GT(moi.atomic_object_capacity(), 0);

    int* device_flags = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flags, flag_count * sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMemset(device_flags, 0, flag_count * sizeof(int)));

    hipLaunchKernelGGL(publish_many_atomic_objects_kernel,
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_flags,
                       flag_count);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());

    testing::internal::CaptureStderr();
    EXPECT_FALSE(moi.check());
    std::string stderr_output = testing::internal::GetCapturedStderr();

    expect_contains(stderr_output, "kind=metadata_full");
    expect_contains(stderr_output, "first_site_id=0x");
    expect_contains(stderr_output, "first_size=4");

    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_flags));
}

TEST(HipMoiAtomicMetadata, AtomicObjectRecordsAreGenerationSeparated)
{
    hip_moi::host_context moi(atomic_metadata_test_options());
    int                   flag_count = moi.atomic_object_capacity();
    ASSERT_GT(flag_count, 0);

    int* device_flags = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flags, 2 * flag_count * sizeof(int)));
    HIP_MOI_TEST_HIP_ASSERT(hipMemset(device_flags, 0, 2 * flag_count * sizeof(int)));

    hipLaunchKernelGGL(publish_many_atomic_objects_kernel,
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_flags,
                       flag_count);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    EXPECT_TRUE(moi.check(nullptr));

    hipLaunchKernelGGL(publish_many_atomic_objects_kernel,
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_flags + flag_count,
                       flag_count);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    EXPECT_TRUE(moi.check(nullptr));

    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_flags));
}
