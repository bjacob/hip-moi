// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/001_single_subgroup_safe_mvp_test.hip
//
// Initial diagnostic-free tests for the hip-moi instrumentation API.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount = 8;

    __global__ void same_thread_store_load_kernel(int* out, hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::thread_level_context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
            /*subgroup_count=*/1,
        };
        hip_moi::thread_level_context ctx(storage, cfg);

        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, 123);
            out[0] = ctx.lds_load(&value);
            out[1] = ctx.error_count();
        }
    }

    TEST(HipMoiInstrumentedSafeMvp, SameThreadStoreLoad)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(16, 4, 1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, 2 * sizeof(int)));

        int initial[2] = {-1, -1};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, initial, sizeof(initial), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(same_thread_store_load_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[2] = {-1, -1};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        int diagnostic_count = -1;
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&diagnostic_count,
                                          storage.diagnostic_count_device(),
                                          sizeof(diagnostic_count),
                                          hipMemcpyDeviceToHost));
        int access_count = -1;
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&access_count,
                                          storage.access_count_device(),
                                          sizeof(access_count),
                                          hipMemcpyDeviceToHost));

        EXPECT_EQ(output[0], 123);
        EXPECT_EQ(output[1], 0);
        EXPECT_EQ(access_count, 2);
        EXPECT_EQ(diagnostic_count, 0);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }
} // namespace
