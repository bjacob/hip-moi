// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/003_host_context_test.hip
//
// User-facing host_context and HIP_MOI_CHECK tests.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

namespace
{
    constexpr int kThreadCount = 8;

    hip_moi::host_context_options small_host_context_options()
    {
        hip_moi::host_context_options options;
        options.access_record_capacity = 16;
        options.diagnostic_capacity    = 4;
        options.subgroup_capacity      = 1;
        return options;
    }

    __global__ void same_thread_store_load_kernel(int* out, hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
            /*subgroup_count=*/1,
        };
        hip_moi::context ctx(storage, cfg);

        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, 123);
            out[0] = ctx.lds_load(&value);
        }
    }

    __global__ void same_epoch_write_read_kernel(int* out, hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
            /*subgroup_count=*/1,
        };
        hip_moi::context ctx(storage, cfg);

        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, 123);
        }
        if(threadIdx.x == 1)
        {
            out[0] = ctx.lds_load(&value);
        }
    }

    void require_hip_success(hipError_t status, const char* expression)
    {
        if(status == hipSuccess)
        {
            return;
        }

        std::fprintf(stderr, "%s failed: %s\n", expression, hipGetErrorString(status));
        std::fflush(stderr);
        std::abort();
    }

    void launch_safe_kernel(hip_moi::host_context& moi, int* device_output)
    {
        hipLaunchKernelGGL(same_thread_store_load_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           moi.device_ref());
    }

    void launch_race_kernel(hip_moi::host_context& moi, int* device_output)
    {
        hipLaunchKernelGGL(same_epoch_write_read_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           moi.device_ref());
    }

    void launch_race_kernel_or_abort(hip_moi::host_context& moi)
    {
        int* device_output = nullptr;
        require_hip_success(hipMalloc(&device_output, sizeof(int)), "hipMalloc");

        int initial = -1;
        require_hip_success(
            hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice),
            "hipMemcpy(device_output)");

        launch_race_kernel(moi, device_output);
        require_hip_success(hipGetLastError(), "hipGetLastError");
    }

    TEST(HipMoiHostContext, CheckMacroSucceedsForSafeKernel)
    {
        hip_moi::host_context moi(small_host_context_options());

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));

        int initial = -1;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice));

        launch_safe_kernel(moi, device_output);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());

        HIP_MOI_CHECK(moi);

        int output = -1;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        EXPECT_EQ(output, 123);
        EXPECT_EQ(moi.last_diagnostic_count(), 0);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiHostContext, ExplicitCheckConsumesRaceDiagnostic)
    {
        hip_moi::host_context moi(small_host_context_options());

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));

        int initial = -1;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice));

        launch_race_kernel(moi, device_output);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());

        EXPECT_FALSE(moi.check(/*stream=*/nullptr));
        EXPECT_EQ(moi.last_diagnostic_count(), 1);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiHostContext, DestructorAbortsOnUnconsumedRaceDiagnostic)
    {
        ASSERT_DEATH(
            {
                hip_moi::host_context moi(small_host_context_options());
                launch_race_kernel_or_abort(moi);
            },
            "unconsumed diagnostics");
    }

    TEST(HipMoiHostContext, DestructorReportingAndAbortCanBeDisabled)
    {
        {
            hip_moi::host_context moi(small_host_context_options());
            moi.disable_destructor_reporting();
            moi.disable_destructor_abort();

            int* device_output = nullptr;
            HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));

            int initial = -1;
            HIP_MOI_TEST_HIP_ASSERT(
                hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice));

            launch_race_kernel(moi, device_output);
            HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
            HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());
            HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
        }
    }
} // namespace
