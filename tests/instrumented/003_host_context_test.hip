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
#include <string>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;

    hip_moi::host_context_options small_context_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes                     = 64 * 1024;
        options.access_record_capacity            = 16;
        options.coalesced_access_record_capacity  = 0;
        options.coalescing_access_record_capacity = 0;
        options.coalescing_group_record_capacity  = 0;
        options.diagnostic_capacity               = 4;
        options.subgroup_capacity                 = kSubgroupCount;
        return options;
    }

    hip_moi::host_context_options byte_budget_context_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes     = 64 * 1024;
        options.subgroup_capacity = kSubgroupCount;
        return options;
    }

    __device__ hip_moi::context make_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    __global__ void cross_subgroup_safe_kernel(int* out, hip_moi::context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&value, 456);
        }

        ctx.syncthreads();

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            out[0] = ctx.lds_load(&value);
        }
        ctx.finish();
    }

    __global__ void cross_subgroup_race_kernel(int* out, hip_moi::context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&value, 789);
        }
        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            out[0] = ctx.lds_load(&value);
        }
        ctx.finish();
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
        hipLaunchKernelGGL(cross_subgroup_safe_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           moi.device_ref());
    }

    void launch_race_kernel(hip_moi::host_context& moi, int* device_output)
    {
        hipLaunchKernelGGL(cross_subgroup_race_kernel,
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

    void expect_contains(const std::string& text, const char* needle)
    {
        EXPECT_NE(text.find(needle), std::string::npos) << "missing stderr text: " << needle;
    }

    TEST(HipMoiHostContext, CheckMacroSucceedsForSafeKernel)
    {
        hip_moi::host_context moi(small_context_options());

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

        EXPECT_EQ(output, 456);
        EXPECT_EQ(moi.last_diagnostic_count(), 0);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiHostContext, ByteBudgetContextRunsSafeKernel)
    {
        hip_moi::host_context moi(byte_budget_context_options());

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

        EXPECT_EQ(output, 456);
        EXPECT_EQ(moi.last_diagnostic_count(), 0);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiHostContext, ByteBudgetContextReportsComputedCapacities)
    {
        hip_moi::host_context_options options = byte_budget_context_options();
        hip_moi::host_context         moi(options);

        EXPECT_EQ(moi.storage_bytes(), options.storage_bytes);
        EXPECT_GT(moi.layout_bytes(), 0u);
        EXPECT_LE(moi.layout_bytes(), moi.storage_bytes());
        EXPECT_GT(moi.access_record_capacity(), 0);
        EXPECT_GT(moi.diagnostic_capacity(), 0);
        EXPECT_EQ(moi.subgroup_capacity(), kSubgroupCount);
        EXPECT_GT(moi.coalesced_access_record_capacity(), 0);
        EXPECT_GT(moi.coalescing_access_record_capacity(), 0);
        EXPECT_GT(moi.coalescing_group_record_capacity(), 0);
    }

    TEST(HipMoiHostContext, ExplicitCapacityOverridesAreHonored)
    {
        hip_moi::host_context_options options;
        options.storage_bytes                     = 64 * 1024;
        options.access_record_capacity            = 19;
        options.coalesced_access_record_capacity  = 7;
        options.coalescing_access_record_capacity = 11;
        options.coalescing_group_record_capacity  = 13;
        options.diagnostic_capacity               = 5;
        options.subgroup_capacity                 = kSubgroupCount;

        hip_moi::host_context moi(options);

        EXPECT_EQ(moi.access_record_capacity(), 19);
        EXPECT_EQ(moi.coalesced_access_record_capacity(), 7);
        EXPECT_EQ(moi.coalescing_access_record_capacity(), 11);
        EXPECT_EQ(moi.coalescing_group_record_capacity(), 13);
        EXPECT_EQ(moi.diagnostic_capacity(), 5);
        EXPECT_EQ(moi.subgroup_capacity(), kSubgroupCount);
        EXPECT_LE(moi.layout_bytes(), moi.storage_bytes());
    }

    TEST(HipMoiHostContext, OptionalCoalescingCapacitiesCanBeDisabled)
    {
        hip_moi::host_context_options options = small_context_options();
        hip_moi::host_context         moi(options);

        EXPECT_EQ(moi.coalesced_access_record_capacity(), 0);
        EXPECT_EQ(moi.coalescing_access_record_capacity(), 0);
        EXPECT_EQ(moi.coalescing_group_record_capacity(), 0);

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

        EXPECT_EQ(output, 456);
        EXPECT_EQ(moi.last_diagnostic_count(), 0);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiHostContext, RejectsZeroStorageBytes)
    {
        ASSERT_DEATH(
            {
                hip_moi::host_context_options options;
                options.storage_bytes = 0;
                hip_moi::host_context moi(options);
            },
            "storage options are invalid");
    }

    TEST(HipMoiHostContext, RejectsZeroRequiredCapacities)
    {
        ASSERT_DEATH(
            {
                hip_moi::host_context_options options;
                options.access_record_capacity = 0;
                options.diagnostic_capacity    = 4;
                hip_moi::host_context moi(options);
            },
            "access and diagnostic capacities");

        ASSERT_DEATH(
            {
                hip_moi::host_context_options options;
                options.access_record_capacity = 16;
                options.diagnostic_capacity    = 0;
                hip_moi::host_context moi(options);
            },
            "access and diagnostic capacities");
    }

    TEST(HipMoiHostContext, RejectsTinyStorageBytes)
    {
        ASSERT_DEATH(
            {
                hip_moi::host_context_options options;
                options.storage_bytes     = 16;
                options.subgroup_capacity = kSubgroupCount;
                hip_moi::host_context moi(options);
            },
            "storage_bytes=.*too small");
    }

    TEST(HipMoiHostContext, ExplicitCheckConsumesRaceDiagnostic)
    {
        hip_moi::host_context moi(small_context_options());

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));

        int initial = -1;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice));

        launch_race_kernel(moi, device_output);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());

        EXPECT_FALSE(moi.check(/*stream=*/nullptr));
        EXPECT_GT(moi.last_diagnostic_count(), 0);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiHostContext, ExplicitCheckPrintsSubgroupDiagnosticToStderr)
    {
        hip_moi::host_context moi(small_context_options());

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));

        int initial = -1;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice));

        launch_race_kernel(moi, device_output);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());

        testing::internal::CaptureStderr();
        EXPECT_FALSE(moi.check());
        std::string stderr_output = testing::internal::GetCapturedStderr();

        expect_contains(stderr_output, "diagnostic(s)");
        expect_contains(stderr_output, "hip-moi diagnostic 0: kind=access_conflict");
        expect_contains(stderr_output, "first_subgroup=");
        expect_contains(stderr_output, "second_subgroup=");
        expect_contains(stderr_output, "first_site_id=");
        expect_contains(stderr_output, "second_site_id=");
        EXPECT_EQ(stderr_output.find("first_thread="), std::string::npos);
        EXPECT_EQ(stderr_output.find("second_thread="), std::string::npos);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiHostContext, CheckMacroAbortsOnRaceDiagnostic)
    {
        ASSERT_DEATH(
            {
                hip_moi::host_context moi(small_context_options());
                launch_race_kernel_or_abort(moi);
                HIP_MOI_CHECK(moi);
            },
            "HIP_MOI_CHECK failed");
    }

    TEST(HipMoiHostContext, CheckMacroPrintsDiagnosticBeforeAbort)
    {
        ASSERT_DEATH(
            {
                hip_moi::host_context moi(small_context_options());
                launch_race_kernel_or_abort(moi);
                HIP_MOI_CHECK(moi);
            },
            "first_subgroup=");
    }

    TEST(HipMoiHostContext, DestructorAbortsOnUnconsumedRaceDiagnostic)
    {
        ASSERT_DEATH(
            {
                hip_moi::host_context moi(small_context_options());
                launch_race_kernel_or_abort(moi);
            },
            "host_context destructor");
    }

    TEST(HipMoiHostContext, DestructorPrintsDiagnosticWhenAbortDisabled)
    {
        testing::internal::CaptureStderr();
        {
            hip_moi::host_context moi(small_context_options());
            moi.disable_destructor_abort();

            int* device_output = nullptr;
            HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));

            int initial = -1;
            HIP_MOI_TEST_HIP_ASSERT(
                hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice));

            launch_race_kernel(moi, device_output);
            HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
            HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
        }
        std::string stderr_output = testing::internal::GetCapturedStderr();

        expect_contains(stderr_output, "diagnostic(s)");
        expect_contains(stderr_output, "first_subgroup=");
        EXPECT_EQ(stderr_output.find("destructor; aborting"), std::string::npos);
    }

    TEST(HipMoiHostContext, DestructorReportingAndAbortCanBeDisabled)
    {
        {
            hip_moi::host_context moi(small_context_options());
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
