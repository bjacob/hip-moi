// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/002_host_context_test.hip
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
        options.storage_bytes       = 64 * 1024;
        options.diagnostic_capacity = 4;
        options.subgroup_capacity   = kSubgroupCount;
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
            ctx.lds_store_at(&value, 456, /*lds_byte_offset=*/0);
        }

        ctx.syncthreads();

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            out[0] = ctx.lds_load_at(&value, /*lds_byte_offset=*/0);
        }
    }

    __global__ void cross_subgroup_race_kernel(int* out, hip_moi::context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&value, 789, /*lds_byte_offset=*/0);
        }
        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            out[0] = ctx.lds_load_at(&value, /*lds_byte_offset=*/0);
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
        hipLaunchKernelGGL(cross_subgroup_safe_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           moi.launch_ref());
    }

    void launch_race_kernel(hip_moi::host_context& moi, int* device_output)
    {
        hipLaunchKernelGGL(cross_subgroup_race_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           moi.launch_ref());
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
        EXPECT_GT(moi.diagnostic_capacity(), 0);
        EXPECT_EQ(moi.subgroup_capacity(), kSubgroupCount);
        EXPECT_GT(moi.exact_shadow_entry_capacity(), 0);
        EXPECT_EQ(moi.sampled_watchpoint_capacity(), 0);

        hip_moi::context::storage_ref ref = moi.launch_ref();
        EXPECT_NE(ref.exact_shadow_entries, nullptr);
        EXPECT_EQ(ref.exact_shadow_entry_capacity, moi.exact_shadow_entry_capacity());
        EXPECT_EQ(ref.sampled_watchpoints, nullptr);
        EXPECT_EQ(ref.sampled_watchpoint_capacity, moi.sampled_watchpoint_capacity());
        EXPECT_NE(ref.generation, 0u);
    }

    TEST(HipMoiHostContext, OptionFactoriesNameCommonModes)
    {
        hip_moi::host_context_options exact = hip_moi::make_exact_shadow_options();
        EXPECT_EQ(exact.backend, hip_moi::backend_kind::exact_shadow);

        hip_moi::host_context_options sampled_reporting =
            hip_moi::make_sampled_watchpoint_reporting_options();
        EXPECT_EQ(sampled_reporting.backend, hip_moi::backend_kind::sampled_watchpoint);
        EXPECT_EQ(sampled_reporting.exact_shadow_entry_capacity, 0);
        EXPECT_EQ(sampled_reporting.sampled_watchpoint_capacity, -1);
        EXPECT_TRUE(sampled_reporting.sampled_watchpoint_reports);

        hip_moi::host_context_options sampled_publish_only =
            hip_moi::make_sampled_watchpoint_publish_only_options();
        EXPECT_EQ(sampled_publish_only.backend, hip_moi::backend_kind::sampled_watchpoint);
        EXPECT_FALSE(sampled_publish_only.sampled_watchpoint_reports);

        hip_moi::host_context_options one_watchpoint =
            hip_moi::make_one_watchpoint_publish_only_options();
        EXPECT_EQ(one_watchpoint.backend, hip_moi::backend_kind::sampled_watchpoint);
        EXPECT_EQ(one_watchpoint.sampled_watchpoint_capacity, 1);
        EXPECT_EQ(one_watchpoint.sampled_watchpoint_sample_skip, 32u);
        EXPECT_EQ(one_watchpoint.sampled_watchpoint_probe_count, 1u);
        EXPECT_EQ(one_watchpoint.sampled_watchpoint_delay_iters, 32u);
        EXPECT_FALSE(one_watchpoint.sampled_watchpoint_reports);
    }

    TEST(HipMoiHostContext, ExplicitCapacityOverridesAreHonored)
    {
        hip_moi::host_context_options options;
        options.storage_bytes               = 64 * 1024;
        options.exact_shadow_entry_capacity = 23;
        options.sampled_watchpoint_capacity = 29;
        options.diagnostic_capacity         = 5;
        options.subgroup_capacity           = kSubgroupCount;

        hip_moi::host_context moi(options);

        EXPECT_EQ(moi.exact_shadow_entry_capacity(), 23);
        EXPECT_EQ(moi.sampled_watchpoint_capacity(), 29);
        EXPECT_EQ(moi.diagnostic_capacity(), 5);
        EXPECT_EQ(moi.subgroup_capacity(), kSubgroupCount);
        EXPECT_LE(moi.layout_bytes(), moi.storage_bytes());
    }

    TEST(HipMoiHostContext, SampledBackendAutoAllocatesWatchpointsOnly)
    {
        hip_moi::host_context_options options = small_context_options();
        options.backend                       = hip_moi::backend_kind::sampled_watchpoint;
        hip_moi::host_context         moi(options);

        EXPECT_EQ(moi.exact_shadow_entry_capacity(), 0);
        EXPECT_GT(moi.sampled_watchpoint_capacity(), 0);

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

    TEST(HipMoiHostContext, RejectsZeroDiagnosticCapacity)
    {
        ASSERT_DEATH(
            {
                hip_moi::host_context_options options;
                options.diagnostic_capacity = 0;
                hip_moi::host_context moi(options);
            },
            "diagnostic capacity");
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
