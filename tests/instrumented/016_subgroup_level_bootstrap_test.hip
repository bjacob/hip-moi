// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/016_subgroup_level_bootstrap_test.hip
//
// Bootstrap tests for subgroup-level instrumentation mode.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kDiagnosticCapacity = 8;
    constexpr int kAccessCapacity     = 16;

    struct observed_metadata
    {
        int                    access_count;
        int                    diagnostic_count;
        hip_moi::diagnostic    diagnostics[kDiagnosticCapacity];
        hip_moi::access_record access_records[kAccessCapacity];
    };

    __device__ hip_moi::context make_context(hip_moi::context_storage_ref storage,
                                             hip_moi::instrumentation_mode mode)
    {
        hip_moi::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
            /*mode=*/mode,
        };
        return hip_moi::context(storage, cfg);
    }

    __global__ void subgroup_level_mode_query_kernel(uint32_t*                    out,
                                                     hip_moi::context_storage_ref storage)
    {
        hip_moi::context ctx = make_context(storage, hip_moi::instrumentation_mode::subgroup_level);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            out[0] = static_cast<uint32_t>(ctx.mode());
        }
    }

    __global__ void same_subgroup_same_epoch_kernel(hip_moi::context_storage_ref  storage,
                                                   hip_moi::instrumentation_mode mode)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage, mode);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&value, 23);
        }
        if(ctx.thread_id() == 1)
        {
            (void)ctx.lds_load(&value);
        }
    }

    __global__ void cross_subgroup_same_epoch_kernel(hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage, hip_moi::instrumentation_mode::subgroup_level);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&value, 17);
        }
        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            (void)ctx.lds_load(&value);
        }
    }

    __global__ void cross_subgroup_separated_by_syncthreads_kernel(
        hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage, hip_moi::instrumentation_mode::subgroup_level);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&value, 31);
        }

        ctx.syncthreads();

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            (void)ctx.lds_load(&value);
        }
    }

    void copy_observed_metadata(const hip_moi::test::device_context_storage& storage,
                                observed_metadata*                           observed)
    {
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->access_count,
                                          storage.access_count_device(),
                                          sizeof(observed->access_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->diagnostic_count,
                                          storage.diagnostic_count_device(),
                                          sizeof(observed->diagnostic_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(observed->diagnostics,
                                          storage.diagnostics_device(),
                                          sizeof(observed->diagnostics),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(observed->access_records,
                                          storage.access_records_device(),
                                          sizeof(observed->access_records),
                                          hipMemcpyDeviceToHost));
    }

    void expect_no_diagnostics(const observed_metadata& observed, int expected_access_count)
    {
        EXPECT_EQ(observed.access_count, expected_access_count);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    void expect_conflict_between_threads(const observed_metadata& observed,
                                         int                      expected_access_count,
                                         uint32_t                 first_thread,
                                         uint32_t                 second_thread)
    {
        EXPECT_EQ(observed.access_count, expected_access_count);
        ASSERT_GT(observed.diagnostic_count, 0);

        int stored_count = observed.diagnostic_count;
        if(stored_count > kDiagnosticCapacity)
        {
            stored_count = kDiagnosticCapacity;
        }

        bool found_expected_conflict = false;
        for(int i = 0; i < stored_count; ++i)
        {
            const hip_moi::diagnostic& diagnostic = observed.diagnostics[i];
            bool has_expected_threads =
                (diagnostic.writer_or_first_thread_id == first_thread
                 && diagnostic.reader_or_second_thread_id == second_thread)
                || (diagnostic.writer_or_first_thread_id == second_thread
                    && diagnostic.reader_or_second_thread_id == first_thread);
            found_expected_conflict |=
                diagnostic.kind == static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict)
                && diagnostic.epoch == 0u && diagnostic.first_addr == diagnostic.second_addr
                && diagnostic.first_size == sizeof(int) && diagnostic.second_size == sizeof(int)
                && has_expected_threads;
        }
        EXPECT_TRUE(found_expected_conflict);
    }

    void expect_accesses_from_subgroups(const observed_metadata& observed,
                                        uint32_t                 first_subgroup,
                                        uint32_t                 second_subgroup)
    {
        bool saw_first_subgroup  = false;
        bool saw_second_subgroup = false;

        for(int i = 0; i < kAccessCapacity; ++i)
        {
            if(observed.access_records[i].valid == 0)
            {
                continue;
            }
            saw_first_subgroup |= observed.access_records[i].subgroup_id == first_subgroup;
            saw_second_subgroup |= observed.access_records[i].subgroup_id == second_subgroup;
        }

        EXPECT_TRUE(saw_first_subgroup);
        EXPECT_TRUE(saw_second_subgroup);
    }

    TEST(HipMoiSubgroupLevelBootstrap, ContextReportsConfiguredMode)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        uint32_t* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(uint32_t)));

        uint32_t initial = 0xffffffffu;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(subgroup_level_mode_query_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        uint32_t output = 0xffffffffu;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
        EXPECT_EQ(output, static_cast<uint32_t>(hip_moi::instrumentation_mode::subgroup_level));

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_no_diagnostics(observed, /*expected_access_count=*/0);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiSubgroupLevelBootstrap, SameSubgroupConflictDoesNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        hipLaunchKernelGGL(same_subgroup_same_epoch_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref(),
                           hip_moi::instrumentation_mode::subgroup_level);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_no_diagnostics(observed, /*expected_access_count=*/2);
        expect_accesses_from_subgroups(observed, /*first_subgroup=*/0, /*second_subgroup=*/0);
    }

    TEST(HipMoiSubgroupLevelBootstrap, ExplicitThreadLevelStillReportsSameSubgroupConflict)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        hipLaunchKernelGGL(same_subgroup_same_epoch_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref(),
                           hip_moi::instrumentation_mode::thread_level);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_conflict_between_threads(
            observed, /*expected_access_count=*/2, /*first_thread=*/0, /*second_thread=*/1);
        expect_accesses_from_subgroups(observed, /*first_subgroup=*/0, /*second_subgroup=*/0);
    }

    TEST(HipMoiSubgroupLevelBootstrap, CrossSubgroupConflictReports)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        hipLaunchKernelGGL(cross_subgroup_same_epoch_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_conflict_between_threads(
            observed, /*expected_access_count=*/2, /*first_thread=*/0,
            /*second_thread=*/kThreadsPerSubgroup);
        expect_accesses_from_subgroups(observed, /*first_subgroup=*/0, /*second_subgroup=*/1);
    }

    TEST(HipMoiSubgroupLevelBootstrap, CrossSubgroupSyncthreadsSeparationDoesNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        hipLaunchKernelGGL(cross_subgroup_separated_by_syncthreads_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_no_diagnostics(observed, /*expected_access_count=*/2);
        EXPECT_EQ(observed.access_records[0].valid, 1u);
        EXPECT_EQ(observed.access_records[0].thread_id,
                  static_cast<uint32_t>(kThreadsPerSubgroup));
        EXPECT_EQ(observed.access_records[0].subgroup_id, 1u);
        EXPECT_EQ(observed.access_records[0].epoch, 1u);
    }
} // namespace
