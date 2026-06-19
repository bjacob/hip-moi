// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/015_thread_level_subgroup_test.hip
//
// Multi-subgroup tests for thread-level instrumentation mode.
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

    __device__ hip_moi::thread_level_context make_context(hip_moi::context_storage_ref storage)
    {
        hip_moi::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::thread_level_context(storage, cfg);
    }

    __global__ void subgroup_identity_kernel(uint32_t*                    out,
                                             hip_moi::context_storage_ref storage)
    {
        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        uint32_t tid  = ctx.thread_id();
        uint32_t base = 3 * tid;
        out[base + 0] = tid;
        out[base + 1] = ctx.subgroup_id();
        out[base + 2] = ctx.thread_rank_in_subgroup();
    }

    __global__ void independent_subgroup_slots_kernel(hip_moi::context_storage_ref storage)
    {
        __shared__ int values[kSubgroupCount];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_rank_in_subgroup() == 0)
        {
            uint32_t subgroup = ctx.subgroup_id();
            ctx.lds_store(&values[subgroup], static_cast<int>(subgroup));
        }
    }

    __global__ void cross_subgroup_same_epoch_kernel(hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::thread_level_context ctx = make_context(storage);
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

    __global__ void same_subgroup_same_epoch_kernel(hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::thread_level_context ctx = make_context(storage);
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

    __global__ void cross_subgroup_separated_by_syncthreads_kernel(
        int* out, hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&value, 31);
        }

        ctx.syncthreads();

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            out[0] = ctx.lds_load(&value);
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

    TEST(HipMoiThreadLevelSubgroup, ContextReportsThreadSubgroupAndRank)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        uint32_t* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, 3 * kThreadCount * sizeof(uint32_t)));

        hipLaunchKernelGGL(subgroup_identity_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        uint32_t output[3 * kThreadCount] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        for(uint32_t tid = 0; tid < static_cast<uint32_t>(kThreadCount); ++tid)
        {
            EXPECT_EQ(output[3 * tid + 0], tid);
            EXPECT_EQ(output[3 * tid + 1], tid / kThreadsPerSubgroup);
            EXPECT_EQ(output[3 * tid + 2], tid % kThreadsPerSubgroup);
        }

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_no_diagnostics(observed, /*expected_access_count=*/0);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiThreadLevelSubgroup, IndependentSubgroupSlotsDoNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        hipLaunchKernelGGL(independent_subgroup_slots_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_no_diagnostics(observed, /*expected_access_count=*/kSubgroupCount);

        bool saw_subgroup_0 = false;
        bool saw_subgroup_1 = false;
        for(int i = 0; i < kSubgroupCount; ++i)
        {
            EXPECT_EQ(observed.access_records[i].valid, 1u);
            EXPECT_EQ(observed.access_records[i].byte_count, sizeof(int));
            EXPECT_EQ(observed.access_records[i].epoch, 0u);
            saw_subgroup_0 |= observed.access_records[i].subgroup_id == 0u
                              && observed.access_records[i].thread_id == 0u;
            saw_subgroup_1 |= observed.access_records[i].subgroup_id == 1u
                              && observed.access_records[i].thread_id
                                     == static_cast<uint32_t>(kThreadsPerSubgroup);
        }
        EXPECT_TRUE(saw_subgroup_0);
        EXPECT_TRUE(saw_subgroup_1);
    }

    TEST(HipMoiThreadLevelSubgroup, CrossSubgroupSameEpochConflictReports)
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
    }

    TEST(HipMoiThreadLevelSubgroup, SameSubgroupSameEpochConflictReports)
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
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_conflict_between_threads(
            observed, /*expected_access_count=*/2, /*first_thread=*/0, /*second_thread=*/1);
    }

    TEST(HipMoiThreadLevelSubgroup, CrossSubgroupSyncthreadsSeparationDoesNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
        int initial = -1;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(cross_subgroup_separated_by_syncthreads_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output = -1;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(output, 31);
        expect_no_diagnostics(observed, /*expected_access_count=*/2);
        EXPECT_EQ(observed.access_records[0].valid, 1u);
        EXPECT_EQ(observed.access_records[0].thread_id, static_cast<uint32_t>(kThreadsPerSubgroup));
        EXPECT_EQ(observed.access_records[0].subgroup_id, 1u);
        EXPECT_EQ(observed.access_records[0].epoch, 1u);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }
} // namespace
