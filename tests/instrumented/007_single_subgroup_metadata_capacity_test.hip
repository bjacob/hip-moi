// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/007_single_subgroup_metadata_capacity_test.hip
//
// Metadata capacity and overflow behavior for the MVP detector.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <string>

namespace
{
    constexpr int kThreadCount                 = 8;
    constexpr int kAccessOverflowCapacity      = 2;
    constexpr int kAccessOverflowAccessCount   = 3;
    constexpr int kStoredDiagnosticCapacity    = 4;
    constexpr int kTruncatedDiagnosticCapacity = 3;
    constexpr int kThreadsPerSubgroup          = 32;
    constexpr int kSubgroupCount               = 2;
    constexpr int kSubgroupThreadCount         = kThreadsPerSubgroup * kSubgroupCount;
    constexpr int kStoredSubgroupDiagnostics   = 16;

    constexpr hip_moi::site_id kHostCoalescingStoreSite(0x701u);
    constexpr hip_moi::site_id kHostCoalescingLoadSite(0x702u);

    struct observed_metadata
    {
        int                 access_count;
        int                 diagnostic_count;
        hip_moi::thread_level_context::diagnostic diagnostics[kStoredDiagnosticCapacity];
    };

    struct observed_subgroup_metadata
    {
        int access_count;
        int coalescing_access_count;
        int coalescing_fallback_count;
        int diagnostic_count;
        hip_moi::subgroup_level_context::diagnostic diagnostics[kStoredSubgroupDiagnostics];
    };

    __device__ hip_moi::thread_level_context make_context(hip_moi::thread_level_context::storage_ref storage)
    {
        hip_moi::thread_level_context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
            /*subgroup_count=*/1,
        };
        return hip_moi::thread_level_context(storage, cfg);
    }

    __device__ hip_moi::thread_level_context make_two_subgroup_thread_level_context(
        hip_moi::thread_level_context::storage_ref storage)
    {
        hip_moi::thread_level_context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::thread_level_context(storage, cfg);
    }

    __device__ hip_moi::subgroup_level_context make_two_subgroup_context(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        hip_moi::subgroup_level_context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::subgroup_level_context(storage, cfg);
    }

    __global__ void access_log_overflow_kernel(hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int values[kAccessOverflowAccessCount];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&values[0], 10);
            ctx.lds_store(&values[1], 20);
            ctx.lds_store(&values[2], 30);
        }
    }

    __global__ void many_neighbor_conflicts_kernel(hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        ctx.lds_store(&values[tid], tid);

        int neighbor = (tid + 1) % kThreadCount;
        (void)ctx.lds_load(&values[neighbor]);
    }

    __global__ void many_neighbor_conflicts_subgroup_level_kernel(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::subgroup_level_context ctx =
            hip_moi::test::make_single_subgroup_context<hip_moi::subgroup_level_context>(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        ctx.lds_store(&values[tid], tid);

        int neighbor = (tid + 1) % kThreadCount;
        (void)ctx.lds_load(&values[neighbor]);
        ctx.finish();
    }

    __global__ void thread_level_subgroup_capacity_mismatch_kernel(
        hip_moi::thread_level_context::storage_ref storage)
    {
        hip_moi::thread_level_context ctx = make_two_subgroup_thread_level_context(storage);
        ctx.init_workgroup();
    }

    __global__ void subgroup_level_subgroup_capacity_mismatch_kernel(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        hip_moi::subgroup_level_context ctx = make_two_subgroup_context(storage);
        ctx.init_workgroup();
        ctx.finish();
    }

    __global__ void host_owned_coalescing_conflict_kernel(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadsPerSubgroup];

        hip_moi::subgroup_level_context ctx = make_two_subgroup_context(storage);
        ctx.init_workgroup();

        int lane = static_cast<int>(ctx.lane_in_subgroup());
        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store(&values[lane], lane, kHostCoalescingStoreSite);
        }
        if(ctx.subgroup_id() == 1)
        {
            (void)ctx.lds_load(&values[lane], kHostCoalescingLoadSite);
        }
        ctx.finish();
    }

    void copy_observed_metadata(const hip_moi::test::device_context_storage& storage,
                                observed_metadata*                           observed,
                                int stored_diagnostic_count)
    {
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->access_count,
                                          storage.access_count_device(),
                                          sizeof(observed->access_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->diagnostic_count,
                                          storage.diagnostic_count_device(),
                                          sizeof(observed->diagnostic_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(observed->diagnostics,
                      storage.diagnostics_device(),
                      stored_diagnostic_count * sizeof(hip_moi::thread_level_context::diagnostic),
                      hipMemcpyDeviceToHost));
    }

    void copy_observed_metadata(hip_moi::thread_level_context::storage_ref storage,
                                observed_metadata*                         observed,
                                int stored_diagnostic_count)
    {
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->access_count,
                                          storage.access_count,
                                          sizeof(observed->access_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->diagnostic_count,
                                          storage.diagnostic_count,
                                          sizeof(observed->diagnostic_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(observed->diagnostics,
                      storage.diagnostics,
                      stored_diagnostic_count * sizeof(hip_moi::thread_level_context::diagnostic),
                      hipMemcpyDeviceToHost));
    }

    void copy_observed_metadata(hip_moi::subgroup_level_context::storage_ref storage,
                                observed_subgroup_metadata*                 observed,
                                int stored_diagnostic_count)
    {
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->access_count,
                                          storage.access_count,
                                          sizeof(observed->access_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->coalescing_access_count,
                                          storage.coalescing_access_count,
                                          sizeof(observed->coalescing_access_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->coalescing_fallback_count,
                                          storage.coalescing_fallback_count,
                                          sizeof(observed->coalescing_fallback_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->diagnostic_count,
                                          storage.diagnostic_count,
                                          sizeof(observed->diagnostic_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(observed->diagnostics,
                      storage.diagnostics,
                      stored_diagnostic_count
                          * sizeof(hip_moi::subgroup_level_context::diagnostic),
                      hipMemcpyDeviceToHost));
    }

    void expect_access_conflict(const hip_moi::thread_level_context::diagnostic& diagnostic)
    {
        EXPECT_EQ(diagnostic.kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
        EXPECT_EQ(diagnostic.epoch, 0u);
    }

    void expect_contains(const std::string& text, const char* needle)
    {
        EXPECT_NE(text.find(needle), std::string::npos) << "missing stderr text: " << needle;
    }

    hip_moi::host_context_options truncating_host_context_options()
    {
        hip_moi::host_context_options options;
        options.access_record_capacity = 32;
        options.diagnostic_capacity    = kTruncatedDiagnosticCapacity;
        options.subgroup_capacity      = 1;
        return options;
    }

    bool has_metadata_full(const observed_metadata& observed, int stored_diagnostic_count)
    {
        for(int i = 0; i < stored_diagnostic_count; ++i)
        {
            if(observed.diagnostics[i].kind
               == static_cast<uint32_t>(hip_moi::diagnostic_kind::metadata_full))
            {
                return true;
            }
        }
        return false;
    }

    bool has_metadata_full(const observed_subgroup_metadata& observed,
                           int                               stored_diagnostic_count)
    {
        for(int i = 0; i < stored_diagnostic_count; ++i)
        {
            if(observed.diagnostics[i].kind
               == static_cast<uint32_t>(hip_moi::diagnostic_kind::metadata_full))
            {
                return true;
            }
        }
        return false;
    }

    TEST(HipMoiMetadataCapacity, AccessLogOverflowReportsMetadataFull)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kAccessOverflowCapacity,
            /*diagnostic_capacity=*/kStoredDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(
            access_log_overflow_kernel, dim3(1), dim3(kThreadCount), 0, 0, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(
            copy_observed_metadata(storage, &observed, /*stored_diagnostic_count=*/1));

        EXPECT_EQ(observed.access_count, kAccessOverflowAccessCount);
        ASSERT_EQ(observed.diagnostic_count, 1);

        const hip_moi::thread_level_context::diagnostic& diagnostic = observed.diagnostics[0];
        EXPECT_EQ(diagnostic.kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::metadata_full));
        EXPECT_EQ(diagnostic.epoch, 0u);
        EXPECT_EQ(diagnostic.first_thread_id, 0u);
        EXPECT_EQ(diagnostic.second_thread_id, 0u);
        EXPECT_EQ(diagnostic.first_size, sizeof(int));
        EXPECT_EQ(diagnostic.second_size, sizeof(int));
    }

    TEST(HipMoiMetadataCapacity, HostOwnedAccessLogOverflowReportsMetadataFull)
    {
        hip_moi::host_context_options options;
        options.access_record_capacity = kAccessOverflowCapacity;
        options.diagnostic_capacity    = kStoredDiagnosticCapacity;
        options.subgroup_capacity      = 1;
        hip_moi::host_context moi(options);

        hip_moi::thread_level_context::storage_ref storage = moi.device_ref();
        hipLaunchKernelGGL(
            access_log_overflow_kernel, dim3(1), dim3(kThreadCount), 0, 0, storage);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(
            copy_observed_metadata(storage, &observed, /*stored_diagnostic_count=*/1));

        EXPECT_EQ(observed.access_count, kAccessOverflowAccessCount);
        ASSERT_EQ(observed.diagnostic_count, 1);
        EXPECT_TRUE(has_metadata_full(observed, /*stored_diagnostic_count=*/1));
        EXPECT_FALSE(moi.check(/*stream=*/nullptr));
    }

    TEST(HipMoiMetadataCapacity, DiagnosticCounterCanExceedStoredCapacity)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/32,
            /*diagnostic_capacity=*/kTruncatedDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(
            many_neighbor_conflicts_kernel, dim3(1), dim3(kThreadCount), 0, 0, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(
            storage, &observed, /*stored_diagnostic_count=*/kTruncatedDiagnosticCapacity));

        EXPECT_EQ(observed.access_count, 2 * kThreadCount);
        EXPECT_EQ(observed.diagnostic_count, kThreadCount);
        for(int i = 0; i < kTruncatedDiagnosticCapacity; ++i)
        {
            expect_access_conflict(observed.diagnostics[i]);
        }
    }

    TEST(HipMoiMetadataCapacity, HostCheckReportsTruncatedDiagnosticBuffer)
    {
        hip_moi::host_context moi(truncating_host_context_options());

        hipLaunchKernelGGL(
            many_neighbor_conflicts_kernel, dim3(1), dim3(kThreadCount), 0, 0, moi.device_ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());

        testing::internal::CaptureStderr();
        EXPECT_FALSE(moi.check());
        std::string stderr_output = testing::internal::GetCapturedStderr();

        EXPECT_EQ(moi.last_diagnostic_count(), kThreadCount);
        expect_contains(stderr_output, "hip-moi: 8 diagnostic(s)");
        expect_contains(stderr_output, "hip-moi diagnostic 0: kind=access_conflict");
        expect_contains(stderr_output,
                        "hip-moi: diagnostic buffer stored 3 of 8 diagnostic(s)");
    }

    TEST(HipMoiMetadataCapacity, HostOwnedThreadLevelSubgroupCapacityMismatchReports)
    {
        hip_moi::host_context_options options;
        options.access_record_capacity = 16;
        options.diagnostic_capacity    = kStoredDiagnosticCapacity;
        options.subgroup_capacity      = 1;
        hip_moi::host_context moi(options);

        hip_moi::thread_level_context::storage_ref storage = moi.device_ref();
        hipLaunchKernelGGL(thread_level_subgroup_capacity_mismatch_kernel,
                           dim3(1),
                           dim3(kSubgroupThreadCount),
                           0,
                           0,
                           storage);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(
            copy_observed_metadata(storage, &observed, /*stored_diagnostic_count=*/1));

        EXPECT_EQ(observed.diagnostic_count, 1);
        EXPECT_TRUE(has_metadata_full(observed, /*stored_diagnostic_count=*/1));
        EXPECT_FALSE(moi.check(/*stream=*/nullptr));
    }

    TEST(HipMoiMetadataCapacity, HostOwnedSubgroupLevelSubgroupCapacityMismatchReports)
    {
        hip_moi::host_context_options options;
        options.access_record_capacity              = 16;
        options.coalesced_access_record_capacity    = 0;
        options.coalescing_access_record_capacity   = 0;
        options.coalescing_group_record_capacity    = 0;
        options.diagnostic_capacity                 = kStoredSubgroupDiagnostics;
        options.subgroup_capacity                   = 1;
        hip_moi::subgroup_level_host_context moi(options);

        hip_moi::subgroup_level_context::storage_ref storage = moi.device_ref();
        hipLaunchKernelGGL(subgroup_level_subgroup_capacity_mismatch_kernel,
                           dim3(1),
                           dim3(kSubgroupThreadCount),
                           0,
                           0,
                           storage);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_subgroup_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(
            storage, &observed, /*stored_diagnostic_count=*/kStoredSubgroupDiagnostics));

        EXPECT_EQ(observed.diagnostic_count, 1);
        EXPECT_TRUE(
            has_metadata_full(observed, /*stored_diagnostic_count=*/kStoredSubgroupDiagnostics));
        EXPECT_EQ(observed.diagnostics[0].first_subgroup_id, 1u);
        EXPECT_EQ(observed.diagnostics[0].second_subgroup_id,
                  static_cast<uint32_t>(kSubgroupCount));
        EXPECT_FALSE(moi.check(/*stream=*/nullptr));
    }

    TEST(HipMoiMetadataCapacity, HostOwnedSubgroupCoalescingOverflowFallsBackToExactRecords)
    {
        constexpr int kTinyCoalescingAccessCapacity = 8;

        hip_moi::host_context_options options;
        options.access_record_capacity              = kSubgroupThreadCount;
        options.coalesced_access_record_capacity    = 8;
        options.coalescing_access_record_capacity   = kTinyCoalescingAccessCapacity;
        options.coalescing_group_record_capacity    = 8;
        options.diagnostic_capacity                 = kStoredSubgroupDiagnostics;
        options.subgroup_capacity                   = kSubgroupCount;
        hip_moi::subgroup_level_host_context moi(options);

        hip_moi::subgroup_level_context::storage_ref storage = moi.device_ref();
        hipLaunchKernelGGL(host_owned_coalescing_conflict_kernel,
                           dim3(1),
                           dim3(kSubgroupThreadCount),
                           0,
                           0,
                           storage);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_subgroup_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(
            storage, &observed, /*stored_diagnostic_count=*/kStoredSubgroupDiagnostics));

        EXPECT_EQ(observed.access_count, kSubgroupThreadCount - kTinyCoalescingAccessCapacity);
        EXPECT_EQ(observed.coalescing_access_count, kSubgroupThreadCount);
        EXPECT_EQ(observed.coalescing_fallback_count,
                  kSubgroupThreadCount - kTinyCoalescingAccessCapacity);
        EXPECT_GT(observed.diagnostic_count, 0);
        EXPECT_FALSE(moi.check(/*stream=*/nullptr));
    }

    TEST(HipMoiMetadataCapacity, HostOwnedCoalescingFallbackCanAlsoFillExactAccessStorage)
    {
        hip_moi::host_context_options options;
        options.access_record_capacity              = 4;
        options.coalesced_access_record_capacity    = 0;
        options.coalescing_access_record_capacity   = 0;
        options.coalescing_group_record_capacity    = 0;
        options.diagnostic_capacity                 = kStoredSubgroupDiagnostics;
        options.subgroup_capacity                   = kSubgroupCount;
        hip_moi::subgroup_level_host_context moi(options);

        hip_moi::subgroup_level_context::storage_ref storage = moi.device_ref();
        hipLaunchKernelGGL(host_owned_coalescing_conflict_kernel,
                           dim3(1),
                           dim3(kSubgroupThreadCount),
                           0,
                           0,
                           storage);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_subgroup_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(
            storage, &observed, /*stored_diagnostic_count=*/kStoredSubgroupDiagnostics));

        EXPECT_EQ(observed.access_count, kSubgroupThreadCount);
        EXPECT_EQ(observed.coalescing_access_count, 0);
        EXPECT_EQ(observed.coalescing_fallback_count, kSubgroupThreadCount);
        EXPECT_GT(observed.diagnostic_count, 0);
        EXPECT_TRUE(
            has_metadata_full(observed, /*stored_diagnostic_count=*/kStoredSubgroupDiagnostics));
        EXPECT_FALSE(moi.check(/*stream=*/nullptr));
    }

    TEST(HipMoiMetadataCapacity, SubgroupLevelSingleSubgroupConflictsDoNotFillDiagnostics)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/32,
            /*diagnostic_capacity=*/kTruncatedDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(many_neighbor_conflicts_subgroup_level_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        ASSERT_NO_FATAL_FAILURE(hip_moi::test::expect_metadata_counts(
            storage, /*expected_access_count=*/2 * kThreadCount,
            /*expected_diagnostic_count=*/0));
    }
} // namespace
