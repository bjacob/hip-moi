// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/023_subgroup_level_coalesced_conflict_test.hip
//
// Tests for subgroup-level conflict diagnostics produced from coalesced
// access summaries.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kThreadCount        = kThreadsPerSubgroup * kSubgroupCount;
    constexpr int kAccessCapacity     = 160;
    constexpr int kProofCapacity      = 160;
    constexpr int kDiagnosticCapacity = 512;

    constexpr uint32_t kContiguousSpan = kThreadsPerSubgroup * sizeof(int);
    constexpr uint32_t kStrideSpan     = (2 * (kThreadsPerSubgroup - 1) + 1) * sizeof(int);

    constexpr hip_moi::site_id kContiguousStoreSite(0x2301u);
    constexpr hip_moi::site_id kContiguousLoadSite(0x2302u);
    constexpr hip_moi::site_id kDisjointStoreSite0(0x2303u);
    constexpr hip_moi::site_id kDisjointStoreSite1(0x2304u);
    constexpr hip_moi::site_id kStrideEvenStoreSite(0x2305u);
    constexpr hip_moi::site_id kStrideOddStoreSite(0x2306u);
    constexpr hip_moi::site_id kStrideEvenLoadSite(0x2307u);
    constexpr hip_moi::site_id kReadSite0(0x2308u);
    constexpr hip_moi::site_id kReadSite1(0x2309u);
    constexpr hip_moi::site_id kSummaryExactStoreSite(0x230au);

    struct observed_metadata
    {
        int                                         access_count;
        int                                         coalescing_proof_count;
        int                                         coalesced_access_count;
        int                                         diagnostic_count;
        hip_moi::subgroup_level_context::diagnostic diagnostics[kDiagnosticCapacity];
    };

    __device__ hip_moi::subgroup_level_context
               make_context(hip_moi::subgroup_level_context::storage_ref storage)
    {
        hip_moi::subgroup_level_context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::subgroup_level_context(storage, cfg);
    }

    __global__ void
        contiguous_summary_conflict_kernel(hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadsPerSubgroup];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane = static_cast<int>(ctx.lane_in_subgroup());
        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store(&values[lane], lane, kContiguousStoreSite);
        }
        if(ctx.subgroup_id() == 1)
        {
            (void)ctx.lds_load(&values[lane], kContiguousLoadSite);
        }
        ctx.syncthreads();
    }

    __global__ void disjoint_summary_kernel(hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane     = static_cast<int>(ctx.lane_in_subgroup());
        int subgroup = static_cast<int>(ctx.subgroup_id());
        if(subgroup == 0)
        {
            ctx.lds_store(&values[lane], lane, kDisjointStoreSite0);
        }
        if(subgroup == 1)
        {
            ctx.lds_store(&values[kThreadsPerSubgroup + lane], lane, kDisjointStoreSite1);
        }
        ctx.syncthreads();
    }

    __global__ void fixed_stride_gap_kernel(hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[2 * kThreadsPerSubgroup];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane = static_cast<int>(ctx.lane_in_subgroup());
        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store(&values[2 * lane], lane, kStrideEvenStoreSite);
        }
        if(ctx.subgroup_id() == 1)
        {
            ctx.lds_store(&values[2 * lane + 1], lane, kStrideOddStoreSite);
        }
        ctx.syncthreads();
    }

    __global__ void
        fixed_stride_summary_conflict_kernel(hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[2 * kThreadsPerSubgroup];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane = static_cast<int>(ctx.lane_in_subgroup());
        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store(&values[2 * lane], lane, kStrideEvenStoreSite);
        }
        if(ctx.subgroup_id() == 1)
        {
            (void)ctx.lds_load(&values[2 * lane], kStrideEvenLoadSite);
        }
        ctx.syncthreads();
    }

    __global__ void coalesced_read_read_kernel(hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadsPerSubgroup];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane = static_cast<int>(ctx.lane_in_subgroup());
        if(ctx.subgroup_id() == 0)
        {
            values[lane] = lane;
        }
        __syncthreads();

        if(ctx.subgroup_id() == 0)
        {
            (void)ctx.lds_load(&values[lane], kReadSite0);
        }
        if(ctx.subgroup_id() == 1)
        {
            (void)ctx.lds_load(&values[lane], kReadSite1);
        }
        ctx.syncthreads();
    }

    __global__ void
        coalesced_summary_vs_exact_kernel(hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadsPerSubgroup];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane = static_cast<int>(ctx.lane_in_subgroup());
        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store(&values[lane], lane, kSummaryExactStoreSite);
        }
        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            (void)ctx.lds_load(&values[7]);
        }
        ctx.syncthreads();
    }

    void copy_observed_metadata(const hip_moi::test::subgroup_level_device_context_storage& storage,
                                observed_metadata* observed)
    {
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->access_count,
                                          storage.access_count_device(),
                                          sizeof(observed->access_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->coalescing_proof_count,
                                          storage.coalescing_proof_count_device(),
                                          sizeof(observed->coalescing_proof_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->coalesced_access_count,
                                          storage.coalesced_access_count_device(),
                                          sizeof(observed->coalesced_access_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->diagnostic_count,
                                          storage.diagnostic_count_device(),
                                          sizeof(observed->diagnostic_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(observed->diagnostics,
                                          storage.diagnostics_device(),
                                          sizeof(observed->diagnostics),
                                          hipMemcpyDeviceToHost));
    }

    template <typename Kernel>
    void run_case(Kernel kernel, observed_metadata* observed)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*coalescing_proof_capacity=*/kProofCapacity);

        hipLaunchKernelGGL(kernel, dim3(1), dim3(kThreadCount), 0, 0, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        copy_observed_metadata(storage, observed);
    }

    int stored_diagnostic_count(const observed_metadata& observed)
    {
        return observed.diagnostic_count < kDiagnosticCapacity ? observed.diagnostic_count
                                                               : kDiagnosticCapacity;
    }

    bool diagnostic_matches_subgroups(const hip_moi::subgroup_level_context::diagnostic& diagnostic,
                                      uint32_t first_subgroup,
                                      uint32_t second_subgroup)
    {
        return (diagnostic.first_subgroup_id == first_subgroup
                && diagnostic.second_subgroup_id == second_subgroup)
               || (diagnostic.first_subgroup_id == second_subgroup
                   && diagnostic.second_subgroup_id == first_subgroup);
    }

    bool diagnostic_matches_sites_and_sizes(
        const hip_moi::subgroup_level_context::diagnostic& diagnostic,
        hip_moi::site_id                                   first_site,
        hip_moi::site_id                                   second_site,
        uint32_t                                           first_size,
        uint32_t                                           second_size)
    {
        bool forward = diagnostic.first_site_id == first_site.value()
                       && diagnostic.second_site_id == second_site.value()
                       && diagnostic.first_size == first_size
                       && diagnostic.second_size == second_size;
        bool reverse = diagnostic.first_site_id == second_site.value()
                       && diagnostic.second_site_id == first_site.value()
                       && diagnostic.first_size == second_size
                       && diagnostic.second_size == first_size;
        return forward || reverse;
    }

    bool has_coalesced_conflict_diagnostic(const observed_metadata& observed,
                                           hip_moi::site_id         first_site,
                                           hip_moi::site_id         second_site,
                                           uint32_t                 first_size,
                                           uint32_t                 second_size)
    {
        int stored_count = stored_diagnostic_count(observed);
        for(int i = 0; i < stored_count; ++i)
        {
            const hip_moi::subgroup_level_context::diagnostic& diagnostic = observed.diagnostics[i];
            if(diagnostic.kind == static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict)
               && diagnostic.epoch == 0u
               && diagnostic_matches_subgroups(diagnostic,
                                               /*first_subgroup=*/0,
                                               /*second_subgroup=*/1)
               && diagnostic_matches_sites_and_sizes(
                   diagnostic, first_site, second_site, first_size, second_size))
            {
                return true;
            }
        }
        return false;
    }

    TEST(HipMoiSubgroupLevelCoalescedConflict, ContiguousSummariesReportConflict)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(contiguous_summary_conflict_kernel, &observed));

        EXPECT_EQ(observed.access_count, kThreadCount);
        EXPECT_EQ(observed.coalescing_proof_count, kThreadCount);
        EXPECT_EQ(observed.coalesced_access_count, kSubgroupCount);
        EXPECT_TRUE(has_coalesced_conflict_diagnostic(
            observed, kContiguousStoreSite, kContiguousLoadSite, kContiguousSpan, kContiguousSpan));
    }

    TEST(HipMoiSubgroupLevelCoalescedConflict, DisjointSummariesDoNotReport)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(disjoint_summary_kernel, &observed));

        EXPECT_EQ(observed.access_count, kThreadCount);
        EXPECT_EQ(observed.coalescing_proof_count, kThreadCount);
        EXPECT_EQ(observed.coalesced_access_count, kSubgroupCount);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    TEST(HipMoiSubgroupLevelCoalescedConflict, FixedStrideGapsDoNotReport)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(fixed_stride_gap_kernel, &observed));

        EXPECT_EQ(observed.access_count, kThreadCount);
        EXPECT_EQ(observed.coalescing_proof_count, kThreadCount);
        EXPECT_EQ(observed.coalesced_access_count, kSubgroupCount);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    TEST(HipMoiSubgroupLevelCoalescedConflict, FixedStrideSummariesReportConflict)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(fixed_stride_summary_conflict_kernel, &observed));

        EXPECT_EQ(observed.access_count, kThreadCount);
        EXPECT_EQ(observed.coalescing_proof_count, kThreadCount);
        EXPECT_EQ(observed.coalesced_access_count, kSubgroupCount);
        EXPECT_TRUE(has_coalesced_conflict_diagnostic(
            observed, kStrideEvenStoreSite, kStrideEvenLoadSite, kStrideSpan, kStrideSpan));
    }

    TEST(HipMoiSubgroupLevelCoalescedConflict, CoalescedReadReadDoesNotReport)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(coalesced_read_read_kernel, &observed));

        EXPECT_EQ(observed.access_count, kThreadCount);
        EXPECT_EQ(observed.coalescing_proof_count, kThreadCount);
        EXPECT_EQ(observed.coalesced_access_count, kSubgroupCount);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    TEST(HipMoiSubgroupLevelCoalescedConflict, SummaryVsExactReportsConflict)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(coalesced_summary_vs_exact_kernel, &observed));

        EXPECT_EQ(observed.access_count, kThreadsPerSubgroup + 1);
        EXPECT_EQ(observed.coalescing_proof_count, kThreadsPerSubgroup);
        EXPECT_EQ(observed.coalesced_access_count, 1);
        EXPECT_TRUE(has_coalesced_conflict_diagnostic(
            observed, kSummaryExactStoreSite, hip_moi::no_site_id, kContiguousSpan, sizeof(int)));
    }
} // namespace
