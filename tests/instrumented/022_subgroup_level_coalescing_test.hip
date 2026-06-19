// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/022_subgroup_level_coalescing_test.hip
//
// Tests for subgroup-level coalescing proof logs and summaries.
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
    constexpr int kGroupCapacity      = 16;
    constexpr int kDiagnosticCapacity = 8;

    constexpr hip_moi::site_id kContiguousSite(0x201u);
    constexpr hip_moi::site_id kFixedStrideSite(0x202u);
    constexpr hip_moi::site_id kRepeatedSite(0x203u);
    constexpr hip_moi::site_id kIrregularSite(0x204u);
    constexpr hip_moi::site_id kTwoSubgroupSite(0x205u);
    constexpr hip_moi::site_id kDescendingSite(0x206u);
    constexpr hip_moi::site_id kManySite0(0x207u);
    constexpr hip_moi::site_id kManySite1(0x208u);
    constexpr hip_moi::site_id kManySite2(0x209u);
    constexpr hip_moi::site_id kManySite3(0x20au);

    struct observed_metadata
    {
        int access_count;
        int coalescing_proof_count;
        int coalescing_group_count;
        int coalesced_access_count;
        int diagnostic_count;
        hip_moi::subgroup_level_context::coalescing_proof_record
            coalescing_proof_records[kProofCapacity];
        hip_moi::subgroup_level_context::coalesced_access_record
            coalesced_access_records[kAccessCapacity];
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

    __global__ void default_site_kernel(hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadsPerSubgroup];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane = static_cast<int>(ctx.lane_in_subgroup());
        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store(&values[lane], lane);
        }
        ctx.syncthreads();
    }

    __global__ void contiguous_site_kernel(hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane = static_cast<int>(ctx.lane_in_subgroup());
        if(ctx.subgroup_id() == 1)
        {
            ctx.lds_store(&values[kThreadsPerSubgroup + lane], lane, kContiguousSite);
        }
        ctx.syncthreads();
    }

    __global__ void fixed_stride_site_kernel(hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[2 * kThreadsPerSubgroup];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane = static_cast<int>(ctx.lane_in_subgroup());
        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store(&values[2 * lane], lane, kFixedStrideSite);
        }
        ctx.syncthreads();
    }

    __global__ void repeated_lane_site_kernel(hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[2 * kThreadsPerSubgroup];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane = static_cast<int>(ctx.lane_in_subgroup());
        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store(&values[lane], lane, kRepeatedSite);
            ctx.lds_store(&values[kThreadsPerSubgroup + lane], lane, kRepeatedSite);
        }
        ctx.syncthreads();
    }

    __global__ void irregular_site_kernel(hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadsPerSubgroup + 1];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane  = static_cast<int>(ctx.lane_in_subgroup());
        int index = lane + (lane >= kThreadsPerSubgroup / 2 ? 1 : 0);
        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store(&values[index], lane, kIrregularSite);
        }
        ctx.syncthreads();
    }

    __global__ void two_subgroup_site_kernel(hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane     = static_cast<int>(ctx.lane_in_subgroup());
        int subgroup = static_cast<int>(ctx.subgroup_id());
        ctx.lds_store(&values[subgroup * kThreadsPerSubgroup + lane], lane, kTwoSubgroupSite);
        ctx.syncthreads();
    }

    __global__ void descending_site_kernel(hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadsPerSubgroup];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane = static_cast<int>(ctx.lane_in_subgroup());
        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store(&values[kThreadsPerSubgroup - 1 - lane], lane, kDescendingSite);
        }
        ctx.syncthreads();
    }

    __global__ void many_site_kernel(hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[4][kThreadsPerSubgroup];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane = static_cast<int>(ctx.lane_in_subgroup());
        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store(&values[0][lane], lane, kManySite0);
            ctx.lds_store(&values[1][lane], lane, kManySite1);
            ctx.lds_store(&values[2][lane], lane, kManySite2);
            ctx.lds_store(&values[3][lane], lane, kManySite3);
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
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->coalescing_group_count,
                                          storage.coalescing_group_count_device(),
                                          sizeof(observed->coalescing_group_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->coalesced_access_count,
                                          storage.coalesced_access_count_device(),
                                          sizeof(observed->coalesced_access_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->diagnostic_count,
                                          storage.diagnostic_count_device(),
                                          sizeof(observed->diagnostic_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(observed->coalescing_proof_records,
                                          storage.coalescing_proof_records_device(),
                                          sizeof(observed->coalescing_proof_records),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(observed->coalesced_access_records,
                                          storage.coalesced_access_records_device(),
                                          sizeof(observed->coalesced_access_records),
                                          hipMemcpyDeviceToHost));
    }

    template <typename Kernel>
    void run_case(Kernel kernel, observed_metadata* observed, int group_capacity = kGroupCapacity)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*coalescing_proof_capacity=*/kProofCapacity,
            /*coalescing_group_capacity=*/group_capacity);

        hipLaunchKernelGGL(kernel, dim3(1), dim3(kThreadCount), 0, 0, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        copy_observed_metadata(storage, observed);
    }

    int stored_proof_count(const observed_metadata& observed)
    {
        return observed.coalescing_proof_count < kProofCapacity ? observed.coalescing_proof_count
                                                                : kProofCapacity;
    }

    int stored_summary_count(const observed_metadata& observed)
    {
        return observed.coalesced_access_count < kAccessCapacity ? observed.coalesced_access_count
                                                                 : kAccessCapacity;
    }

    void expect_one_proof_per_lane(const observed_metadata& observed,
                                   hip_moi::site_id         expected_site,
                                   uint32_t                 expected_subgroup)
    {
        bool seen_lanes[kThreadsPerSubgroup] = {};
        int  proof_count                     = stored_proof_count(observed);
        int  matched_count                   = 0;

        for(int i = 0; i < proof_count; ++i)
        {
            const hip_moi::subgroup_level_context::coalescing_proof_record& record
                = observed.coalescing_proof_records[i];
            if(!record.valid || record.site_id != expected_site.value()
               || record.subgroup_id != expected_subgroup)
            {
                continue;
            }

            ASSERT_LT(record.lane, static_cast<uint32_t>(kThreadsPerSubgroup));
            EXPECT_FALSE(seen_lanes[record.lane]);
            seen_lanes[record.lane] = true;
            EXPECT_EQ(record.byte_count, static_cast<uint32_t>(sizeof(int)));
            EXPECT_EQ(record.kind, static_cast<uint32_t>(hip_moi::access_kind::store));
            EXPECT_EQ(record.epoch, 0u);
            ++matched_count;
        }

        EXPECT_EQ(matched_count, kThreadsPerSubgroup);
        for(bool seen : seen_lanes)
        {
            EXPECT_TRUE(seen);
        }
    }

    bool has_summary(const observed_metadata& observed,
                     hip_moi::site_id         expected_site,
                     uint32_t                 expected_subgroup,
                     int64_t                  expected_stride,
                     uint32_t                 expected_span)
    {
        int summary_count = stored_summary_count(observed);
        for(int i = 0; i < summary_count; ++i)
        {
            const hip_moi::subgroup_level_context::coalesced_access_record& record
                = observed.coalesced_access_records[i];
            if(record.valid && record.site_id == expected_site.value()
               && record.subgroup_id == expected_subgroup && record.first_lane == 0
               && record.participant_count == static_cast<uint32_t>(kThreadsPerSubgroup)
               && record.byte_count == static_cast<uint32_t>(sizeof(int))
               && record.kind == static_cast<uint32_t>(hip_moi::access_kind::store)
               && record.epoch == 0u && record.address_stride == expected_stride
               && record.span_byte_count == expected_span)
            {
                return true;
            }
        }
        return false;
    }

    TEST(HipMoiSubgroupLevelCoalescing, DefaultSiteDoesNotWriteProofRecords)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(default_site_kernel, &observed));

        EXPECT_EQ(observed.access_count, kThreadsPerSubgroup);
        EXPECT_EQ(observed.coalescing_proof_count, 0);
        EXPECT_EQ(observed.coalescing_group_count, 0);
        EXPECT_EQ(observed.coalesced_access_count, 0);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    TEST(HipMoiSubgroupLevelCoalescing, OptedInSiteWritesOneProofRecordPerLane)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(contiguous_site_kernel, &observed));

        EXPECT_EQ(observed.access_count, kThreadsPerSubgroup);
        EXPECT_EQ(observed.coalescing_proof_count, kThreadsPerSubgroup);
        EXPECT_EQ(observed.coalescing_group_count, 1);
        EXPECT_EQ(observed.diagnostic_count, 0);
        expect_one_proof_per_lane(observed, kContiguousSite, /*expected_subgroup=*/1);
    }

    TEST(HipMoiSubgroupLevelCoalescing, ContiguousLanePatternProducesSummary)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(contiguous_site_kernel, &observed));

        EXPECT_EQ(observed.coalesced_access_count, 1);
        EXPECT_EQ(observed.coalescing_group_count, 1);
        EXPECT_TRUE(has_summary(observed,
                                kContiguousSite,
                                /*expected_subgroup=*/1,
                                /*expected_stride=*/static_cast<int64_t>(sizeof(int)),
                                /*expected_span=*/
                                static_cast<uint32_t>(kThreadsPerSubgroup * sizeof(int))));
    }

    TEST(HipMoiSubgroupLevelCoalescing, FixedStrideLanePatternProducesSummary)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(fixed_stride_site_kernel, &observed));

        EXPECT_EQ(observed.coalesced_access_count, 1);
        EXPECT_EQ(observed.coalescing_group_count, 1);
        EXPECT_TRUE(
            has_summary(observed,
                        kFixedStrideSite,
                        /*expected_subgroup=*/0,
                        /*expected_stride=*/static_cast<int64_t>(2 * sizeof(int)),
                        /*expected_span=*/
                        static_cast<uint32_t>((2 * (kThreadsPerSubgroup - 1) + 1) * sizeof(int))));
    }

    TEST(HipMoiSubgroupLevelCoalescing, DescendingLanePatternProducesSummary)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(descending_site_kernel, &observed));

        EXPECT_EQ(observed.coalesced_access_count, 1);
        EXPECT_EQ(observed.coalescing_group_count, 1);
        EXPECT_TRUE(has_summary(observed,
                                kDescendingSite,
                                /*expected_subgroup=*/0,
                                /*expected_stride=*/-static_cast<int64_t>(sizeof(int)),
                                /*expected_span=*/
                                static_cast<uint32_t>(kThreadsPerSubgroup * sizeof(int))));
    }

    TEST(HipMoiSubgroupLevelCoalescing, RepeatedLaneSiteStaysExactOnly)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(repeated_lane_site_kernel, &observed));

        EXPECT_EQ(observed.access_count, 2 * kThreadsPerSubgroup);
        EXPECT_EQ(observed.coalescing_proof_count, 2 * kThreadsPerSubgroup);
        EXPECT_EQ(observed.coalescing_group_count, 1);
        EXPECT_EQ(observed.coalesced_access_count, 0);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    TEST(HipMoiSubgroupLevelCoalescing, IrregularLanePatternStaysExactOnly)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(irregular_site_kernel, &observed));

        EXPECT_EQ(observed.access_count, kThreadsPerSubgroup);
        EXPECT_EQ(observed.coalescing_proof_count, kThreadsPerSubgroup);
        EXPECT_EQ(observed.coalescing_group_count, 1);
        EXPECT_EQ(observed.coalesced_access_count, 0);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    TEST(HipMoiSubgroupLevelCoalescing, IndependentSubgroupsProduceSeparateSummaries)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(two_subgroup_site_kernel, &observed));

        EXPECT_EQ(observed.access_count, kThreadCount);
        EXPECT_EQ(observed.coalescing_proof_count, kThreadCount);
        EXPECT_EQ(observed.coalescing_group_count, kSubgroupCount);
        EXPECT_EQ(observed.coalesced_access_count, kSubgroupCount);
        EXPECT_EQ(observed.diagnostic_count, 0);
        EXPECT_TRUE(has_summary(observed,
                                kTwoSubgroupSite,
                                /*expected_subgroup=*/0,
                                /*expected_stride=*/static_cast<int64_t>(sizeof(int)),
                                /*expected_span=*/
                                static_cast<uint32_t>(kThreadsPerSubgroup * sizeof(int))));
        EXPECT_TRUE(has_summary(observed,
                                kTwoSubgroupSite,
                                /*expected_subgroup=*/1,
                                /*expected_stride=*/static_cast<int64_t>(sizeof(int)),
                                /*expected_span=*/
                                static_cast<uint32_t>(kThreadsPerSubgroup * sizeof(int))));
    }

    TEST(HipMoiSubgroupLevelCoalescing, ManySitesProduceSeparateGroupsAndSummaries)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(many_site_kernel, &observed));

        EXPECT_EQ(observed.access_count, 4 * kThreadsPerSubgroup);
        EXPECT_EQ(observed.coalescing_proof_count, 4 * kThreadsPerSubgroup);
        EXPECT_EQ(observed.coalescing_group_count, 4);
        EXPECT_EQ(observed.coalesced_access_count, 4);
        EXPECT_EQ(observed.diagnostic_count, 0);
        EXPECT_TRUE(has_summary(observed,
                                kManySite0,
                                /*expected_subgroup=*/0,
                                /*expected_stride=*/static_cast<int64_t>(sizeof(int)),
                                /*expected_span=*/
                                static_cast<uint32_t>(kThreadsPerSubgroup * sizeof(int))));
        EXPECT_TRUE(has_summary(observed,
                                kManySite1,
                                /*expected_subgroup=*/0,
                                /*expected_stride=*/static_cast<int64_t>(sizeof(int)),
                                /*expected_span=*/
                                static_cast<uint32_t>(kThreadsPerSubgroup * sizeof(int))));
        EXPECT_TRUE(has_summary(observed,
                                kManySite2,
                                /*expected_subgroup=*/0,
                                /*expected_stride=*/static_cast<int64_t>(sizeof(int)),
                                /*expected_span=*/
                                static_cast<uint32_t>(kThreadsPerSubgroup * sizeof(int))));
        EXPECT_TRUE(has_summary(observed,
                                kManySite3,
                                /*expected_subgroup=*/0,
                                /*expected_stride=*/static_cast<int64_t>(sizeof(int)),
                                /*expected_span=*/
                                static_cast<uint32_t>(kThreadsPerSubgroup * sizeof(int))));
    }

    TEST(HipMoiSubgroupLevelCoalescing, GroupCapacityOverflowFallsBackToProofScan)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(
            run_case(two_subgroup_site_kernel, &observed, /*group_capacity=*/1));

        EXPECT_EQ(observed.access_count, kThreadCount);
        EXPECT_EQ(observed.coalescing_proof_count, kThreadCount);
        EXPECT_EQ(observed.coalescing_group_count, 1);
        EXPECT_EQ(observed.coalesced_access_count, kSubgroupCount);
        EXPECT_EQ(observed.diagnostic_count, 0);
        EXPECT_TRUE(has_summary(observed,
                                kTwoSubgroupSite,
                                /*expected_subgroup=*/0,
                                /*expected_stride=*/static_cast<int64_t>(sizeof(int)),
                                /*expected_span=*/
                                static_cast<uint32_t>(kThreadsPerSubgroup * sizeof(int))));
        EXPECT_TRUE(has_summary(observed,
                                kTwoSubgroupSite,
                                /*expected_subgroup=*/1,
                                /*expected_stride=*/static_cast<int64_t>(sizeof(int)),
                                /*expected_span=*/
                                static_cast<uint32_t>(kThreadsPerSubgroup * sizeof(int))));
    }
} // namespace
