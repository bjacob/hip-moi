// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/021_coalescing_opportunity_test.hip
//
// Tests for conservative access coalescing opportunity detection.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 32;
    constexpr int kAccessCapacity     = 96;
    constexpr int kDiagnosticCapacity = 8;

    constexpr hip_moi::site_id kContiguousSite(0x101u);
    constexpr hip_moi::site_id kFixedStrideSite(0x102u);
    constexpr hip_moi::site_id kRepeatedSite(0x103u);
    constexpr hip_moi::site_id kIrregularSite(0x104u);

    struct observed_metadata
    {
        int access_count;
        int coalesced_access_count;
        int diagnostic_count;
        hip_moi::thread_level_context::coalesced_access_record
            coalesced_access_records[kAccessCapacity];
    };

    __global__ void contiguous_site_kernel(hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::thread_level_context ctx
            = hip_moi::test::make_single_subgroup_context<hip_moi::thread_level_context>(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(ctx.thread_id());
        ctx.lds_store(&values[tid], tid, kContiguousSite);
        ctx.syncthreads();
    }

    __global__ void fixed_stride_site_kernel(hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int values[2 * kThreadCount];

        hip_moi::thread_level_context ctx
            = hip_moi::test::make_single_subgroup_context<hip_moi::thread_level_context>(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(ctx.thread_id());
        ctx.lds_store(&values[2 * tid], tid, kFixedStrideSite);
        ctx.syncthreads();
    }

    __global__ void default_site_kernel(hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::thread_level_context ctx
            = hip_moi::test::make_single_subgroup_context<hip_moi::thread_level_context>(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(ctx.thread_id());
        ctx.lds_store(&values[tid], tid);
        ctx.syncthreads();
    }

    __global__ void repeated_dynamic_site_kernel(hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int values[2 * kThreadCount];

        hip_moi::thread_level_context ctx
            = hip_moi::test::make_single_subgroup_context<hip_moi::thread_level_context>(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(ctx.thread_id());
        ctx.lds_store(&values[tid], tid, kRepeatedSite);
        ctx.lds_store(&values[tid + kThreadCount], tid, kRepeatedSite);
        ctx.syncthreads();
    }

    __global__ void irregular_site_kernel(hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadCount + 1];

        hip_moi::thread_level_context ctx
            = hip_moi::test::make_single_subgroup_context<hip_moi::thread_level_context>(storage);
        ctx.init_workgroup();

        int tid   = static_cast<int>(ctx.thread_id());
        int index = tid + (tid >= kThreadCount / 2 ? 1 : 0);
        ctx.lds_store(&values[index], tid, kIrregularSite);
        ctx.syncthreads();
    }

    void copy_observed_metadata(const hip_moi::test::thread_level_device_context_storage& storage,
                                observed_metadata*                                        observed)
    {
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->access_count,
                                          storage.access_count_device(),
                                          sizeof(observed->access_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->coalesced_access_count,
                                          storage.coalesced_access_count_device(),
                                          sizeof(observed->coalesced_access_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->diagnostic_count,
                                          storage.diagnostic_count_device(),
                                          sizeof(observed->diagnostic_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(observed->coalesced_access_records,
                                          storage.coalesced_access_records_device(),
                                          sizeof(observed->coalesced_access_records),
                                          hipMemcpyDeviceToHost));
    }

    template <typename Kernel>
    void run_case(Kernel kernel, observed_metadata* observed)
    {
        hip_moi::test::thread_level_device_context_storage storage;
        storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1);

        hipLaunchKernelGGL(kernel, dim3(1), dim3(kThreadCount), 0, 0, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        copy_observed_metadata(storage, observed);
    }

    void expect_one_regular_summary(const observed_metadata& observed,
                                    hip_moi::site_id         expected_site,
                                    int64_t                  expected_stride,
                                    uint32_t                 expected_span)
    {
        EXPECT_EQ(observed.access_count, kThreadCount);
        EXPECT_EQ(observed.coalesced_access_count, 1);
        EXPECT_EQ(observed.diagnostic_count, 0);

        const hip_moi::thread_level_context::coalesced_access_record& record
            = observed.coalesced_access_records[0];
        EXPECT_EQ(record.valid, 1u);
        EXPECT_EQ(record.site_id, expected_site.value());
        EXPECT_EQ(record.kind, static_cast<uint32_t>(hip_moi::access_kind::store));
        EXPECT_EQ(record.epoch, 0u);
        EXPECT_EQ(record.subgroup_id, 0u);
        EXPECT_EQ(record.first_thread_id, 0u);
        EXPECT_EQ(record.participant_count, static_cast<uint32_t>(kThreadCount));
        EXPECT_EQ(record.byte_count, static_cast<uint32_t>(sizeof(int)));
        EXPECT_EQ(record.address_stride, expected_stride);
        EXPECT_EQ(record.span_byte_count, expected_span);
    }

    TEST(HipMoiCoalescingOpportunity, ContiguousSiteProducesOneSummary)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(contiguous_site_kernel, &observed));

        expect_one_regular_summary(
            observed,
            kContiguousSite,
            /*expected_stride=*/static_cast<int64_t>(sizeof(int)),
            /*expected_span=*/static_cast<uint32_t>(kThreadCount * sizeof(int)));
    }

    TEST(HipMoiCoalescingOpportunity, FixedStrideSiteProducesOneSummary)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(fixed_stride_site_kernel, &observed));

        expect_one_regular_summary(
            observed,
            kFixedStrideSite,
            /*expected_stride=*/static_cast<int64_t>(2 * sizeof(int)),
            /*expected_span=*/static_cast<uint32_t>((2 * (kThreadCount - 1) + 1) * sizeof(int)));
    }

    TEST(HipMoiCoalescingOpportunity, DefaultSiteStaysExactOnly)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(default_site_kernel, &observed));

        EXPECT_EQ(observed.access_count, kThreadCount);
        EXPECT_EQ(observed.coalesced_access_count, 0);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    TEST(HipMoiCoalescingOpportunity, RepeatedDynamicSiteStaysExactOnly)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(repeated_dynamic_site_kernel, &observed));

        EXPECT_EQ(observed.access_count, 2 * kThreadCount);
        EXPECT_EQ(observed.coalesced_access_count, 0);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    TEST(HipMoiCoalescingOpportunity, IrregularSiteStaysExactOnly)
    {
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(irregular_site_kernel, &observed));

        EXPECT_EQ(observed.access_count, kThreadCount);
        EXPECT_EQ(observed.coalesced_access_count, 0);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }
} // namespace
