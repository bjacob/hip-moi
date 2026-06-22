// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/020_site_id_test.hip
//
// Tests for source-site ids carried by exact access-level instrumentation.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kAccessCapacity     = 8;
    constexpr int kDiagnosticCapacity = 8;

    constexpr hip_moi::site_id kCompileTimeSiteA = HIP_MOI_SITE_ID();
    constexpr hip_moi::site_id kCompileTimeSiteB = HIP_MOI_SITE_ID();

    static_assert(hip_moi::no_site_id.value() == 0);
    static_assert(!hip_moi::no_site_id.allows_coalescing());
    static_assert(hip_moi::site_id(7).value() == 7);
    static_assert(hip_moi::site_id(7).allows_coalescing());
    static_assert(kCompileTimeSiteA.value() != 0);
    static_assert(kCompileTimeSiteB.value() != 0);

    struct observed_metadata
    {
        int                             access_count;
        int                             diagnostic_count;
        hip_moi::context::access_record access_records[kAccessCapacity];
        hip_moi::context::diagnostic    diagnostics[kDiagnosticCapacity];
    };

    __device__ hip_moi::context make_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    __global__ void record_site_ids_kernel(hip_moi::context::storage_ref storage)
    {
        __shared__ int values[3];

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&values[0], 10, HIP_MOI_SITE_ID());
        }
        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            ctx.lds_store(&values[1], 20, HIP_MOI_SITE_ID());
        }
        if(ctx.thread_id() == kThreadsPerSubgroup + 1)
        {
            ctx.lds_store(&values[2], 30);
        }
        ctx.finish();
    }

    __global__ void conflict_site_ids_kernel(hip_moi::context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&value, 10, HIP_MOI_SITE_ID());
        }
        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            (void)ctx.lds_load(&value, HIP_MOI_SITE_ID());
        }
        ctx.finish();
    }

    template <typename Storage>
    void copy_observed_metadata(const Storage& storage, observed_metadata* observed)
    {
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->access_count,
                                          storage.access_count_device(),
                                          sizeof(observed->access_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->diagnostic_count,
                                          storage.diagnostic_count_device(),
                                          sizeof(observed->diagnostic_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(observed->access_records,
                                          storage.access_records_device(),
                                          sizeof(observed->access_records),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(observed->diagnostics,
                                          storage.diagnostics_device(),
                                          sizeof(observed->diagnostics),
                                          hipMemcpyDeviceToHost));
    }

    int valid_access_count_with_site_id(const observed_metadata& observed, uint64_t site_id)
    {
        int count = 0;
        for(int i = 0; i < kAccessCapacity; ++i)
        {
            if(observed.access_records[i].valid && observed.access_records[i].site_id == site_id)
            {
                ++count;
            }
        }
        return count;
    }

    void expect_some_diagnostic_with_nonzero_site_ids(const observed_metadata& observed)
    {
        int stored_count = observed.diagnostic_count;
        if(stored_count > kDiagnosticCapacity)
        {
            stored_count = kDiagnosticCapacity;
        }

        bool found = false;
        for(int i = 0; i < stored_count; ++i)
        {
            const hip_moi::context::diagnostic& diagnostic = observed.diagnostics[i];
            found |= diagnostic.kind
                         == static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict)
                     && diagnostic.first_site_id != 0 && diagnostic.second_site_id != 0
                     && diagnostic.first_site_id != diagnostic.second_site_id;
        }
        EXPECT_TRUE(found);
    }

    TEST(HipMoiSiteId, MacroProducesNonzeroCompileTimeIds)
    {
        EXPECT_NE(kCompileTimeSiteA.value(), 0u);
        EXPECT_NE(kCompileTimeSiteB.value(), 0u);
        EXPECT_NE(kCompileTimeSiteA.value(), kCompileTimeSiteB.value());

        hip_moi::site_id explicit_site(123);
        EXPECT_EQ(explicit_site.value(), 123u);
        EXPECT_TRUE(explicit_site.allows_coalescing());
        EXPECT_FALSE(hip_moi::no_site_id.allows_coalescing());
    }

    TEST(HipMoiSiteId, AccessRecordsCarryExplicitAndDefaultSiteIds)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        hipLaunchKernelGGL(
            record_site_ids_kernel, dim3(1), dim3(kThreadCount), 0, 0, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(observed.access_count, 3);
        EXPECT_EQ(observed.diagnostic_count, 0);

        int zero_site_records    = valid_access_count_with_site_id(observed, /*site_id=*/0);
        int nonzero_site_records = 0;
        for(int i = 0; i < kAccessCapacity; ++i)
        {
            if(observed.access_records[i].valid && observed.access_records[i].site_id != 0)
            {
                ++nonzero_site_records;
            }
        }
        EXPECT_EQ(zero_site_records, 1);
        EXPECT_EQ(nonzero_site_records, 2);
    }

    TEST(HipMoiSiteId, DiagnosticsCarrySiteIds)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        hipLaunchKernelGGL(conflict_site_ids_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(observed.access_count, 2);
        ASSERT_GT(observed.diagnostic_count, 0);

        const hip_moi::context::diagnostic& diagnostic = observed.diagnostics[0];
        EXPECT_EQ(diagnostic.kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
        expect_some_diagnostic_with_nonzero_site_ids(observed);
    }
} // namespace
