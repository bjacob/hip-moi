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

    template <typename Context>
    struct observed_metadata
    {
        using diagnostic_type = typename Context::diagnostic;

        int                             access_count;
        int                             diagnostic_count;
        typename Context::access_record access_records[kAccessCapacity];
        diagnostic_type                 diagnostics[kDiagnosticCapacity];
    };

    using observed_thread_level_metadata   = observed_metadata<hip_moi::thread_level_context>;
    using observed_subgroup_level_metadata = observed_metadata<hip_moi::subgroup_level_context>;

    __device__ hip_moi::thread_level_context
               make_thread_level_context(hip_moi::thread_level_context::storage_ref storage)
    {
        hip_moi::thread_level_context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::thread_level_context(storage, cfg);
    }

    __device__ hip_moi::subgroup_level_context
               make_subgroup_level_context(hip_moi::subgroup_level_context::storage_ref storage)
    {
        hip_moi::subgroup_level_context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::subgroup_level_context(storage, cfg);
    }

    __global__ void
        thread_level_record_site_ids_kernel(hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int values[3];

        hip_moi::thread_level_context ctx = make_thread_level_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&values[0], 10, HIP_MOI_SITE_ID());
        }
        if(ctx.thread_id() == 1)
        {
            ctx.lds_store(&values[1], 20, HIP_MOI_SITE_ID());
        }
        if(ctx.thread_id() == 2)
        {
            ctx.lds_store(&values[2], 30);
        }
    }

    __global__ void
        thread_level_conflict_site_ids_kernel(hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::thread_level_context ctx = make_thread_level_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&value, 10, HIP_MOI_SITE_ID());
        }
        if(ctx.thread_id() == 1)
        {
            (void)ctx.lds_load(&value, HIP_MOI_SITE_ID());
        }
    }

    __global__ void subgroup_level_conflict_site_ids_kernel(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::subgroup_level_context ctx = make_subgroup_level_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&value, 10, HIP_MOI_SITE_ID());
        }
        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            (void)ctx.lds_load(&value, HIP_MOI_SITE_ID());
        }
    }

    template <typename Storage, typename Observed>
    void copy_observed_metadata(const Storage& storage, Observed* observed)
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

    template <typename Observed>
    int valid_access_count_with_site_id(const Observed& observed, uint64_t site_id)
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

    template <typename Observed>
    void expect_some_diagnostic_with_nonzero_site_ids(const Observed& observed)
    {
        int stored_count = observed.diagnostic_count;
        if(stored_count > kDiagnosticCapacity)
        {
            stored_count = kDiagnosticCapacity;
        }

        bool found = false;
        for(int i = 0; i < stored_count; ++i)
        {
            const typename Observed::diagnostic_type& diagnostic = observed.diagnostics[i];
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

    TEST(HipMoiSiteId, ThreadLevelAccessRecordsCarryExplicitAndDefaultSiteIds)
    {
        hip_moi::test::thread_level_device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        hipLaunchKernelGGL(
            thread_level_record_site_ids_kernel, dim3(1), dim3(kThreadCount), 0, 0, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_thread_level_metadata observed = {};
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

    TEST(HipMoiSiteId, ThreadLevelDiagnosticsCarrySiteIds)
    {
        hip_moi::test::thread_level_device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        hipLaunchKernelGGL(thread_level_conflict_site_ids_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_thread_level_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(observed.access_count, 2);
        ASSERT_GT(observed.diagnostic_count, 0);

        const hip_moi::thread_level_context::diagnostic& diagnostic = observed.diagnostics[0];
        EXPECT_EQ(diagnostic.kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
        expect_some_diagnostic_with_nonzero_site_ids(observed);
    }

    TEST(HipMoiSiteId, SubgroupLevelDiagnosticsCarrySiteIds)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        hipLaunchKernelGGL(subgroup_level_conflict_site_ids_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_subgroup_level_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(observed.access_count, 2);
        ASSERT_GT(observed.diagnostic_count, 0);

        const hip_moi::subgroup_level_context::diagnostic& diagnostic = observed.diagnostics[0];
        EXPECT_EQ(diagnostic.kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
        expect_some_diagnostic_with_nonzero_site_ids(observed);
    }
} // namespace
