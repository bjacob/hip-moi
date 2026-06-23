// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/003_site_id_test.hip
//
// Tests for source-site ids carried by shadow instrumentation diagnostics.
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

    constexpr hip_moi::site_id kCompileTimeSiteA = HIP_MOI_SITE_ID();
    constexpr hip_moi::site_id kCompileTimeSiteB = HIP_MOI_SITE_ID();

    static_assert(hip_moi::no_site_id.value() == 0);
    static_assert(hip_moi::site_id(7).value() == 7);
    static_assert(kCompileTimeSiteA.value() != 0);
    static_assert(kCompileTimeSiteB.value() != 0);

    struct observed_metadata
    {
        int                          diagnostic_count;
        hip_moi::context::diagnostic diagnostics[kDiagnosticCapacity];
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

    __global__ void conflict_site_ids_kernel(hip_moi::context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&value, 10, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
        }
        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            (void)ctx.lds_load_at(&value, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
        }
        ctx.finish();
    }

    template <typename Storage>
    void copy_observed_metadata(const Storage& storage, observed_metadata* observed)
    {
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->diagnostic_count,
                                          storage.diagnostic_count_device(),
                                          sizeof(observed->diagnostic_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(observed->diagnostics,
                                          storage.diagnostics_device(),
                                          sizeof(observed->diagnostics),
                                          hipMemcpyDeviceToHost));
    }

    TEST(HipMoiSiteId, MacroProducesNonzeroCompileTimeIds)
    {
        EXPECT_NE(kCompileTimeSiteA.value(), 0u);
        EXPECT_NE(kCompileTimeSiteB.value(), 0u);
        EXPECT_NE(kCompileTimeSiteA.value(), kCompileTimeSiteB.value());

        hip_moi::site_id explicit_site(123);
        EXPECT_EQ(explicit_site.value(), 123u);
        EXPECT_EQ(hip_moi::no_site_id.value(), 0u);
    }

    TEST(HipMoiSiteId, DiagnosticsCarrySiteIds)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_exact_shadow(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*exact_shadow_entry_capacity=*/1));

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

        ASSERT_GT(observed.diagnostic_count, 0);

        const hip_moi::context::diagnostic& diagnostic = observed.diagnostics[0];
        EXPECT_EQ(diagnostic.kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
        EXPECT_NE(diagnostic.first_site_id, 0u);
        EXPECT_NE(diagnostic.second_site_id, 0u);
        EXPECT_NE(diagnostic.first_site_id, diagnostic.second_site_id);
    }
} // namespace
