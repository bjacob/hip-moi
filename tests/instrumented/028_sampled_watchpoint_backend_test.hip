// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/028_sampled_watchpoint_backend_test.hip
//
// Tests for the sampled-watchpoint backend.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int              kThreadCount        = 64;
    constexpr int              kThreadsPerSubgroup = 32;
    constexpr int              kSubgroupCount      = 2;
    constexpr int              kAccessCapacity     = 1;
    constexpr int              kDiagnosticCapacity = 8;
    constexpr int              kWatchpointCapacity = 64;
    constexpr hip_moi::site_id kProducerSite(0x401);
    constexpr hip_moi::site_id kConsumerSite(0x502);

    struct observed_metadata
    {
        int                          access_count;
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

    __global__ void sampled_safe_kernel(hip_moi::context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store_at(&value, 77, /*lds_byte_offset=*/0, kProducerSite);
        }
        ctx.syncthreads();
        if(ctx.subgroup_id() == 1)
        {
            (void)ctx.lds_load_at(&value, /*lds_byte_offset=*/0, kConsumerSite);
        }
        ctx.finish();
    }

    __global__ void sampled_race_kernel(hip_moi::context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store_at(&value, 77, /*lds_byte_offset=*/0, kProducerSite);
        }
        if(ctx.subgroup_id() == 1)
        {
            (void)ctx.lds_load_at(&value, /*lds_byte_offset=*/0, kConsumerSite);
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
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(observed->diagnostics,
                                          storage.diagnostics_device(),
                                          sizeof(observed->diagnostics),
                                          hipMemcpyDeviceToHost));
    }

    hip_moi::context::storage_ref sampled_ref(hip_moi::test::device_context_storage& storage)
    {
        hip_moi::context::storage_ref ref = storage.ref();
        ref.backend = static_cast<uint32_t>(hip_moi::backend_kind::sampled_watchpoint);
        return ref;
    }

    TEST(HipMoiSampledWatchpointBackend, SynchronizedSelectedAccessesDoNotUseDeferredLog)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_sampled_watchpoints(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*sampled_watchpoint_capacity=*/kWatchpointCapacity));

        hipLaunchKernelGGL(
            sampled_safe_kernel, dim3(1), dim3(kThreadCount), 0, 0, sampled_ref(storage));
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(observed.access_count, 0);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    TEST(HipMoiSampledWatchpointBackend, RacySelectedAccessesDiagnoseWithoutDeferredLog)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_sampled_watchpoints(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*sampled_watchpoint_capacity=*/kWatchpointCapacity));

        hipLaunchKernelGGL(
            sampled_race_kernel, dim3(1), dim3(kThreadCount), 0, 0, sampled_ref(storage));
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(observed.access_count, 0);
        ASSERT_GT(observed.diagnostic_count, 0);
        EXPECT_EQ(observed.diagnostics[0].kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
    }
} // namespace
