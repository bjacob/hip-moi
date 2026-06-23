// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/026_lds_offset_api_test.hip
//
// Tests for explicit LDS-byte-offset access APIs.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int              kThreadCount        = 64;
    constexpr int              kThreadsPerSubgroup = 32;
    constexpr int              kSubgroupCount      = 2;
    constexpr int              kDiagnosticCapacity = 8;
    constexpr int              kExactShadowEntries = 1;
    constexpr hip_moi::site_id kProducerSite(0x101);
    constexpr hip_moi::site_id kConsumerSite(0x202);

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

    __global__ void offset_safe_kernel(int* out, hip_moi::context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&value, 77, /*lds_byte_offset=*/0, kProducerSite);
        }
        ctx.syncthreads();
        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            out[0] = ctx.lds_load_at(&value, /*lds_byte_offset=*/0, kConsumerSite);
        }
        ctx.finish();
    }

    __global__ void offset_race_kernel(hip_moi::context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&value, 77, /*lds_byte_offset=*/0, kProducerSite);
        }
        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            (void)ctx.lds_load_at(&value, /*lds_byte_offset=*/0, kConsumerSite);
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

    TEST(HipMoiLdsOffsetApi, OffsetAccessesPreserveCurrentSafeBehavior)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_exact_shadow(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*exact_shadow_entry_capacity=*/kExactShadowEntries));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
        int initial = -1;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(
            offset_safe_kernel, dim3(1), dim3(kThreadCount), 0, 0, device_output, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output = -1;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(output, 77);
        EXPECT_EQ(observed.diagnostic_count, 0);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiLdsOffsetApi, OffsetAccessesPreserveCurrentDiagnosticBehavior)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_exact_shadow(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*exact_shadow_entry_capacity=*/kExactShadowEntries));

        hipLaunchKernelGGL(offset_race_kernel, dim3(1), dim3(kThreadCount), 0, 0, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        ASSERT_GT(observed.diagnostic_count, 0);
        EXPECT_EQ(observed.diagnostics[0].kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
        EXPECT_EQ(observed.diagnostics[0].first_site_id, kProducerSite.value());
        EXPECT_EQ(observed.diagnostics[0].second_site_id, kConsumerSite.value());
    }
} // namespace
