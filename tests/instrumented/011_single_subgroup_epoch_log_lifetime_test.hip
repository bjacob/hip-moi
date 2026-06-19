// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/011_single_subgroup_epoch_log_lifetime_test.hip
//
// Tests for epoch-local access-log lifetime.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 8;
    constexpr int kIterationCount     = 16;
    constexpr int kDiagnosticCapacity = 8;

    struct observed_metadata
    {
        int                 access_count;
        int                 diagnostic_count;
        hip_moi::thread_level_context::diagnostic diagnostics[kDiagnosticCapacity];
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

    __global__ void long_scalar_producer_consumer_kernel(
        int* out, hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        for(int iter = 0; iter < kIterationCount; ++iter)
        {
            if(threadIdx.x == 0)
            {
                ctx.lds_store(&value, 200 + iter);
            }

            ctx.syncthreads();

            if(threadIdx.x == 1)
            {
                out[iter] = ctx.lds_load(&value);
            }

            ctx.syncthreads();
        }
    }

    __global__ void long_all_threads_own_slot_kernel(int*                         out,
                                                     hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        for(int iter = 0; iter < kIterationCount; ++iter)
        {
            ctx.lds_store(&values[tid], 1000 + iter * kThreadCount + tid);

            ctx.syncthreads();

            out[iter * kThreadCount + tid] = ctx.lds_load(&values[tid]);

            ctx.syncthreads();
        }
    }

    __global__ void same_epoch_overflow_then_barrier_kernel(
        hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int values[3];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&values[0], 10);
            ctx.lds_store(&values[1], 20);
            ctx.lds_store(&values[2], 30);
        }

        ctx.syncthreads();
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
    }

    void expect_no_diagnostics(const observed_metadata& observed, int expected_access_count)
    {
        EXPECT_EQ(observed.access_count, expected_access_count);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    TEST(HipMoiEpochLogLifetime, LongScalarLoopReusesSingleAccessSlotPerEpoch)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/1, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kIterationCount * sizeof(int)));

        hipLaunchKernelGGL(long_scalar_producer_consumer_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[kIterationCount] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        for(int iter = 0; iter < kIterationCount; ++iter)
        {
            EXPECT_EQ(output[iter], 200 + iter);
        }
        expect_no_diagnostics(observed, /*expected_access_count=*/2 * kIterationCount);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiEpochLogLifetime, LongAllThreadLoopReusesWorkgroupEpochSlots)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/kThreadCount, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMalloc(&device_output, kIterationCount * kThreadCount * sizeof(int)));

        hipLaunchKernelGGL(long_all_threads_own_slot_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[kIterationCount * kThreadCount] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        for(int iter = 0; iter < kIterationCount; ++iter)
        {
            for(int tid = 0; tid < kThreadCount; ++tid)
            {
                EXPECT_EQ(output[iter * kThreadCount + tid],
                          1000 + iter * kThreadCount + tid);
            }
        }
        expect_no_diagnostics(observed,
                              /*expected_access_count=*/2 * kIterationCount * kThreadCount);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiEpochLogLifetime, SameEpochOverflowStillReportsBeforeEpochReset)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/2, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(same_epoch_overflow_then_barrier_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(observed.access_count, 3);
        ASSERT_EQ(observed.diagnostic_count, 1);
        EXPECT_EQ(observed.diagnostics[0].kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::metadata_full));
        EXPECT_EQ(observed.diagnostics[0].epoch, 0u);
    }

} // namespace
