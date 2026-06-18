// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/005_epoch_boundary_test.hip
//
// Focused tests for ctx.syncthreads() as the MVP epoch boundary.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount = 8;

    struct observed_metadata
    {
        int                 access_count;
        int                 diagnostic_count;
        hip_moi::diagnostic diagnostics[4];
    };

    __device__ hip_moi::context make_context(hip_moi::context_storage_ref storage)
    {
        hip_moi::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
            /*subgroup_count=*/1,
        };
        return hip_moi::context(storage, cfg);
    }

    __global__ void same_epoch_write_read_kernel(int* out, hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, 10);
        }
        if(threadIdx.x == 1)
        {
            out[0] = ctx.lds_load(&value);
        }
    }

    __global__ void separated_write_read_kernel(int* out, hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, 10);
        }

        ctx.syncthreads();

        if(threadIdx.x == 1)
        {
            out[0] = ctx.lds_load(&value);
        }
    }

    __global__ void separated_write_write_kernel(hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, 10);
        }

        ctx.syncthreads();

        if(threadIdx.x == 1)
        {
            ctx.lds_store(&value, 20);
        }
    }

    __global__ void repeated_reuse_across_epochs_kernel(int*                         out,
                                                        hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, 10);
        }

        ctx.syncthreads();

        if(threadIdx.x == 1)
        {
            out[0] = ctx.lds_load(&value);
        }

        ctx.syncthreads();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, 20);
        }

        ctx.syncthreads();

        if(threadIdx.x == 1)
        {
            out[1] = ctx.lds_load(&value);
        }
    }

    __global__ void same_epoch_conflict_after_barrier_kernel(hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, 10);
        }

        ctx.syncthreads();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, 20);
        }
        if(threadIdx.x == 1)
        {
            (void)ctx.lds_load(&value);
        }
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

    void expect_one_access_conflict(const observed_metadata& observed,
                                    int                      expected_access_count,
                                    uint32_t                 expected_epoch)
    {
        EXPECT_EQ(observed.access_count, expected_access_count);
        ASSERT_EQ(observed.diagnostic_count, 1);

        const hip_moi::diagnostic& diagnostic = observed.diagnostics[0];
        EXPECT_EQ(diagnostic.kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
        EXPECT_EQ(diagnostic.epoch, expected_epoch);

        bool threads_are_zero_and_one = (diagnostic.writer_or_first_thread_id == 0u
                                         && diagnostic.reader_or_second_thread_id == 1u)
                                        || (diagnostic.writer_or_first_thread_id == 1u
                                            && diagnostic.reader_or_second_thread_id == 0u);
        EXPECT_TRUE(threads_are_zero_and_one);
    }

    TEST(HipMoiEpochBoundary, SameEpochWriteReadReports)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/16, /*diagnostic_capacity=*/4, /*subgroup_capacity=*/1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));

        int initial = -1;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(same_epoch_write_read_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        expect_one_access_conflict(observed, /*expected_access_count=*/2, /*expected_epoch=*/0);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiEpochBoundary, WriteReadSeparatedBySyncthreadsDoesNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/16, /*diagnostic_capacity=*/4, /*subgroup_capacity=*/1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));

        int initial = -1;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(separated_write_read_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output = -1;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(output, 10);
        expect_no_diagnostics(observed, /*expected_access_count=*/2);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiEpochBoundary, WriteWriteSeparatedBySyncthreadsDoesNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/16, /*diagnostic_capacity=*/4, /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(
            separated_write_write_kernel, dim3(1), dim3(kThreadCount), 0, 0, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        expect_no_diagnostics(observed, /*expected_access_count=*/2);
    }

    TEST(HipMoiEpochBoundary, RepeatedReuseAcrossEpochsDoesNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/16, /*diagnostic_capacity=*/4, /*subgroup_capacity=*/1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, 2 * sizeof(int)));

        int initial[2] = {-1, -1};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, initial, sizeof(initial), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(repeated_reuse_across_epochs_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[2] = {-1, -1};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(output[0], 10);
        EXPECT_EQ(output[1], 20);
        expect_no_diagnostics(observed, /*expected_access_count=*/4);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiEpochBoundary, SameEpochConflictAfterSyncthreadsReportsInNewEpoch)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/16, /*diagnostic_capacity=*/4, /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(same_epoch_conflict_after_barrier_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        expect_one_access_conflict(observed, /*expected_access_count=*/3, /*expected_epoch=*/1);
    }
} // namespace
