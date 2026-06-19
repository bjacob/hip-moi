// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/006_all_thread_array_test.hip
//
// All-thread LDS array patterns for the MVP instrumentation contract.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 8;
    constexpr int kDiagnosticCapacity = 16;

    struct observed_metadata
    {
        int                 access_count;
        int                 diagnostic_count;
        hip_moi::diagnostic diagnostics[kDiagnosticCapacity];
    };

    __device__ hip_moi::thread_level_context make_context(hip_moi::context_storage_ref storage)
    {
        hip_moi::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
            /*subgroup_count=*/1,
        };
        return hip_moi::thread_level_context(storage, cfg);
    }

    __global__ void all_threads_write_own_slot_kernel(hip_moi::context_storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        ctx.lds_store(&values[tid], tid);
    }

    __global__ void all_threads_write_read_own_slot_kernel(
        int* out, hip_moi::context_storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        ctx.lds_store(&values[tid], tid * 3);

        ctx.syncthreads();

        out[tid] = ctx.lds_load(&values[tid]);
    }

    __global__ void all_threads_neighbor_read_kernel(int*                         out,
                                                     hip_moi::context_storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        ctx.lds_store(&values[tid], tid * 5);

        ctx.syncthreads();

        int neighbor = (tid + 1) % kThreadCount;
        out[tid]     = ctx.lds_load(&values[neighbor]);
    }

    __global__ void all_threads_neighbor_read_without_barrier_kernel(
        hip_moi::context_storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        ctx.lds_store(&values[tid], tid * 7);

        int neighbor = (tid + 1) % kThreadCount;
        (void)ctx.lds_load(&values[neighbor]);
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

    TEST(HipMoiAllThreadArray, IndependentWritesDoNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/32, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(
            all_threads_write_own_slot_kernel, dim3(1), dim3(kThreadCount), 0, 0, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        expect_no_diagnostics(observed, /*expected_access_count=*/kThreadCount);
    }

    TEST(HipMoiAllThreadArray, WriteReadOwnSlotAfterSyncthreadsDoesNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/32, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kThreadCount * sizeof(int)));

        hipLaunchKernelGGL(all_threads_write_read_own_slot_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[kThreadCount] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        for(int tid = 0; tid < kThreadCount; ++tid)
        {
            EXPECT_EQ(output[tid], tid * 3);
        }
        expect_no_diagnostics(observed, /*expected_access_count=*/2 * kThreadCount);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiAllThreadArray, NeighborReadAfterSyncthreadsDoesNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/32, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kThreadCount * sizeof(int)));

        hipLaunchKernelGGL(all_threads_neighbor_read_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[kThreadCount] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        for(int tid = 0; tid < kThreadCount; ++tid)
        {
            int neighbor = (tid + 1) % kThreadCount;
            EXPECT_EQ(output[tid], neighbor * 5);
        }
        expect_no_diagnostics(observed, /*expected_access_count=*/2 * kThreadCount);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiAllThreadArray, NeighborReadWithoutSyncthreadsReports)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/32, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(all_threads_neighbor_read_without_barrier_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(observed.access_count, 2 * kThreadCount);
        EXPECT_EQ(observed.diagnostic_count, kThreadCount);
        for(int i = 0; i < kThreadCount; ++i)
        {
            EXPECT_EQ(observed.diagnostics[i].kind,
                      static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
            EXPECT_EQ(observed.diagnostics[i].epoch, 0u);
        }
    }
} // namespace
