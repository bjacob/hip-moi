// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/008_loop_epoch_test.hip
//
// Looped epoch patterns for the MVP synchronization model.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 8;
    constexpr int kIterationCount     = 4;
    constexpr int kDiagnosticCapacity = 40;

    struct observed_metadata
    {
        int                 access_count;
        int                 diagnostic_count;
        hip_moi::diagnostic diagnostics[kDiagnosticCapacity];
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

    __global__ void loop_scalar_producer_consumer_kernel(int*                         out,
                                                         hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        for(int iter = 0; iter < kIterationCount; ++iter)
        {
            if(threadIdx.x == 0)
            {
                ctx.lds_store(&value, 100 + iter);
            }

            ctx.syncthreads();

            if(threadIdx.x == 1)
            {
                out[iter] = ctx.lds_load(&value);
            }

            ctx.syncthreads();
        }
    }

    __global__ void loop_all_threads_own_slot_kernel(int*                         out,
                                                     hip_moi::context_storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        for(int iter = 0; iter < kIterationCount; ++iter)
        {
            ctx.lds_store(&values[tid], iter * 100 + tid);

            ctx.syncthreads();

            out[iter * kThreadCount + tid] = ctx.lds_load(&values[tid]);

            ctx.syncthreads();
        }
    }

    __global__ void loop_neighbor_read_without_inner_barrier_kernel(
        hip_moi::context_storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        for(int iter = 0; iter < kIterationCount; ++iter)
        {
            ctx.lds_store(&values[tid], iter * 100 + tid);

            int neighbor = (tid + 1) % kThreadCount;
            (void)ctx.lds_load(&values[neighbor]);

            ctx.syncthreads();
        }
    }

    __global__ void loop_conflict_after_each_leading_barrier_kernel(
        hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        for(int iter = 0; iter < kIterationCount; ++iter)
        {
            ctx.syncthreads();

            if(threadIdx.x == 0)
            {
                ctx.lds_store(&value, iter);
            }
            if(threadIdx.x == 1)
            {
                (void)ctx.lds_load(&value);
            }

            ctx.syncthreads();
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

    void expect_access_conflict(const hip_moi::diagnostic& diagnostic, uint32_t expected_epoch)
    {
        EXPECT_EQ(diagnostic.kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
        EXPECT_EQ(diagnostic.epoch, expected_epoch);
    }

    TEST(HipMoiLoopEpoch, ScalarProducerConsumerLoopDoesNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/32, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kIterationCount * sizeof(int)));

        hipLaunchKernelGGL(loop_scalar_producer_consumer_kernel,
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
            EXPECT_EQ(output[iter], 100 + iter);
        }
        expect_no_diagnostics(observed, /*expected_access_count=*/2 * kIterationCount);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiLoopEpoch, AllThreadsOwnSlotLoopDoesNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/96, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMalloc(&device_output, kIterationCount * kThreadCount * sizeof(int)));

        hipLaunchKernelGGL(loop_all_threads_own_slot_kernel,
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
                EXPECT_EQ(output[iter * kThreadCount + tid], iter * 100 + tid);
            }
        }
        expect_no_diagnostics(observed,
                              /*expected_access_count=*/2 * kIterationCount * kThreadCount);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiLoopEpoch, MissingInnerBarrierReportsEveryIteration)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/96, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(loop_neighbor_read_without_inner_barrier_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(observed.access_count, 2 * kIterationCount * kThreadCount);
        ASSERT_EQ(observed.diagnostic_count, kIterationCount * kThreadCount);
        for(int iter = 0; iter < kIterationCount; ++iter)
        {
            for(int tid = 0; tid < kThreadCount; ++tid)
            {
                int diagnostic_index = iter * kThreadCount + tid;
                expect_access_conflict(observed.diagnostics[diagnostic_index],
                                       static_cast<uint32_t>(iter));
            }
        }
    }

    TEST(HipMoiLoopEpoch, ConflictEpochAdvancesThroughLoop)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/32, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(loop_conflict_after_each_leading_barrier_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(observed.access_count, 2 * kIterationCount);
        ASSERT_EQ(observed.diagnostic_count, kIterationCount);
        for(int iter = 0; iter < kIterationCount; ++iter)
        {
            expect_access_conflict(observed.diagnostics[iter],
                                   static_cast<uint32_t>(2 * iter + 1));
        }
    }
} // namespace
