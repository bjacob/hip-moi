// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/004_single_subgroup_basic_conflict_predicate_test.hip
//
// Focused tests for the MVP byte-range conflict predicate.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount = 8;

    struct int_pair
    {
        int first;
        int second;
    };

    struct observed_metadata
    {
        int                 access_count;
        int                 diagnostic_count;
        hip_moi::thread_level_context::diagnostic diagnostics[4];
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

    __global__ void same_address_read_read_kernel(int* out, hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            value = 17;
        }
        __syncthreads();

        if(threadIdx.x == 0)
        {
            out[0] = ctx.lds_load(&value);
        }
        if(threadIdx.x == 1)
        {
            out[1] = ctx.lds_load(&value);
        }
    }

    __global__ void same_address_write_write_kernel(hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, 10);
        }
        if(threadIdx.x == 1)
        {
            ctx.lds_store(&value, 20);
        }
    }

    __global__ void non_overlapping_write_write_kernel(hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int values[4];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&values[0], 10);
        }
        if(threadIdx.x == 1)
        {
            ctx.lds_store(&values[2], 20);
        }
    }

    __global__ void adjacent_byte_ranges_write_write_kernel(hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int_pair value;

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value.first, 10);
        }
        if(threadIdx.x == 1)
        {
            ctx.lds_store(&value.second, 20);
        }
    }

    __global__ void overlapping_subobject_write_write_kernel(hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int_pair value;

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, int_pair{10, 20});
        }
        if(threadIdx.x == 1)
        {
            ctx.lds_store(&value.second, 30);
        }
    }

    __global__ void same_address_write_write_subgroup_level_kernel(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::subgroup_level_context ctx =
            hip_moi::test::make_single_subgroup_context<hip_moi::subgroup_level_context>(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, 10);
        }
        if(threadIdx.x == 1)
        {
            ctx.lds_store(&value, 20);
        }
    }

    __global__ void overlapping_subobject_write_write_subgroup_level_kernel(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int_pair value;

        hip_moi::subgroup_level_context ctx =
            hip_moi::test::make_single_subgroup_context<hip_moi::subgroup_level_context>(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, int_pair{10, 20});
        }
        if(threadIdx.x == 1)
        {
            ctx.lds_store(&value.second, 30);
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

    void expect_two_accesses_and_no_diagnostics(const observed_metadata& observed)
    {
        EXPECT_EQ(observed.access_count, 2);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    void expect_one_access_conflict(const observed_metadata& observed)
    {
        EXPECT_EQ(observed.access_count, 2);
        ASSERT_EQ(observed.diagnostic_count, 1);

        const hip_moi::thread_level_context::diagnostic& diagnostic = observed.diagnostics[0];
        EXPECT_EQ(diagnostic.kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
        EXPECT_EQ(diagnostic.epoch, 0u);

        bool threads_are_zero_and_one = (diagnostic.first_thread_id == 0u
                                         && diagnostic.second_thread_id == 1u)
                                        || (diagnostic.first_thread_id == 1u
                                            && diagnostic.second_thread_id == 0u);
        EXPECT_TRUE(threads_are_zero_and_one);
    }

    TEST(HipMoiBasicConflictPredicate, SameAddressReadReadDoesNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/16, /*diagnostic_capacity=*/4, /*subgroup_capacity=*/1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, 2 * sizeof(int)));

        int initial[2] = {-1, -1};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, initial, sizeof(initial), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(same_address_read_read_kernel,
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

        EXPECT_EQ(output[0], 17);
        EXPECT_EQ(output[1], 17);
        expect_two_accesses_and_no_diagnostics(observed);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiBasicConflictPredicate, SameAddressWriteWriteReports)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/16, /*diagnostic_capacity=*/4, /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(
            same_address_write_write_kernel, dim3(1), dim3(kThreadCount), 0, 0, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        expect_one_access_conflict(observed);

        const hip_moi::thread_level_context::diagnostic& diagnostic = observed.diagnostics[0];
        EXPECT_EQ(diagnostic.first_addr, diagnostic.second_addr);
        EXPECT_EQ(diagnostic.first_size, sizeof(int));
        EXPECT_EQ(diagnostic.second_size, sizeof(int));
    }

    TEST(HipMoiBasicConflictPredicate, NonOverlappingWritesDoNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/16, /*diagnostic_capacity=*/4, /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(
            non_overlapping_write_write_kernel, dim3(1), dim3(kThreadCount), 0, 0, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        expect_two_accesses_and_no_diagnostics(observed);
    }

    TEST(HipMoiBasicConflictPredicate, AdjacentByteRangesDoNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/16, /*diagnostic_capacity=*/4, /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(adjacent_byte_ranges_write_write_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        expect_two_accesses_and_no_diagnostics(observed);
    }

    TEST(HipMoiBasicConflictPredicate, OverlappingSubobjectWriteReports)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/16, /*diagnostic_capacity=*/4, /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(overlapping_subobject_write_write_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        expect_one_access_conflict(observed);

        const hip_moi::thread_level_context::diagnostic& diagnostic = observed.diagnostics[0];
        EXPECT_NE(diagnostic.first_addr, diagnostic.second_addr);

        bool sizes_are_pair_and_int
            = (diagnostic.first_size == sizeof(int_pair) && diagnostic.second_size == sizeof(int))
              || (diagnostic.first_size == sizeof(int)
                  && diagnostic.second_size == sizeof(int_pair));
        EXPECT_TRUE(sizes_are_pair_and_int);
    }

    TEST(HipMoiBasicConflictPredicate, SubgroupLevelSameAddressWriteWriteDoesNotReport)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/16, /*diagnostic_capacity=*/4, /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(same_address_write_write_subgroup_level_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        ASSERT_NO_FATAL_FAILURE(hip_moi::test::expect_metadata_counts(
            storage, /*expected_access_count=*/2, /*expected_diagnostic_count=*/0));
    }

    TEST(HipMoiBasicConflictPredicate, SubgroupLevelOverlappingSubobjectWriteDoesNotReport)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/16, /*diagnostic_capacity=*/4, /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(overlapping_subobject_write_write_subgroup_level_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        ASSERT_NO_FATAL_FAILURE(hip_moi::test::expect_metadata_counts(
            storage, /*expected_access_count=*/2, /*expected_diagnostic_count=*/0));
    }
} // namespace
