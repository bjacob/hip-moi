// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/009_single_subgroup_tiled_lds_test.hip
//
// Tiled LDS access patterns under the MVP synchronization contract.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 32;
    constexpr int kDiagnosticCapacity = 48;

    struct observed_metadata
    {
        int                 access_count;
        int                 diagnostic_count;
        hip_moi::thread_level_context::diagnostic diagnostics[kDiagnosticCapacity];
    };

    __host__ __device__ int wrap_power_of_two(int value, int modulus)
    {
        return value & (modulus - 1);
    }

    __device__ hip_moi::thread_level_context make_context(hip_moi::thread_level_context::storage_ref storage)
    {
        hip_moi::thread_level_context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
            /*subgroup_count=*/1,
        };
        return hip_moi::thread_level_context(storage, cfg);
    }

    __global__ void row_major_copy_kernel(int* out, hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int tile[4][8];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        int row = tid / 8;
        int col = tid - 8 * row;
        ctx.lds_store(&tile[row][col], 100 * row + col);

        ctx.syncthreads();

        out[tid] = ctx.lds_load(&tile[row][col]);
    }

    __global__ void transpose_4x8_to_8x4_kernel(int*                         out,
                                                hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int tile[4][8];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        int row = tid / 8;
        int col = tid - 8 * row;
        ctx.lds_store(&tile[row][col], 10 * row + col);

        ctx.syncthreads();

        int out_row = tid / 4;
        int out_col = tid - 4 * out_row;
        out[tid]    = ctx.lds_load(&tile[out_col][out_row]);
    }

    __global__ void skewed_row_stride_kernel(int*                         out,
                                             hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int tile[4 * 9];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid   = static_cast<int>(threadIdx.x);
        int row   = tid / 8;
        int col   = tid - 8 * row;
        int index = col + 9 * row;
        ctx.lds_store(&tile[index], 100 * row + col);

        ctx.syncthreads();

        out[tid] = ctx.lds_load(&tile[index]);
    }

    __global__ void blocked_2x2_layout_kernel(int*                         out,
                                              hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int tile[32];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid       = static_cast<int>(threadIdx.x);
        int row       = tid / 8;
        int col       = tid - 8 * row;
        int block_row = row >> 1;
        int block_col = col >> 1;
        int lane      = (row & 1) * 2 + (col & 1);
        int index     = 4 * (block_col + 4 * block_row) + lane;
        ctx.lds_store(&tile[index], 100 * row + col);

        ctx.syncthreads();

        out[tid] = ctx.lds_load(&tile[index]);
    }

    __global__ void diagonal_gather_kernel(int* out, hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int tile[4][8];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        int row = tid / 8;
        int col = tid - 8 * row;
        ctx.lds_store(&tile[row][col], 10 * row + col);

        ctx.syncthreads();

        out[tid] = ctx.lds_load(&tile[row][wrap_power_of_two(col + row, 8)]);
    }

    __global__ void striped_load_store_kernel(int*                         out,
                                              hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int tile[32];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        if(tid < 16)
        {
            for(int idx = tid; idx < 32; idx += 16)
            {
                ctx.lds_store(&tile[idx], 3 * idx);
            }
        }

        ctx.syncthreads();

        if(tid < 16)
        {
            for(int idx = tid; idx < 32; idx += 16)
            {
                out[idx] = ctx.lds_load(&tile[wrap_power_of_two(idx + 9, 32)]);
            }
        }
    }

    __global__ void transpose_without_barrier_kernel(hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int tile[4][8];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        int row = tid / 8;
        int col = tid - 8 * row;
        ctx.lds_store(&tile[row][col], 10 * row + col);

        int out_row = tid / 4;
        int out_col = tid - 4 * out_row;
        (void)ctx.lds_load(&tile[out_col][out_row]);
    }

    __global__ void transpose_without_barrier_subgroup_level_kernel(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int tile[4][8];

        hip_moi::subgroup_level_context ctx =
            hip_moi::test::make_single_subgroup_context<hip_moi::subgroup_level_context>(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        int row = tid / 8;
        int col = tid - 8 * row;
        ctx.lds_store(&tile[row][col], 10 * row + col);

        int out_row = tid / 4;
        int out_col = tid - 4 * out_row;
        (void)ctx.lds_load(&tile[out_col][out_row]);
        ctx.finish();
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

    void expect_access_conflict(const hip_moi::thread_level_context::diagnostic& diagnostic)
    {
        EXPECT_EQ(diagnostic.kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
        EXPECT_EQ(diagnostic.epoch, 0u);
    }

    template <typename Kernel, typename Expected>
    void run_output_case(Kernel kernel, Expected expected, int output_count, int expected_accesses)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/128, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, output_count * sizeof(int)));

        hipLaunchKernelGGL(kernel, dim3(1), dim3(kThreadCount), 0, 0, device_output, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[32] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, output_count * sizeof(int), hipMemcpyDeviceToHost));

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        for(int i = 0; i < output_count; ++i)
        {
            EXPECT_EQ(output[i], expected(i));
        }
        expect_no_diagnostics(observed, expected_accesses);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiTiledLds, RowMajorCopyDoesNotReport)
    {
        run_output_case(
            row_major_copy_kernel,
            [] __host__ __device__(int i) { return 100 * (i / 8) + (i & 7); },
            /*output_count=*/32,
            /*expected_accesses=*/64);
    }

    TEST(HipMoiTiledLds, TransposeAfterSyncthreadsDoesNotReport)
    {
        run_output_case(
            transpose_4x8_to_8x4_kernel,
            [] __host__ __device__(int i) {
                int out_row = i / 4;
                int out_col = i - 4 * out_row;
                return 10 * out_col + out_row;
            },
            /*output_count=*/32,
            /*expected_accesses=*/64);
    }

    TEST(HipMoiTiledLds, SkewedRowStrideDoesNotReport)
    {
        run_output_case(
            skewed_row_stride_kernel,
            [] __host__ __device__(int i) { return 100 * (i / 8) + (i & 7); },
            /*output_count=*/32,
            /*expected_accesses=*/64);
    }

    TEST(HipMoiTiledLds, Blocked2x2LayoutDoesNotReport)
    {
        run_output_case(
            blocked_2x2_layout_kernel,
            [] __host__ __device__(int i) { return 100 * (i / 8) + (i & 7); },
            /*output_count=*/32,
            /*expected_accesses=*/64);
    }

    TEST(HipMoiTiledLds, DiagonalGatherDoesNotReport)
    {
        run_output_case(
            diagonal_gather_kernel,
            [] __host__ __device__(int i) {
                int row = i / 8;
                int col = i - 8 * row;
                return 10 * row + wrap_power_of_two(col + row, 8);
            },
            /*output_count=*/32,
            /*expected_accesses=*/64);
    }

    TEST(HipMoiTiledLds, StripedLoadStoreDoesNotReport)
    {
        run_output_case(
            striped_load_store_kernel,
            [] __host__ __device__(int i) { return 3 * wrap_power_of_two(i + 9, 32); },
            /*output_count=*/32,
            /*expected_accesses=*/64);
    }

    TEST(HipMoiTiledLds, TransposeWithoutBarrierReportsOverlaps)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/128, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(
            transpose_without_barrier_kernel, dim3(1), dim3(kThreadCount), 0, 0, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(observed.access_count, 64);
        ASSERT_EQ(observed.diagnostic_count, 30);
        for(int i = 0; i < observed.diagnostic_count; ++i)
        {
            expect_access_conflict(observed.diagnostics[i]);
        }
    }

    TEST(HipMoiTiledLds, SubgroupLevelTransposeWithoutBarrierDoesNotReport)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/128, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(transpose_without_barrier_subgroup_level_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        ASSERT_NO_FATAL_FAILURE(hip_moi::test::expect_metadata_counts(
            storage, /*expected_access_count=*/64, /*expected_diagnostic_count=*/0));
    }
} // namespace
