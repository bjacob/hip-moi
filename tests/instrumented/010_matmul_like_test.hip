// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/010_matmul_like_test.hip
//
// Matmul-like LDS patterns under the MVP synchronization contract.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 16;
    constexpr int kDiagnosticCapacity = 32;

    struct observed_metadata
    {
        int                 access_count;
        int                 diagnostic_count;
        hip_moi::diagnostic diagnostics[kDiagnosticCapacity];
    };

    __host__ __device__ int matmul_a_value(int tile_id, int m, int k)
    {
        return 1 + tile_id + 2 * m + k;
    }

    __host__ __device__ int matmul_b_value(int tile_id, int n, int k)
    {
        return 2 + tile_id + n + 3 * k;
    }

    __host__ __device__ int matmul_expected_value(int tile_id, int m, int n, int k_size)
    {
        int acc = 0;
        for(int k = 0; k < k_size; ++k)
        {
            acc += matmul_a_value(tile_id, m, k) * matmul_b_value(tile_id, n, k);
        }
        return acc;
    }

    __device__ hip_moi::context make_context(hip_moi::context_storage_ref storage)
    {
        hip_moi::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
            /*subgroup_count=*/1,
        };
        return hip_moi::context(storage, cfg);
    }

    template <int MTile, int NTile, int KTile>
    __device__ void load_matmul_tile(
        hip_moi::context& ctx, int* a_shared, int* b_shared, int tile_id)
    {
        int tid = static_cast<int>(threadIdx.x);
        for(int idx = tid; idx < MTile * KTile; idx += kThreadCount)
        {
            int m = idx / KTile;
            int k = idx - KTile * m;
            ctx.lds_store(&a_shared[idx], matmul_a_value(tile_id, m, k));
        }
        for(int idx = tid; idx < NTile * KTile; idx += kThreadCount)
        {
            int n = idx / KTile;
            int k = idx - KTile * n;
            ctx.lds_store(&b_shared[idx], matmul_b_value(tile_id, n, k));
        }
    }

    template <int MTile, int NTile, int KTile>
    __device__ int matmul_mac_from_lds(
        hip_moi::context& ctx, const int* a_shared, const int* b_shared, int m, int n)
    {
        int acc = 0;
        for(int k = 0; k < KTile; ++k)
        {
            int a = ctx.lds_load(&a_shared[k + KTile * m]);
            int b = ctx.lds_load(&b_shared[k + KTile * n]);
            acc += a * b;
        }
        return acc;
    }

    template <int MTile, int NTile, int KTile>
    __device__ void run_simple_matmul_body(
        hip_moi::context& ctx, int* out, int tile_id, int* a_shared, int* b_shared)
    {
        int tid = static_cast<int>(threadIdx.x);
        load_matmul_tile<MTile, NTile, KTile>(ctx, a_shared, b_shared, tile_id);

        ctx.syncthreads();

        if(tid < MTile * NTile)
        {
            int m    = tid / NTile;
            int n    = tid - NTile * m;
            out[tid] = matmul_mac_from_lds<MTile, NTile, KTile>(ctx, a_shared, b_shared, m, n);
        }

        ctx.syncthreads();
    }

    __global__ void simple_matmul_2x2_k3_kernel(int*                         out,
                                                hip_moi::context_storage_ref storage)
    {
        __shared__ int a_shared[2 * 3];
        __shared__ int b_shared[2 * 3];

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();
        run_simple_matmul_body<2, 2, 3>(ctx, out, /*tile_id=*/0, a_shared, b_shared);
    }

    __global__ void simple_matmul_4x4_k4_kernel(int*                         out,
                                                hip_moi::context_storage_ref storage)
    {
        __shared__ int a_shared[4 * 4];
        __shared__ int b_shared[4 * 4];

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();
        run_simple_matmul_body<4, 4, 4>(ctx, out, /*tile_id=*/0, a_shared, b_shared);
    }

    __global__ void chunked_matmul_2x2_k2x3_kernel(int*                         out,
                                                   hip_moi::context_storage_ref storage)
    {
        __shared__ int a_shared[2 * 2];
        __shared__ int b_shared[2 * 2];

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        int acc = 0;
        int m   = tid / 2;
        int n   = tid - 2 * m;

        for(int tile_id = 0; tile_id < 3; ++tile_id)
        {
            load_matmul_tile<2, 2, 2>(ctx, a_shared, b_shared, tile_id);

            ctx.syncthreads();

            if(tid < 4)
            {
                acc += matmul_mac_from_lds<2, 2, 2>(ctx, a_shared, b_shared, m, n);
            }

            ctx.syncthreads();
        }

        if(tid < 4)
        {
            out[tid] = acc;
        }
    }

    __global__ void scalar_matmul_missing_barrier_kernel(hip_moi::context_storage_ref storage)
    {
        __shared__ int a_shared;
        __shared__ int b_shared;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&a_shared, 5);
            ctx.lds_store(&b_shared, 7);
        }
        if(threadIdx.x == 1)
        {
            int a = ctx.lds_load(&a_shared);
            int b = ctx.lds_load(&b_shared);
            (void)(a * b);
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

    template <int MTile, int NTile, int KTile>
    void expect_simple_matmul_output(const int* output, int tile_id)
    {
        for(int m = 0; m < MTile; ++m)
        {
            for(int n = 0; n < NTile; ++n)
            {
                EXPECT_EQ(output[n + NTile * m],
                          matmul_expected_value(tile_id, m, n, KTile));
            }
        }
    }

    template <int MTile, int NTile, int KTile>
    void expect_chunked_matmul_output(const int* output, int tile_count)
    {
        for(int m = 0; m < MTile; ++m)
        {
            for(int n = 0; n < NTile; ++n)
            {
                int expected = 0;
                for(int tile_id = 0; tile_id < tile_count; ++tile_id)
                {
                    expected += matmul_expected_value(tile_id, m, n, KTile);
                }
                EXPECT_EQ(output[n + NTile * m], expected);
            }
        }
    }

    TEST(HipMoiMatmulLike, Simple2x2K3DoesNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/128, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, 4 * sizeof(int)));

        hipLaunchKernelGGL(simple_matmul_2x2_k3_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[4] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        expect_simple_matmul_output<2, 2, 3>(output, /*tile_id=*/0);
        expect_no_diagnostics(observed, /*expected_access_count=*/36);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiMatmulLike, Simple4x4K4DoesNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/256, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, 16 * sizeof(int)));

        hipLaunchKernelGGL(simple_matmul_4x4_k4_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[16] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        expect_simple_matmul_output<4, 4, 4>(output, /*tile_id=*/0);
        expect_no_diagnostics(observed, /*expected_access_count=*/160);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiMatmulLike, Chunked2x2K2x3DoesNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/128, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, 4 * sizeof(int)));

        hipLaunchKernelGGL(chunked_matmul_2x2_k2x3_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[4] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        expect_chunked_matmul_output<2, 2, 2>(output, /*tile_count=*/3);
        expect_no_diagnostics(observed, /*expected_access_count=*/72);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiMatmulLike, ScalarMissingBarrierReports)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/16, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        hipLaunchKernelGGL(scalar_matmul_missing_barrier_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(observed.access_count, 4);
        ASSERT_EQ(observed.diagnostic_count, 2);
        for(int i = 0; i < observed.diagnostic_count; ++i)
        {
            EXPECT_EQ(observed.diagnostics[i].kind,
                      static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
            EXPECT_EQ(observed.diagnostics[i].epoch, 0u);
        }
    }
} // namespace
