// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/012_matmul_pipeline_test.hip
//
// Double-buffered and pipeline-like matmul LDS instrumentation tests.
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
        hip_moi::thread_level_context::diagnostic diagnostics[kDiagnosticCapacity];
    };

    __host__ __device__ int matmul_a_value(int tile_id, int m, int k)
    {
        return 1 + tile_id + 2 * m + k;
    }

    __host__ __device__ int matmul_b_value(int tile_id, int n, int k)
    {
        return 2 + tile_id + n + 3 * k;
    }

    template <int MTile, int KTile>
    __host__ __device__ int matmul_a_offset(int tile_id, int m, int k)
    {
        return k + KTile * (m + MTile * tile_id);
    }

    template <int NTile, int KTile>
    __host__ __device__ int matmul_b_offset(int tile_id, int n, int k)
    {
        return k + KTile * (n + NTile * tile_id);
    }

    template <int MTile, int NTile, int KTile>
    void fill_matmul_inputs(int* a, int* b, int tile_count)
    {
        for(int tile_id = 0; tile_id < tile_count; ++tile_id)
        {
            for(int m = 0; m < MTile; ++m)
            {
                for(int k = 0; k < KTile; ++k)
                {
                    a[matmul_a_offset<MTile, KTile>(tile_id, m, k)]
                        = matmul_a_value(tile_id, m, k);
                }
            }
            for(int n = 0; n < NTile; ++n)
            {
                for(int k = 0; k < KTile; ++k)
                {
                    b[matmul_b_offset<NTile, KTile>(tile_id, n, k)]
                        = matmul_b_value(tile_id, n, k);
                }
            }
        }
    }

    template <int MTile, int NTile, int KTile>
    int reference_matmul_value(
        const int* a, const int* b, int tile_count, int m, int n)
    {
        int acc = 0;
        for(int tile_id = 0; tile_id < tile_count; ++tile_id)
        {
            for(int k = 0; k < KTile; ++k)
            {
                acc += a[matmul_a_offset<MTile, KTile>(tile_id, m, k)]
                       * b[matmul_b_offset<NTile, KTile>(tile_id, n, k)];
            }
        }
        return acc;
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

    template <int MTile, int NTile, int KTile>
    __device__ void load_matmul_tile(hip_moi::thread_level_context& ctx,
                                     int*              a_shared,
                                     int*              b_shared,
                                     const int*        a_global,
                                     const int*        b_global,
                                     int               tile_id)
    {
        int tid = static_cast<int>(threadIdx.x);
        for(int idx = tid; idx < MTile * KTile; idx += kThreadCount)
        {
            int m = idx / KTile;
            int k = idx - KTile * m;
            ctx.lds_store(&a_shared[idx], a_global[matmul_a_offset<MTile, KTile>(tile_id, m, k)]);
        }
        for(int idx = tid; idx < NTile * KTile; idx += kThreadCount)
        {
            int n = idx / KTile;
            int k = idx - KTile * n;
            ctx.lds_store(&b_shared[idx], b_global[matmul_b_offset<NTile, KTile>(tile_id, n, k)]);
        }
    }

    template <int MTile, int NTile, int KTile>
    __device__ int matmul_mac_from_lds(
        hip_moi::thread_level_context& ctx, const int* a_shared, const int* b_shared, int m, int n)
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

    __global__ void double_buffered_matmul_2x2_k2x3_kernel(
        const int* a_global, const int* b_global, int* out, hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int a_shared[2][2 * 2];
        __shared__ int b_shared[2][2 * 2];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        int acc = 0;
        int m   = tid / 2;
        int n   = tid - 2 * m;

        for(int tile_id = 0; tile_id < 3; ++tile_id)
        {
            int buffer = tile_id & 1;
            load_matmul_tile<2, 2, 2>(
                ctx, a_shared[buffer], b_shared[buffer], a_global, b_global, tile_id);

            ctx.syncthreads();

            if(tid < 4)
            {
                acc += matmul_mac_from_lds<2, 2, 2>(
                    ctx, a_shared[buffer], b_shared[buffer], m, n);
            }

            ctx.syncthreads();
        }

        if(tid < 4)
        {
            out[tid] = acc;
        }
    }

    __global__ void preloaded_ping_pong_matmul_2x2_k2x3_kernel(
        const int* a_global, const int* b_global, int* out, hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int a_shared[2][2 * 2];
        __shared__ int b_shared[2][2 * 2];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int tid = static_cast<int>(threadIdx.x);
        int acc = 0;
        int m   = tid / 2;
        int n   = tid - 2 * m;

        load_matmul_tile<2, 2, 2>(
            ctx, a_shared[0], b_shared[0], a_global, b_global, /*tile_id=*/0);
        ctx.syncthreads();

        for(int tile_id = 0; tile_id < 3; ++tile_id)
        {
            int buffer = tile_id & 1;
            if(tid < 4)
            {
                acc += matmul_mac_from_lds<2, 2, 2>(
                    ctx, a_shared[buffer], b_shared[buffer], m, n);
            }

            ctx.syncthreads();

            int next_tile = tile_id + 1;
            if(next_tile < 3)
            {
                int next_buffer = next_tile & 1;
                load_matmul_tile<2, 2, 2>(
                    ctx, a_shared[next_buffer], b_shared[next_buffer], a_global, b_global,
                    next_tile);
            }

            ctx.syncthreads();
        }

        if(tid < 4)
        {
            out[tid] = acc;
        }
    }

    __global__ void single_output_reuse_without_barrier_kernel(
        const int* a_global, const int* b_global, hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int a_shared[2];
        __shared__ int b_shared[2];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&a_shared[0], a_global[matmul_a_offset<1, 2>(0, 0, 0)]);
            ctx.lds_store(&a_shared[1], a_global[matmul_a_offset<1, 2>(0, 0, 1)]);
            ctx.lds_store(&b_shared[0], b_global[matmul_b_offset<1, 2>(0, 0, 0)]);
            ctx.lds_store(&b_shared[1], b_global[matmul_b_offset<1, 2>(0, 0, 1)]);
        }

        ctx.syncthreads();

        if(threadIdx.x == 0)
        {
            int acc = 0;
            for(int k = 0; k < 2; ++k)
            {
                int a = ctx.lds_load(&a_shared[k]);
                int b = ctx.lds_load(&b_shared[k]);
                acc += a * b;
            }
            (void)acc;
        }
        if(threadIdx.x == 1)
        {
            ctx.lds_store(&a_shared[0], a_global[matmul_a_offset<1, 2>(1, 0, 0)]);
            ctx.lds_store(&a_shared[1], a_global[matmul_a_offset<1, 2>(1, 0, 1)]);
            ctx.lds_store(&b_shared[0], b_global[matmul_b_offset<1, 2>(1, 0, 0)]);
            ctx.lds_store(&b_shared[1], b_global[matmul_b_offset<1, 2>(1, 0, 1)]);
        }
    }

    __global__ void partial_a_tile_overwrite_without_barrier_kernel(
        const int* a_global, const int* b_global, hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int a_shared[2 * 2];
        __shared__ int b_shared[2 * 2];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        load_matmul_tile<2, 2, 2>(
            ctx, a_shared, b_shared, a_global, b_global, /*tile_id=*/0);
        ctx.syncthreads();

        int tid = static_cast<int>(threadIdx.x);
        if(tid < 4)
        {
            int m = tid / 2;
            int n = tid - 2 * m;
            (void)matmul_mac_from_lds<2, 2, 2>(ctx, a_shared, b_shared, m, n);
        }
        if(tid >= 4 && tid < 8)
        {
            int idx = tid - 4;
            int m   = idx / 2;
            int k   = idx - 2 * m;
            ctx.lds_store(&a_shared[idx], a_global[matmul_a_offset<2, 2>(1, m, k)]);
        }
    }

    __global__ void partial_b_tile_overwrite_without_barrier_kernel(
        const int* a_global, const int* b_global, hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int a_shared[2 * 2];
        __shared__ int b_shared[2 * 2];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        load_matmul_tile<2, 2, 2>(
            ctx, a_shared, b_shared, a_global, b_global, /*tile_id=*/0);
        ctx.syncthreads();

        int tid = static_cast<int>(threadIdx.x);
        if(tid < 4)
        {
            int m = tid / 2;
            int n = tid - 2 * m;
            (void)matmul_mac_from_lds<2, 2, 2>(ctx, a_shared, b_shared, m, n);
        }
        if(tid >= 4 && tid < 8)
        {
            int idx = tid - 4;
            int n   = idx / 2;
            int k   = idx - 2 * n;
            ctx.lds_store(&b_shared[idx], b_global[matmul_b_offset<2, 2>(1, n, k)]);
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
    void expect_reference_matmul_output(
        const int* output, const int* a, const int* b, int tile_count)
    {
        for(int m = 0; m < MTile; ++m)
        {
            for(int n = 0; n < NTile; ++n)
            {
                int expected
                    = reference_matmul_value<MTile, NTile, KTile>(a, b, tile_count, m, n);
                EXPECT_EQ(output[n + NTile * m], expected);
            }
        }
    }

    void expect_access_conflicts(
        const observed_metadata& observed, int expected_access_count, int expected_diagnostic_count)
    {
        EXPECT_EQ(observed.access_count, expected_access_count);
        ASSERT_EQ(observed.diagnostic_count, expected_diagnostic_count);
        for(int i = 0; i < expected_diagnostic_count; ++i)
        {
            EXPECT_EQ(observed.diagnostics[i].kind,
                      static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
            EXPECT_EQ(observed.diagnostics[i].epoch, 1u);
        }
    }

    template <typename Kernel>
    void run_safe_output_case(Kernel kernel)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/32, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        constexpr int MTile     = 2;
        constexpr int NTile     = 2;
        constexpr int KTile     = 2;
        constexpr int TileCount = 3;

        int host_a[TileCount * MTile * KTile];
        int host_b[TileCount * NTile * KTile];
        fill_matmul_inputs<MTile, NTile, KTile>(host_a, host_b, TileCount);

        int* device_a      = nullptr;
        int* device_b      = nullptr;
        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_a, sizeof(host_a)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_b, sizeof(host_b)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, MTile * NTile * sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_a, host_a, sizeof(host_a), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_b, host_b, sizeof(host_b), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(
            kernel, dim3(1), dim3(kThreadCount), 0, 0, device_a, device_b, device_output,
            storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[MTile * NTile] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        expect_reference_matmul_output<MTile, NTile, KTile>(
            output, host_a, host_b, TileCount);
        expect_no_diagnostics(observed, /*expected_access_count=*/72);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_a));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_b));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    template <int MTile, int NTile, int KTile, int TileCount, typename Kernel>
    void run_diagnostic_case(Kernel kernel, int expected_access_count, int expected_diagnostic_count)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/32, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        int host_a[TileCount * MTile * KTile];
        int host_b[TileCount * NTile * KTile];
        fill_matmul_inputs<MTile, NTile, KTile>(host_a, host_b, TileCount);

        int* device_a = nullptr;
        int* device_b = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_a, sizeof(host_a)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_b, sizeof(host_b)));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_a, host_a, sizeof(host_a), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_b, host_b, sizeof(host_b), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(
            kernel, dim3(1), dim3(kThreadCount), 0, 0, device_a, device_b, storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        expect_access_conflicts(observed, expected_access_count, expected_diagnostic_count);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_a));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_b));
    }

    TEST(HipMoiMatmulPipeline, DoubleBuffered2x2K2x3DoesNotReport)
    {
        run_safe_output_case(double_buffered_matmul_2x2_k2x3_kernel);
    }

    TEST(HipMoiMatmulPipeline, PreloadedPingPong2x2K2x3DoesNotReport)
    {
        run_safe_output_case(preloaded_ping_pong_matmul_2x2_k2x3_kernel);
    }

    TEST(HipMoiMatmulPipeline, SingleOutputReuseWithoutBarrierReports)
    {
        run_diagnostic_case<1, 1, 2, 2>(
            single_output_reuse_without_barrier_kernel,
            /*expected_access_count=*/12,
            /*expected_diagnostic_count=*/4);
    }

    TEST(HipMoiMatmulPipeline, PartialATileOverwriteWithoutBarrierReports)
    {
        run_diagnostic_case<2, 2, 2, 2>(
            partial_a_tile_overwrite_without_barrier_kernel,
            /*expected_access_count=*/28,
            /*expected_diagnostic_count=*/8);
    }

    TEST(HipMoiMatmulPipeline, PartialBTileOverwriteWithoutBarrierReports)
    {
        run_diagnostic_case<2, 2, 2, 2>(
            partial_b_tile_overwrite_without_barrier_kernel,
            /*expected_access_count=*/28,
            /*expected_diagnostic_count=*/8);
    }
} // namespace
