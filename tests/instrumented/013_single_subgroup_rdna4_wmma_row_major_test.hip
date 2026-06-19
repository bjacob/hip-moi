// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/013_single_subgroup_rdna4_wmma_row_major_test.hip
//
// RDNA4 WMMA tests using conventional row-major LDS tiles.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 32;
    constexpr int kTileM              = 16;
    constexpr int kTileN              = 16;
    constexpr int kTileK              = 16;
    constexpr int kTileElementCount   = 256;
    constexpr int kDiagnosticCapacity = 64;

    using f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
    using f32x8_t = float __attribute__((ext_vector_type(8)));

    struct observed_metadata
    {
        int                 access_count;
        int                 diagnostic_count;
        hip_moi::thread_level_context::diagnostic diagnostics[kDiagnosticCapacity];
    };

    __host__ __device__ int row_major(int row, int col, int cols)
    {
        return col + cols * row;
    }

    __host__ __device__ int matmul_a_value(int tile_id, int m, int k)
    {
        return ((tile_id + 2 * m + 3 * k) % 5) - 2;
    }

    __host__ __device__ int matmul_b_value(int tile_id, int n, int k)
    {
        return ((1 + 2 * tile_id + 3 * n + k) % 5) - 2;
    }

    __device__ int rdna4_wmma_f16_k(int lane, int idx)
    {
        int reg  = idx >> 1;
        int half = idx & 1;
        return half | ((reg & 1) << 1) | ((lane >> 4) << 2) | ((reg >> 1) << 3);
    }

    __device__ int rdna4_wmma_acc_m(int lane, int elem)
    {
        return elem + 8 * (lane >> 4);
    }

    float reference_matmul_value(int tile_count, int m, int n)
    {
        int acc = 0;
        for(int tile_id = 0; tile_id < tile_count; ++tile_id)
        {
            for(int k = 0; k < kTileK; ++k)
            {
                acc += matmul_a_value(tile_id, m, k) * matmul_b_value(tile_id, n, k);
            }
        }
        return static_cast<float>(acc);
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

    template <typename Context>
    __device__ void load_row_major_tile_to_lds(Context&          ctx,
                                               _Float16*         a_shared,
                                               _Float16*         b_shared,
                                               const _Float16*   a_global,
                                               const _Float16*   b_global,
                                               int               tile_id)
    {
        int tid = static_cast<int>(threadIdx.x);
        for(int idx = tid; idx < kTileM * kTileK; idx += kThreadCount)
        {
            ctx.lds_store(&a_shared[idx], a_global[tile_id * kTileElementCount + idx]);
        }
        for(int idx = tid; idx < kTileN * kTileK; idx += kThreadCount)
        {
            ctx.lds_store(&b_shared[idx], b_global[tile_id * kTileElementCount + idx]);
        }
    }

    template <typename Context>
    __device__ void wmma_f16_mac_from_row_major_lds(Context&          ctx,
                                                    f32x8_t&          acc,
                                                    const _Float16*   a_shared,
                                                    const _Float16*   b_shared)
    {
        int lane = static_cast<int>(threadIdx.x) & 31;
        int m    = lane & 15;
        int n    = lane & 15;

        f16x8_t a;
        f16x8_t b;
        for(int idx = 0; idx < 8; ++idx)
        {
            int k  = rdna4_wmma_f16_k(lane, idx);
            a[idx] = ctx.lds_load(&a_shared[row_major(m, k, kTileK)]);
            b[idx] = ctx.lds_load(&b_shared[row_major(n, k, kTileK)]);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }

    __device__ void store_accumulator(const f32x8_t& acc, float* out)
    {
        int lane = static_cast<int>(threadIdx.x) & 31;
        int n    = lane & 15;
        for(int elem = 0; elem < 8; ++elem)
        {
            int m                         = rdna4_wmma_acc_m(lane, elem);
            out[row_major(m, n, kTileN)] = acc[elem];
        }
    }

    __global__ void row_major_single_buffer_kernel(const _Float16*             a_global,
                                                   const _Float16*             b_global,
                                                   float*                      out,
                                                   hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ _Float16 a_shared[kTileM * kTileK];
        __shared__ _Float16 b_shared[kTileN * kTileK];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        f32x8_t acc = {};
        load_row_major_tile_to_lds(ctx, a_shared, b_shared, a_global, b_global, /*tile_id=*/0);
        ctx.syncthreads();
        wmma_f16_mac_from_row_major_lds(ctx, acc, a_shared, b_shared);
        ctx.syncthreads();
        store_accumulator(acc, out);
    }

    __global__ void row_major_double_buffer_kernel(const _Float16*             a_global,
                                                   const _Float16*             b_global,
                                                   float*                      out,
                                                   hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ _Float16 a_shared[2][kTileM * kTileK];
        __shared__ _Float16 b_shared[2][kTileN * kTileK];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        f32x8_t acc = {};
        for(int tile_id = 0; tile_id < 2; ++tile_id)
        {
            int buffer = tile_id & 1;
            load_row_major_tile_to_lds(
                ctx, a_shared[buffer], b_shared[buffer], a_global, b_global, tile_id);
            ctx.syncthreads();
            wmma_f16_mac_from_row_major_lds(ctx, acc, a_shared[buffer], b_shared[buffer]);
            ctx.syncthreads();
        }
        store_accumulator(acc, out);
    }

    __global__ void row_major_reuse_without_barrier_kernel(
        const _Float16* a_global, const _Float16* b_global, hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ _Float16 a_shared[kTileM * kTileK];
        __shared__ _Float16 b_shared[kTileN * kTileK];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        f32x8_t acc = {};
        load_row_major_tile_to_lds(ctx, a_shared, b_shared, a_global, b_global, /*tile_id=*/0);
        ctx.syncthreads();
        wmma_f16_mac_from_row_major_lds(ctx, acc, a_shared, b_shared);

        if(threadIdx.x == 0)
        {
            for(int k = 0; k < kTileK; ++k)
            {
                ctx.lds_store(&a_shared[row_major(/*row=*/1, k, kTileK)],
                              static_cast<_Float16>(2.0f));
            }
        }
    }

    __global__ void row_major_reuse_without_barrier_subgroup_level_kernel(
        const _Float16*                 a_global,
        const _Float16*                 b_global,
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ _Float16 a_shared[kTileM * kTileK];
        __shared__ _Float16 b_shared[kTileN * kTileK];

        hip_moi::subgroup_level_context ctx =
            hip_moi::test::make_single_subgroup_context<hip_moi::subgroup_level_context>(storage);
        ctx.init_workgroup();

        f32x8_t acc = {};
        load_row_major_tile_to_lds(ctx, a_shared, b_shared, a_global, b_global, /*tile_id=*/0);
        ctx.syncthreads();
        wmma_f16_mac_from_row_major_lds(ctx, acc, a_shared, b_shared);

        if(threadIdx.x == 0)
        {
            for(int k = 0; k < kTileK; ++k)
            {
                ctx.lds_store(&a_shared[row_major(/*row=*/1, k, kTileK)],
                              static_cast<_Float16>(2.0f));
            }
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

    void fill_row_major_inputs(_Float16* a, _Float16* b, int tile_count)
    {
        for(int tile_id = 0; tile_id < tile_count; ++tile_id)
        {
            for(int m = 0; m < kTileM; ++m)
            {
                for(int k = 0; k < kTileK; ++k)
                {
                    a[tile_id * kTileElementCount + row_major(m, k, kTileK)]
                        = static_cast<_Float16>(matmul_a_value(tile_id, m, k));
                }
            }
            for(int n = 0; n < kTileN; ++n)
            {
                for(int k = 0; k < kTileK; ++k)
                {
                    b[tile_id * kTileElementCount + row_major(n, k, kTileK)]
                        = static_cast<_Float16>(matmul_b_value(tile_id, n, k));
                }
            }
        }
    }

    void expect_reference_output(float* device_output, int tile_count)
    {
        float output[kTileElementCount] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));
        for(int m = 0; m < kTileM; ++m)
        {
            for(int n = 0; n < kTileN; ++n)
            {
                EXPECT_EQ(output[row_major(m, n, kTileN)],
                          reference_matmul_value(tile_count, m, n));
            }
        }
    }

    void expect_no_diagnostics(const observed_metadata& observed, int expected_access_count)
    {
        EXPECT_EQ(observed.access_count, expected_access_count);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    void expect_row_reuse_diagnostics(const observed_metadata& observed)
    {
        EXPECT_EQ(observed.access_count, 1040);
        ASSERT_EQ(observed.diagnostic_count, 16);
        for(int i = 0; i < observed.diagnostic_count; ++i)
        {
            EXPECT_EQ(observed.diagnostics[i].kind,
                      static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
            EXPECT_EQ(observed.diagnostics[i].epoch, 1u);
        }
    }

    template <typename Kernel>
    void run_safe_case(Kernel kernel, int tile_count, int expected_access_count)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/640, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        _Float16 host_a[2 * kTileElementCount];
        _Float16 host_b[2 * kTileElementCount];
        fill_row_major_inputs(host_a, host_b, tile_count);

        _Float16* device_a = nullptr;
        _Float16* device_b = nullptr;
        float*    device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMalloc(&device_a, tile_count * kTileElementCount * sizeof(_Float16)));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMalloc(&device_b, tile_count * kTileElementCount * sizeof(_Float16)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kTileElementCount * sizeof(float)));

        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(device_a,
                                          host_a,
                                          tile_count * kTileElementCount * sizeof(_Float16),
                                          hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(device_b,
                                          host_b,
                                          tile_count * kTileElementCount * sizeof(_Float16),
                                          hipMemcpyHostToDevice));

        hipLaunchKernelGGL(
            kernel, dim3(1), dim3(kThreadCount), 0, 0, device_a, device_b, device_output,
            storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_reference_output(device_output, tile_count);
        expect_no_diagnostics(observed, expected_access_count);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_a));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_b));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiRdna4WmmaRowMajor, SingleBufferUsesWmmaIntrinsic)
    {
        run_safe_case(row_major_single_buffer_kernel,
                      /*tile_count=*/1,
                      /*expected_access_count=*/1024);
    }

    TEST(HipMoiRdna4WmmaRowMajor, DoubleBufferUsesWmmaIntrinsic)
    {
        run_safe_case(row_major_double_buffer_kernel,
                      /*tile_count=*/2,
                      /*expected_access_count=*/2048);
    }

    TEST(HipMoiRdna4WmmaRowMajor, ReuseWithoutBarrierReports)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/640, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        _Float16 host_a[kTileElementCount];
        _Float16 host_b[kTileElementCount];
        fill_row_major_inputs(host_a, host_b, /*tile_count=*/1);

        _Float16* device_a = nullptr;
        _Float16* device_b = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_a, sizeof(host_a)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_b, sizeof(host_b)));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_a, host_a, sizeof(host_a), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_b, host_b, sizeof(host_b), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(row_major_reuse_without_barrier_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_a,
                           device_b,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_row_reuse_diagnostics(observed);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_a));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_b));
    }

    TEST(HipMoiRdna4WmmaRowMajor, SubgroupLevelReuseWithoutBarrierDoesNotReport)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/640, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        _Float16 host_a[kTileElementCount];
        _Float16 host_b[kTileElementCount];
        fill_row_major_inputs(host_a, host_b, /*tile_count=*/1);

        _Float16* device_a = nullptr;
        _Float16* device_b = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_a, sizeof(host_a)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_b, sizeof(host_b)));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_a, host_a, sizeof(host_a), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_b, host_b, sizeof(host_b), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(row_major_reuse_without_barrier_subgroup_level_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_a,
                           device_b,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        ASSERT_NO_FATAL_FAILURE(hip_moi::test::expect_metadata_counts(
            storage, /*expected_access_count=*/1040, /*expected_diagnostic_count=*/0));

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_a));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_b));
    }
} // namespace
