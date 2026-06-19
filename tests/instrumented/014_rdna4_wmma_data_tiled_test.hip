// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/014_rdna4_wmma_data_tiled_test.hip
//
// RDNA4 WMMA tests using data-tiled LDS fragments. Each thread owns one
// contiguous 16-byte fragment per matrix, at byte offset lane * 16.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 32;
    constexpr int kTileN              = 16;
    constexpr int kFragmentElements   = 8;
    constexpr int kPackedTileElements = kThreadCount * kFragmentElements;
    constexpr int kOutputElementCount = 16 * 16;
    constexpr int kDiagnosticCapacity = 64;

    using f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
    using f32x8_t = float __attribute__((ext_vector_type(8)));

    struct observed_metadata
    {
        int                 access_count;
        int                 diagnostic_count;
        hip_moi::diagnostic diagnostics[kDiagnosticCapacity];
    };

    __host__ __device__ int row_major(int row, int col, int cols)
    {
        return col + cols * row;
    }

    __device__ int rdna4_wmma_acc_m(int lane, int elem)
    {
        return elem + 8 * (lane >> 4);
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

    __device__ f16x8_t load_global_fragment(const _Float16* global_fragments, int fragment_index)
    {
        f16x8_t result;
        int     base = fragment_index * kFragmentElements;
        for(int i = 0; i < kFragmentElements; ++i)
        {
            result[i] = global_fragments[base + i];
        }
        return result;
    }

    __device__ void load_data_tiled_fragments_to_lds(hip_moi::context& ctx,
                                                     f16x8_t*          a_shared,
                                                     f16x8_t*          b_shared,
                                                     const _Float16*   a_global,
                                                     const _Float16*   b_global,
                                                     int               tile_id)
    {
        int lane = static_cast<int>(threadIdx.x) & 31;
        int base = tile_id * kThreadCount;
        ctx.lds_store(&a_shared[lane], load_global_fragment(a_global, base + lane));
        ctx.lds_store(&b_shared[lane], load_global_fragment(b_global, base + lane));
    }

    __device__ void wmma_f16_mac_from_data_tiled_lds(hip_moi::context& ctx,
                                                     f32x8_t&          acc,
                                                     const f16x8_t*    a_shared,
                                                     const f16x8_t*    b_shared)
    {
        int     lane = static_cast<int>(threadIdx.x) & 31;
        f16x8_t a    = ctx.lds_load(&a_shared[lane]);
        f16x8_t b    = ctx.lds_load(&b_shared[lane]);
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

    __global__ void data_tiled_single_buffer_kernel(const _Float16*             a_global,
                                                    const _Float16*             b_global,
                                                    float*                      out,
                                                    hip_moi::context_storage_ref storage)
    {
        __shared__ f16x8_t a_shared[kThreadCount];
        __shared__ f16x8_t b_shared[kThreadCount];

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        f32x8_t acc = {};
        load_data_tiled_fragments_to_lds(
            ctx, a_shared, b_shared, a_global, b_global, /*tile_id=*/0);
        ctx.syncthreads();
        wmma_f16_mac_from_data_tiled_lds(ctx, acc, a_shared, b_shared);
        ctx.syncthreads();
        store_accumulator(acc, out);
    }

    __global__ void data_tiled_preloaded_ping_pong_kernel(
        const _Float16* a_global, const _Float16* b_global, float* out,
        hip_moi::context_storage_ref storage)
    {
        __shared__ f16x8_t a_shared[2][kThreadCount];
        __shared__ f16x8_t b_shared[2][kThreadCount];

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        f32x8_t acc = {};
        load_data_tiled_fragments_to_lds(
            ctx, a_shared[0], b_shared[0], a_global, b_global, /*tile_id=*/0);
        ctx.syncthreads();

        for(int tile_id = 0; tile_id < 2; ++tile_id)
        {
            int buffer = tile_id & 1;
            wmma_f16_mac_from_data_tiled_lds(ctx, acc, a_shared[buffer], b_shared[buffer]);
            ctx.syncthreads();

            int next_tile = tile_id + 1;
            if(next_tile < 2)
            {
                int next_buffer = next_tile & 1;
                load_data_tiled_fragments_to_lds(
                    ctx, a_shared[next_buffer], b_shared[next_buffer], a_global, b_global,
                    next_tile);
            }
            ctx.syncthreads();
        }

        store_accumulator(acc, out);
    }

    __global__ void data_tiled_neighbor_fragment_overwrite_without_barrier_kernel(
        const _Float16* a_global, const _Float16* b_global, hip_moi::context_storage_ref storage)
    {
        __shared__ f16x8_t a_shared[kThreadCount];
        __shared__ f16x8_t b_shared[kThreadCount];

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        f32x8_t acc = {};
        load_data_tiled_fragments_to_lds(
            ctx, a_shared, b_shared, a_global, b_global, /*tile_id=*/0);
        ctx.syncthreads();
        wmma_f16_mac_from_data_tiled_lds(ctx, acc, a_shared, b_shared);

        if(threadIdx.x == 0)
        {
            f16x8_t replacement;
            for(int i = 0; i < kFragmentElements; ++i)
            {
                replacement[i] = static_cast<_Float16>(2.0f);
            }
            ctx.lds_store(&a_shared[1], replacement);
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

    void fill_ones(_Float16* values, int count)
    {
        for(int i = 0; i < count; ++i)
        {
            values[i] = static_cast<_Float16>(1.0f);
        }
    }

    void expect_output_equals(float* device_output, float expected)
    {
        float output[kOutputElementCount] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));
        for(float value : output)
        {
            EXPECT_EQ(value, expected);
        }
    }

    void expect_no_diagnostics(const observed_metadata& observed, int expected_access_count)
    {
        EXPECT_EQ(observed.access_count, expected_access_count);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    void expect_neighbor_fragment_reuse_diagnostics(const observed_metadata& observed)
    {
        EXPECT_EQ(observed.access_count, 129);
        ASSERT_EQ(observed.diagnostic_count, 1);

        const hip_moi::diagnostic& diagnostic = observed.diagnostics[0];
        EXPECT_EQ(diagnostic.kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
        EXPECT_EQ(diagnostic.epoch, 1u);
        EXPECT_EQ(diagnostic.first_size, sizeof(f16x8_t));
        EXPECT_EQ(diagnostic.second_size, sizeof(f16x8_t));
    }

    template <typename Kernel>
    void run_safe_case(Kernel kernel, int tile_count, float expected_value, int expected_access_count)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/128, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        _Float16 host_a[2 * kPackedTileElements];
        _Float16 host_b[2 * kPackedTileElements];
        fill_ones(host_a, tile_count * kPackedTileElements);
        fill_ones(host_b, tile_count * kPackedTileElements);

        _Float16* device_a      = nullptr;
        _Float16* device_b      = nullptr;
        float*    device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMalloc(&device_a, tile_count * kPackedTileElements * sizeof(_Float16)));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMalloc(&device_b, tile_count * kPackedTileElements * sizeof(_Float16)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kOutputElementCount * sizeof(float)));

        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(device_a,
                                          host_a,
                                          tile_count * kPackedTileElements * sizeof(_Float16),
                                          hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(device_b,
                                          host_b,
                                          tile_count * kPackedTileElements * sizeof(_Float16),
                                          hipMemcpyHostToDevice));

        hipLaunchKernelGGL(
            kernel, dim3(1), dim3(kThreadCount), 0, 0, device_a, device_b, device_output,
            storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_output_equals(device_output, expected_value);
        expect_no_diagnostics(observed, expected_access_count);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_a));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_b));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiRdna4WmmaDataTiled, SingleBufferUsesPackedFragments)
    {
        run_safe_case(data_tiled_single_buffer_kernel,
                      /*tile_count=*/1,
                      /*expected_value=*/16.0f,
                      /*expected_access_count=*/128);
    }

    TEST(HipMoiRdna4WmmaDataTiled, PreloadedPingPongUsesPackedFragments)
    {
        run_safe_case(data_tiled_preloaded_ping_pong_kernel,
                      /*tile_count=*/2,
                      /*expected_value=*/32.0f,
                      /*expected_access_count=*/256);
    }

    TEST(HipMoiRdna4WmmaDataTiled, NeighborFragmentOverwriteWithoutBarrierReports)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(
            /*access_capacity=*/128, /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/1));

        _Float16 host_a[kPackedTileElements];
        _Float16 host_b[kPackedTileElements];
        fill_ones(host_a, kPackedTileElements);
        fill_ones(host_b, kPackedTileElements);

        _Float16* device_a = nullptr;
        _Float16* device_b = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_a, sizeof(host_a)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_b, sizeof(host_b)));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_a, host_a, sizeof(host_a), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_b, host_b, sizeof(host_b), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(data_tiled_neighbor_fragment_overwrite_without_barrier_kernel,
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
        expect_neighbor_fragment_reuse_diagnostics(observed);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_a));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_b));
    }
} // namespace
