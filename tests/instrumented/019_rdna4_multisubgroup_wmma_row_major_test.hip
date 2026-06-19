// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/019_rdna4_multisubgroup_wmma_row_major_test.hip
//
// RDNA4 WMMA tests using two subgroups in one workgroup. Each subgroup owns a
// conventional row-major A/B/C tile and stages scalar elements through LDS.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kThreadCount        = kThreadsPerSubgroup * kSubgroupCount;
    constexpr int kTileCount          = 2;
    constexpr int kTileM              = 16;
    constexpr int kTileN              = 16;
    constexpr int kTileK              = 16;
    constexpr int kTileElementCount   = kTileM * kTileK;
    constexpr int kInputElementCount  = kTileCount * kSubgroupCount * kTileElementCount;
    constexpr int kOutputElementCount = kSubgroupCount * kTileM * kTileN;
    constexpr int kExpectedSafeAccessCount
        = kTileCount * kSubgroupCount * (2 * kTileElementCount) + kTileCount * kThreadCount * 16;
    constexpr int kExpectedRaceAccessCount = 2 * kTileElementCount + kThreadsPerSubgroup * 16;
    constexpr int kAccessCapacity          = 1280;
    constexpr int kDiagnosticCapacity      = 128;

    using f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
    using f32x8_t = float __attribute__((ext_vector_type(8)));

    template <typename Diagnostic>
    struct observed_metadata
    {
        int        access_count;
        int        diagnostic_count;
        Diagnostic diagnostics[kDiagnosticCapacity];
    };

    using thread_observed_metadata = observed_metadata<hip_moi::thread_level_context::diagnostic>;
    using subgroup_observed_metadata
        = observed_metadata<hip_moi::subgroup_level_context::diagnostic>;

    __host__ __device__ int row_major(int row, int col, int cols)
    {
        return col + cols * row;
    }

    __host__ __device__ int tile_offset(int tile_id, int subgroup, int element)
    {
        return element + kTileElementCount * (subgroup + kSubgroupCount * tile_id);
    }

    __host__ __device__ int output_offset(int subgroup, int row, int col)
    {
        return row_major(row, col, kTileN) + kTileM * kTileN * subgroup;
    }

    __host__ __device__ int matmul_a_value(int tile_id, int subgroup, int m, int k)
    {
        return ((1 + tile_id + 2 * subgroup + 2 * m + 3 * k) % 5) - 2;
    }

    __host__ __device__ int matmul_b_value(int tile_id, int subgroup, int n, int k)
    {
        return ((2 + 2 * tile_id + subgroup + 3 * n + k) % 5) - 2;
    }

    __host__ __device__ int rdna4_wmma_f16_k(int lane, int idx)
    {
        int reg  = idx >> 1;
        int half = idx & 1;
        return half | ((reg & 1) << 1) | ((lane >> 4) << 2) | ((reg >> 1) << 3);
    }

    __host__ __device__ int rdna4_wmma_acc_m(int lane, int elem)
    {
        return elem + 8 * (lane >> 4);
    }

    float reference_matmul_value(int subgroup, int m, int n)
    {
        int acc = 0;
        for(int tile_id = 0; tile_id < kTileCount; ++tile_id)
        {
            for(int k = 0; k < kTileK; ++k)
            {
                acc += matmul_a_value(tile_id, subgroup, m, k)
                       * matmul_b_value(tile_id, subgroup, n, k);
            }
        }
        return static_cast<float>(acc);
    }

    __device__ hip_moi::thread_level_context
               make_thread_level_context(hip_moi::thread_level_context::storage_ref storage)
    {
        hip_moi::thread_level_context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::thread_level_context(storage, cfg);
    }

    __device__ hip_moi::subgroup_level_context
               make_subgroup_level_context(hip_moi::subgroup_level_context::storage_ref storage)
    {
        hip_moi::subgroup_level_context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::subgroup_level_context(storage, cfg);
    }

    __device__ int subgroup_tile_base(int subgroup)
    {
        return subgroup * kTileElementCount;
    }

    template <typename Context>
    __device__ void load_row_major_tile_to_lds(Context&        ctx,
                                               _Float16*       a_shared,
                                               _Float16*       b_shared,
                                               const _Float16* a_global,
                                               const _Float16* b_global,
                                               int             tile_id)
    {
        int subgroup    = static_cast<int>(ctx.subgroup_id());
        int rank        = static_cast<int>(ctx.thread_rank_in_subgroup());
        int shared_base = subgroup_tile_base(subgroup);
        for(int idx = rank; idx < kTileElementCount; idx += kThreadsPerSubgroup)
        {
            int offset = tile_offset(tile_id, subgroup, idx);
            ctx.lds_store(&a_shared[shared_base + idx], a_global[offset]);
            ctx.lds_store(&b_shared[shared_base + idx], b_global[offset]);
        }
    }

    template <typename Context>
    __device__ void wmma_f16_mac_from_row_major_lds_at_base(Context&        ctx,
                                                            f32x8_t&        acc,
                                                            const _Float16* a_shared,
                                                            const _Float16* b_shared,
                                                            int             shared_base)
    {
        int lane = static_cast<int>(ctx.thread_rank_in_subgroup());
        int m    = lane & 15;
        int n    = lane & 15;

        f16x8_t a;
        f16x8_t b;
        for(int idx = 0; idx < 8; ++idx)
        {
            int k  = rdna4_wmma_f16_k(lane, idx);
            a[idx] = ctx.lds_load(&a_shared[shared_base + row_major(m, k, kTileK)]);
            b[idx] = ctx.lds_load(&b_shared[shared_base + row_major(n, k, kTileK)]);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }

    template <typename Context>
    __device__ void wmma_f16_mac_from_row_major_lds(Context&        ctx,
                                                    f32x8_t&        acc,
                                                    const _Float16* a_shared,
                                                    const _Float16* b_shared)
    {
        int shared_base = subgroup_tile_base(static_cast<int>(ctx.subgroup_id()));
        wmma_f16_mac_from_row_major_lds_at_base(ctx, acc, a_shared, b_shared, shared_base);
    }

    __device__ void store_accumulator(const f32x8_t& acc, float* out, int subgroup, int lane)
    {
        int n = lane & 15;
        for(int elem = 0; elem < 8; ++elem)
        {
            int m                              = rdna4_wmma_acc_m(lane, elem);
            out[output_offset(subgroup, m, n)] = acc[elem];
        }
    }

    template <typename Context>
    __device__ void run_safe_double_buffered_kernel_body(Context&        ctx,
                                                         const _Float16* a_global,
                                                         const _Float16* b_global,
                                                         float*          out)
    {
        __shared__ _Float16 a_shared[2][kSubgroupCount * kTileElementCount];
        __shared__ _Float16 b_shared[2][kSubgroupCount * kTileElementCount];

        ctx.init_workgroup();

        int subgroup = static_cast<int>(ctx.subgroup_id());
        int lane     = static_cast<int>(ctx.thread_rank_in_subgroup());

        f32x8_t acc = {};
        load_row_major_tile_to_lds(
            ctx, a_shared[0], b_shared[0], a_global, b_global, /*tile_id=*/0);
        ctx.syncthreads();

        for(int tile_id = 0; tile_id < kTileCount; ++tile_id)
        {
            int buffer = tile_id & 1;
            wmma_f16_mac_from_row_major_lds(ctx, acc, a_shared[buffer], b_shared[buffer]);
            ctx.syncthreads();

            int next_tile = tile_id + 1;
            if(next_tile < kTileCount)
            {
                int next_buffer = next_tile & 1;
                load_row_major_tile_to_lds(ctx,
                                           a_shared[next_buffer],
                                           b_shared[next_buffer],
                                           a_global,
                                           b_global,
                                           next_tile);
            }
            ctx.syncthreads();
        }

        store_accumulator(acc, out, subgroup, lane);
    }

    template <typename Context>
    __device__ void run_cross_subgroup_missing_barrier_body(Context&        ctx,
                                                            const _Float16* a_global,
                                                            const _Float16* b_global)
    {
        __shared__ _Float16 a_shared[kTileElementCount];
        __shared__ _Float16 b_shared[kTileElementCount];

        ctx.init_workgroup();

        uint32_t subgroup = ctx.subgroup_id();
        uint32_t rank     = ctx.thread_rank_in_subgroup();
        if(subgroup == 0)
        {
            for(int idx = static_cast<int>(rank); idx < kTileElementCount;
                idx += kThreadsPerSubgroup)
            {
                ctx.lds_store(&a_shared[idx], a_global[idx]);
                ctx.lds_store(&b_shared[idx], b_global[idx]);
            }
        }
        if(subgroup == 1)
        {
            f32x8_t acc = {};
            wmma_f16_mac_from_row_major_lds_at_base(
                ctx, acc, a_shared, b_shared, /*shared_base=*/0);
            (void)acc;
        }
    }

    __global__ void
        safe_double_buffered_thread_level_kernel(const _Float16* a_global,
                                                 const _Float16* b_global,
                                                 float*          out,
                                                 hip_moi::thread_level_context::storage_ref storage)
    {
        hip_moi::thread_level_context ctx = make_thread_level_context(storage);
        run_safe_double_buffered_kernel_body(ctx, a_global, b_global, out);
    }

    __global__ void safe_double_buffered_subgroup_level_kernel(
        const _Float16*                              a_global,
        const _Float16*                              b_global,
        float*                                       out,
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        hip_moi::subgroup_level_context ctx = make_subgroup_level_context(storage);
        run_safe_double_buffered_kernel_body(ctx, a_global, b_global, out);
    }

    __global__ void
        missing_barrier_thread_level_kernel(const _Float16*                            a_global,
                                            const _Float16*                            b_global,
                                            hip_moi::thread_level_context::storage_ref storage)
    {
        hip_moi::thread_level_context ctx = make_thread_level_context(storage);
        run_cross_subgroup_missing_barrier_body(ctx, a_global, b_global);
    }

    __global__ void
        missing_barrier_subgroup_level_kernel(const _Float16*                              a_global,
                                              const _Float16*                              b_global,
                                              hip_moi::subgroup_level_context::storage_ref storage)
    {
        hip_moi::subgroup_level_context ctx = make_subgroup_level_context(storage);
        run_cross_subgroup_missing_barrier_body(ctx, a_global, b_global);
        ctx.finish();
    }

    void fill_row_major_inputs(_Float16* a, _Float16* b)
    {
        for(int tile_id = 0; tile_id < kTileCount; ++tile_id)
        {
            for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
            {
                for(int m = 0; m < kTileM; ++m)
                {
                    for(int k = 0; k < kTileK; ++k)
                    {
                        int element = row_major(m, k, kTileK);
                        a[tile_offset(tile_id, subgroup, element)]
                            = static_cast<_Float16>(matmul_a_value(tile_id, subgroup, m, k));
                    }
                }
                for(int n = 0; n < kTileN; ++n)
                {
                    for(int k = 0; k < kTileK; ++k)
                    {
                        int element = row_major(n, k, kTileK);
                        b[tile_offset(tile_id, subgroup, element)]
                            = static_cast<_Float16>(matmul_b_value(tile_id, subgroup, n, k));
                    }
                }
            }
        }
    }

    void expect_reference_output(float* device_output)
    {
        float output[kOutputElementCount] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));

        for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
        {
            for(int m = 0; m < kTileM; ++m)
            {
                for(int n = 0; n < kTileN; ++n)
                {
                    EXPECT_EQ(output[output_offset(subgroup, m, n)],
                              reference_matmul_value(subgroup, m, n));
                }
            }
        }
    }

    template <typename Storage, typename Observed>
    void copy_observed_metadata(const Storage& storage, Observed* observed)
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

    int stored_diagnostic_count(int diagnostic_count)
    {
        return diagnostic_count < kDiagnosticCapacity ? diagnostic_count : kDiagnosticCapacity;
    }

    void expect_thread_level_cross_subgroup_conflict(const thread_observed_metadata& observed)
    {
        EXPECT_EQ(observed.access_count, kExpectedRaceAccessCount);
        ASSERT_GT(observed.diagnostic_count, 0);

        bool found_expected_conflict = false;
        for(int i = 0; i < stored_diagnostic_count(observed.diagnostic_count); ++i)
        {
            const hip_moi::thread_level_context::diagnostic& diagnostic = observed.diagnostics[i];
            bool has_subgroup0_thread = diagnostic.first_thread_id < kThreadsPerSubgroup
                                        || diagnostic.second_thread_id < kThreadsPerSubgroup;
            bool has_subgroup1_thread = diagnostic.first_thread_id >= kThreadsPerSubgroup
                                        || diagnostic.second_thread_id >= kThreadsPerSubgroup;
            found_expected_conflict
                |= diagnostic.kind
                       == static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict)
                   && diagnostic.epoch == 0 && diagnostic.first_size == sizeof(_Float16)
                   && diagnostic.second_size == sizeof(_Float16) && has_subgroup0_thread
                   && has_subgroup1_thread;
        }
        EXPECT_TRUE(found_expected_conflict);
    }

    void expect_subgroup_level_cross_subgroup_conflict(const subgroup_observed_metadata& observed)
    {
        EXPECT_EQ(observed.access_count, kExpectedRaceAccessCount);
        ASSERT_GT(observed.diagnostic_count, 0);

        bool found_expected_conflict = false;
        for(int i = 0; i < stored_diagnostic_count(observed.diagnostic_count); ++i)
        {
            const hip_moi::subgroup_level_context::diagnostic& diagnostic = observed.diagnostics[i];
            bool                                               matches_subgroups
                = (diagnostic.first_subgroup_id == 0 && diagnostic.second_subgroup_id == 1)
                  || (diagnostic.first_subgroup_id == 1 && diagnostic.second_subgroup_id == 0);
            found_expected_conflict
                |= diagnostic.kind
                       == static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict)
                   && diagnostic.epoch == 0 && diagnostic.first_size == sizeof(_Float16)
                   && diagnostic.second_size == sizeof(_Float16) && matches_subgroups;
        }
        EXPECT_TRUE(found_expected_conflict);
    }

    template <typename Storage, typename Kernel>
    void run_safe_case(Kernel kernel, Storage* storage)
    {
        ASSERT_NO_FATAL_FAILURE(storage->allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        _Float16 host_a[kInputElementCount];
        _Float16 host_b[kInputElementCount];
        fill_row_major_inputs(host_a, host_b);

        _Float16* device_a      = nullptr;
        _Float16* device_b      = nullptr;
        float*    device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_a, sizeof(host_a)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_b, sizeof(host_b)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kOutputElementCount * sizeof(float)));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(device_a, host_a, sizeof(host_a), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(device_b, host_b, sizeof(host_b), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_a,
                           device_b,
                           device_output,
                           storage->ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        expect_reference_output(device_output);
        ASSERT_NO_FATAL_FAILURE(hip_moi::test::expect_metadata_counts(
            *storage,
            /*expected_access_count=*/kExpectedSafeAccessCount,
            /*expected_diagnostic_count=*/0));

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_a));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_b));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    template <typename Storage, typename Kernel>
    void run_missing_barrier_case(Kernel kernel, Storage* storage)
    {
        ASSERT_NO_FATAL_FAILURE(storage->allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        _Float16 host_a[kInputElementCount];
        _Float16 host_b[kInputElementCount];
        fill_row_major_inputs(host_a, host_b);

        _Float16* device_a = nullptr;
        _Float16* device_b = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_a, sizeof(host_a)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_b, sizeof(host_b)));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(device_a, host_a, sizeof(host_a), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(device_b, host_b, sizeof(host_b), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(
            kernel, dim3(1), dim3(kThreadCount), 0, 0, device_a, device_b, storage->ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_a));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_b));
    }

    TEST(HipMoiRdna4WmmaMultiSubgroupRowMajor, ThreadLevelDoubleBufferedDoesNotReport)
    {
        hip_moi::test::thread_level_device_context_storage storage;
        run_safe_case(safe_double_buffered_thread_level_kernel, &storage);
    }

    TEST(HipMoiRdna4WmmaMultiSubgroupRowMajor, SubgroupLevelDoubleBufferedDoesNotReport)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        run_safe_case(safe_double_buffered_subgroup_level_kernel, &storage);
    }

    TEST(HipMoiRdna4WmmaMultiSubgroupRowMajor, ThreadLevelCrossSubgroupMissingBarrierReports)
    {
        hip_moi::test::thread_level_device_context_storage storage;
        run_missing_barrier_case(missing_barrier_thread_level_kernel, &storage);

        thread_observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_thread_level_cross_subgroup_conflict(observed);
    }

    TEST(HipMoiRdna4WmmaMultiSubgroupRowMajor, SubgroupLevelCrossSubgroupMissingBarrierReports)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        run_missing_barrier_case(missing_barrier_subgroup_level_kernel, &storage);

        subgroup_observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_subgroup_level_cross_subgroup_conflict(observed);
    }
} // namespace
