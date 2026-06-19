// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/018_rdna4_multisubgroup_wmma_data_tiled_test.hip
//
// RDNA4 WMMA tests using two subgroups in one workgroup. Each subgroup owns
// data-tiled A/B/C fragments and stages vector fragments through LDS.
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
    constexpr int kTileK              = 16;
    constexpr int kFragmentElements   = 8;
    constexpr int kInputElementCount
        = kTileCount * kSubgroupCount * kThreadsPerSubgroup * kFragmentElements;
    constexpr int kOutputElementCount = kSubgroupCount * kThreadsPerSubgroup * kFragmentElements;
    constexpr int kExpectedSafeAccessCount = 2 * kTileCount * kThreadCount * 2;
    constexpr int kExpectedRaceAccessCount = kThreadCount * 2;
    constexpr int kAccessCapacity          = 256;
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

    __host__ __device__ int fragment_offset(int tile_id, int subgroup, int lane, int elem)
    {
        return elem
               + kFragmentElements
                     * (lane + kThreadsPerSubgroup * (subgroup + kSubgroupCount * tile_id));
    }

    __host__ __device__ int output_offset(int subgroup, int lane, int elem)
    {
        return elem + kFragmentElements * (lane + kThreadsPerSubgroup * subgroup);
    }

    __host__ __device__ int matmul_a_value(int tile_id, int subgroup, int m, int k)
    {
        return ((1 + tile_id + 2 * subgroup + 2 * m + 3 * k) % 5) - 2;
    }

    __host__ __device__ int matmul_b_value(int tile_id, int subgroup, int n, int k)
    {
        return ((2 + 2 * tile_id + subgroup + 3 * n + k) % 5) - 2;
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

    __device__ f16x8_t load_global_fragment(const _Float16* global_fragments, int fragment_index)
    {
        return reinterpret_cast<const f16x8_t*>(global_fragments)[fragment_index];
    }

    __device__ void
        store_global_fragment(float* global_fragments, int fragment_index, f32x8_t fragment)
    {
        reinterpret_cast<f32x8_t*>(global_fragments)[fragment_index] = fragment;
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

    template <typename Context>
    __device__ int lds_fragment_index(Context& ctx)
    {
        return static_cast<int>(ctx.thread_rank_in_subgroup()
                                + kThreadsPerSubgroup * ctx.subgroup_id());
    }

    template <typename Context>
    __device__ void load_subgroup_fragments_to_lds(Context&        ctx,
                                                   f16x8_t*        a_shared,
                                                   f16x8_t*        b_shared,
                                                   const _Float16* a_global,
                                                   const _Float16* b_global,
                                                   int             tile_id)
    {
        int subgroup        = static_cast<int>(ctx.subgroup_id());
        int lane            = static_cast<int>(ctx.thread_rank_in_subgroup());
        int global_fragment = subgroup + kSubgroupCount * tile_id;
        global_fragment     = lane + kThreadsPerSubgroup * global_fragment;
        int shared_fragment = lds_fragment_index(ctx);

        ctx.lds_store(&a_shared[shared_fragment], load_global_fragment(a_global, global_fragment));
        ctx.lds_store(&b_shared[shared_fragment], load_global_fragment(b_global, global_fragment));
    }

    template <typename Context>
    __device__ void wmma_f16_mac_from_data_tiled_lds(Context&       ctx,
                                                     f32x8_t&       acc,
                                                     const f16x8_t* a_shared,
                                                     const f16x8_t* b_shared)
    {
        int     shared_fragment = lds_fragment_index(ctx);
        f16x8_t a               = ctx.lds_load(&a_shared[shared_fragment]);
        f16x8_t b               = ctx.lds_load(&b_shared[shared_fragment]);
        acc                     = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }

    __device__ void store_accumulator(const f32x8_t& acc, float* out, int subgroup, int lane)
    {
        store_global_fragment(out, subgroup * kThreadsPerSubgroup + lane, acc);
    }

    template <typename Context>
    __device__ void run_safe_double_buffered_kernel_body(Context&        ctx,
                                                         const _Float16* a_global,
                                                         const _Float16* b_global,
                                                         float*          out)
    {
        __shared__ f16x8_t a_shared[2][kThreadCount];
        __shared__ f16x8_t b_shared[2][kThreadCount];

        ctx.init_workgroup();

        int subgroup = static_cast<int>(ctx.subgroup_id());
        int lane     = static_cast<int>(ctx.thread_rank_in_subgroup());

        f32x8_t acc = {};
        load_subgroup_fragments_to_lds(
            ctx, a_shared[0], b_shared[0], a_global, b_global, /*tile_id=*/0);
        ctx.syncthreads();

        for(int tile_id = 0; tile_id < kTileCount; ++tile_id)
        {
            int buffer = tile_id & 1;
            wmma_f16_mac_from_data_tiled_lds(ctx, acc, a_shared[buffer], b_shared[buffer]);
            ctx.syncthreads();

            int next_tile = tile_id + 1;
            if(next_tile < kTileCount)
            {
                int next_buffer = next_tile & 1;
                load_subgroup_fragments_to_lds(ctx,
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
        __shared__ f16x8_t a_shared[kThreadsPerSubgroup];
        __shared__ f16x8_t b_shared[kThreadsPerSubgroup];

        ctx.init_workgroup();

        uint32_t subgroup = ctx.subgroup_id();
        uint32_t lane     = ctx.thread_rank_in_subgroup();
        if(subgroup == 0)
        {
            int global_fragment = static_cast<int>(lane);
            ctx.lds_store(&a_shared[lane], load_global_fragment(a_global, global_fragment));
            ctx.lds_store(&b_shared[lane], load_global_fragment(b_global, global_fragment));
        }
        if(subgroup == 1)
        {
            f32x8_t acc = {};
            f16x8_t a   = ctx.lds_load(&a_shared[lane]);
            f16x8_t b   = ctx.lds_load(&b_shared[lane]);
            acc         = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
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
    }

    void fill_data_tiled_inputs(_Float16* a, _Float16* b)
    {
        for(int tile_id = 0; tile_id < kTileCount; ++tile_id)
        {
            for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
            {
                for(int lane = 0; lane < kThreadsPerSubgroup; ++lane)
                {
                    int m = lane & 15;
                    int n = lane & 15;
                    for(int elem = 0; elem < kFragmentElements; ++elem)
                    {
                        int k      = rdna4_wmma_f16_k(lane, elem);
                        int offset = fragment_offset(tile_id, subgroup, lane, elem);
                        a[offset]  = static_cast<_Float16>(matmul_a_value(tile_id, subgroup, m, k));
                        b[offset]  = static_cast<_Float16>(matmul_b_value(tile_id, subgroup, n, k));
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
            for(int lane = 0; lane < kThreadsPerSubgroup; ++lane)
            {
                int n = lane & 15;
                for(int elem = 0; elem < kFragmentElements; ++elem)
                {
                    int m = rdna4_wmma_acc_m(lane, elem);
                    EXPECT_EQ(output[output_offset(subgroup, lane, elem)],
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
                   && diagnostic.epoch == 0 && diagnostic.first_size == sizeof(f16x8_t)
                   && diagnostic.second_size == sizeof(f16x8_t) && has_subgroup0_thread
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
                   && diagnostic.epoch == 0 && diagnostic.first_size == sizeof(f16x8_t)
                   && diagnostic.second_size == sizeof(f16x8_t) && matches_subgroups;
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
        fill_data_tiled_inputs(host_a, host_b);

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
        fill_data_tiled_inputs(host_a, host_b);

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

    TEST(HipMoiRdna4WmmaMultiSubgroup, ThreadLevelDoubleBufferedDoesNotReport)
    {
        hip_moi::test::thread_level_device_context_storage storage;
        run_safe_case(safe_double_buffered_thread_level_kernel, &storage);
    }

    TEST(HipMoiRdna4WmmaMultiSubgroup, SubgroupLevelDoubleBufferedDoesNotReport)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        run_safe_case(safe_double_buffered_subgroup_level_kernel, &storage);
    }

    TEST(HipMoiRdna4WmmaMultiSubgroup, ThreadLevelCrossSubgroupMissingBarrierReports)
    {
        hip_moi::test::thread_level_device_context_storage storage;
        run_missing_barrier_case(missing_barrier_thread_level_kernel, &storage);

        thread_observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_thread_level_cross_subgroup_conflict(observed);
    }

    TEST(HipMoiRdna4WmmaMultiSubgroup, SubgroupLevelCrossSubgroupMissingBarrierReports)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        run_missing_barrier_case(missing_barrier_subgroup_level_kernel, &storage);

        subgroup_observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_subgroup_level_cross_subgroup_conflict(observed);
    }
} // namespace
