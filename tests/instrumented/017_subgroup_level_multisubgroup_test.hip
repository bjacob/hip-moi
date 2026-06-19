// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/017_subgroup_level_multisubgroup_test.hip
//
// Richer multi-subgroup tests for subgroup-level instrumentation mode.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 128;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 4;
    constexpr int kDiagnosticCapacity = 128;
    constexpr int kAccessCapacity     = 128;
    constexpr int kLoopCount          = 3;
    constexpr int kTileCols           = 8;
    constexpr int kMatmulM            = 2;
    constexpr int kMatmulN            = 2;
    constexpr int kMatmulK            = 4;

    struct observed_metadata
    {
        int                      access_count;
        int                      diagnostic_count;
        hip_moi::subgroup_level_context::diagnostic    diagnostics[kDiagnosticCapacity];
        hip_moi::subgroup_level_context::access_record access_records[kAccessCapacity];
    };

    __device__ hip_moi::subgroup_level_context make_context(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        hip_moi::subgroup_level_context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::subgroup_level_context(storage, cfg);
    }

    __device__ int matmul_a_value(int m, int k)
    {
        return 1 + m + k;
    }

    __device__ int matmul_b_value(int k, int n)
    {
        return 2 + k - n;
    }

    __global__ void independent_subgroup_array_slots_kernel(
        int* out, hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[kSubgroupCount];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        uint32_t subgroup = ctx.subgroup_id();
        uint32_t rank     = ctx.thread_rank_in_subgroup();
        if(rank == 0)
        {
            ctx.lds_store(&values[subgroup], static_cast<int>(10 + subgroup));
        }

        ctx.syncthreads();

        if(rank == 1)
        {
            out[subgroup] = ctx.lds_load(&values[subgroup]);
        }
        ctx.finish();
    }

    __global__ void cross_subgroup_array_overlap_kernel(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[2];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&values[0], 11);
        }
        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            (void)ctx.lds_load(&values[0]);
        }
        if(ctx.thread_id() == 2 * kThreadsPerSubgroup)
        {
            ctx.lds_store(&values[1], 22);
        }
        if(ctx.thread_id() == 3 * kThreadsPerSubgroup)
        {
            (void)ctx.lds_load(&values[1]);
        }
        ctx.finish();
    }

    __global__ void cross_subgroup_array_separated_kernel(
        int* out, hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int values[2];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&values[0], 11);
        }
        if(ctx.thread_id() == 2 * kThreadsPerSubgroup)
        {
            ctx.lds_store(&values[1], 22);
        }

        ctx.syncthreads();

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            out[0] = ctx.lds_load(&values[0]);
        }
        if(ctx.thread_id() == 3 * kThreadsPerSubgroup)
        {
            out[1] = ctx.lds_load(&values[1]);
        }
        ctx.finish();
    }

    __global__ void looped_cross_subgroup_conflict_kernel(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        for(int iteration = 0; iteration < kLoopCount; ++iteration)
        {
            if(ctx.thread_id() == 0)
            {
                ctx.lds_store(&value, iteration);
            }
            if(ctx.thread_id() == kThreadsPerSubgroup)
            {
                (void)ctx.lds_load(&value);
            }
            ctx.syncthreads();
        }
    }

    __global__ void tiled_independent_rows_kernel(
        int* out, hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int tile[kSubgroupCount][kTileCols];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        uint32_t subgroup = ctx.subgroup_id();
        uint32_t rank     = ctx.thread_rank_in_subgroup();
        if(rank < kTileCols)
        {
            ctx.lds_store(&tile[subgroup][rank],
                          static_cast<int>(100 * subgroup + rank));
        }

        ctx.syncthreads();

        if(rank < kTileCols)
        {
            out[subgroup * kTileCols + rank] = ctx.lds_load(&tile[subgroup][rank]);
        }
        ctx.finish();
    }

    __global__ void tiled_cross_subgroup_row_collision_kernel(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int row[kTileCols];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        uint32_t subgroup = ctx.subgroup_id();
        uint32_t rank     = ctx.thread_rank_in_subgroup();
        if(subgroup == 0 && rank < kTileCols)
        {
            ctx.lds_store(&row[rank], static_cast<int>(rank));
        }
        if(subgroup == 1 && rank < kTileCols)
        {
            (void)ctx.lds_load(&row[rank]);
        }
        ctx.finish();
    }

    __global__ void matmul_shape_cooperative_tile_kernel(
        int* out, hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int a_tile[kMatmulM][kMatmulK];
        __shared__ int b_tile[kMatmulK][kMatmulN];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        uint32_t subgroup = ctx.subgroup_id();
        uint32_t rank     = ctx.thread_rank_in_subgroup();
        if(subgroup == 0 && rank < kMatmulM * kMatmulK)
        {
            int m = static_cast<int>(rank) / kMatmulK;
            int k = static_cast<int>(rank) % kMatmulK;
            ctx.lds_store(&a_tile[m][k], matmul_a_value(m, k));
        }
        if(subgroup == 2 && rank < kMatmulK * kMatmulN)
        {
            int k = static_cast<int>(rank) / kMatmulN;
            int n = static_cast<int>(rank) % kMatmulN;
            ctx.lds_store(&b_tile[k][n], matmul_b_value(k, n));
        }

        ctx.syncthreads();

        if(subgroup == 1 && rank < kMatmulM * kMatmulN)
        {
            int m   = static_cast<int>(rank) / kMatmulN;
            int n   = static_cast<int>(rank) % kMatmulN;
            int acc = 0;
            for(int k = 0; k < kMatmulK; ++k)
            {
                acc += ctx.lds_load(&a_tile[m][k]) * ctx.lds_load(&b_tile[k][n]);
            }
            out[m * kMatmulN + n] = acc;
        }
        ctx.finish();
    }

    __global__ void matmul_shape_missing_barrier_kernel(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int a_tile[kMatmulM][kMatmulK];
        __shared__ int b_tile[kMatmulK][kMatmulN];

        hip_moi::subgroup_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        uint32_t subgroup = ctx.subgroup_id();
        uint32_t rank     = ctx.thread_rank_in_subgroup();
        if(subgroup == 0 && rank < kMatmulM * kMatmulK)
        {
            int m = static_cast<int>(rank) / kMatmulK;
            int k = static_cast<int>(rank) % kMatmulK;
            ctx.lds_store(&a_tile[m][k], matmul_a_value(m, k));
        }
        if(subgroup == 2 && rank < kMatmulK * kMatmulN)
        {
            int k = static_cast<int>(rank) / kMatmulN;
            int n = static_cast<int>(rank) % kMatmulN;
            ctx.lds_store(&b_tile[k][n], matmul_b_value(k, n));
        }
        if(subgroup == 1 && rank < kMatmulM * kMatmulN)
        {
            int m = static_cast<int>(rank) / kMatmulN;
            int n = static_cast<int>(rank) % kMatmulN;
            for(int k = 0; k < kMatmulK; ++k)
            {
                (void)ctx.lds_load(&a_tile[m][k]);
                (void)ctx.lds_load(&b_tile[k][n]);
            }
        }
        ctx.finish();
    }

    void copy_observed_metadata(const hip_moi::test::subgroup_level_device_context_storage& storage,
                                observed_metadata* observed)
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
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(observed->access_records,
                                          storage.access_records_device(),
                                          sizeof(observed->access_records),
                                          hipMemcpyDeviceToHost));
    }

    int stored_diagnostic_count(const observed_metadata& observed)
    {
        return observed.diagnostic_count < kDiagnosticCapacity ? observed.diagnostic_count
                                                              : kDiagnosticCapacity;
    }

    bool diagnostic_matches_subgroups(const hip_moi::subgroup_level_context::diagnostic& diagnostic,
                                      uint32_t first_subgroup,
                                      uint32_t second_subgroup)
    {
        return (diagnostic.first_subgroup_id == first_subgroup
                && diagnostic.second_subgroup_id == second_subgroup)
               || (diagnostic.first_subgroup_id == second_subgroup
                   && diagnostic.second_subgroup_id == first_subgroup);
    }

    void expect_no_diagnostics(const observed_metadata& observed, int expected_access_count)
    {
        EXPECT_EQ(observed.access_count, expected_access_count);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    void expect_conflict_between_subgroups(const observed_metadata& observed,
                                           uint32_t                 first_subgroup,
                                           uint32_t                 second_subgroup)
    {
        ASSERT_GT(observed.diagnostic_count, 0);

        bool found_expected_conflict = false;
        int  stored_count            = stored_diagnostic_count(observed);
        for(int i = 0; i < stored_count; ++i)
        {
            const hip_moi::subgroup_level_context::diagnostic& diagnostic =
                observed.diagnostics[i];
            found_expected_conflict |=
                diagnostic.kind == static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict)
                && diagnostic_matches_subgroups(diagnostic, first_subgroup, second_subgroup);
        }
        EXPECT_TRUE(found_expected_conflict);
    }

    void expect_conflict_between_subgroups_in_epoch(const observed_metadata& observed,
                                                    uint32_t                 first_subgroup,
                                                    uint32_t                 second_subgroup,
                                                    uint32_t                 epoch)
    {
        ASSERT_GT(observed.diagnostic_count, 0);

        bool found_expected_conflict = false;
        int  stored_count            = stored_diagnostic_count(observed);
        for(int i = 0; i < stored_count; ++i)
        {
            const hip_moi::subgroup_level_context::diagnostic& diagnostic =
                observed.diagnostics[i];
            found_expected_conflict |=
                diagnostic.kind == static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict)
                && diagnostic.epoch == epoch
                && diagnostic_matches_subgroups(diagnostic, first_subgroup, second_subgroup);
        }
        EXPECT_TRUE(found_expected_conflict);
    }

    void allocate_storage(hip_moi::test::subgroup_level_device_context_storage* storage)
    {
        ASSERT_NO_FATAL_FAILURE(storage->allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));
    }

    int matmul_reference(int m, int n)
    {
        int acc = 0;
        for(int k = 0; k < kMatmulK; ++k)
        {
            acc += (1 + m + k) * (2 + k - n);
        }
        return acc;
    }

    TEST(HipMoiSubgroupLevelMultiSubgroup, IndependentArraySlotsDoNotReport)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        allocate_storage(&storage);

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kSubgroupCount * sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMemset(device_output, 0xff, kSubgroupCount * sizeof(int)));

        hipLaunchKernelGGL(independent_subgroup_array_slots_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[kSubgroupCount] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));
        for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
        {
            EXPECT_EQ(output[subgroup], 10 + subgroup);
        }

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_no_diagnostics(observed, /*expected_access_count=*/2 * kSubgroupCount);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiSubgroupLevelMultiSubgroup, CrossSubgroupArrayOverlapReports)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        allocate_storage(&storage);

        hipLaunchKernelGGL(cross_subgroup_array_overlap_kernel,
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
        expect_conflict_between_subgroups(observed, /*first_subgroup=*/0, /*second_subgroup=*/1);
        expect_conflict_between_subgroups(observed, /*first_subgroup=*/2, /*second_subgroup=*/3);
    }

    TEST(HipMoiSubgroupLevelMultiSubgroup, CrossSubgroupArraySyncthreadsSeparationDoesNotReport)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        allocate_storage(&storage);

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, 2 * sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMemset(device_output, 0xff, 2 * sizeof(int)));

        hipLaunchKernelGGL(cross_subgroup_array_separated_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[2] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));
        EXPECT_EQ(output[0], 11);
        EXPECT_EQ(output[1], 22);

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_no_diagnostics(observed, /*expected_access_count=*/4);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiSubgroupLevelMultiSubgroup, LoopedCrossSubgroupConflictReportsEachEpoch)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        allocate_storage(&storage);

        hipLaunchKernelGGL(looped_cross_subgroup_conflict_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        EXPECT_EQ(observed.access_count, 2 * kLoopCount);
        for(uint32_t epoch = 0; epoch < static_cast<uint32_t>(kLoopCount); ++epoch)
        {
            expect_conflict_between_subgroups_in_epoch(
                observed, /*first_subgroup=*/0, /*second_subgroup=*/1, epoch);
        }
    }

    TEST(HipMoiSubgroupLevelMultiSubgroup, TiledIndependentRowsDoNotReport)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        allocate_storage(&storage);

        constexpr int kOutputCount = kSubgroupCount * kTileCols;
        int*          device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kOutputCount * sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMemset(device_output, 0xff, kOutputCount * sizeof(int)));

        hipLaunchKernelGGL(tiled_independent_rows_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[kOutputCount] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));
        for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
        {
            for(int col = 0; col < kTileCols; ++col)
            {
                EXPECT_EQ(output[subgroup * kTileCols + col], 100 * subgroup + col);
            }
        }

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_no_diagnostics(observed,
                              /*expected_access_count=*/2 * kSubgroupCount * kTileCols);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiSubgroupLevelMultiSubgroup, TiledCrossSubgroupRowCollisionReports)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        allocate_storage(&storage);

        hipLaunchKernelGGL(tiled_cross_subgroup_row_collision_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        EXPECT_EQ(observed.access_count, 2 * kTileCols);
        expect_conflict_between_subgroups(observed, /*first_subgroup=*/0, /*second_subgroup=*/1);
    }

    TEST(HipMoiSubgroupLevelMultiSubgroup, MatmulShapeCooperativeTileDoesNotReport)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        allocate_storage(&storage);

        constexpr int kOutputCount = kMatmulM * kMatmulN;
        int*          device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kOutputCount * sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMemset(device_output, 0xff, kOutputCount * sizeof(int)));

        hipLaunchKernelGGL(matmul_shape_cooperative_tile_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output[kOutputCount] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));
        for(int m = 0; m < kMatmulM; ++m)
        {
            for(int n = 0; n < kMatmulN; ++n)
            {
                EXPECT_EQ(output[m * kMatmulN + n], matmul_reference(m, n));
            }
        }

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        expect_no_diagnostics(observed,
                              /*expected_access_count=*/kMatmulM * kMatmulK
                                  + kMatmulK * kMatmulN
                                  + 2 * kMatmulM * kMatmulN * kMatmulK);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiSubgroupLevelMultiSubgroup, MatmulShapeMissingBarrierReportsCrossSubgroupConsumers)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        allocate_storage(&storage);

        hipLaunchKernelGGL(matmul_shape_missing_barrier_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));
        EXPECT_EQ(observed.access_count,
                  kMatmulM * kMatmulK + kMatmulK * kMatmulN
                      + 2 * kMatmulM * kMatmulN * kMatmulK);
        expect_conflict_between_subgroups(observed, /*first_subgroup=*/0, /*second_subgroup=*/1);
        expect_conflict_between_subgroups(observed, /*first_subgroup=*/1, /*second_subgroup=*/2);
    }
} // namespace
