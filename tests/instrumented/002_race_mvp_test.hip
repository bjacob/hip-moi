// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/002_race_mvp_test.hip
//
// Initial diagnostic-positive tests for the hip-moi instrumentation API.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount = 8;

    __global__ void same_epoch_write_read_kernel(int* out, hip_moi::context_storage_ref storage)
    {
        __shared__ int value;

        hip_moi::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
            /*subgroup_count=*/1,
        };
        hip_moi::thread_level_context ctx(storage, cfg);

        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            ctx.lds_store(&value, 123);
        }
        if(threadIdx.x == 1)
        {
            out[0] = ctx.lds_load(&value);
        }
    }

    TEST(HipMoiInstrumentedRaceMvp, SameEpochWriteReadReportsDiagnostic)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate(16, 4, 1));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));

        int initial = -1;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(same_epoch_write_read_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int access_count = -1;
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&access_count,
                                          storage.access_count_device(),
                                          sizeof(access_count),
                                          hipMemcpyDeviceToHost));

        int diagnostic_count = -1;
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&diagnostic_count,
                                          storage.diagnostic_count_device(),
                                          sizeof(diagnostic_count),
                                          hipMemcpyDeviceToHost));

        hip_moi::diagnostic diagnostics[4] = {};
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(
            diagnostics, storage.diagnostics_device(), sizeof(diagnostics), hipMemcpyDeviceToHost));

        EXPECT_EQ(access_count, 2);
        ASSERT_EQ(diagnostic_count, 1);

        const hip_moi::diagnostic& diagnostic = diagnostics[0];
        EXPECT_EQ(diagnostic.kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
        EXPECT_EQ(diagnostic.epoch, 0u);
        EXPECT_EQ(diagnostic.first_addr, diagnostic.second_addr);
        EXPECT_EQ(diagnostic.first_size, sizeof(int));
        EXPECT_EQ(diagnostic.second_size, sizeof(int));

        bool threads_are_zero_and_one = (diagnostic.writer_or_first_thread_id == 0u
                                         && diagnostic.reader_or_second_thread_id == 1u)
                                        || (diagnostic.writer_or_first_thread_id == 1u
                                            && diagnostic.reader_or_second_thread_id == 0u);
        EXPECT_TRUE(threads_are_zero_and_one);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }
} // namespace
