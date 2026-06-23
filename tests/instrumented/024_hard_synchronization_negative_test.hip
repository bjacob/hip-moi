// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/024_hard_synchronization_negative_test.hip
//
// Non-hanging tests for synchronization patterns that would be hard to exercise
// with a real divergent __syncthreads().
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <string>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kDiagnosticCapacity = 8;

    constexpr hip_moi::site_id kSimulatedBarrierSite(0x2401u);

    hip_moi::host_context_options small_host_context_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes       = 64 * 1024;
        options.diagnostic_capacity = kDiagnosticCapacity;
        options.subgroup_capacity   = kSubgroupCount;
        return options;
    }

    struct observed_metadata
    {
        int                          diagnostic_count;
        hip_moi::context::diagnostic diagnostics[kDiagnosticCapacity];
    };

    __device__ hip_moi::context make_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    __global__ void simulated_barrier_safe_kernel(hip_moi::context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&value, 23, /*lds_byte_offset=*/0);
        }

        ctx.simulate_syncthreads(/*participates=*/true, kSimulatedBarrierSite);

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            (void)ctx.lds_load_at(&value, /*lds_byte_offset=*/0);
        }
        ctx.finish();
    }

    __global__ void
        simulated_barrier_missing_participant_kernel(hip_moi::context::storage_ref storage)
    {
        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        ctx.simulate_syncthreads(/*participates=*/ctx.thread_id() != 0, kSimulatedBarrierSite);
        ctx.finish();
    }

    template <typename Storage>
    void copy_observed_metadata(const Storage& storage, observed_metadata* observed)
    {
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&observed->diagnostic_count,
                                          storage.diagnostic_count_device(),
                                          sizeof(observed->diagnostic_count),
                                          hipMemcpyDeviceToHost));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(observed->diagnostics,
                                          storage.diagnostics_device(),
                                          sizeof(observed->diagnostics),
                                          hipMemcpyDeviceToHost));
    }

    template <typename Kernel>
    void run_case(Kernel kernel,
                  hip_moi::test::device_context_storage* storage,
                  observed_metadata*                     observed)
    {
        ASSERT_NO_FATAL_FAILURE(storage->allocate_with_exact_shadow(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*exact_shadow_entry_capacity=*/1));

        hipLaunchKernelGGL(kernel, dim3(1), dim3(kThreadCount), 0, 0, storage->ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(*storage, observed));
    }

    void expect_barrier_divergence_diagnostic(const hip_moi::context::diagnostic& diagnostic)
    {
        EXPECT_EQ(diagnostic.kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::barrier_divergence));
        EXPECT_EQ(diagnostic.epoch, 0u);
        EXPECT_EQ(diagnostic.expected_thread_count, static_cast<uint32_t>(kThreadCount));
        EXPECT_EQ(diagnostic.observed_thread_count, static_cast<uint32_t>(kThreadCount - 1));
        EXPECT_EQ(diagnostic.first_site_id, kSimulatedBarrierSite.value());
        EXPECT_EQ(diagnostic.second_site_id, kSimulatedBarrierSite.value());
    }

    void expect_contains(const std::string& text, const char* needle)
    {
        EXPECT_NE(text.find(needle), std::string::npos) << "missing stderr text: " << needle;
    }

    TEST(HipMoiHardSynchronizationNegative, AllParticipantsSimulationAdvancesEpoch)
    {
        hip_moi::test::device_context_storage storage;
        observed_metadata                     observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(simulated_barrier_safe_kernel, &storage, &observed));

        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    TEST(HipMoiHardSynchronizationNegative, MissingParticipantReportsDiagnostic)
    {
        hip_moi::test::device_context_storage storage;
        observed_metadata                     observed = {};
        ASSERT_NO_FATAL_FAILURE(
            run_case(simulated_barrier_missing_participant_kernel, &storage, &observed));

        ASSERT_EQ(observed.diagnostic_count, 1);
        expect_barrier_divergence_diagnostic(observed.diagnostics[0]);
    }

    TEST(HipMoiHardSynchronizationNegative, HostReportNamesBarrierDivergence)
    {
        hip_moi::host_context moi(small_host_context_options());

        hipLaunchKernelGGL(simulated_barrier_missing_participant_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           moi.device_ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());

        testing::internal::CaptureStderr();
        EXPECT_FALSE(moi.check());
        std::string stderr_output = testing::internal::GetCapturedStderr();

        expect_contains(stderr_output, "kind=barrier_divergence");
        expect_contains(stderr_output, "expected_threads=64");
        expect_contains(stderr_output, "observed_threads=63");
        expect_contains(stderr_output, "site_id=0x2401");
    }
} // namespace
