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
    constexpr int kAccessCapacity     = 8;
    constexpr int kDiagnosticCapacity = 8;

    constexpr hip_moi::site_id kSimulatedBarrierSite(0x2401u);

    hip_moi::host_context_options small_host_context_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes           = 64 * 1024;
        options.access_record_capacity  = kAccessCapacity;
        options.diagnostic_capacity     = kDiagnosticCapacity;
        options.subgroup_capacity       = kSubgroupCount;
        return options;
    }

    template <typename Context>
    struct observed_metadata
    {
        using diagnostic_type = typename Context::diagnostic;

        int             access_count;
        int             diagnostic_count;
        diagnostic_type diagnostics[kDiagnosticCapacity];
    };

    using observed_thread_level_metadata   = observed_metadata<hip_moi::thread_level_context>;
    using observed_subgroup_level_metadata = observed_metadata<hip_moi::subgroup_level_context>;

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

    __global__ void thread_level_simulated_barrier_safe_kernel(
        hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::thread_level_context ctx = make_thread_level_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&value, 17);
        }

        ctx.simulate_syncthreads(/*participates=*/true, kSimulatedBarrierSite);

        if(ctx.thread_id() == 1)
        {
            (void)ctx.lds_load(&value);
        }
    }

    __global__ void thread_level_simulated_barrier_missing_participant_kernel(
        hip_moi::thread_level_context::storage_ref storage)
    {
        hip_moi::thread_level_context ctx = make_thread_level_context(storage);
        ctx.init_workgroup();

        ctx.simulate_syncthreads(/*participates=*/ctx.thread_id() != 0, kSimulatedBarrierSite);
    }

    __global__ void subgroup_level_simulated_barrier_safe_kernel(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::subgroup_level_context ctx = make_subgroup_level_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store(&value, 23);
        }

        ctx.simulate_syncthreads(/*participates=*/true, kSimulatedBarrierSite);

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            (void)ctx.lds_load(&value);
        }
        ctx.finish();
    }

    __global__ void subgroup_level_simulated_barrier_missing_participant_kernel(
        hip_moi::subgroup_level_context::storage_ref storage)
    {
        hip_moi::subgroup_level_context ctx = make_subgroup_level_context(storage);
        ctx.init_workgroup();

        ctx.simulate_syncthreads(/*participates=*/ctx.thread_id() != 0, kSimulatedBarrierSite);
        ctx.finish();
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

    template <typename Kernel, typename Storage, typename Observed>
    void run_case(Kernel kernel, Storage* storage, Observed* observed)
    {
        ASSERT_NO_FATAL_FAILURE(storage->allocate(
            /*access_capacity=*/kAccessCapacity,
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount));

        hipLaunchKernelGGL(kernel, dim3(1), dim3(kThreadCount), 0, 0, storage->ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(*storage, observed));
    }

    template <typename Diagnostic>
    void expect_barrier_divergence_diagnostic(const Diagnostic& diagnostic)
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

    TEST(HipMoiHardSynchronizationNegative, ThreadLevelAllParticipantsSimulationAdvancesEpoch)
    {
        hip_moi::test::thread_level_device_context_storage storage;
        observed_thread_level_metadata                     observed = {};
        ASSERT_NO_FATAL_FAILURE(
            run_case(thread_level_simulated_barrier_safe_kernel, &storage, &observed));

        EXPECT_EQ(observed.access_count, 2);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    TEST(HipMoiHardSynchronizationNegative, ThreadLevelMissingParticipantReportsDiagnostic)
    {
        hip_moi::test::thread_level_device_context_storage storage;
        observed_thread_level_metadata                     observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(
            thread_level_simulated_barrier_missing_participant_kernel, &storage, &observed));

        EXPECT_EQ(observed.access_count, 0);
        ASSERT_EQ(observed.diagnostic_count, 1);
        expect_barrier_divergence_diagnostic(observed.diagnostics[0]);
    }

    TEST(HipMoiHardSynchronizationNegative, ThreadLevelHostReportNamesBarrierDivergence)
    {
        hip_moi::host_context moi(small_host_context_options());

        hipLaunchKernelGGL(thread_level_simulated_barrier_missing_participant_kernel,
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

    TEST(HipMoiHardSynchronizationNegative, SubgroupLevelAllParticipantsSimulationAdvancesEpoch)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        observed_subgroup_level_metadata                     observed = {};
        ASSERT_NO_FATAL_FAILURE(
            run_case(subgroup_level_simulated_barrier_safe_kernel, &storage, &observed));

        EXPECT_EQ(observed.access_count, 2);
        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    TEST(HipMoiHardSynchronizationNegative, SubgroupLevelMissingParticipantReportsDiagnostic)
    {
        hip_moi::test::subgroup_level_device_context_storage storage;
        observed_subgroup_level_metadata                     observed = {};
        ASSERT_NO_FATAL_FAILURE(run_case(
            subgroup_level_simulated_barrier_missing_participant_kernel, &storage, &observed));

        EXPECT_EQ(observed.access_count, 0);
        ASSERT_EQ(observed.diagnostic_count, 1);
        expect_barrier_divergence_diagnostic(observed.diagnostics[0]);
    }
} // namespace
