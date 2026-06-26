// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/008_sampled_watchpoint_backend_test.hip
//
// Tests for the sampled-watchpoint backend.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int              kThreadCount        = 64;
    constexpr int              kThreadsPerSubgroup = 32;
    constexpr int              kSubgroupCount      = 2;
    constexpr int              kDiagnosticCapacity = 8;
    constexpr int              kWatchpointCapacity = 64;
    constexpr int              kPayloadValue       = 117;
    constexpr int              kInitialOutput      = -1;
    constexpr int              kFlagReady          = 0;
    constexpr int              kFlagWaiting        = 1;
    constexpr hip_moi::site_id kProducerSite(0x401);
    constexpr hip_moi::site_id kConsumerSite(0x502);

    using TestSampledPolicy = hip_moi::sampled_watchpoint_policy<
        /*SampleSkip=*/1,
        /*ProbeCount=*/0,
        /*DelayIters=*/0,
        /*ReportConflicts=*/true>;
    using FastPublishOnlyPolicy = hip_moi::sampled_watchpoint_policy<
        /*SampleSkip=*/1,
        /*ProbeCount=*/1,
        /*DelayIters=*/0,
        /*ReportConflicts=*/false>;

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

    __device__ hip_moi::sampled_watchpoint_context
               make_fast_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::sampled_watchpoint_context::config cfg{
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
        };
        return hip_moi::make_sampled_watchpoint_context(storage, cfg);
    }

    __global__ void sampled_safe_kernel(hip_moi::context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store_at<hip_moi::backend_kind::sampled_watchpoint, TestSampledPolicy>(
                &value, 77, /*lds_byte_offset=*/0, kProducerSite);
        }
        ctx.syncthreads();
        if(ctx.subgroup_id() == 1)
        {
            (void)ctx.lds_load_at<hip_moi::backend_kind::sampled_watchpoint, TestSampledPolicy>(
                &value, /*lds_byte_offset=*/0, kConsumerSite);
        }
    }

    __global__ void sampled_fast_context_smoke_kernel(hip_moi::context::storage_ref storage,
                                                      int*                          out)
    {
        __shared__ int values[kThreadCount];

        hip_moi::sampled_watchpoint_context ctx = make_fast_context(storage);
        ctx.init_workgroup();

        int index = static_cast<int>(threadIdx.x);
        ctx.lds_store_at<FastPublishOnlyPolicy>(&values[index],
                                                index,
                                                /*lds_byte_offset=*/index * sizeof(int),
                                                kProducerSite);
        ctx.syncthreads();

        int loaded = ctx.lds_load_at<FastPublishOnlyPolicy>(
            &values[index], /*lds_byte_offset=*/index * sizeof(int), kConsumerSite);
        if(index == 0)
        {
            int tail = ctx.lds_load_at<FastPublishOnlyPolicy>(&values[kThreadCount - 1],
                                                              /*lds_byte_offset=*/(kThreadCount - 1)
                                                                  * sizeof(int),
                                                              kConsumerSite);
            out[0]   = loaded + tail;
        }
    }

    __global__ void sampled_race_kernel(hip_moi::context::storage_ref storage)
    {
        __shared__ int value;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store_at<hip_moi::backend_kind::sampled_watchpoint, TestSampledPolicy>(
                &value, 77, /*lds_byte_offset=*/0, kProducerSite);
        }
        if(ctx.subgroup_id() == 1)
        {
            (void)ctx.lds_load_at<hip_moi::backend_kind::sampled_watchpoint, TestSampledPolicy>(
                &value, /*lds_byte_offset=*/0, kConsumerSite);
        }
    }

    template <hip_moi::atomic_memory_order StoreOrder>
    __global__ void sampled_atomic_flag_orders_lds_payload_kernel(
        hip_moi::context::storage_ref storage, int* flag, int* out)
    {
        __shared__ int payload;

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store_at<hip_moi::backend_kind::sampled_watchpoint, TestSampledPolicy>(
                &payload, kPayloadValue, /*lds_byte_offset=*/0, kProducerSite);
            ctx.atomic_store(flag,
                             kFlagReady,
                             StoreOrder,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }
        if(ctx.subgroup_id() == 1)
        {
            while(ctx.atomic_load(flag,
                                  hip_moi::atomic_memory_order::acquire,
                                  hip_moi::atomic_memory_scope::agent,
                                  HIP_MOI_SITE_ID())
                  != kFlagReady)
            {
            }
            int loaded
                = ctx.lds_load_at<hip_moi::backend_kind::sampled_watchpoint, TestSampledPolicy>(
                    &payload, /*lds_byte_offset=*/0, kConsumerSite);
            if(ctx.thread_id() == kThreadsPerSubgroup)
            {
                out[0] = loaded;
            }
        }
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

    hip_moi::context::storage_ref sampled_ref(hip_moi::test::device_context_storage& storage)
    {
        hip_moi::context::storage_ref ref = storage.ref();
        ref.backend = static_cast<uint32_t>(hip_moi::backend_kind::sampled_watchpoint);
        ref.sampled_watchpoint_sample_skip      = 1;
        ref.sampled_watchpoint_probe_count      = 0;
        ref.sampled_watchpoint_report_conflicts = 1;
        return ref;
    }

    hip_moi::host_context_options sampled_reporting_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes                  = 64 * 1024;
        options.backend                        = hip_moi::backend_kind::sampled_watchpoint;
        options.exact_shadow_entry_capacity    = 0;
        options.sampled_watchpoint_capacity    = kWatchpointCapacity;
        options.sampled_watchpoint_sample_skip = 1;
        options.sampled_watchpoint_probe_count = 0;
        options.sampled_watchpoint_reports     = true;
        options.diagnostic_capacity            = kDiagnosticCapacity;
        options.subgroup_capacity              = kSubgroupCount;
        return options;
    }

    void reset_flag_and_output(int* device_flag, int* device_output)
    {
        int flag   = kFlagWaiting;
        int output = kInitialOutput;
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(device_flag, &flag, sizeof(flag), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, &output, sizeof(output), hipMemcpyHostToDevice));
    }

    TEST(HipMoiSampledWatchpointBackend, SynchronizedSelectedAccessesDoNotReport)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_sampled_watchpoints(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*sampled_watchpoint_capacity=*/kWatchpointCapacity));

        hipLaunchKernelGGL(
            sampled_safe_kernel, dim3(1), dim3(kThreadCount), 0, 0, sampled_ref(storage));
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        EXPECT_EQ(observed.diagnostic_count, 0);
    }

    TEST(HipMoiSampledWatchpointBackend, RacySelectedAccessesDiagnose)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_sampled_watchpoints(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*sampled_watchpoint_capacity=*/kWatchpointCapacity));

        hipLaunchKernelGGL(
            sampled_race_kernel, dim3(1), dim3(kThreadCount), 0, 0, sampled_ref(storage));
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        observed_metadata observed = {};
        ASSERT_NO_FATAL_FAILURE(copy_observed_metadata(storage, &observed));

        ASSERT_GT(observed.diagnostic_count, 0);
        EXPECT_EQ(observed.diagnostics[0].kind,
                  static_cast<uint32_t>(hip_moi::diagnostic_kind::access_conflict));
    }

    TEST(HipMoiSampledWatchpointBackend, FastContextRunsPublishOnlyAccesses)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_sampled_watchpoints(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*sampled_watchpoint_capacity=*/kWatchpointCapacity));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMemset(device_output, 0, sizeof(int)));

        hipLaunchKernelGGL(sampled_fast_context_smoke_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           sampled_ref(storage),
                           device_output);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output = 0;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
        EXPECT_EQ(output, kThreadCount - 1);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }

    TEST(HipMoiSampledWatchpointBackend, AtomicReleaseAcquireOrdersSampledDiagnostics)
    {
        hip_moi::host_context moi(sampled_reporting_options());
        int*                  device_flag   = nullptr;
        int*                  device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flag, sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
        reset_flag_and_output(device_flag, device_output);

        hipLaunchKernelGGL(
            (sampled_atomic_flag_orders_lds_payload_kernel<hip_moi::atomic_memory_order::release>),
            dim3(1),
            dim3(kThreadCount),
            0,
            0,
            moi.launch_ref(),
            device_flag,
            device_output);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output = 0;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
        EXPECT_EQ(output, kPayloadValue);
        EXPECT_TRUE(moi.check(nullptr));

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_flag));
    }

    TEST(HipMoiSampledWatchpointBackend, RelaxedAtomicDoesNotOrderSampledDiagnostics)
    {
        hip_moi::host_context moi(sampled_reporting_options());
        int*                  device_flag   = nullptr;
        int*                  device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_flag, sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
        reset_flag_and_output(device_flag, device_output);

        hipLaunchKernelGGL(
            (sampled_atomic_flag_orders_lds_payload_kernel<hip_moi::atomic_memory_order::relaxed>),
            dim3(1),
            dim3(kThreadCount),
            0,
            0,
            moi.launch_ref(),
            device_flag,
            device_output);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output = 0;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
        EXPECT_EQ(output, kPayloadValue);
        EXPECT_FALSE(moi.check(nullptr));
        EXPECT_GT(moi.last_diagnostic_count(), 0);

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_flag));
    }
} // namespace
