// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/019_atomic_api_test.hip
//
// Stage 2 atomics tests: public hip_moi::context atomic wrappers lower to HIP
// atomics and preserve ordinary kernel results. These tests intentionally do
// not claim release/acquire diagnostic semantics yet.
//
#include "test_support.hpp"

#include <array>

namespace
{
    constexpr int kOutputCount        = 64;
    constexpr int kSentinel           = -1234567;
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kCountBlocks        = 4;
    constexpr int kCountThreads       = 32;
    constexpr int kHandoffFlag        = 16;

    __device__ hip_moi::context make_atomic_test_context(hip_moi::context::storage_ref storage,
                                                         int                           thread_count)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/thread_count,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/(thread_count + kThreadsPerSubgroup - 1) / kThreadsPerSubgroup,
        };
        return hip_moi::context(storage, cfg);
    }

    __global__ void atomic_fetch_add_counter_kernel(hip_moi::context::storage_ref storage, int* out)
    {
        hip_moi::context ctx = make_atomic_test_context(storage, static_cast<int>(blockDim.x));

        int tid = static_cast<int>(threadIdx.x);
        int gid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
        if(gid % 3 == 0)
        {
            (void)ctx.atomic_fetch_add(&out[0],
                                       1,
                                       hip_moi::atomic_memory_order::relaxed,
                                       hip_moi::atomic_memory_scope::agent,
                                       HIP_MOI_SITE_ID());
        }
    }

    __global__ void release_acquire_raw_lds_handoff_kernel(hip_moi::context::storage_ref storage,
                                                           int*                          out)
    {
        hip_moi::context ctx = make_atomic_test_context(storage, kThreadCount);
        __shared__ int   payload;

        int tid = static_cast<int>(threadIdx.x);
        if(tid == 0)
        {
            payload = 0;
            ctx.atomic_store(&out[kHandoffFlag],
                             1,
                             hip_moi::atomic_memory_order::relaxed,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }
        __syncthreads();

        if(tid == 0)
        {
            payload = 37;
            ctx.atomic_store(&out[kHandoffFlag],
                             0,
                             hip_moi::atomic_memory_order::release,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(tid == kThreadsPerSubgroup)
        {
            while(ctx.atomic_load(&out[kHandoffFlag],
                                  hip_moi::atomic_memory_order::acquire,
                                  hip_moi::atomic_memory_scope::agent,
                                  HIP_MOI_SITE_ID())
                  != 0)
            {
            }
            out[0] = payload;
        }
    }

    __global__ void atomic_fetch_or_bitmask_kernel(hip_moi::context::storage_ref storage, int* out)
    {
        hip_moi::context ctx = make_atomic_test_context(storage, kThreadCount);
        int              tid = static_cast<int>(threadIdx.x);

        if(tid == 0)
        {
            ctx.atomic_store(&out[0],
                             0,
                             hip_moi::atomic_memory_order::relaxed,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }
        __syncthreads();

        if(tid == 0)
        {
            out[1] = ctx.atomic_fetch_or(&out[0],
                                         1,
                                         hip_moi::atomic_memory_order::acq_rel,
                                         hip_moi::atomic_memory_scope::agent,
                                         HIP_MOI_SITE_ID());
        }
        __syncthreads();

        if(tid == kThreadsPerSubgroup)
        {
            out[2] = ctx.atomic_fetch_or(&out[0],
                                         2,
                                         hip_moi::atomic_memory_order::acq_rel,
                                         hip_moi::atomic_memory_scope::agent,
                                         HIP_MOI_SITE_ID());
        }
    }

    void reset_output(int* device_output)
    {
        std::array<int, kOutputCount> initial{};
        initial.fill(kSentinel);
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_output, initial.data(), sizeof(initial), hipMemcpyHostToDevice));
    }

    void expect_output(int* device_output, const std::array<int, kOutputCount>& expected)
    {
        std::array<int, kOutputCount> actual{};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(actual.data(), device_output, sizeof(actual), hipMemcpyDeviceToHost));
        EXPECT_EQ(actual, expected);
    }
} // namespace

TEST(HipMoiAtomicApi, AtomicFetchAddCounterMatchesExpectedOutput)
{
    hip_moi::host_context moi;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kOutputCount * sizeof(int)));
    reset_output(device_output);
    HIP_MOI_TEST_HIP_ASSERT(hipMemset(device_output, 0, sizeof(int)));

    hipLaunchKernelGGL(atomic_fetch_add_counter_kernel,
                       dim3(kCountBlocks),
                       dim3(kCountThreads),
                       0,
                       0,
                       moi.launch_ref(),
                       device_output);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

    std::array<int, kOutputCount> expected{};
    expected.fill(kSentinel);
    expected[0] = 43;
    expect_output(device_output, expected);
    EXPECT_TRUE(moi.check(nullptr));
    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
}

TEST(HipMoiAtomicApi, ReleaseAcquireGlobalFlagOrdersRawLdsPayload)
{
    hip_moi::host_context moi;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kOutputCount * sizeof(int)));
    reset_output(device_output);

    hipLaunchKernelGGL(release_acquire_raw_lds_handoff_kernel,
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_output);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

    std::array<int, kOutputCount> expected{};
    expected.fill(kSentinel);
    expected[0]            = 37;
    expected[kHandoffFlag] = 0;
    expect_output(device_output, expected);
    EXPECT_TRUE(moi.check(nullptr));
    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
}

TEST(HipMoiAtomicApi, AtomicFetchOrBitmaskMatchesExpectedOutput)
{
    hip_moi::host_context moi;
    int*                  device_output = nullptr;
    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, kOutputCount * sizeof(int)));
    reset_output(device_output);

    hipLaunchKernelGGL(atomic_fetch_or_bitmask_kernel,
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       moi.launch_ref(),
                       device_output);
    HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
    HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

    std::array<int, kOutputCount> expected{};
    expected.fill(kSentinel);
    expected[0] = 3;
    expected[1] = 0;
    expected[2] = 1;
    expect_output(device_output, expected);
    EXPECT_TRUE(moi.check(nullptr));
    HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
}
