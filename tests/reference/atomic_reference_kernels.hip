// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/reference/atomic_reference_kernels.hip
//
// Uninstrumented HIP atomics reference kernels. The synchronization shapes are
// derived from rocjitsu-test-corpus:
//   * llama.cpp count-equal.cu for a tiny global atomicAdd reduction;
//   * hip-stream-k device_locks.hpp for release/acquire flag handoff.
//
#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <array>

#define HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(expr)                         \
    do                                                                    \
    {                                                                     \
        hipError_t hip_moi_atomic_reference_status = (expr);              \
        ASSERT_EQ(hip_moi_atomic_reference_status, hipSuccess)            \
            << #expr << ": "                                             \
            << hipGetErrorString(hip_moi_atomic_reference_status);        \
    } while(false)

namespace
{
    constexpr int kOutputCount       = 64;
    constexpr int kSentinel          = -1234567;
    constexpr int kCountEqualThreads = 32;
    constexpr int kCountEqualBlocks  = 4;
    constexpr int kHandoffFlag0      = 16;
    constexpr int kHandoffFlag1      = 17;

    __device__ void release_store_global_flag(int* flag, int value)
    {
        __hip_atomic_store(flag, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }

    __device__ void relaxed_store_global_flag(int* flag, int value)
    {
        __hip_atomic_store(flag, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    __device__ int acquire_load_global_flag(int* flag)
    {
        return __hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    }

    __device__ int relaxed_load_global_flag(int* flag)
    {
        return __hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    __global__ void global_atomic_add_counter(int* out)
    {
        __shared__ int partial[kCountEqualThreads];

        int tid = static_cast<int>(threadIdx.x);
        int gid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;

        int x        = (gid * 7 + 3) & 15;
        int y        = (gid % 3 == 0) ? x : x + 1;
        partial[tid] = (x == y) ? 1 : 0;
        __syncthreads();

        for(int stride = kCountEqualThreads / 2; stride > 0; stride >>= 1)
        {
            if(tid < stride)
            {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }

        if(tid == 0)
        {
            atomicAdd(&out[0], partial[0]);
        }
    }

    __global__ void release_acquire_flag_handoff(int* out)
    {
        __shared__ int payload;

        int tid = static_cast<int>(threadIdx.x);
        if(tid == 0)
        {
            payload = 0;
            relaxed_store_global_flag(&out[kHandoffFlag0], 1);
        }
        __syncthreads();

        if(tid == 0)
        {
            payload = 37;
            release_store_global_flag(&out[kHandoffFlag0], 0);
        }

        if(tid == 32)
        {
            while(acquire_load_global_flag(&out[kHandoffFlag0]) != 0)
            {
            }
            out[0] = payload;
        }
    }

    __global__ void two_helper_flag_handoff(int* out)
    {
        __shared__ int payload[2];

        int tid = static_cast<int>(threadIdx.x);
        if(tid == 0)
        {
            payload[0] = 0;
            payload[1] = 0;
            relaxed_store_global_flag(&out[kHandoffFlag0], 1);
            relaxed_store_global_flag(&out[kHandoffFlag1], 1);
        }
        __syncthreads();

        if(tid == 0)
        {
            payload[0] = 11;
            release_store_global_flag(&out[kHandoffFlag0], 0);
        }
        if(tid == 32)
        {
            payload[1] = 31;
            release_store_global_flag(&out[kHandoffFlag1], 0);
        }

        if(tid == 64)
        {
            while(acquire_load_global_flag(&out[kHandoffFlag0]) != 0)
            {
            }
            while(acquire_load_global_flag(&out[kHandoffFlag1]) != 0)
            {
            }
            out[0] = payload[0] + payload[1];
        }
    }

    __global__ void plain_flag_handoff_compile_only(int* out)
    {
        __shared__ int payload;

        int tid = static_cast<int>(threadIdx.x);
        if(tid == 0)
        {
            payload              = 0;
            out[kHandoffFlag0]   = 1;
        }
        __syncthreads();

        volatile int* flag = &out[kHandoffFlag0];
        if(tid == 0)
        {
            payload = 37;
            *flag   = 0;
        }
        if(tid == 32)
        {
            while(*flag != 0)
            {
            }
            out[0] = payload;
        }
    }

    __global__ void relaxed_flag_handoff_compile_only(int* out)
    {
        __shared__ int payload;

        int tid = static_cast<int>(threadIdx.x);
        if(tid == 0)
        {
            payload = 0;
            relaxed_store_global_flag(&out[kHandoffFlag0], 1);
        }
        __syncthreads();

        if(tid == 0)
        {
            payload = 37;
            relaxed_store_global_flag(&out[kHandoffFlag0], 0);
        }
        if(tid == 32)
        {
            while(relaxed_load_global_flag(&out[kHandoffFlag0]) != 0)
            {
            }
            out[0] = payload;
        }
    }

    void reset_output(int* device_output)
    {
        std::array<int, kOutputCount> initial{};
        initial.fill(kSentinel);
        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(
            hipMemcpy(device_output, initial.data(), sizeof(initial), hipMemcpyHostToDevice));
    }

    void expect_output(int* device_output, const std::array<int, kOutputCount>& expected)
    {
        std::array<int, kOutputCount> actual{};
        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(
            hipMemcpy(actual.data(), device_output, sizeof(actual), hipMemcpyDeviceToHost));
        EXPECT_EQ(actual, expected);
    }

    void run_and_check_global_atomic_add_counter()
    {
        int* device_output = nullptr;
        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipMalloc(&device_output, kOutputCount * sizeof(int)));
        reset_output(device_output);
        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipMemset(device_output, 0, sizeof(int)));

        hipLaunchKernelGGL(global_atomic_add_counter,
                           dim3(kCountEqualBlocks),
                           dim3(kCountEqualThreads),
                           0,
                           0,
                           device_output);
        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipGetLastError());
        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipDeviceSynchronize());

        std::array<int, kOutputCount> expected{};
        expected.fill(kSentinel);
        expected[0] = 43;
        expect_output(device_output, expected);

        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipFree(device_output));
    }

    void run_and_check_release_acquire_flag_handoff()
    {
        int* device_output = nullptr;
        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipMalloc(&device_output, kOutputCount * sizeof(int)));
        reset_output(device_output);

        hipLaunchKernelGGL(release_acquire_flag_handoff, dim3(1), dim3(64), 0, 0, device_output);
        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipGetLastError());
        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipDeviceSynchronize());

        std::array<int, kOutputCount> expected{};
        expected.fill(kSentinel);
        expected[0]             = 37;
        expected[kHandoffFlag0] = 0;
        expect_output(device_output, expected);

        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipFree(device_output));
    }

    void run_and_check_two_helper_flag_handoff()
    {
        int* device_output = nullptr;
        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipMalloc(&device_output, kOutputCount * sizeof(int)));
        reset_output(device_output);

        hipLaunchKernelGGL(two_helper_flag_handoff, dim3(1), dim3(96), 0, 0, device_output);
        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipGetLastError());
        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipDeviceSynchronize());

        std::array<int, kOutputCount> expected{};
        expected.fill(kSentinel);
        expected[0]             = 42;
        expected[kHandoffFlag0] = 0;
        expected[kHandoffFlag1] = 0;
        expect_output(device_output, expected);

        HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipFree(device_output));
    }

    void instantiate_compile_only_kernels(int* device_output)
    {
        if(false)
        {
            hipLaunchKernelGGL(
                plain_flag_handoff_compile_only, dim3(1), dim3(64), 0, 0, device_output);
            hipLaunchKernelGGL(
                relaxed_flag_handoff_compile_only, dim3(1), dim3(64), 0, 0, device_output);
        }
    }
} // namespace

TEST(HipMoiAtomicReferenceKernels, GlobalAtomicAddCounterMatchesExpectedOutput)
{
    run_and_check_global_atomic_add_counter();
}

TEST(HipMoiAtomicReferenceKernels, ReleaseAcquireFlagHandoffMatchesExpectedOutput)
{
    run_and_check_release_acquire_flag_handoff();
}

TEST(HipMoiAtomicReferenceKernels, TwoHelperFlagHandoffMatchesExpectedOutput)
{
    run_and_check_two_helper_flag_handoff();
}

TEST(HipMoiAtomicReferenceKernels, BrokenFlagHandoffsCompileOnly)
{
    int* device_output = nullptr;
    HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipMalloc(&device_output, kOutputCount * sizeof(int)));
    instantiate_compile_only_kernels(device_output);
    HIP_MOI_ATOMIC_REFERENCE_HIP_ASSERT(hipFree(device_output));
}
