// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// docs/tutorial/001_passing_syncthreads.hip
//
// A minimal passing hip-moi program: a store and a load separated by
// ctx.syncthreads().
#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace
{
    constexpr int kThreadCount = 8;

    void check_hip(hipError_t status, const char* expression)
    {
        if(status == hipSuccess)
        {
            return;
        }

        std::fprintf(stderr, "%s failed: %s\n", expression, hipGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }

    hip_moi::host_context_options small_context_options()
    {
        hip_moi::host_context_options options;
        options.access_record_capacity = 16;
        options.diagnostic_capacity    = 4;
        options.subgroup_capacity      = 1;
        return options;
    }

    __global__ void plain_producer_consumer_kernel(int* out)
    {
        __shared__ int value;

        if(threadIdx.x == 0)
        {
            value = 41;
        }

        __syncthreads();

        if(threadIdx.x == 1)
        {
            out[0] = value + 1;
        }
    }

    __global__ void producer_consumer_kernel(int* out, hip_moi::context_storage_ref storage)
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
            ctx.lds_store(&value, 41);
        }

        ctx.syncthreads();

        if(threadIdx.x == 1)
        {
            out[0] = ctx.lds_load(&value) + 1;
        }
    }
} // namespace

int main()
{
    hip_moi::host_context moi(small_context_options());

    int* device_output = nullptr;
    check_hip(hipMalloc(&device_output, 2 * sizeof(int)), "hipMalloc");

    int initial[2] = {-1, -1};
    check_hip(hipMemcpy(device_output, initial, sizeof(initial), hipMemcpyHostToDevice),
              "hipMemcpy(device_output)");

    hipLaunchKernelGGL(plain_producer_consumer_kernel,
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       device_output);
    check_hip(hipGetLastError(), "hipGetLastError(plain)");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(plain)");

    hipLaunchKernelGGL(producer_consumer_kernel,
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       device_output + 1,
                       moi.device_ref());
    check_hip(hipGetLastError(), "hipGetLastError(instrumented)");

    HIP_MOI_CHECK(moi);

    int output[2] = {-1, -1};
    check_hip(hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost),
              "hipMemcpy(output)");
    check_hip(hipFree(device_output), "hipFree");

    if(output[0] != 42 || output[1] != 42)
    {
        std::fprintf(stderr,
                     "expected outputs 42 and 42, got %d and %d\n",
                     output[0],
                     output[1]);
        return EXIT_FAILURE;
    }

    std::printf("hip-moi tutorial 001: plain=%d instrumented=%d\n", output[0], output[1]);
    return EXIT_SUCCESS;
}
