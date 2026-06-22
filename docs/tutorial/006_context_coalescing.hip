// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// docs/tutorial/006_context_coalescing.hip
//
// A passing context hip-moi program with site ids and coalescing storage.
#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace
{
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kThreadCount        = kThreadsPerSubgroup * kSubgroupCount;

    void check_hip(hipError_t status, const char* expression)
    {
        if(status == hipSuccess)
        {
            return;
        }

        std::fprintf(stderr, "%s failed: %s\n", expression, hipGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }

    hip_moi::host_context_options coalescing_context_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes     = 64 * 1024;
        options.subgroup_capacity = kSubgroupCount;
        return options;
    }

    [[maybe_unused]] __global__ void plain_subgroup_tile_kernel(int* out)
    {
        __shared__ int values[kThreadCount];

        int thread = static_cast<int>(threadIdx.x);
        values[thread] = thread + 7;

        __syncthreads();

        out[thread] = values[thread];
    }

    __global__ void subgroup_tile_kernel(
        int* out, hip_moi::context::storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        hip_moi::context ctx(storage, cfg);

        ctx.init_workgroup();

        int thread = static_cast<int>(threadIdx.x);
        ctx.lds_store(&values[thread], thread + 7, HIP_MOI_SITE_ID());

        ctx.syncthreads();

        out[thread] = ctx.lds_load(&values[thread], HIP_MOI_SITE_ID());
        ctx.finish();
    }
} // namespace

int main()
{
    hip_moi::host_context moi(coalescing_context_options());

    int* device_output = nullptr;
    check_hip(hipMalloc(&device_output, kThreadCount * sizeof(int)), "hipMalloc");

    hipLaunchKernelGGL(subgroup_tile_kernel,
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       device_output,
                       moi.device_ref());
    check_hip(hipGetLastError(), "hipGetLastError");

    HIP_MOI_CHECK(moi);

    int output[kThreadCount] = {};
    check_hip(hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost),
              "hipMemcpy(output)");
    check_hip(hipFree(device_output), "hipFree");

    for(int i = 0; i < kThreadCount; ++i)
    {
        if(output[i] != i + 7)
        {
            std::fprintf(stderr, "output[%d]: expected %d, got %d\n", i, i + 7, output[i]);
            return EXIT_FAILURE;
        }
    }

    std::printf("hip-moi tutorial 006: context coalescing example passed\n");
    return EXIT_SUCCESS;
}
