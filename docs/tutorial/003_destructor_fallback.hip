// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// docs/tutorial/003_destructor_fallback.hip
//
// A failing hip-moi program that uses scope-based checking. The host_context
// destructor reports and aborts on the unconsumed diagnostic.
#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;

    void check_hip(hipError_t status, const char* expression)
    {
        if(status == hipSuccess)
        {
            return;
        }

        std::fprintf(stderr, "%s failed: %s\n", expression, hipGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }

    // Same small metadata allocation as the explicit-check failing example:
    // one workgroup, two subgroups, and enough storage for the diagnostic path.
    hip_moi::host_context_options small_context_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes     = 64 * 1024;
        options.subgroup_capacity = kSubgroupCount;
        return options;
    }

    // Plain HIP version of the race that the instrumented kernel below wraps.
    [[maybe_unused]] __global__ void plain_cross_subgroup_race_kernel(int* out)
    {
        __shared__ int value;

        if(threadIdx.x == 0)
        {
            value = 123;
        }
        if(threadIdx.x == kThreadsPerSubgroup)
        {
            out[0] = value;
        }
    }

    // Instrumented race kernel. This file demonstrates host_context destructor
    // handling, so main intentionally does not call HIP_MOI_CHECK.
    __global__ void cross_subgroup_race_kernel(int* out, hip_moi::context::storage_ref storage)
    {
        __shared__ int value;

        // The two active threads live in different subgroups, which is the
        // granularity hip-moi currently diagnoses.
        hip_moi::context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        hip_moi::context ctx(storage, cfg);

        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            // The shared scalar value starts at LDS byte offset 0.
            ctx.lds_store_at(&value, 123, /*lds_byte_offset=*/0);
        }
        if(threadIdx.x == kThreadsPerSubgroup)
        {
            out[0] = ctx.lds_load_at(&value, /*lds_byte_offset=*/0);
        }
        // Close the final epoch so the unconsumed diagnostic is visible to the
        // host_context destructor.
        ctx.finish();
    }
} // namespace

int main()
{
    int* device_output = nullptr;
    check_hip(hipMalloc(&device_output, sizeof(int)), "hipMalloc");

    int initial = -1;
    check_hip(hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice),
              "hipMemcpy(device_output)");

    {
        // When this object leaves scope, its destructor checks for unconsumed
        // diagnostics, prints them, and aborts by default.
        hip_moi::host_context moi(small_context_options());

        hipLaunchKernelGGL(cross_subgroup_race_kernel,
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           moi.device_ref());
        check_hip(hipGetLastError(), "hipGetLastError");
    }

    std::fprintf(stderr, "expected host_context destructor to abort\n");
    check_hip(hipFree(device_output), "hipFree");
    return EXIT_FAILURE;
}
