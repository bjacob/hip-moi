// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// docs/tutorial/002_failing_cross_subgroup_race.hip
//
// A failing hip-moi program: two subgroups access the same LDS object in one
// epoch, and one access is a write.
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

    // Allocate enough metadata for the two subgroups in this one-workgroup
    // example. The storage byte budget is deliberately small but not part of
    // the behavior being demonstrated.
    hip_moi::host_context_options small_context_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes     = 64 * 1024;
        options.subgroup_capacity = kSubgroupCount;
        return options;
    }

    // Plain HIP version of the bug: subgroup 0 writes and subgroup 1 reads the
    // same LDS scalar without a barrier between them.
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

    // Instrumented version: the same LDS accesses are wrapped so hip-moi can
    // compare the subgroups, byte range, access kind, and synchronization epoch.
    __global__ void cross_subgroup_race_kernel(int* out, hip_moi::context::storage_ref storage)
    {
        __shared__ int value;

        // Make the subgroup partition explicit. Thread 0 is in subgroup 0 and
        // thread 32 is in subgroup 1 on this 64-thread workgroup.
        hip_moi::context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        hip_moi::context ctx(storage, cfg);

        ctx.init_workgroup();

        if(threadIdx.x == 0)
        {
            // value is the only LDS object in this tutorial kernel, so both
            // instrumented accesses use byte offset 0.
            ctx.lds_store_at(&value, 123, /*lds_byte_offset=*/0);
        }
        if(threadIdx.x == kThreadsPerSubgroup)
        {
            out[0] = ctx.lds_load_at(&value, /*lds_byte_offset=*/0);
        }
        // No ctx.syncthreads() occurred, so finish() closes the final epoch and
        // lets the host-side check observe the conflict.
        ctx.finish();
    }
} // namespace

int main()
{
    hip_moi::host_context moi(small_context_options());

    int* device_output = nullptr;
    check_hip(hipMalloc(&device_output, sizeof(int)), "hipMalloc");

    int initial = -1;
    check_hip(hipMemcpy(device_output, &initial, sizeof(initial), hipMemcpyHostToDevice),
              "hipMemcpy(device_output)");

    hipLaunchKernelGGL(
        cross_subgroup_race_kernel, dim3(1), dim3(kThreadCount), 0, 0, device_output, moi.device_ref());
    check_hip(hipGetLastError(), "hipGetLastError");

    // This call is expected to print the race diagnostic and abort. The CTest
    // wrapper treats that abort as the success condition for this tutorial.
    HIP_MOI_CHECK(moi);

    std::fprintf(stderr, "expected HIP_MOI_CHECK to abort\n");
    check_hip(hipFree(device_output), "hipFree");
    return EXIT_FAILURE;
}
