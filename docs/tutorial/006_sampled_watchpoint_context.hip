// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// docs/tutorial/006_sampled_watchpoint_context.hip
//
// A passing publish-only sampled-watchpoint fast-view program. This is the
// small tutorial analogue of the high-performance benchmark rows that use
// hip_moi::sampled_watchpoint_context directly.
#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace
{
    constexpr int              kThreadCount        = 64;
    constexpr int              kThreadsPerSubgroup = 32;
    constexpr int              kSubgroupCount      = 2;
    constexpr hip_moi::site_id kStoreSite(0x601);
    constexpr hip_moi::site_id kLoadSite(0x602);

    // Publish-only uses publish as a verb: selected lanes publish compact
    // watchpoint metadata, but this fast path does not report diagnostics to
    // the host. That is why ReportConflicts is false.
    using publish_only_policy = hip_moi::sampled_watchpoint_policy<
        /*SampleSkip=*/32,
        /*ProbeCount=*/1,
        /*DelayIters=*/32,
        /*ReportConflicts=*/false,
        /*StaticWatchpointCapacity=*/1>;

    void check_hip(hipError_t status, const char* expression)
    {
        if(status == hipSuccess)
        {
            return;
        }

        std::fprintf(stderr, "%s failed: %s\n", expression, hipGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }

    // Publish-only context options allocate the sampled-watchpoint storage used
    // by sampled_watchpoint_context. They intentionally disable host-visible
    // reports to match the low-overhead benchmark path.
    hip_moi::host_context_options publish_only_context_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes                       = 64 * 1024;
        options.subgroup_capacity                   = kSubgroupCount;
        options.backend                             = hip_moi::backend_kind::sampled_watchpoint;
        options.sampled_watchpoint_capacity         = 1;
        options.sampled_watchpoint_sample_skip      = 32;
        options.sampled_watchpoint_probe_count      = 1;
        options.sampled_watchpoint_delay_iters      = 32;
        options.sampled_watchpoint_reports          = false;
        return options;
    }

    // Convert the general storage_ref produced by host_context into the smaller
    // storage view required by sampled_watchpoint_context. This avoids carrying
    // the general diagnostic context through the hot kernel.
    __device__ hip_moi::sampled_watchpoint_context
        make_fast_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::sampled_watchpoint_context::config cfg{
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
        };
        // The fast context only needs the epoch word, watchpoint table, table
        // size, and launch generation. It has no diagnostic buffer.
        hip_moi::sampled_watchpoint_context::storage_ref fast_storage{
            /*workgroup_epoch=*/
            storage.subgroup_states ? &storage.subgroup_states[0].epoch : nullptr,
            /*sampled_watchpoints=*/storage.sampled_watchpoints,
            /*sampled_watchpoint_capacity=*/storage.sampled_watchpoint_capacity,
            /*generation=*/storage.generation,
        };
        return hip_moi::sampled_watchpoint_context(fast_storage, cfg);
    }

    // Plain HIP version: each thread writes one LDS element, synchronizes, then
    // reads its neighbor's element.
    __global__ void plain_neighbor_kernel(int* out)
    {
        __shared__ int values[kThreadCount];

        int index    = static_cast<int>(threadIdx.x);
        int neighbor = (index + 1) % kThreadCount;

        values[index] = index * 3;
        __syncthreads();
        out[index] = values[neighbor] + 7;
    }

    // Fast sampled-watchpoint version: LDS operations are instrumented with a
    // compile-time policy, but the kernel remains a passing computation test.
    __global__ void publish_only_neighbor_kernel(int*                          out,
                                                 hip_moi::context::storage_ref storage)
    {
        __shared__ int values[kThreadCount];

        hip_moi::sampled_watchpoint_context ctx = make_fast_context(storage);
        ctx.init_workgroup();

        int index    = static_cast<int>(threadIdx.x);
        int neighbor = (index + 1) % kThreadCount;

        // The site id distinguishes this static store site from the load site.
        // Nonzero site ids let the backend apply site-specific optimizations.
        ctx.lds_store_at<publish_only_policy>(&values[index],
                                              index * 3,
                                              /*lds_byte_offset=*/index * sizeof(int),
                                              kStoreSite);

        ctx.syncthreads();

        // This synchronized load should observe the neighbor's value and should
        // not produce any user-facing diagnostic in publish-only mode.
        out[index] = ctx.lds_load_at<publish_only_policy>(
                         &values[neighbor],
                         /*lds_byte_offset=*/neighbor * sizeof(int),
                         kLoadSite)
                     + 7;
    }

    bool check_output(const int* output, const char* label)
    {
        for(int index = 0; index < kThreadCount; ++index)
        {
            int neighbor = (index + 1) % kThreadCount;
            int expected = neighbor * 3 + 7;
            if(output[index] != expected)
            {
                std::fprintf(stderr,
                             "%s[%d] mismatch: expected %d, got %d\n",
                             label,
                             index,
                             expected,
                             output[index]);
                return false;
            }
        }
        return true;
    }
} // namespace

int main()
{
    int* device_plain_output        = nullptr;
    int* device_instrumented_output = nullptr;
    check_hip(hipMalloc(&device_plain_output, kThreadCount * sizeof(int)), "hipMalloc(plain)");
    check_hip(hipMalloc(&device_instrumented_output, kThreadCount * sizeof(int)),
              "hipMalloc(instrumented)");

    plain_neighbor_kernel<<<dim3(1), dim3(kThreadCount)>>>(device_plain_output);
    check_hip(hipGetLastError(), "hipGetLastError(plain)");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(plain)");

    // The host context still allocates and initializes device metadata, but
    // there is no HIP_MOI_CHECK because this fast view does not consume reports.
    hip_moi::host_context moi(publish_only_context_options());
    publish_only_neighbor_kernel<<<dim3(1), dim3(kThreadCount)>>>(
        device_instrumented_output, moi.device_ref());
    check_hip(hipGetLastError(), "hipGetLastError(publish_only)");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(publish_only)");

    int plain_output[kThreadCount]        = {};
    int instrumented_output[kThreadCount] = {};
    check_hip(hipMemcpy(plain_output,
                        device_plain_output,
                        sizeof(plain_output),
                        hipMemcpyDeviceToHost),
              "hipMemcpy(plain_output)");
    check_hip(hipMemcpy(instrumented_output,
                        device_instrumented_output,
                        sizeof(instrumented_output),
                        hipMemcpyDeviceToHost),
              "hipMemcpy(instrumented_output)");

    check_hip(hipFree(device_plain_output), "hipFree(plain)");
    check_hip(hipFree(device_instrumented_output), "hipFree(instrumented)");

    if(!check_output(plain_output, "plain") || !check_output(instrumented_output, "instrumented"))
    {
        return EXIT_FAILURE;
    }

    std::printf("hip-moi tutorial 006: sampled_watchpoint_context publish-only ok\n");
    return EXIT_SUCCESS;
}
