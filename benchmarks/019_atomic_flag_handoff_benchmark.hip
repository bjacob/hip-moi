// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// benchmarks/019_atomic_flag_handoff_benchmark.hip
//
// Stage 2 atomics benchmark. A global release/acquire flag orders a raw LDS
// payload handoff inside each workgroup. This measures the pass-through HIP
// atomic path against hip_moi::context atomic wrappers before hip-moi models
// atomics as diagnostic synchronization metadata.
#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace
{
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kThreadCount        = kThreadsPerSubgroup * kSubgroupCount;
    constexpr int kDefaultBlocks      = 4096;
    constexpr int kDefaultWarmupMs    = 100;
    constexpr int kDefaultMeasureMs   = 100;

    struct DisplayLatency
    {
        double      value;
        const char* unit;
    };

    DisplayLatency display_latency(double ms)
    {
        if(ms < 1.0)
        {
            return DisplayLatency{/*value=*/ms * 1000.0, /*unit=*/"µs"};
        }
        return DisplayLatency{/*value=*/ms, /*unit=*/"ms"};
    }

    [[noreturn]] void fail_hip(hipError_t status, const char* expr, const char* file, int line)
    {
        std::fprintf(
            stderr, "%s:%d: HIP error from %s: %s\n", file, line, expr, hipGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }

#define HIP_MOI_ATOMIC_BENCH_HIP_CHECK(expr)                                  \
    do                                                                        \
    {                                                                         \
        hipError_t hip_moi_atomic_bench_status = (expr);                      \
        if(hip_moi_atomic_bench_status != hipSuccess)                         \
        {                                                                     \
            fail_hip(hip_moi_atomic_bench_status, #expr, __FILE__, __LINE__); \
        }                                                                     \
    } while(false)

    int get_int_env(const char* name, int default_value)
    {
        const char* value = std::getenv(name);
        return value ? std::atoi(value) : default_value;
    }

    __device__ int payload_value(int block)
    {
        return 37 + ((block * 17) & 255);
    }

    __global__ void atomic_flag_handoff_pass_through(int* flags, int* out)
    {
        __shared__ int payload;

        int tid   = static_cast<int>(threadIdx.x);
        int block = static_cast<int>(blockIdx.x);
        if(tid == 0)
        {
            payload = 0;
            __hip_atomic_store(&flags[block], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();

        if(tid == 0)
        {
            payload = payload_value(block);
            __hip_atomic_store(&flags[block], 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }

        if(tid == kThreadsPerSubgroup)
        {
            while(__hip_atomic_load(&flags[block], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != 0)
            {
            }
            out[block] = payload;
        }
    }

    __global__ void
        atomic_flag_handoff_context(hip_moi::context::storage_ref storage, int* flags, int* out)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        hip_moi::context ctx(storage, cfg);
        __shared__ int   payload;

        int tid   = static_cast<int>(threadIdx.x);
        int block = static_cast<int>(blockIdx.x);
        if(tid == 0)
        {
            payload = 0;
            ctx.atomic_store(&flags[block],
                             1,
                             hip_moi::atomic_memory_order::relaxed,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }
        __syncthreads();

        if(tid == 0)
        {
            payload = payload_value(block);
            ctx.atomic_store(&flags[block],
                             0,
                             hip_moi::atomic_memory_order::release,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(tid == kThreadsPerSubgroup)
        {
            while(ctx.atomic_load(&flags[block],
                                  hip_moi::atomic_memory_order::acquire,
                                  hip_moi::atomic_memory_scope::agent,
                                  HIP_MOI_SITE_ID())
                  != 0)
            {
            }
            out[block] = payload;
        }
    }

    struct DeviceBuffers
    {
        int* flags = nullptr;
        int* out   = nullptr;

        explicit DeviceBuffers(int blocks)
        {
            HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipMalloc(&flags, blocks * sizeof(int)));
            HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipMalloc(&out, blocks * sizeof(int)));
        }

        DeviceBuffers(const DeviceBuffers&)            = delete;
        DeviceBuffers& operator=(const DeviceBuffers&) = delete;

        ~DeviceBuffers()
        {
            (void)hipFree(flags);
            (void)hipFree(out);
        }
    };

    template <typename Launch>
    double run_launches(Launch launch, int iterations)
    {
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipEventCreate(&start));
        HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipEventCreate(&stop));
        HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipEventRecord(start));
        for(int i = 0; i < iterations; ++i)
        {
            launch();
        }
        HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipEventRecord(stop));
        HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipEventDestroy(stop));
        HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipEventDestroy(start));
        return static_cast<double>(ms);
    }

    template <typename Launch>
    int choose_iterations(Launch launch, int warmup_ms)
    {
        int iterations = 1;
        for(;;)
        {
            double ms = run_launches(launch, iterations);
            if(ms >= static_cast<double>(warmup_ms))
            {
                return iterations;
            }
            if(ms <= 0.0)
            {
                iterations *= 2;
            }
            else
            {
                double scale = static_cast<double>(warmup_ms) / ms;
                int    next  = static_cast<int>(static_cast<double>(iterations) * scale * 1.1);
                iterations   = next > iterations ? next : iterations * 2;
            }
        }
    }

    template <typename Launch>
    double benchmark_mode(const char* name, Launch launch, int warmup_ms, int measure_ms)
    {
        int warmup_iterations = choose_iterations(launch, warmup_ms);
        (void)warmup_iterations;
        int            measure_iterations = choose_iterations(launch, measure_ms);
        double         elapsed_ms         = run_launches(launch, measure_iterations);
        double         latency_ms         = elapsed_ms / static_cast<double>(measure_iterations);
        DisplayLatency latency            = display_latency(latency_ms);
        std::printf("%-44s %7.3g %s\n", name, latency.value, latency.unit);
        return latency_ms;
    }

    void verify_outputs(int* device_output, int blocks)
    {
        std::vector<int> host_output(static_cast<std::size_t>(blocks));
        HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipMemcpy(host_output.data(),
                                                 device_output,
                                                 host_output.size() * sizeof(int),
                                                 hipMemcpyDeviceToHost));
        for(int block = 0; block < blocks; ++block)
        {
            int expected = 37 + ((block * 17) & 255);
            if(host_output[static_cast<std::size_t>(block)] != expected)
            {
                std::fprintf(stderr,
                             "block %d expected %d, got %d\n",
                             block,
                             expected,
                             host_output[static_cast<std::size_t>(block)]);
                std::exit(EXIT_FAILURE);
            }
        }
    }
} // namespace

int main()
{
    int blocks     = get_int_env("BLOCKS", kDefaultBlocks);
    int warmup_ms  = get_int_env("WARMUP_MS", kDefaultWarmupMs);
    int measure_ms = get_int_env("MIN_MS", kDefaultMeasureMs);

    DeviceBuffers buffers(blocks);

    hip_moi::host_context_options options;
    options.storage_bytes               = 128 * 1024;
    options.exact_shadow_entry_capacity = 1;
    options.sampled_watchpoint_capacity = 0;
    options.diagnostic_capacity         = 64;
    options.subgroup_capacity           = kSubgroupCount;
    hip_moi::host_context         moi(options);
    hip_moi::context::storage_ref storage = moi.launch_ref();

    auto pass_through_launch = [&]() {
        hipLaunchKernelGGL(atomic_flag_handoff_pass_through,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           buffers.flags,
                           buffers.out);
    };
    auto context_launch = [&]() {
        hipLaunchKernelGGL(atomic_flag_handoff_context,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage,
                           buffers.flags,
                           buffers.out);
    };

    pass_through_launch();
    HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    context_launch();
    HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    std::printf("hip-moi atomic flag handoff benchmark: blocks=%d warmup_ms=%d min_ms=%d\n",
                blocks,
                warmup_ms,
                measure_ms);
    benchmark_mode("atomic-flag-handoff_pass-through", pass_through_launch, warmup_ms, measure_ms);
    benchmark_mode("atomic-flag-handoff_context", context_launch, warmup_ms, measure_ms);

    if(!moi.check(nullptr))
    {
        std::fprintf(stderr, "unexpected hip-moi diagnostics in Stage 2 atomic benchmark\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
