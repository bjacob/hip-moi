// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// benchmarks/025_atomic_fence_happens_before_benchmark.hip
//
// Atomic fence handoff benchmark. A release fence before a relaxed global
// atomic publication, paired with an acquire fence after a relaxed observation,
// orders an instrumented LDS payload handoff.
#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace
{
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kThreadCount        = kThreadsPerSubgroup * kSubgroupCount;
    constexpr int kDefaultBlocks      = 256;
    constexpr int kDefaultWarmupMs    = 100;
    constexpr int kDefaultMeasureMs   = 100;
    constexpr int kFlagReady          = 0;
    constexpr int kFlagWaiting        = 1;

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

#define HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(expr)                                  \
    do                                                                              \
    {                                                                               \
        hipError_t hip_moi_atomic_fence_bench_status = (expr);                      \
        if(hip_moi_atomic_fence_bench_status != hipSuccess)                         \
        {                                                                           \
            fail_hip(hip_moi_atomic_fence_bench_status, #expr, __FILE__, __LINE__); \
        }                                                                           \
    } while(false)

    int get_int_env(const char* name, int default_value)
    {
        const char* value = std::getenv(name);
        return value ? std::atoi(value) : default_value;
    }

    __host__ __device__ int payload_value(int block)
    {
        return 83 + ((block * 17) & 255);
    }

    __global__ void atomic_fence_lds_handoff_pass_through(int* flags, int* out)
    {
        __shared__ int payload;

        int tid   = static_cast<int>(threadIdx.x);
        int block = static_cast<int>(blockIdx.x);
        if(tid == 0)
        {
            __hip_atomic_store(
                &flags[block], kFlagWaiting, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();

        if(tid == 0)
        {
            payload = payload_value(block);
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            __hip_atomic_store(
                &flags[block], kFlagReady, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }

        if(tid == kThreadsPerSubgroup)
        {
            while(__hip_atomic_load(&flags[block], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
                  != kFlagReady)
            {
            }
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            out[block] = payload;
        }
    }

    __global__ void atomic_fence_lds_handoff_context(hip_moi::context::storage_ref* storage_refs,
                                                     uint64_t generation,
                                                     int*     flags,
                                                     int*     out)
    {
        __shared__ int payload;

        int                           block   = static_cast<int>(blockIdx.x);
        hip_moi::context::storage_ref storage = storage_refs[block];
        storage.generation                    = generation;
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        hip_moi::context ctx(storage, cfg);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.atomic_store(&flags[block],
                             kFlagWaiting,
                             hip_moi::atomic_memory_order::relaxed,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }
        __syncthreads();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&payload,
                             payload_value(block),
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
            ctx.atomic_fence(hip_moi::atomic_memory_order::release,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
            ctx.atomic_store(&flags[block],
                             kFlagReady,
                             hip_moi::atomic_memory_order::relaxed,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            while(ctx.atomic_load(&flags[block],
                                  hip_moi::atomic_memory_order::relaxed,
                                  hip_moi::atomic_memory_scope::agent,
                                  HIP_MOI_SITE_ID())
                  != kFlagReady)
            {
            }
            ctx.atomic_fence(hip_moi::atomic_memory_order::acquire,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
            out[block] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
        }
    }

    struct DeviceBuffers
    {
        int* flags = nullptr;
        int* out   = nullptr;

        explicit DeviceBuffers(int blocks)
        {
            HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipMalloc(&flags, blocks * sizeof(int)));
            HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipMalloc(&out, blocks * sizeof(int)));
        }

        DeviceBuffers(const DeviceBuffers&)            = delete;
        DeviceBuffers& operator=(const DeviceBuffers&) = delete;

        ~DeviceBuffers()
        {
            (void)hipFree(flags);
            (void)hipFree(out);
        }
    };

    struct ContextStorageSet
    {
        std::vector<std::unique_ptr<hip_moi::host_context>> contexts;
        hip_moi::context::storage_ref*                      device_refs = nullptr;

        explicit ContextStorageSet(int blocks)
        {
            hip_moi::host_context_options options;
            options.storage_bytes               = 64 * 1024;
            options.exact_shadow_entry_capacity = 16;
            options.sampled_watchpoint_capacity = 0;
            options.diagnostic_capacity         = 8;
            options.subgroup_capacity           = kSubgroupCount;

            contexts.reserve(static_cast<std::size_t>(blocks));
            std::vector<hip_moi::context::storage_ref> host_refs;
            host_refs.reserve(static_cast<std::size_t>(blocks));
            for(int block = 0; block < blocks; ++block)
            {
                contexts.push_back(std::make_unique<hip_moi::host_context>(options));
                host_refs.push_back(contexts.back()->launch_ref());
            }

            HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(
                hipMalloc(&device_refs, host_refs.size() * sizeof(host_refs[0])));
            HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipMemcpy(device_refs,
                                                           host_refs.data(),
                                                           host_refs.size()
                                                               * sizeof(host_refs[0]),
                                                           hipMemcpyHostToDevice));
        }

        ContextStorageSet(const ContextStorageSet&)            = delete;
        ContextStorageSet& operator=(const ContextStorageSet&) = delete;

        ~ContextStorageSet()
        {
            (void)hipFree(device_refs);
        }

        bool check_all()
        {
            bool ok = true;
            for(auto& context : contexts)
            {
                ok = context->check(nullptr) && ok;
            }
            return ok;
        }
    };

    template <typename Launch>
    double run_launches(Launch launch, int iterations)
    {
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipEventCreate(&start));
        HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipEventCreate(&stop));
        HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipEventRecord(start, 0));
        for(int i = 0; i < iterations; ++i)
        {
            launch();
        }
        HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipEventRecord(stop, 0));
        HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipEventDestroy(stop));
        HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipEventDestroy(start));
        return static_cast<double>(elapsed_ms);
    }

    template <typename Launch>
    void warmup_for(Launch launch, int warmup_ms)
    {
        int iterations = 1;
        while(true)
        {
            double elapsed = run_launches(launch, iterations);
            if(elapsed >= static_cast<double>(warmup_ms))
            {
                return;
            }
            iterations *= 2;
        }
    }

    template <typename Launch>
    double benchmark_for(Launch launch, int measure_ms)
    {
        int iterations = 1;
        while(true)
        {
            double elapsed = run_launches(launch, iterations);
            if(elapsed >= static_cast<double>(measure_ms))
            {
                return elapsed / static_cast<double>(iterations);
            }
            iterations *= 2;
        }
    }

    template <typename Launch>
    void benchmark_mode(const char* name, Launch launch, int warmup_ms, int measure_ms)
    {
        warmup_for(launch, warmup_ms);
        double         latency_ms = benchmark_for(launch, measure_ms);
        DisplayLatency display    = display_latency(latency_ms);
        std::printf("%-56s %.3g %s\n", name, display.value, display.unit);
    }

    void verify_outputs(const int* device_output, int blocks)
    {
        std::vector<int> host_output(static_cast<std::size_t>(blocks), 0);
        HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipMemcpy(host_output.data(),
                                                       device_output,
                                                       host_output.size()
                                                           * sizeof(host_output[0]),
                                                       hipMemcpyDeviceToHost));
        for(int block = 0; block < blocks; ++block)
        {
            int expected = payload_value(block);
            if(host_output[static_cast<std::size_t>(block)] != expected)
            {
                std::fprintf(stderr,
                             "output mismatch at block %d: got %d expected %d\n",
                             block,
                             host_output[static_cast<std::size_t>(block)],
                             expected);
                std::exit(EXIT_FAILURE);
            }
        }
    }
} // namespace

int main()
{
    int blocks     = get_int_env("HIP_MOI_ATOMIC_FENCE_BENCH_BLOCKS", kDefaultBlocks);
    int warmup_ms  = get_int_env("HIP_MOI_ATOMIC_FENCE_BENCH_WARMUP_MS", kDefaultWarmupMs);
    int measure_ms = get_int_env("HIP_MOI_ATOMIC_FENCE_BENCH_MEASURE_MS", kDefaultMeasureMs);

    DeviceBuffers    buffers(blocks);
    ContextStorageSet context_storage(blocks);
    uint64_t          generation = 1;

    auto pass_through_launch = [&]() {
        hipLaunchKernelGGL(atomic_fence_lds_handoff_pass_through,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           buffers.flags,
                           buffers.out);
    };
    auto context_launch = [&]() {
        ++generation;
        hipLaunchKernelGGL(atomic_fence_lds_handoff_context,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           context_storage.device_refs,
                           generation,
                           buffers.flags,
                           buffers.out);
    };

    pass_through_launch();
    HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    context_launch();
    HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_FENCE_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    std::printf("hip-moi atomic fence benchmark: blocks=%d warmup_ms=%d min_ms=%d\n",
                blocks,
                warmup_ms,
                measure_ms);
    benchmark_mode("atomic-fence-handoff_pass-through", pass_through_launch, warmup_ms, measure_ms);
    benchmark_mode("atomic-fence-handoff_context", context_launch, warmup_ms, measure_ms);

    if(!context_storage.check_all())
    {
        std::fprintf(stderr, "unexpected hip-moi diagnostic\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
