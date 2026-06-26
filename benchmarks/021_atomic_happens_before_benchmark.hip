// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// benchmarks/021_atomic_happens_before_benchmark.hip
//
// Atomic release/acquire handoff benchmark. A global flag orders an
// instrumented LDS payload handoff, so the context path exercises both
// atomic-object metadata and the acquired-epoch query in the exact-shadow
// conflict predicate.
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

    using SampledReportingPolicy = hip_moi::sampled_watchpoint_policy<
        /*SampleSkip=*/1,
        /*ProbeCount=*/0,
        /*DelayIters=*/0,
        /*ReportConflicts=*/true>;

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

#define HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(expr)                                  \
    do                                                                           \
    {                                                                            \
        hipError_t hip_moi_atomic_hb_bench_status = (expr);                      \
        if(hip_moi_atomic_hb_bench_status != hipSuccess)                         \
        {                                                                        \
            fail_hip(hip_moi_atomic_hb_bench_status, #expr, __FILE__, __LINE__); \
        }                                                                        \
    } while(false)

    int get_int_env(const char* name, int default_value)
    {
        const char* value = std::getenv(name);
        return value ? std::atoi(value) : default_value;
    }

    __host__ __device__ int payload_value(int block)
    {
        return 91 + ((block * 13) & 255);
    }

    __global__ void atomic_hb_lds_handoff_pass_through(int* flags, int* out)
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
            __hip_atomic_store(
                &flags[block], kFlagReady, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }

        if(tid == kThreadsPerSubgroup)
        {
            while(__hip_atomic_load(&flags[block], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)
                  != kFlagReady)
            {
            }
            out[block] = payload;
        }
    }

    __global__ void atomic_hb_lds_handoff_context(hip_moi::context::storage_ref* storage_refs,
                                                  uint64_t                       generation,
                                                  int*                           flags,
                                                  int*                           out)
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
            ctx.atomic_store(&flags[block],
                             kFlagReady,
                             hip_moi::atomic_memory_order::release,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            while(ctx.atomic_load(&flags[block],
                                  hip_moi::atomic_memory_order::acquire,
                                  hip_moi::atomic_memory_scope::agent,
                                  HIP_MOI_SITE_ID())
                  != kFlagReady)
            {
            }
            out[block] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
        }
    }

    __global__ void
        atomic_hb_lds_handoff_sampled_reporting(hip_moi::context::storage_ref* storage_refs,
                                                uint64_t                       generation,
                                                int*                           flags,
                                                int*                           out)
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

        if(ctx.subgroup_id() == 0)
        {
            ctx.lds_store_at<hip_moi::backend_kind::sampled_watchpoint, SampledReportingPolicy>(
                &payload, payload_value(block), /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            ctx.atomic_store(&flags[block],
                             kFlagReady,
                             hip_moi::atomic_memory_order::release,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(ctx.subgroup_id() == 1)
        {
            while(ctx.atomic_load(&flags[block],
                                  hip_moi::atomic_memory_order::acquire,
                                  hip_moi::atomic_memory_scope::agent,
                                  HIP_MOI_SITE_ID())
                  != kFlagReady)
            {
            }
            int loaded
                = ctx.lds_load_at<hip_moi::backend_kind::sampled_watchpoint,
                                  SampledReportingPolicy>(
                    &payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            if(ctx.thread_id() == kThreadsPerSubgroup)
            {
                out[block] = loaded;
            }
        }
    }

    struct DeviceBuffers
    {
        int* flags = nullptr;
        int* out   = nullptr;

        explicit DeviceBuffers(int blocks)
        {
            HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipMalloc(&flags, blocks * sizeof(int)));
            HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipMalloc(&out, blocks * sizeof(int)));
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

        explicit ContextStorageSet(int blocks,
                                   hip_moi::backend_kind backend,
                                   int exact_shadow_entry_capacity,
                                   int sampled_watchpoint_capacity)
        {
            hip_moi::host_context_options options;
            options.storage_bytes                  = 64 * 1024;
            options.backend                        = backend;
            options.exact_shadow_entry_capacity    = exact_shadow_entry_capacity;
            options.sampled_watchpoint_capacity    = sampled_watchpoint_capacity;
            options.sampled_watchpoint_sample_skip = 1;
            options.sampled_watchpoint_probe_count = 0;
            options.sampled_watchpoint_reports     = true;
            options.diagnostic_capacity            = 8;
            options.subgroup_capacity              = kSubgroupCount;

            contexts.reserve(static_cast<std::size_t>(blocks));
            std::vector<hip_moi::context::storage_ref> host_refs;
            host_refs.reserve(static_cast<std::size_t>(blocks));
            for(int block = 0; block < blocks; ++block)
            {
                contexts.push_back(std::make_unique<hip_moi::host_context>(options));
                host_refs.push_back(contexts.back()->launch_ref());
            }

            HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(
                hipMalloc(&device_refs, host_refs.size() * sizeof(host_refs[0])));
            HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipMemcpy(device_refs,
                                                        host_refs.data(),
                                                        host_refs.size() * sizeof(host_refs[0]),
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
        HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipEventCreate(&start));
        HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipEventCreate(&stop));
        HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipEventRecord(start));
        for(int i = 0; i < iterations; ++i)
        {
            launch();
        }
        HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipEventRecord(stop));
        HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipEventDestroy(stop));
        HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipEventDestroy(start));
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
        HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipMemcpy(host_output.data(),
                                                    device_output,
                                                    host_output.size() * sizeof(int),
                                                    hipMemcpyDeviceToHost));
        for(int block = 0; block < blocks; ++block)
        {
            int expected = payload_value(block);
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

    DeviceBuffers     buffers(blocks);
    ContextStorageSet context_storage(blocks,
                                      hip_moi::backend_kind::exact_shadow,
                                      /*exact_shadow_entry_capacity=*/16,
                                      /*sampled_watchpoint_capacity=*/0);
    ContextStorageSet sampled_reporting_storage(blocks,
                                                hip_moi::backend_kind::sampled_watchpoint,
                                                /*exact_shadow_entry_capacity=*/0,
                                                /*sampled_watchpoint_capacity=*/64);
    uint64_t          generation = 1;

    auto pass_through_launch = [&]() {
        hipLaunchKernelGGL(atomic_hb_lds_handoff_pass_through,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           buffers.flags,
                           buffers.out);
    };
    auto context_launch = [&]() {
        ++generation;
        hipLaunchKernelGGL(atomic_hb_lds_handoff_context,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           context_storage.device_refs,
                           generation,
                           buffers.flags,
                           buffers.out);
    };
    auto sampled_reporting_launch = [&]() {
        ++generation;
        hipLaunchKernelGGL(atomic_hb_lds_handoff_sampled_reporting,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           sampled_reporting_storage.device_refs,
                           generation,
                           buffers.flags,
                           buffers.out);
    };

    pass_through_launch();
    HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    context_launch();
    HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    sampled_reporting_launch();
    HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_HB_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    std::printf("hip-moi atomic happens-before benchmark: blocks=%d warmup_ms=%d min_ms=%d\n",
                blocks,
                warmup_ms,
                measure_ms);
    benchmark_mode(
        "atomic-hb-lds-handoff_pass-through", pass_through_launch, warmup_ms, measure_ms);
    benchmark_mode("atomic-hb-lds-handoff_context", context_launch, warmup_ms, measure_ms);
    benchmark_mode("atomic-hb-lds-handoff_context-sampled-reporting",
                   sampled_reporting_launch,
                   warmup_ms,
                   measure_ms);

    if(!context_storage.check_all())
    {
        std::fprintf(stderr, "unexpected hip-moi diagnostics in atomic handoff benchmark\n");
        return EXIT_FAILURE;
    }
    if(!sampled_reporting_storage.check_all())
    {
        std::fprintf(
            stderr, "unexpected sampled hip-moi diagnostics in atomic handoff benchmark\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
