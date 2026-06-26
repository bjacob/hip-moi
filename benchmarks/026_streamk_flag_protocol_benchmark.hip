// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// benchmarks/026_streamk_flag_protocol_benchmark.hip
//
// Stream-K flag-protocol benchmark. A RocJITsu hip-stream-k-style owner
// subgroup waits on two helper release/acquire flags and folds their LDS
// partial payloads through the same fixup loop shape as the instrumented test.
#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace
{
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 3;
    constexpr int kThreadCount        = kThreadsPerSubgroup * kSubgroupCount;
    constexpr int kHelperCount        = 2;
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

#define HIP_MOI_STREAMK_BENCH_HIP_CHECK(expr)                                  \
    do                                                                         \
    {                                                                          \
        hipError_t hip_moi_streamk_bench_status = (expr);                      \
        if(hip_moi_streamk_bench_status != hipSuccess)                         \
        {                                                                      \
            fail_hip(hip_moi_streamk_bench_status, #expr, __FILE__, __LINE__); \
        }                                                                      \
    } while(false)

    int get_int_env(const char* name, int default_value)
    {
        const char* value = std::getenv(name);
        return value ? std::atoi(value) : default_value;
    }

    __host__ __device__ int owner_partial_value(int block)
    {
        return 19 + ((block * 3) & 255);
    }

    __host__ __device__ int helper_partial_value(int block, int helper)
    {
        return 31 + helper * 17 + ((block * (helper + 5)) & 255);
    }

    __host__ __device__ int expected_streamk_sum(int block)
    {
        int sum = owner_partial_value(block);
        for(int helper = 0; helper < kHelperCount; ++helper)
        {
            sum += helper_partial_value(block, helper);
        }
        return sum;
    }

    __global__ void streamk_flag_fixup_pass_through(int* flags, int* out)
    {
        __shared__ int partials[1 + kHelperCount];

        int tid   = static_cast<int>(threadIdx.x);
        int block = static_cast<int>(blockIdx.x);
        if(tid == 0)
        {
            for(int helper = 0; helper < kHelperCount; ++helper)
            {
                __hip_atomic_store(&flags[block * kHelperCount + helper],
                                   kFlagWaiting,
                                   __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
            }
            partials[0] = owner_partial_value(block);
        }
        __syncthreads();

        for(int helper = 0; helper < kHelperCount; ++helper)
        {
            int helper_tid = (helper + 1) * kThreadsPerSubgroup;
            if(tid == helper_tid)
            {
                partials[helper + 1] = helper_partial_value(block, helper);
                __hip_atomic_store(&flags[block * kHelperCount + helper],
                                   kFlagReady,
                                   __ATOMIC_RELEASE,
                                   __HIP_MEMORY_SCOPE_AGENT);
            }
        }

        if(tid == 0)
        {
            int sum = partials[0];
            for(int helper = 0; helper < kHelperCount; ++helper)
            {
                while(__hip_atomic_load(&flags[block * kHelperCount + helper],
                                        __ATOMIC_ACQUIRE,
                                        __HIP_MEMORY_SCOPE_AGENT)
                      != kFlagReady)
                {
                }
                sum += partials[helper + 1];
            }
            out[block] = sum;
        }
    }

    __global__ void streamk_flag_fixup_context(hip_moi::context::storage_ref* storage_refs,
                                               uint64_t                       generation,
                                               int*                           flags,
                                               int*                           out)
    {
        __shared__ int partials[1 + kHelperCount];

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
            for(int helper = 0; helper < kHelperCount; ++helper)
            {
                ctx.atomic_store(&flags[block * kHelperCount + helper],
                                 kFlagWaiting,
                                 hip_moi::atomic_memory_order::relaxed,
                                 hip_moi::atomic_memory_scope::agent,
                                 HIP_MOI_SITE_ID());
            }
            ctx.lds_store_at(&partials[0],
                             owner_partial_value(block),
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
        }
        __syncthreads();

        for(int helper = 0; helper < kHelperCount; ++helper)
        {
            int helper_tid = (helper + 1) * kThreadsPerSubgroup;
            if(ctx.thread_id() == static_cast<uint32_t>(helper_tid))
            {
                uint32_t partial_offset = static_cast<uint32_t>((helper + 1) * sizeof(int));
                ctx.lds_store_at(&partials[helper + 1],
                                 helper_partial_value(block, helper),
                                 partial_offset,
                                 HIP_MOI_SITE_ID());
                ctx.atomic_store(&flags[block * kHelperCount + helper],
                                 kFlagReady,
                                 hip_moi::atomic_memory_order::release,
                                 hip_moi::atomic_memory_scope::agent,
                                 HIP_MOI_SITE_ID());
            }
        }

        if(ctx.thread_id() == 0)
        {
            int sum = ctx.lds_load_at(&partials[0], /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            for(int helper = 0; helper < kHelperCount; ++helper)
            {
                while(ctx.atomic_load(&flags[block * kHelperCount + helper],
                                      hip_moi::atomic_memory_order::acquire,
                                      hip_moi::atomic_memory_scope::agent,
                                      HIP_MOI_SITE_ID())
                      != kFlagReady)
                {
                }
                uint32_t partial_offset = static_cast<uint32_t>((helper + 1) * sizeof(int));
                sum += ctx.lds_load_at(&partials[helper + 1],
                                       partial_offset,
                                       HIP_MOI_SITE_ID());
            }
            out[block] = sum;
        }
    }

    struct DeviceBuffers
    {
        int* flags = nullptr;
        int* out   = nullptr;

        explicit DeviceBuffers(int blocks)
        {
            HIP_MOI_STREAMK_BENCH_HIP_CHECK(
                hipMalloc(&flags, blocks * kHelperCount * sizeof(int)));
            HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipMalloc(&out, blocks * sizeof(int)));
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

            HIP_MOI_STREAMK_BENCH_HIP_CHECK(
                hipMalloc(&device_refs, host_refs.size() * sizeof(host_refs[0])));
            HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipMemcpy(device_refs,
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
        HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipEventCreate(&start));
        HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipEventCreate(&stop));
        HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipEventRecord(start, 0));
        for(int i = 0; i < iterations; ++i)
        {
            launch();
        }
        HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipEventRecord(stop, 0));
        HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipEventDestroy(stop));
        HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipEventDestroy(start));
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
        HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipMemcpy(host_output.data(),
                                                  device_output,
                                                  host_output.size() * sizeof(host_output[0]),
                                                  hipMemcpyDeviceToHost));
        for(int block = 0; block < blocks; ++block)
        {
            int expected = expected_streamk_sum(block);
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
    int blocks     = get_int_env("HIP_MOI_STREAMK_BENCH_BLOCKS", kDefaultBlocks);
    int warmup_ms  = get_int_env("HIP_MOI_STREAMK_BENCH_WARMUP_MS", kDefaultWarmupMs);
    int measure_ms = get_int_env("HIP_MOI_STREAMK_BENCH_MEASURE_MS", kDefaultMeasureMs);

    DeviceBuffers    buffers(blocks);
    ContextStorageSet context_storage(blocks);
    uint64_t          generation = 1;

    auto pass_through_launch = [&]() {
        hipLaunchKernelGGL(streamk_flag_fixup_pass_through,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           buffers.flags,
                           buffers.out);
    };
    auto context_launch = [&]() {
        ++generation;
        hipLaunchKernelGGL(streamk_flag_fixup_context,
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
    HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    context_launch();
    HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_STREAMK_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    std::printf("hip-moi Stream-K flag protocol benchmark: blocks=%d warmup_ms=%d min_ms=%d\n",
                blocks,
                warmup_ms,
                measure_ms);
    benchmark_mode("streamk-flag-fixup_pass-through", pass_through_launch, warmup_ms, measure_ms);
    benchmark_mode("streamk-flag-fixup_context", context_launch, warmup_ms, measure_ms);

    if(!context_storage.check_all())
    {
        std::fprintf(stderr, "unexpected hip-moi diagnostic\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
