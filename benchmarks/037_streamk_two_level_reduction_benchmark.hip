// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// benchmarks/037_streamk_two_level_reduction_benchmark.hip
//
// Stream-K-style two-level reduction benchmark. Four producer subgroups write
// LDS partials, two pair-reducer subgroups consume those partials after
// release/acquire flags, and one reducer subgroup consumes the pair partials
// after a second release/acquire flag handoff.
#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace
{
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kProducerCount      = 4;
    constexpr int kPairReducerCount   = 2;
    constexpr int kSubgroupCount      = kProducerCount + kPairReducerCount;
    constexpr int kThreadCount        = kThreadsPerSubgroup * kSubgroupCount;
    constexpr int kProducerFlagCount  = kProducerCount;
    constexpr int kPairFlagCount      = kPairReducerCount;
    constexpr int kFlagCount          = kProducerFlagCount + kPairFlagCount;
    constexpr int kPairFlagBase       = kProducerFlagCount;
    constexpr int kFlagReady          = 1;
    constexpr int kDefaultBlocks      = 256;
    constexpr int kDefaultWarmupMs    = 100;
    constexpr int kDefaultMeasureMs   = 100;

    constexpr uint32_t kPartialsByteOffset = 0;
    constexpr uint32_t kPairPartialsByteOffset
        = kPartialsByteOffset
          + static_cast<uint32_t>(kProducerCount * kThreadsPerSubgroup * sizeof(int));
    constexpr int kSharedBytes = static_cast<int>(
        kPairPartialsByteOffset + kPairReducerCount * kThreadsPerSubgroup * sizeof(int));
    constexpr int kExactShadowEntries
        = (kSharedBytes + static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes) - 1)
          / static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes);

    constexpr hip_moi::site_id kFlagResetSite(0x2601);
    constexpr hip_moi::site_id kPartialStoreSite(0x2602);
    constexpr hip_moi::site_id kPartialLoadSite(0x2603);
    constexpr hip_moi::site_id kPairPartialStoreSite(0x2604);
    constexpr hip_moi::site_id kPairPartialLoadSite(0x2605);
    constexpr hip_moi::site_id kFlagSite(0x2606);

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

#define HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(expr)                                  \
    do                                                                                   \
    {                                                                                    \
        hipError_t hip_moi_streamk_two_level_bench_status = (expr);                      \
        if(hip_moi_streamk_two_level_bench_status != hipSuccess)                         \
        {                                                                                \
            fail_hip(hip_moi_streamk_two_level_bench_status, #expr, __FILE__, __LINE__); \
        }                                                                                \
    } while(false)

    int get_int_env(const char* name, int default_value)
    {
        const char* value = std::getenv(name);
        return value ? std::atoi(value) : default_value;
    }

    __host__ __device__ int pair_reducer_subgroup(int pair)
    {
        return kProducerCount + pair;
    }

    __host__ __device__ int producer_flag_index(int producer)
    {
        return producer;
    }

    __host__ __device__ int pair_flag_index(int pair)
    {
        return kPairFlagBase + pair;
    }

    __host__ __device__ int producer_value(int block, int producer, int lane)
    {
        return (producer + 1) * 100 + lane + ((block * 7) & 31);
    }

    __host__ __device__ int expected_value(int block, int lane)
    {
        int value = 0;
        for(int producer = 0; producer < kProducerCount; ++producer)
        {
            value += producer_value(block, producer, lane);
        }
        return value;
    }

    __host__ __device__ uint32_t partial_offset(int producer, int lane)
    {
        return kPartialsByteOffset
               + static_cast<uint32_t>((producer * kThreadsPerSubgroup + lane) * sizeof(int));
    }

    __host__ __device__ uint32_t pair_partial_offset(int pair, int lane)
    {
        return kPairPartialsByteOffset
               + static_cast<uint32_t>((pair * kThreadsPerSubgroup + lane) * sizeof(int));
    }

    __global__ void streamk_two_level_reduction_pass_through(int* flags, int* out)
    {
        __shared__ int partials[kProducerCount][kThreadsPerSubgroup];
        __shared__ int pair_partials[kPairReducerCount][kThreadsPerSubgroup];

        int block    = static_cast<int>(blockIdx.x);
        int tid      = static_cast<int>(threadIdx.x);
        int subgroup = tid / kThreadsPerSubgroup;
        int lane     = tid & (kThreadsPerSubgroup - 1);

        if(tid == 0)
        {
            for(int flag = 0; flag < kFlagCount; ++flag)
            {
                __hip_atomic_store(&flags[block * kFlagCount + flag],
                                   0,
                                   __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
            }
        }
        __syncthreads();

        if(subgroup < kProducerCount)
        {
            partials[subgroup][lane] = producer_value(block, subgroup, lane);
            if(lane == 0)
            {
                __hip_atomic_store(&flags[block * kFlagCount + producer_flag_index(subgroup)],
                                   kFlagReady,
                                   __ATOMIC_RELEASE,
                                   __HIP_MEMORY_SCOPE_AGENT);
            }
        }

        if(subgroup >= kProducerCount)
        {
            int pair = subgroup - kProducerCount;
            if(lane == 0)
            {
                for(int producer = pair * 2; producer < pair * 2 + 2; ++producer)
                {
                    while(__hip_atomic_load(
                              &flags[block * kFlagCount + producer_flag_index(producer)],
                              __ATOMIC_ACQUIRE,
                              __HIP_MEMORY_SCOPE_AGENT)
                          != kFlagReady)
                    {
                    }
                }
            }

            int first_producer        = pair * 2;
            int second_producer       = first_producer + 1;
            int first                 = partials[first_producer][lane];
            int second                = partials[second_producer][lane];
            pair_partials[pair][lane] = first + second;

            if(lane == 0)
            {
                __hip_atomic_store(&flags[block * kFlagCount + pair_flag_index(pair)],
                                   kFlagReady,
                                   __ATOMIC_RELEASE,
                                   __HIP_MEMORY_SCOPE_AGENT);
            }
        }

        if(subgroup == pair_reducer_subgroup(/*pair=*/0))
        {
            if(lane == 0)
            {
                for(int pair = 0; pair < kPairReducerCount; ++pair)
                {
                    while(__hip_atomic_load(&flags[block * kFlagCount + pair_flag_index(pair)],
                                            __ATOMIC_ACQUIRE,
                                            __HIP_MEMORY_SCOPE_AGENT)
                          != kFlagReady)
                    {
                    }
                }
            }

            out[block * kThreadsPerSubgroup + lane]
                = pair_partials[0][lane] + pair_partials[1][lane];
        }
    }

    __global__ void streamk_two_level_reduction_context(hip_moi::context::storage_ref* storage_refs,
                                                        uint64_t                       generation,
                                                        int*                           flags,
                                                        int*                           out)
    {
        __shared__ int partials[kProducerCount][kThreadsPerSubgroup];
        __shared__ int pair_partials[kPairReducerCount][kThreadsPerSubgroup];

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

        int tid      = static_cast<int>(threadIdx.x);
        int subgroup = tid / kThreadsPerSubgroup;
        int lane     = tid & (kThreadsPerSubgroup - 1);

        if(tid == 0)
        {
            for(int flag = 0; flag < kFlagCount; ++flag)
            {
                ctx.atomic_store(&flags[block * kFlagCount + flag],
                                 0,
                                 hip_moi::atomic_memory_order::relaxed,
                                 hip_moi::atomic_memory_scope::agent,
                                 kFlagResetSite);
            }
        }
        ctx.syncthreads();

        if(subgroup < kProducerCount)
        {
            ctx.lds_store_at(&partials[subgroup][lane],
                             producer_value(block, subgroup, lane),
                             partial_offset(subgroup, lane),
                             kPartialStoreSite);
            if(lane == 0)
            {
                ctx.atomic_store(&flags[block * kFlagCount + producer_flag_index(subgroup)],
                                 kFlagReady,
                                 hip_moi::atomic_memory_order::release,
                                 hip_moi::atomic_memory_scope::agent,
                                 kFlagSite);
            }
        }

        if(subgroup >= kProducerCount)
        {
            int pair = subgroup - kProducerCount;
            if(lane == 0)
            {
                for(int producer = pair * 2; producer < pair * 2 + 2; ++producer)
                {
                    while(
                        ctx.atomic_load(&flags[block * kFlagCount + producer_flag_index(producer)],
                                        hip_moi::atomic_memory_order::acquire,
                                        hip_moi::atomic_memory_scope::agent,
                                        kFlagSite)
                        != kFlagReady)
                    {
                    }
                }
            }

            int first_producer  = pair * 2;
            int second_producer = first_producer + 1;
            int first           = ctx.lds_load_at(&partials[first_producer][lane],
                                        partial_offset(first_producer, lane),
                                        kPartialLoadSite);
            int second          = ctx.lds_load_at(&partials[second_producer][lane],
                                         partial_offset(second_producer, lane),
                                         kPartialLoadSite);
            ctx.lds_store_at(&pair_partials[pair][lane],
                             first + second,
                             pair_partial_offset(pair, lane),
                             kPairPartialStoreSite);

            if(lane == 0)
            {
                ctx.atomic_store(&flags[block * kFlagCount + pair_flag_index(pair)],
                                 kFlagReady,
                                 hip_moi::atomic_memory_order::release,
                                 hip_moi::atomic_memory_scope::agent,
                                 kFlagSite);
            }
        }

        if(subgroup == pair_reducer_subgroup(/*pair=*/0))
        {
            if(lane == 0)
            {
                for(int pair = 0; pair < kPairReducerCount; ++pair)
                {
                    while(ctx.atomic_load(&flags[block * kFlagCount + pair_flag_index(pair)],
                                          hip_moi::atomic_memory_order::acquire,
                                          hip_moi::atomic_memory_scope::agent,
                                          kFlagSite)
                          != kFlagReady)
                    {
                    }
                }
            }

            int value0                              = ctx.lds_load_at(&pair_partials[0][lane],
                                         pair_partial_offset(/*pair=*/0, lane),
                                         kPairPartialLoadSite);
            int value1                              = ctx.lds_load_at(&pair_partials[1][lane],
                                         pair_partial_offset(/*pair=*/1, lane),
                                         kPairPartialLoadSite);
            out[block * kThreadsPerSubgroup + lane] = value0 + value1;
        }
    }

    struct DeviceBuffers
    {
        int* flags = nullptr;
        int* out   = nullptr;

        explicit DeviceBuffers(int blocks)
        {
            HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(
                hipMalloc(&flags, blocks * kFlagCount * sizeof(int)));
            HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(
                hipMalloc(&out, blocks * kThreadsPerSubgroup * sizeof(int)));
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
            options.storage_bytes               = 128 * 1024;
            options.exact_shadow_entry_capacity = kExactShadowEntries;
            options.sampled_watchpoint_capacity = 0;
            options.diagnostic_capacity         = 16;
            options.subgroup_capacity           = kSubgroupCount;

            contexts.reserve(static_cast<std::size_t>(blocks));
            std::vector<hip_moi::context::storage_ref> host_refs;
            host_refs.reserve(static_cast<std::size_t>(blocks));
            for(int block = 0; block < blocks; ++block)
            {
                contexts.push_back(std::make_unique<hip_moi::host_context>(options));
                host_refs.push_back(contexts.back()->launch_ref());
            }

            HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(
                hipMalloc(&device_refs, host_refs.size() * sizeof(host_refs[0])));
            HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(
                hipMemcpy(device_refs,
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

        bool check_all(std::FILE* stream)
        {
            bool ok = true;
            for(auto& context : contexts)
            {
                ok = context->check(stream) && ok;
            }
            return ok;
        }
    };

    template <typename Launch>
    double run_launches(Launch launch, int iterations)
    {
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(hipEventCreate(&start));
        HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(hipEventCreate(&stop));
        HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(hipEventRecord(start, 0));
        for(int i = 0; i < iterations; ++i)
        {
            launch();
        }
        HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(hipEventRecord(stop, 0));
        HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(hipEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(hipEventDestroy(stop));
        HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(hipEventDestroy(start));
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
        std::vector<int> host_output(static_cast<std::size_t>(blocks * kThreadsPerSubgroup), 0);
        HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(
            hipMemcpy(host_output.data(),
                      device_output,
                      host_output.size() * sizeof(host_output[0]),
                      hipMemcpyDeviceToHost));
        for(int block = 0; block < blocks; ++block)
        {
            for(int lane = 0; lane < kThreadsPerSubgroup; ++lane)
            {
                int offset = block * kThreadsPerSubgroup + lane;
                int actual = host_output[static_cast<std::size_t>(offset)];
                int expect = expected_value(block, lane);
                if(actual != expect)
                {
                    std::fprintf(stderr,
                                 "output mismatch at block %d lane %d: got %d expected %d\n",
                                 block,
                                 lane,
                                 actual,
                                 expect);
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    }
} // namespace

int main()
{
    int blocks     = get_int_env("HIP_MOI_STREAMK_TWO_LEVEL_BENCH_BLOCKS", kDefaultBlocks);
    int warmup_ms  = get_int_env("HIP_MOI_STREAMK_TWO_LEVEL_BENCH_WARMUP_MS", kDefaultWarmupMs);
    int measure_ms = get_int_env("HIP_MOI_STREAMK_TWO_LEVEL_BENCH_MEASURE_MS", kDefaultMeasureMs);

    DeviceBuffers     buffers(blocks);
    ContextStorageSet context_storage(blocks);
    uint64_t          generation = 1;

    auto pass_through_launch = [&]() {
        hipLaunchKernelGGL(streamk_two_level_reduction_pass_through,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           buffers.flags,
                           buffers.out);
    };
    auto context_launch = [&]() {
        ++generation;
        hipLaunchKernelGGL(streamk_two_level_reduction_context,
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
    HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    context_launch();
    HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);
    if(!context_storage.check_all(stderr))
    {
        return EXIT_FAILURE;
    }

    std::printf(
        "hip-moi Stream-K two-level reduction benchmark: blocks=%d warmup_ms=%d min_ms=%d\n",
        blocks,
        warmup_ms,
        measure_ms);
    benchmark_mode(
        "streamk-two-level-reduction_pass-through", pass_through_launch, warmup_ms, measure_ms);
    benchmark_mode("streamk-two-level-reduction_context", context_launch, warmup_ms, measure_ms);

    HIP_MOI_STREAMK_TWO_LEVEL_BENCH_HIP_CHECK(hipDeviceSynchronize());
    if(!context_storage.check_all(stderr))
    {
        return EXIT_FAILURE;
    }
    verify_outputs(buffers.out, blocks);
    return EXIT_SUCCESS;
}
