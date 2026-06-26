// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// benchmarks/034_atomic_bitwise_happens_before_benchmark.hip
//
// Stage 16 atomics benchmark. Bitwise RMWs beyond atomicOr order an
// instrumented LDS payload handoff through old-value-dependent bitmask control
// flow.
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
    constexpr int kProducerBit        = 1;
    constexpr int kInitialAndMask     = kProducerBit;
    constexpr int kInitialXorMask     = 0;

    enum class bitwise_operation
    {
        fetch_and_clear,
        fetch_xor_toggle,
    };

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

#define HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(expr)                                  \
    do                                                                                \
    {                                                                                 \
        hipError_t hip_moi_atomic_bitwise_bench_status = (expr);                      \
        if(hip_moi_atomic_bitwise_bench_status != hipSuccess)                         \
        {                                                                             \
            fail_hip(hip_moi_atomic_bitwise_bench_status, #expr, __FILE__, __LINE__); \
        }                                                                             \
    } while(false)

    int get_int_env(const char* name, int default_value)
    {
        const char* value = std::getenv(name);
        return value ? std::atoi(value) : default_value;
    }

    __host__ __device__ int payload_value(int block)
    {
        return 127 + ((block * 17) & 255);
    }

    template <bitwise_operation Operation>
    __host__ __device__ int initial_bitmask()
    {
        if constexpr(Operation == bitwise_operation::fetch_and_clear)
        {
            return kInitialAndMask;
        }
        else
        {
            return kInitialXorMask;
        }
    }

    template <bitwise_operation Operation>
    __device__ int pass_through_publish(int* bitmask, int block)
    {
        if constexpr(Operation == bitwise_operation::fetch_and_clear)
        {
            return __hip_atomic_fetch_and(
                &bitmask[block], ~kProducerBit, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        else
        {
            return __hip_atomic_fetch_xor(
                &bitmask[block], kProducerBit, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <bitwise_operation Operation>
    __device__ int pass_through_observe(int* bitmask, int block)
    {
        if constexpr(Operation == bitwise_operation::fetch_and_clear)
        {
            return __hip_atomic_fetch_and(
                &bitmask[block], ~0, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        }
        else
        {
            return __hip_atomic_fetch_xor(
                &bitmask[block], 0, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <bitwise_operation Operation>
    __device__ bool producer_done(int old_mask)
    {
        if constexpr(Operation == bitwise_operation::fetch_and_clear)
        {
            return (old_mask & kProducerBit) == 0;
        }
        else
        {
            return (old_mask & kProducerBit) != 0;
        }
    }

    template <bitwise_operation Operation>
    __global__ void atomic_bitwise_pass_through(int* bitmasks, int* out)
    {
        __shared__ int payload;

        int tid   = static_cast<int>(threadIdx.x);
        int block = static_cast<int>(blockIdx.x);
        if(tid == 0)
        {
            __hip_atomic_store(&bitmasks[block],
                               initial_bitmask<Operation>(),
                               __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();

        if(tid == 0)
        {
            payload = payload_value(block);
            (void)pass_through_publish<Operation>(bitmasks, block);
        }

        if(tid == kThreadsPerSubgroup)
        {
            int old_mask = 0;
            do
            {
                old_mask = pass_through_observe<Operation>(bitmasks, block);
            } while(!producer_done<Operation>(old_mask));
            out[block] = payload + old_mask;
        }
    }

    template <bitwise_operation Operation>
    __global__ void atomic_bitwise_context(hip_moi::context::storage_ref* storage_refs,
                                           uint64_t                       generation,
                                           int*                           bitmasks,
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
            ctx.atomic_store(&bitmasks[block],
                             initial_bitmask<Operation>(),
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
            if constexpr(Operation == bitwise_operation::fetch_and_clear)
            {
                (void)ctx.atomic_fetch_and(&bitmasks[block],
                                           ~kProducerBit,
                                           hip_moi::atomic_memory_order::release,
                                           hip_moi::atomic_memory_scope::agent,
                                           HIP_MOI_SITE_ID());
            }
            else
            {
                (void)ctx.atomic_fetch_xor(&bitmasks[block],
                                           kProducerBit,
                                           hip_moi::atomic_memory_order::release,
                                           hip_moi::atomic_memory_scope::agent,
                                           HIP_MOI_SITE_ID());
            }
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            int old_mask = 0;
            do
            {
                if constexpr(Operation == bitwise_operation::fetch_and_clear)
                {
                    old_mask = ctx.atomic_fetch_and(&bitmasks[block],
                                                    ~0,
                                                    hip_moi::atomic_memory_order::acquire,
                                                    hip_moi::atomic_memory_scope::agent,
                                                    HIP_MOI_SITE_ID());
                }
                else
                {
                    old_mask = ctx.atomic_fetch_xor(&bitmasks[block],
                                                    0,
                                                    hip_moi::atomic_memory_order::acquire,
                                                    hip_moi::atomic_memory_scope::agent,
                                                    HIP_MOI_SITE_ID());
                }
            } while(!producer_done<Operation>(old_mask));
            int value  = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            out[block] = value + old_mask;
        }
    }

    struct DeviceBuffers
    {
        int* bitmasks = nullptr;
        int* out      = nullptr;

        explicit DeviceBuffers(int blocks)
        {
            HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(hipMalloc(&bitmasks, blocks * sizeof(int)));
            HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(hipMalloc(&out, blocks * sizeof(int)));
        }

        DeviceBuffers(const DeviceBuffers&)            = delete;
        DeviceBuffers& operator=(const DeviceBuffers&) = delete;

        ~DeviceBuffers()
        {
            (void)hipFree(out);
            (void)hipFree(bitmasks);
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

            HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(
                hipMalloc(&device_refs, host_refs.size() * sizeof(host_refs[0])));
            HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(
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
        HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(hipEventCreate(&start));
        HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(hipEventCreate(&stop));
        HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(hipEventRecord(start, 0));
        for(int i = 0; i < iterations; ++i)
        {
            launch();
        }
        HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(hipEventRecord(stop, 0));
        HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(hipEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(hipEventDestroy(stop));
        HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(hipEventDestroy(start));
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

    template <bitwise_operation Operation>
    void verify_outputs(const int* device_output, int blocks)
    {
        std::vector<int> host_output(static_cast<std::size_t>(blocks), 0);
        HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(
            hipMemcpy(host_output.data(),
                      device_output,
                      host_output.size() * sizeof(host_output[0]),
                      hipMemcpyDeviceToHost));
        for(int block = 0; block < blocks; ++block)
        {
            int old_mask = Operation == bitwise_operation::fetch_and_clear ? 0 : kProducerBit;
            int expected = payload_value(block) + old_mask;
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

    template <bitwise_operation Operation>
    void run_case(const char* pass_through_name,
                  const char* context_name,
                  int         blocks,
                  int         warmup_ms,
                  int         measure_ms)
    {
        DeviceBuffers     buffers(blocks);
        ContextStorageSet context_storage(blocks);
        uint64_t          generation = 1;

        auto pass_through_launch = [&]() {
            hipLaunchKernelGGL((atomic_bitwise_pass_through<Operation>),
                               dim3(blocks),
                               dim3(kThreadCount),
                               0,
                               0,
                               buffers.bitmasks,
                               buffers.out);
        };
        auto context_launch = [&]() {
            ++generation;
            hipLaunchKernelGGL((atomic_bitwise_context<Operation>),
                               dim3(blocks),
                               dim3(kThreadCount),
                               0,
                               0,
                               context_storage.device_refs,
                               generation,
                               buffers.bitmasks,
                               buffers.out);
        };

        pass_through_launch();
        HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(hipGetLastError());
        HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(hipDeviceSynchronize());
        verify_outputs<Operation>(buffers.out, blocks);

        context_launch();
        HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(hipGetLastError());
        HIP_MOI_ATOMIC_BITWISE_BENCH_HIP_CHECK(hipDeviceSynchronize());
        verify_outputs<Operation>(buffers.out, blocks);

        benchmark_mode(pass_through_name, pass_through_launch, warmup_ms, measure_ms);
        benchmark_mode(context_name, context_launch, warmup_ms, measure_ms);

        if(!context_storage.check_all())
        {
            std::fprintf(stderr, "unexpected hip-moi diagnostic\n");
            std::exit(EXIT_FAILURE);
        }
    }
} // namespace

int main()
{
    int blocks     = get_int_env("HIP_MOI_ATOMIC_BITWISE_BENCH_BLOCKS", kDefaultBlocks);
    int warmup_ms  = get_int_env("HIP_MOI_ATOMIC_BITWISE_BENCH_WARMUP_MS", kDefaultWarmupMs);
    int measure_ms = get_int_env("HIP_MOI_ATOMIC_BITWISE_BENCH_MEASURE_MS", kDefaultMeasureMs);

    std::printf("hip-moi atomic bitwise benchmark: blocks=%d warmup_ms=%d min_ms=%d\n",
                blocks,
                warmup_ms,
                measure_ms);
    run_case<bitwise_operation::fetch_and_clear>("atomic-and-bitmask-handoff_pass-through",
                                                 "atomic-and-bitmask-handoff_context",
                                                 blocks,
                                                 warmup_ms,
                                                 measure_ms);
    run_case<bitwise_operation::fetch_xor_toggle>("atomic-xor-bitmask-handoff_pass-through",
                                                  "atomic-xor-bitmask-handoff_context",
                                                  blocks,
                                                  warmup_ms,
                                                  measure_ms);

    return EXIT_SUCCESS;
}
