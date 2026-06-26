// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// benchmarks/033_atomic_fence_extended_benchmark.hip
//
// Extended fence-plus-atomic benchmark. These rows cover relaxed exchange and
// relaxed compare-exchange operations paired with explicit release/acquire
// fences, plus a seq_cst load/store sanity row.
//
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
    constexpr int kInitialFlag        = 0;
    constexpr int kPublishedFlag      = 1;
    constexpr int kConsumedFlag       = 2;
    constexpr int kLocked             = 1;
    constexpr int kUnlocked           = 0;

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

#define HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(expr)                                  \
    do                                                                                  \
    {                                                                                   \
        hipError_t hip_moi_atomic_fence_ext_bench_status = (expr);                      \
        if(hip_moi_atomic_fence_ext_bench_status != hipSuccess)                         \
        {                                                                               \
            fail_hip(hip_moi_atomic_fence_ext_bench_status, #expr, __FILE__, __LINE__); \
        }                                                                               \
    } while(false)

    int get_int_env(const char* name, int default_value)
    {
        const char* value = std::getenv(name);
        return value ? std::atoi(value) : default_value;
    }

    __host__ __device__ int payload_value(int block)
    {
        return 409 + ((block * 13) & 255);
    }

    __device__ void initialize_flag(int* flags, int block, int value)
    {
        if(static_cast<int>(threadIdx.x) == 0)
        {
            __hip_atomic_store(&flags[block], value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();
    }

    __global__ void fence_exchange_pass_through(int* flags, int* out)
    {
        __shared__ int payload;

        int tid   = static_cast<int>(threadIdx.x);
        int block = static_cast<int>(blockIdx.x);
        initialize_flag(flags, block, kInitialFlag);

        if(tid == 0)
        {
            payload = payload_value(block);
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            (void)__hip_atomic_exchange(
                &flags[block], kPublishedFlag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }

        if(tid == kThreadsPerSubgroup)
        {
            while(__hip_atomic_load(&flags[block], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
                  != kPublishedFlag)
            {
            }
            int old_value = __hip_atomic_exchange(
                &flags[block], kConsumedFlag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            if(old_value == kPublishedFlag)
            {
                out[block] = payload;
            }
        }
    }

    __global__ void fence_successful_cas_pass_through(int* locks, int* out)
    {
        __shared__ int payload;

        int tid   = static_cast<int>(threadIdx.x);
        int block = static_cast<int>(blockIdx.x);
        initialize_flag(locks, block, kLocked);

        if(tid == 0)
        {
            payload = payload_value(block);
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            __hip_atomic_store(
                &locks[block], kUnlocked, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }

        if(tid == kThreadsPerSubgroup)
        {
            int expected = kUnlocked;
            while(!__hip_atomic_compare_exchange_strong(&locks[block],
                                                        &expected,
                                                        kLocked,
                                                        __ATOMIC_RELAXED,
                                                        __ATOMIC_RELAXED,
                                                        __HIP_MEMORY_SCOPE_AGENT))
            {
                expected = kUnlocked;
            }
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            out[block] = payload;
        }
    }

    __global__ void fence_failed_cas_pass_through(int* flags, int* out)
    {
        __shared__ int payload;

        int tid   = static_cast<int>(threadIdx.x);
        int block = static_cast<int>(blockIdx.x);
        initialize_flag(flags, block, kInitialFlag);

        if(tid == 0)
        {
            payload = payload_value(block);
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            __hip_atomic_store(
                &flags[block], kPublishedFlag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }

        if(tid == kThreadsPerSubgroup)
        {
            while(__hip_atomic_load(&flags[block], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
                  != kPublishedFlag)
            {
            }
            int  expected  = kInitialFlag;
            bool exchanged = __hip_atomic_compare_exchange_strong(&flags[block],
                                                                  &expected,
                                                                  kConsumedFlag,
                                                                  __ATOMIC_RELAXED,
                                                                  __ATOMIC_RELAXED,
                                                                  __HIP_MEMORY_SCOPE_AGENT);
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            if(!exchanged && expected == kPublishedFlag)
            {
                out[block] = payload;
            }
        }
    }

    __global__ void seq_cst_pass_through(int* flags, int* out)
    {
        __shared__ int payload;

        int tid   = static_cast<int>(threadIdx.x);
        int block = static_cast<int>(blockIdx.x);
        initialize_flag(flags, block, kInitialFlag);

        if(tid == 0)
        {
            payload = payload_value(block);
            __hip_atomic_store(
                &flags[block], kPublishedFlag, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
        }

        if(tid == kThreadsPerSubgroup)
        {
            while(__hip_atomic_load(&flags[block], __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT)
                  != kPublishedFlag)
            {
            }
            out[block] = payload;
        }
    }

    __device__ hip_moi::context make_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    __global__ void fence_exchange_context(hip_moi::context::storage_ref* storage_refs,
                                           uint64_t                       generation,
                                           int*                           flags,
                                           int*                           out)
    {
        __shared__ int payload;

        int                           block   = static_cast<int>(blockIdx.x);
        hip_moi::context::storage_ref storage = storage_refs[block];
        storage.generation                    = generation;
        hip_moi::context ctx                  = make_context(storage);
        ctx.init_workgroup();
        initialize_flag(flags, block, kInitialFlag);

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(
                &payload, payload_value(block), /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            ctx.atomic_fence(hip_moi::atomic_memory_order::release,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
            (void)ctx.atomic_exchange(&flags[block],
                                      kPublishedFlag,
                                      hip_moi::atomic_memory_order::relaxed,
                                      hip_moi::atomic_memory_scope::agent,
                                      HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            while(__hip_atomic_load(&flags[block], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
                  != kPublishedFlag)
            {
            }
            int old_value = ctx.atomic_exchange(&flags[block],
                                                kConsumedFlag,
                                                hip_moi::atomic_memory_order::relaxed,
                                                hip_moi::atomic_memory_scope::agent,
                                                HIP_MOI_SITE_ID());
            ctx.atomic_fence(hip_moi::atomic_memory_order::acquire,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
            if(old_value == kPublishedFlag)
            {
                out[block] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            }
        }
    }

    __global__ void fence_successful_cas_context(hip_moi::context::storage_ref* storage_refs,
                                                 uint64_t                       generation,
                                                 int*                           locks,
                                                 int*                           out)
    {
        __shared__ int payload;

        int                           block   = static_cast<int>(blockIdx.x);
        hip_moi::context::storage_ref storage = storage_refs[block];
        storage.generation                    = generation;
        hip_moi::context ctx                  = make_context(storage);
        ctx.init_workgroup();
        initialize_flag(locks, block, kLocked);

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(
                &payload, payload_value(block), /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            ctx.atomic_fence(hip_moi::atomic_memory_order::release,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
            ctx.atomic_store(&locks[block],
                             kUnlocked,
                             hip_moi::atomic_memory_order::relaxed,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            int expected = kUnlocked;
            while(!ctx.atomic_compare_exchange_strong(&locks[block],
                                                      &expected,
                                                      kLocked,
                                                      hip_moi::atomic_memory_order::relaxed,
                                                      hip_moi::atomic_memory_order::relaxed,
                                                      hip_moi::atomic_memory_scope::agent,
                                                      HIP_MOI_SITE_ID()))
            {
                expected = kUnlocked;
            }
            ctx.atomic_fence(hip_moi::atomic_memory_order::acquire,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
            out[block] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
        }
    }

    __global__ void fence_failed_cas_context(hip_moi::context::storage_ref* storage_refs,
                                             uint64_t                       generation,
                                             int*                           flags,
                                             int*                           out)
    {
        __shared__ int payload;

        int                           block   = static_cast<int>(blockIdx.x);
        hip_moi::context::storage_ref storage = storage_refs[block];
        storage.generation                    = generation;
        hip_moi::context ctx                  = make_context(storage);
        ctx.init_workgroup();
        initialize_flag(flags, block, kInitialFlag);

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(
                &payload, payload_value(block), /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            ctx.atomic_fence(hip_moi::atomic_memory_order::release,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
            ctx.atomic_store(&flags[block],
                             kPublishedFlag,
                             hip_moi::atomic_memory_order::relaxed,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            while(__hip_atomic_load(&flags[block], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
                  != kPublishedFlag)
            {
            }
            int  expected = kInitialFlag;
            bool exchanged
                = ctx.atomic_compare_exchange_strong(&flags[block],
                                                     &expected,
                                                     kConsumedFlag,
                                                     hip_moi::atomic_memory_order::relaxed,
                                                     hip_moi::atomic_memory_order::relaxed,
                                                     hip_moi::atomic_memory_scope::agent,
                                                     HIP_MOI_SITE_ID());
            ctx.atomic_fence(hip_moi::atomic_memory_order::acquire,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
            if(!exchanged && expected == kPublishedFlag)
            {
                out[block] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            }
        }
    }

    __global__ void seq_cst_context(hip_moi::context::storage_ref* storage_refs,
                                    uint64_t                       generation,
                                    int*                           flags,
                                    int*                           out)
    {
        __shared__ int payload;

        int                           block   = static_cast<int>(blockIdx.x);
        hip_moi::context::storage_ref storage = storage_refs[block];
        storage.generation                    = generation;
        hip_moi::context ctx                  = make_context(storage);
        ctx.init_workgroup();
        initialize_flag(flags, block, kInitialFlag);

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(
                &payload, payload_value(block), /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
            ctx.atomic_store(&flags[block],
                             kPublishedFlag,
                             hip_moi::atomic_memory_order::seq_cst,
                             hip_moi::atomic_memory_scope::agent,
                             HIP_MOI_SITE_ID());
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            while(ctx.atomic_load(&flags[block],
                                  hip_moi::atomic_memory_order::seq_cst,
                                  hip_moi::atomic_memory_scope::agent,
                                  HIP_MOI_SITE_ID())
                  != kPublishedFlag)
            {
            }
            out[block] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
        }
    }

    struct DeviceBuffers
    {
        int* flags = nullptr;
        int* out   = nullptr;

        explicit DeviceBuffers(int blocks)
        {
            HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipMalloc(&flags, blocks * sizeof(int)));
            HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipMalloc(&out, blocks * sizeof(int)));
        }

        DeviceBuffers(const DeviceBuffers&)            = delete;
        DeviceBuffers& operator=(const DeviceBuffers&) = delete;

        ~DeviceBuffers()
        {
            (void)hipFree(out);
            (void)hipFree(flags);
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

            HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(
                hipMalloc(&device_refs, host_refs.size() * sizeof(host_refs[0])));
            HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(
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
        HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipEventCreate(&start));
        HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipEventCreate(&stop));
        HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipEventRecord(start));
        for(int i = 0; i < iterations; ++i)
        {
            launch();
        }
        HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipEventRecord(stop));
        HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipEventDestroy(stop));
        HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipEventDestroy(start));
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
            double scale = ms <= 0.0 ? 2.0 : static_cast<double>(warmup_ms) / ms;
            int    next  = static_cast<int>(static_cast<double>(iterations) * scale * 1.1);
            iterations   = next > iterations ? next : iterations * 2;
        }
    }

    template <typename Launch>
    void benchmark_mode(const char* name, Launch launch, int warmup_ms, int measure_ms)
    {
        int warmup_iterations = choose_iterations(launch, warmup_ms);
        (void)warmup_iterations;
        int            measure_iterations = choose_iterations(launch, measure_ms);
        double         elapsed_ms         = run_launches(launch, measure_iterations);
        double         latency_ms         = elapsed_ms / static_cast<double>(measure_iterations);
        DisplayLatency latency            = display_latency(latency_ms);
        std::printf("%-52s %7.3g %s\n", name, latency.value, latency.unit);
    }

    void verify_outputs(int* device_output, int blocks)
    {
        std::vector<int> host_output(static_cast<std::size_t>(blocks));
        HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipMemcpy(host_output.data(),
                                                           device_output,
                                                           host_output.size() * sizeof(int),
                                                           hipMemcpyDeviceToHost));
        for(int block = 0; block < blocks; ++block)
        {
            int expected = payload_value(block);
            int actual   = host_output[static_cast<std::size_t>(block)];
            if(actual != expected)
            {
                std::fprintf(stderr, "block %d expected %d, got %d\n", block, expected, actual);
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
    ContextStorageSet context_storage(blocks);
    uint64_t          generation = 1;

    auto exchange_pass_through_launch = [&]() {
        hipLaunchKernelGGL(fence_exchange_pass_through,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           buffers.flags,
                           buffers.out);
    };
    auto exchange_context_launch = [&]() {
        ++generation;
        hipLaunchKernelGGL(fence_exchange_context,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           context_storage.device_refs,
                           generation,
                           buffers.flags,
                           buffers.out);
    };
    auto successful_cas_pass_through_launch = [&]() {
        hipLaunchKernelGGL(fence_successful_cas_pass_through,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           buffers.flags,
                           buffers.out);
    };
    auto successful_cas_context_launch = [&]() {
        ++generation;
        hipLaunchKernelGGL(fence_successful_cas_context,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           context_storage.device_refs,
                           generation,
                           buffers.flags,
                           buffers.out);
    };
    auto failed_cas_pass_through_launch = [&]() {
        hipLaunchKernelGGL(fence_failed_cas_pass_through,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           buffers.flags,
                           buffers.out);
    };
    auto failed_cas_context_launch = [&]() {
        ++generation;
        hipLaunchKernelGGL(fence_failed_cas_context,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           context_storage.device_refs,
                           generation,
                           buffers.flags,
                           buffers.out);
    };
    auto seq_cst_pass_through_launch = [&]() {
        hipLaunchKernelGGL(seq_cst_pass_through,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           buffers.flags,
                           buffers.out);
    };
    auto seq_cst_context_launch = [&]() {
        ++generation;
        hipLaunchKernelGGL(seq_cst_context,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           context_storage.device_refs,
                           generation,
                           buffers.flags,
                           buffers.out);
    };

    exchange_pass_through_launch();
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    exchange_context_launch();
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    successful_cas_pass_through_launch();
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    successful_cas_context_launch();
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    failed_cas_pass_through_launch();
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    failed_cas_context_launch();
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    seq_cst_pass_through_launch();
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    seq_cst_context_launch();
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATOMIC_FENCE_EXT_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    std::printf("hip-moi atomic fence extended benchmark: blocks=%d warmup_ms=%d min_ms=%d\n",
                blocks,
                warmup_ms,
                measure_ms);
    benchmark_mode(
        "atomic-fence-exchange_pass-through", exchange_pass_through_launch, warmup_ms, measure_ms);
    benchmark_mode("atomic-fence-exchange_context", exchange_context_launch, warmup_ms, measure_ms);
    benchmark_mode("atomic-fence-successful-cas_pass-through",
                   successful_cas_pass_through_launch,
                   warmup_ms,
                   measure_ms);
    benchmark_mode("atomic-fence-successful-cas_context",
                   successful_cas_context_launch,
                   warmup_ms,
                   measure_ms);
    benchmark_mode("atomic-fence-failed-cas_pass-through",
                   failed_cas_pass_through_launch,
                   warmup_ms,
                   measure_ms);
    benchmark_mode(
        "atomic-fence-failed-cas_context", failed_cas_context_launch, warmup_ms, measure_ms);
    benchmark_mode(
        "atomic-seq-cst-handoff_pass-through", seq_cst_pass_through_launch, warmup_ms, measure_ms);
    benchmark_mode("atomic-seq-cst-handoff_context", seq_cst_context_launch, warmup_ms, measure_ms);

    if(!context_storage.check_all())
    {
        std::fprintf(stderr, "unexpected hip-moi diagnostics in extended fence benchmark\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
