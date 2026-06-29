// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// benchmarks/036_cdna4_mfma_attention_lds_alias_handoff_benchmark.hip
//
// CDNA4 MFMA attention LDS-alias handoff benchmark. The kernel models a
// synchronized production-style handoff where one logical tile is consumed from
// an LDS slot before the next logical tile is written into the same slot.
#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace
{
    using half_t  = _Float16;
    using f16x4_t = half_t __attribute__((ext_vector_type(4)));
    using f32x4_t = float __attribute__((ext_vector_type(4)));

    constexpr int kWaveSize           = 64;
    constexpr int kSubgroupCount      = 2;
    constexpr int kThreadCount        = kWaveSize * kSubgroupCount;
    constexpr int kThreadsPerSubgroup = kWaveSize;
    constexpr int kTileM              = 16;
    constexpr int kTileN              = 16;
    constexpr int kKTile              = 16;
    constexpr int kFragmentElements   = 4;
    constexpr int kDefaultBlocks      = 256;
    constexpr int kDefaultWarmupMs    = 100;
    constexpr int kDefaultMeasureMs   = 100;
    constexpr int kDefaultWatchpoints = 1;

    constexpr std::size_t kFragmentsPerBlock = kWaveSize;
    constexpr std::size_t kOutputPerBlock    = kTileM * kTileN;

    constexpr hip_moi::site_id kTailKStoreSite(0x3601);
    constexpr hip_moi::site_id kTailKLoadSite(0x3602);
    constexpr hip_moi::site_id kNextVStoreSite(0x3603);
    constexpr hip_moi::site_id kNextVLoadSite(0x3604);

    using StaticFastPolicy = hip_moi::sampled_watchpoint_policy<
        /*SampleSkip=*/32,
        /*ProbeCount=*/1,
        /*DelayIters=*/32,
        /*ReportConflicts=*/false>;

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

#define HIP_MOI_ALIAS_BENCH_HIP_CHECK(expr)                                  \
    do                                                                       \
    {                                                                        \
        hipError_t hip_moi_alias_bench_status = (expr);                      \
        if(hip_moi_alias_bench_status != hipSuccess)                         \
        {                                                                    \
            fail_hip(hip_moi_alias_bench_status, #expr, __FILE__, __LINE__); \
        }                                                                    \
    } while(false)

    int get_int_env(const char* name, int default_value)
    {
        const char* value = std::getenv(name);
        return value ? std::atoi(value) : default_value;
    }

    __host__ __device__ float input_value(int row, int col, int seed)
    {
        int value = (row * 7 + col * 5 + seed * 13 + (row ^ (col * 3))) % 9;
        return static_cast<float>(value - 4) * 0.25f;
    }

    __host__ __device__ int cdna4_mfma_f16_k(int lane, int elem)
    {
        return elem + 4 * (lane >> 4);
    }

    __host__ __device__ int cdna4_mfma_acc_m(int lane, int elem)
    {
        return elem + 4 * (lane >> 4);
    }

    __host__ __device__ uint32_t fragment_offset(int lane)
    {
        return static_cast<uint32_t>(lane * sizeof(f16x4_t));
    }

    __device__ f32x4_t zero_f32x4()
    {
        f32x4_t value;
#pragma unroll
        for(int i = 0; i < kFragmentElements; ++i)
        {
            value[i] = 0.0f;
        }
        return value;
    }

    struct Runtime
    {
        hip_moi::context::storage_ref* storage_refs;
        uint64_t                       generation;
    };

    __device__ hip_moi::context::storage_ref workgroup_storage(Runtime runtime)
    {
        hip_moi::context::storage_ref storage = runtime.storage_refs[blockIdx.x];
        storage.generation                    = runtime.generation;
        return storage;
    }

    struct PassThroughPolicy
    {
        struct context_type
        {
        };

        __device__ static context_type make_context(Runtime)
        {
            return {};
        }

        __device__ static void init_workgroup(context_type&) {}

        __device__ static void syncthreads(context_type&)
        {
            __syncthreads();
        }

        template <typename T>
        __device__ static void store(context_type&, T* ptr, T value, uint32_t, hip_moi::site_id)
        {
            *ptr = value;
        }

        template <typename T>
        __device__ static T load(context_type&, const T* ptr, uint32_t, hip_moi::site_id)
        {
            return *ptr;
        }
    };

    struct ContextSampledPolicy
    {
        using context_type = hip_moi::context;

        __device__ static context_type make_context(Runtime runtime)
        {
            hip_moi::context::config cfg{
                /*thread_count=*/kThreadCount,
                /*threads_per_subgroup=*/kThreadsPerSubgroup,
                /*subgroup_count=*/kSubgroupCount,
            };
            return hip_moi::context(workgroup_storage(runtime), cfg);
        }

        __device__ static void init_workgroup(context_type& ctx)
        {
            ctx.init_workgroup();
        }

        __device__ static void syncthreads(context_type& ctx)
        {
            ctx.syncthreads();
        }

        template <typename T>
        __device__ static void
            store(context_type& ctx, T* ptr, T value, uint32_t offset, hip_moi::site_id site)
        {
            ctx.template lds_store_at<hip_moi::backend_kind::sampled_watchpoint>(
                ptr, value, offset, site);
        }

        template <typename T>
        __device__ static T
            load(context_type& ctx, const T* ptr, uint32_t offset, hip_moi::site_id site)
        {
            return ctx.template lds_load_at<hip_moi::backend_kind::sampled_watchpoint>(
                ptr, offset, site);
        }
    };

    struct SampledWatchpointContextPolicy
    {
        using context_type = hip_moi::sampled_watchpoint_context;

        __device__ static context_type make_context(Runtime runtime)
        {
            hip_moi::sampled_watchpoint_context::config cfg{
                /*threads_per_subgroup=*/kThreadsPerSubgroup,
            };
            return hip_moi::make_sampled_watchpoint_context(workgroup_storage(runtime), cfg);
        }

        __device__ static void init_workgroup(context_type& ctx)
        {
            ctx.init_workgroup();
        }

        __device__ static void syncthreads(context_type& ctx)
        {
            ctx.syncthreads();
        }

        template <typename T>
        __device__ static void
            store(context_type& ctx, T* ptr, T value, uint32_t offset, hip_moi::site_id site)
        {
            ctx.template lds_store_at<StaticFastPolicy>(ptr, value, offset, site);
        }

        template <typename T>
        __device__ static T
            load(context_type& ctx, const T* ptr, uint32_t offset, hip_moi::site_id site)
        {
            return ctx.template lds_load_at<StaticFastPolicy>(ptr, offset, site);
        }
    };

    template <typename Policy>
    __global__ __launch_bounds__(kThreadCount) void mfma_attention_alias_handoff_kernel(
        const f16x4_t* __restrict__ q_fragments,
        const f16x4_t* __restrict__ k_fragments,
        const f16x4_t* __restrict__ v_fragments,
        float* __restrict__ out,
        float* __restrict__ checksum,
        Runtime runtime)
    {
        __shared__ f16x4_t aliased_slot[kWaveSize];

        typename Policy::context_type ctx = Policy::make_context(runtime);
        Policy::init_workgroup(ctx);

        int block    = static_cast<int>(blockIdx.x);
        int tid      = static_cast<int>(threadIdx.x);
        int subgroup = tid / kWaveSize;
        int lane     = tid & (kWaveSize - 1);

        std::size_t base = static_cast<std::size_t>(block) * kFragmentsPerBlock;

        if(subgroup == 0)
        {
            Policy::store(ctx,
                          &aliased_slot[lane],
                          k_fragments[base + lane],
                          fragment_offset(lane),
                          kTailKStoreSite);
        }
        Policy::syncthreads(ctx);

        f32x4_t acc = zero_f32x4();
        if(subgroup == 1)
        {
            f16x4_t tail_k
                = Policy::load(ctx, &aliased_slot[lane], fragment_offset(lane), kTailKLoadSite);
            acc = __builtin_amdgcn_mfma_f32_16x16x16f16(
                q_fragments[base + lane], tail_k, acc, 0, 0, 0);
        }

        Policy::syncthreads(ctx);

        if(subgroup == 0)
        {
            Policy::store(ctx,
                          &aliased_slot[lane],
                          v_fragments[base + lane],
                          fragment_offset(lane),
                          kNextVStoreSite);
        }
        Policy::syncthreads(ctx);

        if(subgroup == 1)
        {
            f16x4_t next_v
                = Policy::load(ctx, &aliased_slot[lane], fragment_offset(lane), kNextVLoadSite);
            if(lane == 0)
            {
                checksum[block] = static_cast<float>(next_v[0]);
            }

            int col = lane & 15;
#pragma unroll
            for(int elem = 0; elem < kFragmentElements; ++elem)
            {
                int row = cdna4_mfma_acc_m(lane, elem);
                out[static_cast<std::size_t>(block) * kOutputPerBlock + row * kTileN + col]
                    = acc[elem];
            }
        }
    }

    struct DeviceBuffers
    {
        f16x4_t* q        = nullptr;
        f16x4_t* k        = nullptr;
        f16x4_t* v        = nullptr;
        float*   out      = nullptr;
        float*   checksum = nullptr;

        explicit DeviceBuffers(int blocks)
        {
            std::size_t fragments = static_cast<std::size_t>(blocks) * kFragmentsPerBlock;
            std::size_t outputs   = static_cast<std::size_t>(blocks) * kOutputPerBlock;
            HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipMalloc(&q, fragments * sizeof(f16x4_t)));
            HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipMalloc(&k, fragments * sizeof(f16x4_t)));
            HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipMalloc(&v, fragments * sizeof(f16x4_t)));
            HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipMalloc(&out, outputs * sizeof(float)));
            HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipMalloc(&checksum, blocks * sizeof(float)));
        }

        DeviceBuffers(const DeviceBuffers&)            = delete;
        DeviceBuffers& operator=(const DeviceBuffers&) = delete;

        ~DeviceBuffers()
        {
            (void)hipFree(checksum);
            (void)hipFree(out);
            (void)hipFree(v);
            (void)hipFree(k);
            (void)hipFree(q);
        }
    };

    struct StorageSet
    {
        std::vector<std::unique_ptr<hip_moi::host_context>> contexts;
        hip_moi::context::storage_ref*                      device_refs = nullptr;
        uint64_t                                            generation  = 1;

        StorageSet() = default;

        explicit StorageSet(int blocks, int watchpoints)
        {
            hip_moi::host_context_options options;
            options.storage_bytes                  = 64 * 1024;
            options.diagnostic_capacity            = 16;
            options.subgroup_capacity              = kSubgroupCount;
            options.exact_shadow_entry_capacity    = 0;
            options.sampled_watchpoint_capacity    = watchpoints;
            options.sampled_watchpoint_sample_skip = 32;
            options.sampled_watchpoint_probe_count = 1;
            options.sampled_watchpoint_delay_iters = 32;
            options.sampled_watchpoint_reports     = false;
            options.backend                        = hip_moi::backend_kind::sampled_watchpoint;
            options.destructor_reports             = false;
            options.destructor_aborts              = false;
            options.diagnostic_stream              = nullptr;

            std::vector<hip_moi::context::storage_ref> host_refs;
            contexts.reserve(static_cast<std::size_t>(blocks));
            host_refs.reserve(static_cast<std::size_t>(blocks));
            for(int block = 0; block < blocks; ++block)
            {
                contexts.push_back(std::make_unique<hip_moi::host_context>(options));
                contexts.back()->disable_destructor_check();
                host_refs.push_back(contexts.back()->launch_ref());
            }

            HIP_MOI_ALIAS_BENCH_HIP_CHECK(
                hipMalloc(&device_refs, host_refs.size() * sizeof(host_refs[0])));
            HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipMemcpy(device_refs,
                                                    host_refs.data(),
                                                    host_refs.size() * sizeof(host_refs[0]),
                                                    hipMemcpyHostToDevice));
        }

        StorageSet(const StorageSet&)            = delete;
        StorageSet& operator=(const StorageSet&) = delete;

        ~StorageSet()
        {
            (void)hipFree(device_refs);
        }

        Runtime next_runtime()
        {
            Runtime runtime{
                device_refs,
                generation,
            };
            ++generation;
            return runtime;
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

    void fill_fragments(int                   blocks,
                        std::vector<f16x4_t>* q,
                        std::vector<f16x4_t>* k,
                        std::vector<f16x4_t>* v)
    {
        std::size_t fragments = static_cast<std::size_t>(blocks) * kFragmentsPerBlock;
        q->resize(fragments);
        k->resize(fragments);
        v->resize(fragments);
        for(int block = 0; block < blocks; ++block)
        {
            std::size_t base = static_cast<std::size_t>(block) * kFragmentsPerBlock;
            for(int lane = 0; lane < kWaveSize; ++lane)
            {
                int row = lane & 15;
                int col = lane & 15;
                for(int elem = 0; elem < kFragmentElements; ++elem)
                {
                    int k_index = cdna4_mfma_f16_k(lane, elem);
                    (*q)[base + lane][elem]
                        = static_cast<half_t>(input_value(row, k_index, /*seed=*/1));
                    (*k)[base + lane][elem]
                        = static_cast<half_t>(input_value(k_index, col, /*seed=*/2));
                    (*v)[base + lane][elem]
                        = static_cast<half_t>(input_value(k_index, col, /*seed=*/3));
                }
            }
        }
    }

    float quantized_input_value(int row, int col, int seed)
    {
        return static_cast<float>(static_cast<half_t>(input_value(row, col, seed)));
    }

    std::vector<float> host_reference(int blocks)
    {
        std::vector<float> expected(static_cast<std::size_t>(blocks) * kOutputPerBlock, 0.0f);
        for(int block = 0; block < blocks; ++block)
        {
            std::size_t base = static_cast<std::size_t>(block) * kOutputPerBlock;
            for(int m = 0; m < kTileM; ++m)
            {
                for(int n = 0; n < kTileN; ++n)
                {
                    float acc = 0.0f;
                    for(int k = 0; k < kKTile; ++k)
                    {
                        acc += quantized_input_value(m, k, /*seed=*/1)
                               * quantized_input_value(k, n, /*seed=*/2);
                    }
                    expected[base + m * kTileN + n] = acc;
                }
            }
        }
        return expected;
    }

    float host_checksum_reference()
    {
        return quantized_input_value(/*row=*/0, /*col=*/0, /*seed=*/3);
    }

    void copy_inputs(const std::vector<f16x4_t>& q,
                     const std::vector<f16x4_t>& k,
                     const std::vector<f16x4_t>& v,
                     DeviceBuffers*              buffers)
    {
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(
            hipMemcpy(buffers->q, q.data(), q.size() * sizeof(q[0]), hipMemcpyHostToDevice));
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(
            hipMemcpy(buffers->k, k.data(), k.size() * sizeof(k[0]), hipMemcpyHostToDevice));
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(
            hipMemcpy(buffers->v, v.data(), v.size() * sizeof(v[0]), hipMemcpyHostToDevice));
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipMemset(
            buffers->out, 0, q.size() / kFragmentsPerBlock * kOutputPerBlock * sizeof(float)));
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(
            hipMemset(buffers->checksum, 0, q.size() / kFragmentsPerBlock * sizeof(float)));
    }

    template <typename Policy>
    void launch_kernel(const DeviceBuffers& buffers, int blocks, Runtime runtime)
    {
        hipLaunchKernelGGL((mfma_attention_alias_handoff_kernel<Policy>),
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           buffers.q,
                           buffers.k,
                           buffers.v,
                           buffers.out,
                           buffers.checksum,
                           runtime);
    }

    template <typename Policy>
    double
        run_launches(const DeviceBuffers& buffers, int blocks, StorageSet* storage, int iterations)
    {
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipEventCreate(&start));
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipEventCreate(&stop));
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipEventRecord(start, 0));
        for(int i = 0; i < iterations; ++i)
        {
            Runtime runtime{};
            if(storage)
            {
                runtime = storage->next_runtime();
            }
            launch_kernel<Policy>(buffers, blocks, runtime);
        }
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipEventRecord(stop, 0));
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipEventDestroy(stop));
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipEventDestroy(start));
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipGetLastError());
        return static_cast<double>(elapsed_ms);
    }

    template <typename Policy>
    void warmup_for(const DeviceBuffers& buffers, int blocks, StorageSet* storage, int warmup_ms)
    {
        int iterations = 1;
        while(true)
        {
            double elapsed = run_launches<Policy>(buffers, blocks, storage, iterations);
            if(elapsed >= static_cast<double>(warmup_ms))
            {
                return;
            }
            iterations *= 2;
        }
    }

    template <typename Policy>
    double
        benchmark_for(const DeviceBuffers& buffers, int blocks, StorageSet* storage, int measure_ms)
    {
        int iterations = 1;
        while(true)
        {
            double elapsed = run_launches<Policy>(buffers, blocks, storage, iterations);
            if(elapsed >= static_cast<double>(measure_ms))
            {
                return elapsed / static_cast<double>(iterations);
            }
            iterations *= 2;
        }
    }

    template <typename Policy>
    void benchmark_mode(const char*          name,
                        const DeviceBuffers& buffers,
                        int                  blocks,
                        StorageSet*          storage,
                        int                  warmup_ms,
                        int                  measure_ms)
    {
        warmup_for<Policy>(buffers, blocks, storage, warmup_ms);
        double         latency_ms = benchmark_for<Policy>(buffers, blocks, storage, measure_ms);
        DisplayLatency display    = display_latency(latency_ms);
        std::printf("%-64s %.3g %s\n", name, display.value, display.unit);
    }

    void
        verify_outputs(const DeviceBuffers& buffers, int blocks, const std::vector<float>& expected)
    {
        std::vector<float> actual(expected.size(), 0.0f);
        std::vector<float> checksum(static_cast<std::size_t>(blocks), 0.0f);
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipMemcpy(
            actual.data(), buffers.out, actual.size() * sizeof(actual[0]), hipMemcpyDeviceToHost));
        HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipMemcpy(checksum.data(),
                                                buffers.checksum,
                                                checksum.size() * sizeof(checksum[0]),
                                                hipMemcpyDeviceToHost));

        for(std::size_t i = 0; i < actual.size(); ++i)
        {
            if(std::fabs(actual[i] - expected[i]) > 1.0e-4f)
            {
                std::fprintf(stderr,
                             "output mismatch at element %zu: got %.9g expected %.9g\n",
                             i,
                             actual[i],
                             expected[i]);
                std::exit(EXIT_FAILURE);
            }
        }

        float expected_checksum = host_checksum_reference();
        for(int block = 0; block < blocks; ++block)
        {
            float value = checksum[static_cast<std::size_t>(block)];
            if(std::fabs(value - expected_checksum) > 1.0e-4f)
            {
                std::fprintf(stderr,
                             "checksum mismatch at block %d: got %.9g expected %.9g\n",
                             block,
                             value,
                             expected_checksum);
                std::exit(EXIT_FAILURE);
            }
        }
    }
} // namespace

int main()
{
    int blocks     = get_int_env("HIP_MOI_MFMA_ALIAS_HANDOFF_BENCH_BLOCKS", kDefaultBlocks);
    int warmup_ms  = get_int_env("HIP_MOI_MFMA_ALIAS_HANDOFF_BENCH_WARMUP_MS", kDefaultWarmupMs);
    int measure_ms = get_int_env("HIP_MOI_MFMA_ALIAS_HANDOFF_BENCH_MEASURE_MS", kDefaultMeasureMs);
    int watchpoints
        = get_int_env("HIP_MOI_MFMA_ALIAS_HANDOFF_BENCH_WATCHPOINTS", kDefaultWatchpoints);

    std::vector<f16x4_t> q;
    std::vector<f16x4_t> k;
    std::vector<f16x4_t> v;
    fill_fragments(blocks, &q, &k, &v);
    std::vector<float> expected = host_reference(blocks);

    DeviceBuffers buffers(blocks);
    copy_inputs(q, k, v, &buffers);

    StorageSet context_storage(blocks, watchpoints);
    StorageSet fast_storage(blocks, watchpoints);

    launch_kernel<PassThroughPolicy>(buffers, blocks, Runtime{});
    HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers, blocks, expected);

    launch_kernel<ContextSampledPolicy>(buffers, blocks, context_storage.next_runtime());
    HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers, blocks, expected);

    launch_kernel<SampledWatchpointContextPolicy>(buffers, blocks, fast_storage.next_runtime());
    HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_ALIAS_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers, blocks, expected);

    std::printf("hip-moi CDNA4 MFMA attention LDS alias handoff benchmark: "
                "blocks=%d warmup_ms=%d min_ms=%d watchpoints=%d\n",
                blocks,
                warmup_ms,
                measure_ms,
                watchpoints);
    benchmark_mode<PassThroughPolicy>("cdna4-mfma-attention-lds-alias-handoff_pass-through",
                                      buffers,
                                      blocks,
                                      nullptr,
                                      warmup_ms,
                                      measure_ms);
    benchmark_mode<ContextSampledPolicy>(
        "cdna4-mfma-attention-lds-alias-handoff_context+sampled-watchpoint",
        buffers,
        blocks,
        &context_storage,
        warmup_ms,
        measure_ms);
    benchmark_mode<SampledWatchpointContextPolicy>(
        "cdna4-mfma-attention-lds-alias-handoff_sampled-watchpoint-context",
        buffers,
        blocks,
        &fast_storage,
        warmup_ms,
        measure_ms);

    if(!context_storage.check_all() || !fast_storage.check_all())
    {
        std::fprintf(stderr, "unexpected hip-moi diagnostic\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
