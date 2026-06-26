// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// benchmarks/029_rdna4_wmma_streamk_tree_atomic_or_benchmark.hip
//
// RDNA4 WMMA Stream-K tree atomicOr benchmark. Four subgroups compute four
// K-slice partials for one C tile. The first three subgroups publish LDS
// partials by setting bits in a global bitmask, and the final subgroup observes
// those bits before folding all partials.
#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace
{
    using half_t  = _Float16;
    using f16x8_t = half_t __attribute__((ext_vector_type(8)));
    using f32x8_t = float __attribute__((ext_vector_type(8)));

    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 4;
    constexpr int kFinalSubgroup      = kSubgroupCount - 1;
    constexpr int kThreadCount        = kThreadsPerSubgroup * kSubgroupCount;
    constexpr int kFragmentElements   = 8;
    constexpr int kTileM              = 16;
    constexpr int kTileN              = 16;
    constexpr int kKSlice             = 16;
    constexpr int kCElementCount      = kTileM * kTileN;
    constexpr int kDefaultBlocks      = 256;
    constexpr int kDefaultWarmupMs    = 100;
    constexpr int kDefaultMeasureMs   = 100;

    constexpr std::size_t kInputFragmentCount = kSubgroupCount * kThreadsPerSubgroup;
    constexpr std::size_t kInputBytes         = kInputFragmentCount * sizeof(f16x8_t);

    constexpr uint32_t kASharedByteOffset = 0;
    constexpr uint32_t kBSharedByteOffset
        = kASharedByteOffset
          + static_cast<uint32_t>(kSubgroupCount * kThreadsPerSubgroup * sizeof(f16x8_t));
    constexpr uint32_t kPartialsSharedByteOffset
        = kBSharedByteOffset
          + static_cast<uint32_t>(kSubgroupCount * kThreadsPerSubgroup * sizeof(f16x8_t));
    constexpr int kSharedBytes
        = static_cast<int>(kPartialsSharedByteOffset
                           + kSubgroupCount * kThreadsPerSubgroup * kFragmentElements
                                 * sizeof(float));
    constexpr int kExactShadowEntries
        = (kSharedBytes + static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes) - 1)
          / static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes);

    constexpr hip_moi::site_id kAStoreSite(0x1f01);
    constexpr hip_moi::site_id kBStoreSite(0x1f02);
    constexpr hip_moi::site_id kALoadSite(0x1f03);
    constexpr hip_moi::site_id kBLoadSite(0x1f04);
    constexpr hip_moi::site_id kPartialStoreSite(0x1f05);
    constexpr hip_moi::site_id kPartialLoadSite(0x1f06);
    constexpr hip_moi::site_id kBitmaskResetSite(0x1f07);
    constexpr hip_moi::site_id kBitmaskSite(0x1f08);

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

#define HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(expr)                                  \
    do                                                                              \
    {                                                                               \
        hipError_t hip_moi_wmma_tree_or_bench_status = (expr);                      \
        if(hip_moi_wmma_tree_or_bench_status != hipSuccess)                         \
        {                                                                           \
            fail_hip(hip_moi_wmma_tree_or_bench_status, #expr, __FILE__, __LINE__); \
        }                                                                           \
    } while(false)

    int get_int_env(const char* name, int default_value)
    {
        const char* value = std::getenv(name);
        return value ? std::atoi(value) : default_value;
    }

    __host__ __device__ int rdna4_wmma_acc_m(int lane, int elem)
    {
        return elem + 8 * (lane >> 4);
    }

    __host__ __device__ int subgroup_bit(int subgroup)
    {
        return 1 << subgroup;
    }

    __host__ __device__ int producer_bits()
    {
        return (1 << kFinalSubgroup) - 1;
    }

    __host__ __device__ uint32_t a_fragment_offset(int subgroup, int lane)
    {
        return kASharedByteOffset
               + static_cast<uint32_t>((subgroup * kThreadsPerSubgroup + lane)
                                       * sizeof(f16x8_t));
    }

    __host__ __device__ uint32_t b_fragment_offset(int subgroup, int lane)
    {
        return kBSharedByteOffset
               + static_cast<uint32_t>((subgroup * kThreadsPerSubgroup + lane)
                                       * sizeof(f16x8_t));
    }

    __host__ __device__ uint32_t partial_offset(int subgroup, int lane, int elem)
    {
        return kPartialsSharedByteOffset
               + static_cast<uint32_t>(
                   ((subgroup * kThreadsPerSubgroup + lane) * kFragmentElements + elem)
                   * sizeof(float));
    }

    __host__ __device__ float expected_c_value()
    {
        float value = 0.0f;
        for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
        {
            value += static_cast<float>((subgroup + 1) * kKSlice);
        }
        return value;
    }

    __device__ f32x8_t zero_f32x8()
    {
        f32x8_t value;
#pragma unroll
        for(int elem = 0; elem < kFragmentElements; ++elem)
        {
            value[elem] = 0.0f;
        }
        return value;
    }

    __global__ void wmma_streamk_tree_atomic_or_pass_through(
        const f16x8_t* __restrict__ a_fragments,
        const f16x8_t* __restrict__ b_fragments,
        int* __restrict__ bitmasks,
        float* __restrict__ out)
    {
        __shared__ f16x8_t a_shared[kSubgroupCount][kThreadsPerSubgroup];
        __shared__ f16x8_t b_shared[kSubgroupCount][kThreadsPerSubgroup];
        __shared__ float   partials[kSubgroupCount][kThreadsPerSubgroup][kFragmentElements];

        int block    = static_cast<int>(blockIdx.x);
        int tid      = static_cast<int>(threadIdx.x);
        int subgroup = tid / kThreadsPerSubgroup;
        int lane     = tid & (kThreadsPerSubgroup - 1);

        if(tid == 0)
        {
            __hip_atomic_store(&bitmasks[block], 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();

        a_shared[subgroup][lane] = a_fragments[subgroup * kThreadsPerSubgroup + lane];
        b_shared[subgroup][lane] = b_fragments[subgroup * kThreadsPerSubgroup + lane];

        __syncthreads();

        f16x8_t a_fragment = a_shared[subgroup][lane];
        f16x8_t b_fragment = b_shared[subgroup][lane];
        f32x8_t acc        = zero_f32x8();
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_fragment, b_fragment, acc);

#pragma unroll
        for(int elem = 0; elem < kFragmentElements; ++elem)
        {
            partials[subgroup][lane][elem] = acc[elem];
        }

        int old_mask = 0;
        if(subgroup < kFinalSubgroup)
        {
            if(lane == 0)
            {
                (void)__hip_atomic_fetch_or(&bitmasks[block],
                                            subgroup_bit(subgroup),
                                            __ATOMIC_RELEASE,
                                            __HIP_MEMORY_SCOPE_AGENT);
            }
        }
        else
        {
            if(lane == 0)
            {
                while((__hip_atomic_load(&bitmasks[block],
                                         __ATOMIC_RELAXED,
                                         __HIP_MEMORY_SCOPE_AGENT)
                       & producer_bits())
                      != producer_bits())
                {
                }
                old_mask = __hip_atomic_fetch_or(&bitmasks[block],
                                                 subgroup_bit(subgroup),
                                                 __ATOMIC_ACQ_REL,
                                                 __HIP_MEMORY_SCOPE_AGENT);
            }

            uint32_t old_mask_bits
                = __builtin_amdgcn_ds_bpermute(/*byte_offset=*/0,
                                               static_cast<uint32_t>(old_mask));
            if((static_cast<int>(old_mask_bits) & producer_bits()) == producer_bits())
            {
#pragma unroll
                for(int elem = 0; elem < kFragmentElements; ++elem)
                {
                    float value = 0.0f;
#pragma unroll
                    for(int partial_subgroup = 0; partial_subgroup < kSubgroupCount;
                        ++partial_subgroup)
                    {
                        value += partials[partial_subgroup][lane][elem];
                    }
                    int row                                         = rdna4_wmma_acc_m(lane, elem);
                    int column                                      = lane & 15;
                    out[block * kCElementCount + row * kTileN + column] = value;
                }
            }
        }
    }

    __global__ void wmma_streamk_tree_atomic_or_context(
        hip_moi::context::storage_ref* storage_refs,
        uint64_t                       generation,
        const f16x8_t* __restrict__ a_fragments,
        const f16x8_t* __restrict__ b_fragments,
        int* __restrict__ bitmasks,
        float* __restrict__ out)
    {
        __shared__ f16x8_t a_shared[kSubgroupCount][kThreadsPerSubgroup];
        __shared__ f16x8_t b_shared[kSubgroupCount][kThreadsPerSubgroup];
        __shared__ float   partials[kSubgroupCount][kThreadsPerSubgroup][kFragmentElements];

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
            ctx.atomic_store(&bitmasks[block],
                             0,
                             hip_moi::atomic_memory_order::relaxed,
                             hip_moi::atomic_memory_scope::agent,
                             kBitmaskResetSite);
        }
        ctx.syncthreads();

        ctx.lds_store_at(&a_shared[subgroup][lane],
                         a_fragments[subgroup * kThreadsPerSubgroup + lane],
                         a_fragment_offset(subgroup, lane),
                         kAStoreSite);
        ctx.lds_store_at(&b_shared[subgroup][lane],
                         b_fragments[subgroup * kThreadsPerSubgroup + lane],
                         b_fragment_offset(subgroup, lane),
                         kBStoreSite);

        ctx.syncthreads();

        f16x8_t a_fragment = ctx.lds_load_at(&a_shared[subgroup][lane],
                                             a_fragment_offset(subgroup, lane),
                                             kALoadSite);
        f16x8_t b_fragment = ctx.lds_load_at(&b_shared[subgroup][lane],
                                             b_fragment_offset(subgroup, lane),
                                             kBLoadSite);
        f32x8_t acc        = zero_f32x8();
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_fragment, b_fragment, acc);

#pragma unroll
        for(int elem = 0; elem < kFragmentElements; ++elem)
        {
            ctx.lds_store_at(&partials[subgroup][lane][elem],
                             acc[elem],
                             partial_offset(subgroup, lane, elem),
                             kPartialStoreSite);
        }

        int old_mask = 0;
        if(subgroup < kFinalSubgroup)
        {
            if(lane == 0)
            {
                (void)ctx.atomic_fetch_or(&bitmasks[block],
                                          subgroup_bit(subgroup),
                                          hip_moi::atomic_memory_order::release,
                                          hip_moi::atomic_memory_scope::agent,
                                          kBitmaskSite);
            }
        }
        else
        {
            if(lane == 0)
            {
                while((__hip_atomic_load(&bitmasks[block],
                                         __ATOMIC_RELAXED,
                                         __HIP_MEMORY_SCOPE_AGENT)
                       & producer_bits())
                      != producer_bits())
                {
                }
                old_mask = ctx.atomic_fetch_or(&bitmasks[block],
                                               subgroup_bit(subgroup),
                                               hip_moi::atomic_memory_order::acq_rel,
                                               hip_moi::atomic_memory_scope::agent,
                                               kBitmaskSite);
            }

            uint32_t old_mask_bits
                = __builtin_amdgcn_ds_bpermute(/*byte_offset=*/0,
                                               static_cast<uint32_t>(old_mask));
            if((static_cast<int>(old_mask_bits) & producer_bits()) == producer_bits())
            {
#pragma unroll
                for(int elem = 0; elem < kFragmentElements; ++elem)
                {
                    float value = 0.0f;
#pragma unroll
                    for(int partial_subgroup = 0; partial_subgroup < kSubgroupCount;
                        ++partial_subgroup)
                    {
                        value += ctx.lds_load_at(&partials[partial_subgroup][lane][elem],
                                                 partial_offset(partial_subgroup, lane, elem),
                                                 kPartialLoadSite);
                    }
                    int row                                         = rdna4_wmma_acc_m(lane, elem);
                    int column                                      = lane & 15;
                    out[block * kCElementCount + row * kTileN + column] = value;
                }
            }
        }
    }

    struct DeviceBuffers
    {
        f16x8_t* a        = nullptr;
        f16x8_t* b        = nullptr;
        int*     bitmasks = nullptr;
        float*   out      = nullptr;

        explicit DeviceBuffers(int blocks)
        {
            std::vector<f16x8_t> host_a(kInputFragmentCount);
            std::vector<f16x8_t> host_b(kInputFragmentCount);
            for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
            {
                for(int lane = 0; lane < kThreadsPerSubgroup; ++lane)
                {
                    f16x8_t a_fragment;
                    f16x8_t b_fragment;
#pragma unroll
                    for(int elem = 0; elem < kFragmentElements; ++elem)
                    {
                        a_fragment[elem] = static_cast<half_t>(subgroup + 1);
                        b_fragment[elem] = static_cast<half_t>(1);
                    }
                    host_a[static_cast<std::size_t>(subgroup * kThreadsPerSubgroup + lane)]
                        = a_fragment;
                    host_b[static_cast<std::size_t>(subgroup * kThreadsPerSubgroup + lane)]
                        = b_fragment;
                }
            }

            HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(hipMalloc(&a, kInputBytes));
            HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(hipMalloc(&b, kInputBytes));
            HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(
                hipMalloc(&bitmasks, blocks * sizeof(int)));
            HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(
                hipMalloc(&out, blocks * kCElementCount * sizeof(float)));
            HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(
                hipMemcpy(a, host_a.data(), kInputBytes, hipMemcpyHostToDevice));
            HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(
                hipMemcpy(b, host_b.data(), kInputBytes, hipMemcpyHostToDevice));
        }

        DeviceBuffers(const DeviceBuffers&)            = delete;
        DeviceBuffers& operator=(const DeviceBuffers&) = delete;

        ~DeviceBuffers()
        {
            (void)hipFree(a);
            (void)hipFree(b);
            (void)hipFree(bitmasks);
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
            options.storage_bytes               = 512 * 1024;
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

            HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(
                hipMalloc(&device_refs, host_refs.size() * sizeof(host_refs[0])));
            HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(
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
        HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(hipEventCreate(&start));
        HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(hipEventCreate(&stop));
        HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(hipEventRecord(start, 0));
        for(int i = 0; i < iterations; ++i)
        {
            launch();
        }
        HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(hipEventRecord(stop, 0));
        HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(hipEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(hipEventDestroy(stop));
        HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(hipEventDestroy(start));
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

    void verify_outputs(const float* device_output, int blocks)
    {
        std::vector<float> host_output(static_cast<std::size_t>(blocks * kCElementCount), 0.0f);
        HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(
            hipMemcpy(host_output.data(),
                      device_output,
                      host_output.size() * sizeof(host_output[0]),
                      hipMemcpyDeviceToHost));
        for(int block = 0; block < blocks; ++block)
        {
            for(int element = 0; element < kCElementCount; ++element)
            {
                int   offset = block * kCElementCount + element;
                float actual = host_output[static_cast<std::size_t>(offset)];
                if(actual != expected_c_value())
                {
                    std::fprintf(stderr,
                                 "output mismatch at block %d element %d: got %g expected %g\n",
                                 block,
                                 element,
                                 static_cast<double>(actual),
                                 static_cast<double>(expected_c_value()));
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    }
} // namespace

int main()
{
    int blocks     = get_int_env("HIP_MOI_WMMA_TREE_OR_BENCH_BLOCKS", kDefaultBlocks);
    int warmup_ms  = get_int_env("HIP_MOI_WMMA_TREE_OR_BENCH_WARMUP_MS", kDefaultWarmupMs);
    int measure_ms = get_int_env("HIP_MOI_WMMA_TREE_OR_BENCH_MEASURE_MS", kDefaultMeasureMs);

    DeviceBuffers    buffers(blocks);
    ContextStorageSet context_storage(blocks);
    uint64_t          generation = 1;

    auto pass_through_launch = [&]() {
        hipLaunchKernelGGL(wmma_streamk_tree_atomic_or_pass_through,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           buffers.a,
                           buffers.b,
                           buffers.bitmasks,
                           buffers.out);
    };
    auto context_launch = [&]() {
        ++generation;
        hipLaunchKernelGGL(wmma_streamk_tree_atomic_or_context,
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           context_storage.device_refs,
                           generation,
                           buffers.a,
                           buffers.b,
                           buffers.bitmasks,
                           buffers.out);
    };

    pass_through_launch();
    HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    context_launch();
    HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(hipGetLastError());
    HIP_MOI_WMMA_TREE_OR_BENCH_HIP_CHECK(hipDeviceSynchronize());
    verify_outputs(buffers.out, blocks);

    std::printf(
        "hip-moi RDNA4 WMMA Stream-K-tree atomicOr benchmark: blocks=%d warmup_ms=%d min_ms=%d\n",
        blocks,
        warmup_ms,
        measure_ms);
    benchmark_mode("rdna4-wmma-streamk-tree-atomic-or_pass-through",
                   pass_through_launch,
                   warmup_ms,
                   measure_ms);
    benchmark_mode(
        "rdna4-wmma-streamk-tree-atomic-or_context", context_launch, warmup_ms, measure_ms);

    if(!context_storage.check_all(stderr))
    {
        std::fprintf(stderr, "unexpected hip-moi diagnostic\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
