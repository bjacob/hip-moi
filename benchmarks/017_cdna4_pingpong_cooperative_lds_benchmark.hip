// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// benchmarks/017_cdna4_pingpong_cooperative_lds_benchmark.hip
//
// CDNA4 ping-pong-shaped benchmark with real cross-subgroup LDS sharing. Two
// subgroups form one cooperating pair; subgroup 0 stages B fragments in LDS
// and both subgroups consume them.
#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace
{
    using half_t  = _Float16;
    using f16x4_t = half_t __attribute__((ext_vector_type(4)));
    using f32x4_t = float __attribute__((ext_vector_type(4)));

    constexpr int kWaveSize           = 64;
    constexpr int kSubgroupCount      = 2;
    constexpr int kSubgroupPairCount  = kSubgroupCount / 2;
    constexpr int kThreadCount        = kWaveSize * kSubgroupCount;
    constexpr int kThreadsPerSubgroup = kWaveSize;
    constexpr int kTileM              = 16;
    constexpr int kTileN              = 16;
    constexpr int kKTile              = 16;
    constexpr int kKTileCount         = 4;
    constexpr int kBufferCount        = 2;
    constexpr int kFragmentElements   = 4;
    constexpr int kDefaultBlocks      = 256;

    constexpr uint32_t kBSharedByteOffset
        = kBufferCount * kSubgroupCount * kWaveSize * sizeof(f16x4_t);

    constexpr hip_moi::site_id kAStoreSite(0x1701);
    constexpr hip_moi::site_id kBStoreSite(0x1702);
    constexpr hip_moi::site_id kALoadSite(0x1703);
    constexpr hip_moi::site_id kBLoadSite(0x1704);

    using FastPublishOnlyPolicy = hip_moi::sampled_watchpoint_policy<
        /*SampleSkip=*/1,
        /*ProbeCount=*/1,
        /*DelayIters=*/0,
        /*ReportConflicts=*/false>;

    enum class Mode
    {
        CooperativePassThrough,
        CooperativeContextSampled,
        CooperativeSampled,
        Benchmark,
    };

    struct DeviceBuffers
    {
        f16x4_t* a_fragments = nullptr;
        f16x4_t* b_fragments = nullptr;
        float*   output      = nullptr;

        DeviceBuffers()                                = default;
        DeviceBuffers(const DeviceBuffers&)            = delete;
        DeviceBuffers& operator=(const DeviceBuffers&) = delete;

        DeviceBuffers(DeviceBuffers&& other) noexcept
            : a_fragments(other.a_fragments)
            , b_fragments(other.b_fragments)
            , output(other.output)
        {
            other.a_fragments = nullptr;
            other.b_fragments = nullptr;
            other.output      = nullptr;
        }

        DeviceBuffers& operator=(DeviceBuffers&& other) noexcept
        {
            if(this != &other)
            {
                (void)hipFree(a_fragments);
                (void)hipFree(b_fragments);
                (void)hipFree(output);
                a_fragments       = other.a_fragments;
                b_fragments       = other.b_fragments;
                output            = other.output;
                other.a_fragments = nullptr;
                other.b_fragments = nullptr;
                other.output      = nullptr;
            }
            return *this;
        }

        ~DeviceBuffers()
        {
            (void)hipFree(a_fragments);
            (void)hipFree(b_fragments);
            (void)hipFree(output);
        }
    };

    struct HipMoiStorageRefs
    {
        hip_moi::context::storage_ref* refs = nullptr;

        HipMoiStorageRefs()                                    = default;
        HipMoiStorageRefs(const HipMoiStorageRefs&)            = delete;
        HipMoiStorageRefs& operator=(const HipMoiStorageRefs&) = delete;

        HipMoiStorageRefs(HipMoiStorageRefs&& other) noexcept
            : refs(other.refs)
        {
            other.refs = nullptr;
        }

        HipMoiStorageRefs& operator=(HipMoiStorageRefs&& other) noexcept
        {
            if(this != &other)
            {
                (void)hipFree(refs);
                refs       = other.refs;
                other.refs = nullptr;
            }
            return *this;
        }

        ~HipMoiStorageRefs()
        {
            (void)hipFree(refs);
        }
    };

    __device__ hip_moi::context::storage_ref* g_hip_moi_storage_refs  = nullptr;
    __device__ uint64_t                       g_hip_moi_storage_count = 0;

    [[noreturn]] void fail_hip(hipError_t status, const char* expr, const char* file, int line)
    {
        std::fprintf(
            stderr, "%s:%d: HIP error from %s: %s\n", file, line, expr, hipGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }

#define HIP_MOI_ATT_HIP_CHECK(expr)                                  \
    do                                                               \
    {                                                                \
        hipError_t hip_moi_att_status = (expr);                      \
        if(hip_moi_att_status != hipSuccess)                         \
        {                                                            \
            fail_hip(hip_moi_att_status, #expr, __FILE__, __LINE__); \
        }                                                            \
    } while(false)

    __host__ __device__ float input_value(int block, int row, int col, int seed)
    {
        int value = (block * 3 + row * 7 + col * 5 + seed * 13 + (row ^ (col * 3))) % 9;
        return static_cast<float>(value - 4) * 0.125f;
    }

    __host__ __device__ int cdna4_mfma_f16_k(int lane, int elem)
    {
        return elem + 4 * (lane >> 4);
    }

    __host__ __device__ int cdna4_mfma_acc_m(int lane, int elem)
    {
        return elem + 4 * (lane >> 4);
    }

    __host__ __device__ std::size_t a_fragment_index(int block, int subgroup, int k_tile, int lane)
    {
        return (((static_cast<std::size_t>(block) * kKTileCount + k_tile) * kSubgroupCount
                 + subgroup)
                * kWaveSize)
               + lane;
    }

    __host__ __device__ std::size_t b_fragment_index(int block, int pair, int k_tile, int lane)
    {
        return (((static_cast<std::size_t>(block) * kKTileCount + k_tile) * kSubgroupPairCount
                 + pair)
                * kWaveSize)
               + lane;
    }

    __host__ __device__ std::size_t output_index(int block, int subgroup, int row, int col)
    {
        return ((static_cast<std::size_t>(block) * kSubgroupCount + subgroup) * kTileM + row)
                   * kTileN
               + col;
    }

    __host__ __device__ uint32_t a_shared_offset(int buffer, int subgroup, int lane)
    {
        return static_cast<uint32_t>(((buffer * kSubgroupCount + subgroup) * kWaveSize + lane)
                                     * sizeof(f16x4_t));
    }

    __host__ __device__ uint32_t b_shared_offset(int buffer, int pair, int lane)
    {
        return kBSharedByteOffset
               + static_cast<uint32_t>(((buffer * kSubgroupPairCount + pair) * kWaveSize + lane)
                                       * sizeof(f16x4_t));
    }

    __device__ __forceinline__ f32x4_t zero_f32x4()
    {
        f32x4_t value;
#pragma unroll
        for(int i = 0; i < kFragmentElements; ++i)
        {
            value[i] = 0.0f;
        }
        return value;
    }

    __device__ __forceinline__ void set_pingpong_priority(int subgroup, int k_tile)
    {
        // s_setprio is scalar, so make the wave-uniform condition scalar before branching.
        int uniform_subgroup = __builtin_amdgcn_readfirstlane(subgroup);
        if(((uniform_subgroup + k_tile) & 1) == 0)
        {
            __builtin_amdgcn_s_setprio(1);
        }
        else
        {
            __builtin_amdgcn_s_setprio(0);
        }
    }

    __device__ __forceinline__ hip_moi::context::storage_ref current_storage_ref()
    {
        uint64_t                      workgroup_id = static_cast<uint64_t>(blockIdx.x);
        hip_moi::context::storage_ref storage{};
        if(g_hip_moi_storage_refs && workgroup_id < g_hip_moi_storage_count)
        {
            storage = g_hip_moi_storage_refs[workgroup_id];
        }
        return storage;
    }

    __device__ __forceinline__ hip_moi::context make_general_context()
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(current_storage_ref(), cfg);
    }

    __device__ __forceinline__ hip_moi::sampled_watchpoint_context make_sampled_context()
    {
        hip_moi::sampled_watchpoint_context::config cfg{
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
        };
        return hip_moi::make_sampled_watchpoint_context(current_storage_ref(), cfg);
    }

    struct PassThroughPolicy
    {
        struct context_type
        {
        };

        __device__ static context_type make_context()
        {
            return {};
        }

        __device__ static void init_workgroup(context_type&) {}

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

        __device__ static void syncthreads(context_type&)
        {
            __syncthreads();
        }
    };

    struct ContextSampledPolicy
    {
        using context_type = hip_moi::context;

        __device__ static context_type make_context()
        {
            return make_general_context();
        }

        __device__ static void init_workgroup(context_type& ctx)
        {
            ctx.init_workgroup();
        }

        template <typename T>
        __device__ static void
            store(context_type& ctx, T* ptr, T value, uint32_t offset, hip_moi::site_id site)
        {
            ctx.template lds_store_at<hip_moi::backend_kind::sampled_watchpoint,
                                      FastPublishOnlyPolicy>(ptr, value, offset, site);
        }

        template <typename T>
        __device__ static T
            load(context_type& ctx, const T* ptr, uint32_t offset, hip_moi::site_id site)
        {
            return ctx.template lds_load_at<hip_moi::backend_kind::sampled_watchpoint,
                                            FastPublishOnlyPolicy>(ptr, offset, site);
        }

        __device__ static void syncthreads(context_type& ctx)
        {
            ctx.syncthreads();
        }
    };

    struct SampledContextPolicy
    {
        using context_type = hip_moi::sampled_watchpoint_context;

        __device__ static context_type make_context()
        {
            return make_sampled_context();
        }

        __device__ static void init_workgroup(context_type& ctx)
        {
            ctx.init_workgroup();
        }

        template <typename T>
        __device__ static void
            store(context_type& ctx, T* ptr, T value, uint32_t offset, hip_moi::site_id site)
        {
            ctx.template lds_store_at<FastPublishOnlyPolicy>(ptr, value, offset, site);
        }

        template <typename T>
        __device__ static T
            load(context_type& ctx, const T* ptr, uint32_t offset, hip_moi::site_id site)
        {
            return ctx.template lds_load_at<FastPublishOnlyPolicy>(ptr, offset, site);
        }

        __device__ static void syncthreads(context_type& ctx)
        {
            ctx.syncthreads();
        }
    };

    template <typename InstrumentationPolicy>
    __global__ void cooperative_pingpong_kernel(const f16x4_t* __restrict__ a_fragments,
                                                const f16x4_t* __restrict__ b_fragments,
                                                float* __restrict__ out)
    {
        __shared__ f16x4_t a_shared[kBufferCount][kSubgroupCount][kWaveSize];
        __shared__ f16x4_t b_shared[kBufferCount][kSubgroupPairCount][kWaveSize];

        typename InstrumentationPolicy::context_type ctx = InstrumentationPolicy::make_context();
        InstrumentationPolicy::init_workgroup(ctx);

        int tid      = static_cast<int>(threadIdx.x);
        int subgroup = tid / kWaveSize;
        int lane     = tid & (kWaveSize - 1);
        int pair     = subgroup / 2;
        int block    = static_cast<int>(blockIdx.x);

        f32x4_t acc = zero_f32x4();
#pragma unroll
        for(int k_tile = 0; k_tile < kKTileCount; ++k_tile)
        {
            int buffer = k_tile & (kBufferCount - 1);

            set_pingpong_priority(subgroup, k_tile);
            InstrumentationPolicy::store(
                ctx,
                &a_shared[buffer][subgroup][lane],
                a_fragments[a_fragment_index(block, subgroup, k_tile, lane)],
                a_shared_offset(buffer, subgroup, lane),
                kAStoreSite);
            if((subgroup & 1) == 0)
            {
                InstrumentationPolicy::store(
                    ctx,
                    &b_shared[buffer][pair][lane],
                    b_fragments[b_fragment_index(block, pair, k_tile, lane)],
                    b_shared_offset(buffer, pair, lane),
                    kBStoreSite);
            }

            __builtin_amdgcn_sched_barrier(0);
            InstrumentationPolicy::syncthreads(ctx);

            f16x4_t a_fragment
                = InstrumentationPolicy::load(ctx,
                                              &a_shared[buffer][subgroup][lane],
                                              a_shared_offset(buffer, subgroup, lane),
                                              kALoadSite);
            f16x4_t b_fragment = InstrumentationPolicy::load(ctx,
                                                             &b_shared[buffer][pair][lane],
                                                             b_shared_offset(buffer, pair, lane),
                                                             kBLoadSite);

            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(0);
            acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_fragment, b_fragment, acc, 0, 0, 0);
            InstrumentationPolicy::syncthreads(ctx);
        }

        __builtin_amdgcn_s_setprio(0);

        int col = lane & 15;
#pragma unroll
        for(int elem = 0; elem < kFragmentElements; ++elem)
        {
            int row                                      = cdna4_mfma_acc_m(lane, elem);
            out[output_index(block, subgroup, row, col)] = acc[elem];
        }
    }

    int get_int_env(const char* name, int default_value)
    {
        const char* value = std::getenv(name);
        return value ? std::atoi(value) : default_value;
    }

    double get_double_env(const char* name, double default_value)
    {
        const char* value = std::getenv(name);
        return value ? std::atof(value) : default_value;
    }

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

    const char* mode_name(Mode mode)
    {
        switch(mode)
        {
        case Mode::CooperativePassThrough:
            return "cdna4-cooperative-pass-through";
        case Mode::CooperativeContextSampled:
            return "cdna4-cooperative-context-sampled";
        case Mode::CooperativeSampled:
            return "cdna4-cooperative-sampled";
        case Mode::Benchmark:
            return "benchmark";
        }
        return "unknown";
    }

    Mode parse_mode(const char* mode)
    {
        if(std::strcmp(mode, "cdna4-cooperative-pass-through") == 0)
        {
            return Mode::CooperativePassThrough;
        }
        if(std::strcmp(mode, "cdna4-cooperative-context-sampled") == 0)
        {
            return Mode::CooperativeContextSampled;
        }
        if(std::strcmp(mode, "cdna4-cooperative-sampled") == 0)
        {
            return Mode::CooperativeSampled;
        }
        if(std::strcmp(mode, "benchmark") == 0)
        {
            return Mode::Benchmark;
        }

        std::fprintf(stderr,
                     "usage: cdna4_pingpong_cooperative_lds_benchmark "
                     "[cdna4-cooperative-pass-through|cdna4-cooperative-context-sampled|cdna4-"
                     "cooperative-sampled|benchmark]\n");
        std::exit(EXIT_FAILURE);
    }

    void fill_fragments(int                   blocks,
                        std::vector<f16x4_t>* a_fragments,
                        std::vector<f16x4_t>* b_fragments)
    {
        std::size_t a_fragment_count
            = static_cast<std::size_t>(blocks) * kKTileCount * kSubgroupCount * kWaveSize;
        std::size_t b_fragment_count
            = static_cast<std::size_t>(blocks) * kKTileCount * kSubgroupPairCount * kWaveSize;
        a_fragments->resize(a_fragment_count);
        b_fragments->resize(b_fragment_count);
        for(int block = 0; block < blocks; ++block)
        {
            for(int k_tile = 0; k_tile < kKTileCount; ++k_tile)
            {
                for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
                {
                    for(int lane = 0; lane < kWaveSize; ++lane)
                    {
                        int row = subgroup * kTileM + (lane & 15);
                        for(int elem = 0; elem < kFragmentElements; ++elem)
                        {
                            int k = k_tile * kKTile + cdna4_mfma_f16_k(lane, elem);
                            (*a_fragments)[a_fragment_index(block, subgroup, k_tile, lane)][elem]
                                = static_cast<half_t>(input_value(block, row, k, /*seed=*/1));
                        }
                    }
                }

                for(int pair = 0; pair < kSubgroupPairCount; ++pair)
                {
                    for(int lane = 0; lane < kWaveSize; ++lane)
                    {
                        int col = lane & 15;
                        for(int elem = 0; elem < kFragmentElements; ++elem)
                        {
                            int k = k_tile * kKTile + cdna4_mfma_f16_k(lane, elem);
                            (*b_fragments)[b_fragment_index(block, pair, k_tile, lane)][elem]
                                = static_cast<half_t>(
                                    input_value(block, k, pair * kTileN + col, /*seed=*/2));
                        }
                    }
                }
            }
        }
    }

    DeviceBuffers copy_inputs(const std::vector<f16x4_t>& a_fragments,
                              const std::vector<f16x4_t>& b_fragments,
                              int                         blocks)
    {
        DeviceBuffers buffers;
        std::size_t   a_fragment_bytes = a_fragments.size() * sizeof(a_fragments[0]);
        std::size_t   b_fragment_bytes = b_fragments.size() * sizeof(b_fragments[0]);
        std::size_t   output_bytes
            = static_cast<std::size_t>(blocks) * kSubgroupCount * kTileM * kTileN * sizeof(float);

        HIP_MOI_ATT_HIP_CHECK(hipMalloc(&buffers.a_fragments, a_fragment_bytes));
        HIP_MOI_ATT_HIP_CHECK(hipMalloc(&buffers.b_fragments, b_fragment_bytes));
        HIP_MOI_ATT_HIP_CHECK(hipMalloc(&buffers.output, output_bytes));
        HIP_MOI_ATT_HIP_CHECK(hipMemcpy(
            buffers.a_fragments, a_fragments.data(), a_fragment_bytes, hipMemcpyHostToDevice));
        HIP_MOI_ATT_HIP_CHECK(hipMemcpy(
            buffers.b_fragments, b_fragments.data(), b_fragment_bytes, hipMemcpyHostToDevice));
        HIP_MOI_ATT_HIP_CHECK(hipMemset(buffers.output, 0, output_bytes));
        return buffers;
    }

    HipMoiStorageRefs
        install_storage_refs(int                                                  blocks,
                             std::vector<std::unique_ptr<hip_moi::host_context>>* contexts)
    {
        HipMoiStorageRefs             refs;
        hip_moi::host_context_options options = hip_moi::make_one_watchpoint_publish_only_options();
        options.destructor_reports            = false;
        options.destructor_aborts             = false;
        options.diagnostic_stream             = nullptr;
        options.subgroup_capacity             = kSubgroupCount;

        std::vector<hip_moi::context::storage_ref> host_refs;
        host_refs.reserve(static_cast<std::size_t>(blocks));
        contexts->reserve(static_cast<std::size_t>(blocks));
        for(int i = 0; i < blocks; ++i)
        {
            auto context = std::make_unique<hip_moi::host_context>(options);
            context->disable_destructor_check();
            host_refs.push_back(context->launch_ref());
            contexts->push_back(std::move(context));
        }

        HIP_MOI_ATT_HIP_CHECK(
            hipMalloc(&refs.refs, host_refs.size() * sizeof(hip_moi::context::storage_ref)));
        HIP_MOI_ATT_HIP_CHECK(hipMemcpy(refs.refs,
                                        host_refs.data(),
                                        host_refs.size() * sizeof(hip_moi::context::storage_ref),
                                        hipMemcpyHostToDevice));

        hip_moi::context::storage_ref* device_refs = refs.refs;
        uint64_t                       count       = static_cast<uint64_t>(blocks);
        HIP_MOI_ATT_HIP_CHECK(hipMemcpyToSymbol(
            HIP_SYMBOL(g_hip_moi_storage_refs), &device_refs, sizeof(device_refs)));
        HIP_MOI_ATT_HIP_CHECK(
            hipMemcpyToSymbol(HIP_SYMBOL(g_hip_moi_storage_count), &count, sizeof(count)));
        return refs;
    }

    double checksum_output(const float* device_output, int blocks)
    {
        std::vector<float> host_output(static_cast<std::size_t>(blocks) * kSubgroupCount * kTileM
                                       * kTileN);
        HIP_MOI_ATT_HIP_CHECK(hipMemcpy(host_output.data(),
                                        device_output,
                                        host_output.size() * sizeof(float),
                                        hipMemcpyDeviceToHost));
        double checksum = 0.0;
        for(float value : host_output)
        {
            checksum += static_cast<double>(value);
        }
        return checksum;
    }

    void launch_kernel(Mode mode, const DeviceBuffers& buffers, int blocks, hipStream_t stream)
    {
        dim3 grid(static_cast<unsigned>(blocks));
        dim3 block(kThreadCount);
        if(mode == Mode::CooperativePassThrough)
        {
            hipLaunchKernelGGL((cooperative_pingpong_kernel<PassThroughPolicy>),
                               grid,
                               block,
                               0,
                               stream,
                               buffers.a_fragments,
                               buffers.b_fragments,
                               buffers.output);
        }
        else if(mode == Mode::CooperativeContextSampled)
        {
            hipLaunchKernelGGL((cooperative_pingpong_kernel<ContextSampledPolicy>),
                               grid,
                               block,
                               0,
                               stream,
                               buffers.a_fragments,
                               buffers.b_fragments,
                               buffers.output);
        }
        else if(mode == Mode::CooperativeSampled)
        {
            hipLaunchKernelGGL((cooperative_pingpong_kernel<SampledContextPolicy>),
                               grid,
                               block,
                               0,
                               stream,
                               buffers.a_fragments,
                               buffers.b_fragments,
                               buffers.output);
        }
        else
        {
            std::fprintf(stderr, "internal error: cannot launch mode=%s\n", mode_name(mode));
            std::exit(EXIT_FAILURE);
        }
    }

    double run_one(const char*          name,
                   Mode                 mode,
                   const DeviceBuffers& buffers,
                   int                  blocks,
                   double               min_ms,
                   double               warmup_ms)
    {
        std::vector<std::unique_ptr<hip_moi::host_context>> contexts;
        HipMoiStorageRefs                                   storage_refs;
        if(mode == Mode::CooperativeContextSampled || mode == Mode::CooperativeSampled)
        {
            storage_refs = install_storage_refs(blocks, &contexts);
        }

        hipStream_t stream = nullptr;
        for(int i = 0; i < 5; ++i)
        {
            launch_kernel(mode, buffers, blocks, stream);
        }
        HIP_MOI_ATT_HIP_CHECK(hipGetLastError());
        HIP_MOI_ATT_HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start;
        hipEvent_t stop;
        HIP_MOI_ATT_HIP_CHECK(hipEventCreate(&start));
        HIP_MOI_ATT_HIP_CHECK(hipEventCreate(&stop));

        int calibration_iterations
            = std::max(1, get_int_env("HIP_MOI_CDNA4_PINGPONG_COOPERATIVE_CALIBRATION_ITERS", 10));
        int max_iterations
            = std::max(1, get_int_env("HIP_MOI_CDNA4_PINGPONG_COOPERATIVE_MAX_ITERS", 100000));

        auto run_timed_iterations = [&](int iteration_count) {
            HIP_MOI_ATT_HIP_CHECK(hipEventRecord(start, stream));
            for(int i = 0; i < iteration_count; ++i)
            {
                launch_kernel(mode, buffers, blocks, stream);
            }
            HIP_MOI_ATT_HIP_CHECK(hipEventRecord(stop, stream));
            HIP_MOI_ATT_HIP_CHECK(hipEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            HIP_MOI_ATT_HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
            HIP_MOI_ATT_HIP_CHECK(hipGetLastError());
            return elapsed_ms;
        };

        auto target_iterations = [&](double target_ms, double estimate_ms) {
            if(target_ms <= 0.0)
            {
                return 0;
            }
            int iterations = std::max(
                1, static_cast<int>(std::ceil(target_ms / std::max(0.001, estimate_ms))));
            return std::min(iterations, max_iterations);
        };

        auto run_until_target = [&](double target_ms, int* iteration_count) {
            if(target_ms <= 0.0 || *iteration_count <= 0)
            {
                *iteration_count = 0;
                return 0.0f;
            }

            float elapsed_ms = 0.0f;
            for(;;)
            {
                elapsed_ms = run_timed_iterations(*iteration_count);
                if(elapsed_ms >= target_ms || *iteration_count == max_iterations)
                {
                    return elapsed_ms;
                }

                double scale = target_ms / std::max(0.001f, elapsed_ms);
                int    next_iterations
                    = static_cast<int>(std::ceil(static_cast<double>(*iteration_count) * scale));
                *iteration_count
                    = std::min(max_iterations, std::max(*iteration_count + 1, next_iterations));
            }
        };

        float calibration_ms = run_timed_iterations(calibration_iterations);
        float one_ms         = calibration_ms / calibration_iterations;

        int    warmup_iterations = target_iterations(warmup_ms, one_ms);
        float  warmup_elapsed_ms = run_until_target(warmup_ms, &warmup_iterations);
        double measurement_guess = warmup_iterations > 0 && warmup_elapsed_ms > 0.0f
                                       ? static_cast<double>(warmup_elapsed_ms) / warmup_iterations
                                       : one_ms;
        int    iterations        = target_iterations(min_ms, measurement_guess);
        float  elapsed_ms        = run_until_target(min_ms, &iterations);

        HIP_MOI_ATT_HIP_CHECK(hipEventDestroy(start));
        HIP_MOI_ATT_HIP_CHECK(hipEventDestroy(stop));

        uint64_t reports = 0;
        for(const auto& context : contexts)
        {
            int count = context->consume_diagnostics(nullptr);
            if(count < 0)
            {
                std::fprintf(stderr, "error: failed to read hip-moi diagnostics\n");
                std::exit(EXIT_FAILURE);
            }
            reports += static_cast<uint64_t>(count);
        }
        if(reports != 0)
        {
            std::printf("%-58s reports=%llu\n", name, static_cast<unsigned long long>(reports));
        }

        double         ms       = elapsed_ms / iterations;
        DisplayLatency latency  = display_latency(ms);
        double         checksum = checksum_output(buffers.output, blocks);
        std::printf("%-58s %#9.3g %s  checksum=%12.6f"
                    "  total=%8.3f ms  iters=%d  warmup=%8.3f ms  warmup_iters=%d\n",
                    name,
                    latency.value,
                    latency.unit,
                    checksum,
                    elapsed_ms,
                    iterations,
                    warmup_elapsed_ms,
                    warmup_iterations);
        return ms;
    }

    void print_device()
    {
        int device = 0;
        HIP_MOI_ATT_HIP_CHECK(hipGetDevice(&device));
        hipDeviceProp_t prop;
        HIP_MOI_ATT_HIP_CHECK(hipGetDeviceProperties(&prop, device));
        std::printf("device %d: %s, gcnArch=%s, CUs=%d\n",
                    device,
                    prop.name,
                    prop.gcnArchName,
                    prop.multiProcessorCount);
    }
}

int main(int argc, char** argv)
{
    Mode   mode      = parse_mode(argc >= 2 ? argv[1] : "cdna4-cooperative-pass-through");
    int    blocks    = get_int_env("HIP_MOI_CDNA4_PINGPONG_COOPERATIVE_BLOCKS", kDefaultBlocks);
    double min_ms    = get_double_env("HIP_MOI_CDNA4_PINGPONG_COOPERATIVE_MIN_MS", 100.0);
    double warmup_ms = get_double_env("HIP_MOI_CDNA4_PINGPONG_COOPERATIVE_WARMUP_MS", min_ms);
    if(blocks <= 0)
    {
        std::fprintf(stderr, "HIP_MOI_CDNA4_PINGPONG_COOPERATIVE_BLOCKS must be positive\n");
        return EXIT_FAILURE;
    }
    if(min_ms <= 0.0)
    {
        std::fprintf(stderr, "HIP_MOI_CDNA4_PINGPONG_COOPERATIVE_MIN_MS must be positive\n");
        return EXIT_FAILURE;
    }
    if(warmup_ms < 0.0)
    {
        std::fprintf(stderr, "HIP_MOI_CDNA4_PINGPONG_COOPERATIVE_WARMUP_MS must be nonnegative\n");
        return EXIT_FAILURE;
    }

    std::vector<f16x4_t> a_fragments;
    std::vector<f16x4_t> b_fragments;
    fill_fragments(blocks, &a_fragments, &b_fragments);
    DeviceBuffers buffers = copy_inputs(a_fragments, b_fragments, blocks);

    if(mode == Mode::Benchmark)
    {
        print_device();
        std::printf("cdna4_pingpong_cooperative_lds: blocks=%d waves=%d pairs=%d k_tiles=%d"
                    " min_ms=%.1f warmup_ms=%.1f\n",
                    blocks,
                    kSubgroupCount,
                    kSubgroupPairCount,
                    kKTileCount,
                    min_ms,
                    warmup_ms);
        run_one("cdna4_pingpong_cooperative_lds_pass_through",
                Mode::CooperativePassThrough,
                buffers,
                blocks,
                min_ms,
                warmup_ms);
        run_one("cdna4_pingpong_cooperative_lds_context_sampled_watchpoint",
                Mode::CooperativeContextSampled,
                buffers,
                blocks,
                min_ms,
                warmup_ms);
        run_one("cdna4_pingpong_cooperative_lds_sampled_watchpoint_context",
                Mode::CooperativeSampled,
                buffers,
                blocks,
                min_ms,
                warmup_ms);
        return EXIT_SUCCESS;
    }

    std::vector<std::unique_ptr<hip_moi::host_context>> contexts;
    HipMoiStorageRefs                                   storage_refs;
    if(mode == Mode::CooperativeContextSampled || mode == Mode::CooperativeSampled)
    {
        storage_refs = install_storage_refs(blocks, &contexts);
    }

    launch_kernel(mode, buffers, blocks, nullptr);
    HIP_MOI_ATT_HIP_CHECK(hipGetLastError());
    HIP_MOI_ATT_HIP_CHECK(hipDeviceSynchronize());

    std::printf("mode=%s blocks=%d checksum=%.6f\n",
                mode_name(mode),
                blocks,
                checksum_output(buffers.output, blocks));
    return EXIT_SUCCESS;
}
