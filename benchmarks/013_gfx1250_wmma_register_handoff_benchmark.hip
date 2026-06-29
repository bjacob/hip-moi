// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// benchmarks/013_gfx1250_wmma_register_handoff_benchmark.hip
//
// gfx1250 WMMA register-handoff benchmark. The kernel keeps the QK tile in WMMA
// accumulator registers, reshapes it into the PV A operand with ds_bpermute lane
// reads, and optionally runs the second PV WMMA without materializing scores in
// LDS.
#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace
{
    using half_t   = _Float16;
    using f16x16_t = half_t __attribute__((ext_vector_type(16)));
    using f32x8_t  = float __attribute__((ext_vector_type(8)));

    constexpr int   kWaveSize            = 32;
    constexpr int   kSubgroupCount       = 2;
    constexpr int   kThreadCount         = kWaveSize * kSubgroupCount;
    constexpr int   kThreadsPerSubgroup  = kWaveSize;
    constexpr int   kQueryTile           = 16;
    constexpr int   kKeyTile             = 16;
    constexpr int   kHeadDim             = 32;
    constexpr int   kValueDim            = 16;
    constexpr int   kOperandElements     = 16;
    constexpr int   kAccumulatorElements = 8;
    constexpr int   kDefaultBlocks       = 256;
    constexpr int   kDefaultWarmupMs     = 100;
    constexpr int   kDefaultMeasureMs    = 100;
    constexpr int   kDefaultWatchpoints  = 1;
    constexpr float kQkScale             = 0.125f;

    constexpr std::size_t kQFragmentsPerBlock  = kSubgroupCount * kWaveSize;
    constexpr std::size_t kKVFragmentsPerBlock = kWaveSize;
    constexpr std::size_t kOutputPerBlock      = kSubgroupCount * kQueryTile * kValueDim;

    constexpr uint32_t kKSharedByteOffset = 0;
    constexpr uint32_t kVSharedByteOffset = kWaveSize * sizeof(f16x16_t);

    constexpr hip_moi::site_id kKStoreSite(0x1301);
    constexpr hip_moi::site_id kVStoreSite(0x1302);
    constexpr hip_moi::site_id kKLoadSite(0x1303);
    constexpr hip_moi::site_id kVLoadSite(0x1304);

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
            return DisplayLatency{/*value=*/ms * 1000.0, /*unit=*/"us"};
        }
        return DisplayLatency{/*value=*/ms, /*unit=*/"ms"};
    }

    [[noreturn]] void fail_hip(hipError_t status, const char* expr, const char* file, int line)
    {
        std::fprintf(
            stderr, "%s:%d: HIP error from %s: %s\n", file, line, expr, hipGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }

#define HIP_MOI_REGISTER_BENCH_HIP_CHECK(expr)                                  \
    do                                                                          \
    {                                                                           \
        hipError_t hip_moi_register_bench_status = (expr);                      \
        if(hip_moi_register_bench_status != hipSuccess)                         \
        {                                                                       \
            fail_hip(hip_moi_register_bench_status, #expr, __FILE__, __LINE__); \
        }                                                                       \
    } while(false)

    int get_int_env(const char* name, int default_value)
    {
        const char* value = std::getenv(name);
        return value ? std::atoi(value) : default_value;
    }

    __host__ __device__ float input_value(int block, int row, int col, int seed)
    {
        int value = (block * 3 + row * 7 + col * 5 + seed * 11 + (row ^ (col * 3))) % 9;
        return static_cast<float>(value - 4) * 0.0625f;
    }

    __host__ __device__ int gfx1250_wmma_f16_k(int lane, int idx)
    {
        int half = idx & 1;
        int reg  = idx >> 1;
        return half | ((reg & 1) << 1) | (((lane >> 4) & 1) << 2) | (((reg >> 1) & 1) << 3)
               | (((reg >> 2) & 1) << 4);
    }

    __host__ __device__ int gfx1250_wmma_acc_m(int lane, int elem)
    {
        return elem + 8 * (lane >> 4);
    }

    __host__ __device__ uint32_t k_fragment_offset(int lane)
    {
        return kKSharedByteOffset + static_cast<uint32_t>(lane * sizeof(f16x16_t));
    }

    __host__ __device__ uint32_t v_fragment_offset(int lane)
    {
        return kVSharedByteOffset + static_cast<uint32_t>(lane * sizeof(f16x16_t));
    }

    __device__ f32x8_t zero_f32x8()
    {
        f32x8_t value;
#pragma unroll
        for(int i = 0; i < kAccumulatorElements; ++i)
        {
            value[i] = 0.0f;
        }
        return value;
    }

    __device__ float read_float_lane(float value, int source_lane)
    {
        uint32_t bits      = __builtin_bit_cast(uint32_t, value);
        uint32_t lane_bits = __builtin_amdgcn_ds_bpermute(source_lane * sizeof(uint32_t), bits);
        return __builtin_bit_cast(float, lane_bits);
    }

    __device__ float read_accumulator_element_from_lane(f32x8_t acc, int elem, int source_lane)
    {
        float value0 = read_float_lane(acc[0], source_lane);
        float value1 = read_float_lane(acc[1], source_lane);
        float value2 = read_float_lane(acc[2], source_lane);
        float value3 = read_float_lane(acc[3], source_lane);
        float value4 = read_float_lane(acc[4], source_lane);
        float value5 = read_float_lane(acc[5], source_lane);
        float value6 = read_float_lane(acc[6], source_lane);
        float value7 = read_float_lane(acc[7], source_lane);
        switch(elem)
        {
        case 0:
            return value0;
        case 1:
            return value1;
        case 2:
            return value2;
        case 3:
            return value3;
        case 4:
            return value4;
        case 5:
            return value5;
        case 6:
            return value6;
        default:
            return value7;
        }
    }

    __device__ f16x16_t qk_accumulator_to_pv_fragment(f32x8_t qk_acc, int lane)
    {
        f16x16_t p_fragment;
        int      query_row       = lane & 15;
        int      source_acc_elem = query_row & 7;
        int      source_lane_row = query_row >= 8 ? 16 : 0;
#pragma unroll
        for(int elem = 0; elem < kOperandElements; ++elem)
        {
            int key = gfx1250_wmma_f16_k(lane, elem);
            if(key < kKeyTile)
            {
                int   source_lane = source_lane_row + key;
                float qk_value
                    = read_accumulator_element_from_lane(qk_acc, source_acc_elem, source_lane);
                p_fragment[elem] = static_cast<half_t>(qk_value * kQkScale);
            }
            else
            {
                p_fragment[elem] = static_cast<half_t>(0.0f);
            }
        }
        return p_fragment;
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

    template <typename Policy, bool RunPvMma>
    __global__ __launch_bounds__(kThreadCount) void register_handoff_kernel(
        const f16x16_t* __restrict__ q_fragments,
        const f16x16_t* __restrict__ k_fragments,
        const f16x16_t* __restrict__ v_fragments,
        float* __restrict__ out,
        Runtime runtime)
    {
        __shared__ f16x16_t k_shared[kWaveSize];
        __shared__ f16x16_t v_shared[kWaveSize];

        typename Policy::context_type ctx = Policy::make_context(runtime);
        Policy::init_workgroup(ctx);

        int block    = static_cast<int>(blockIdx.x);
        int tid      = static_cast<int>(threadIdx.x);
        int subgroup = tid / kWaveSize;
        int lane     = tid & (kWaveSize - 1);

        std::size_t q_base   = static_cast<std::size_t>(block) * kQFragmentsPerBlock;
        std::size_t kv_base  = static_cast<std::size_t>(block) * kKVFragmentsPerBlock;
        std::size_t out_base = static_cast<std::size_t>(block) * kOutputPerBlock;

        if(subgroup == 0)
        {
            Policy::store(ctx,
                          &k_shared[lane],
                          k_fragments[kv_base + lane],
                          k_fragment_offset(lane),
                          kKStoreSite);
            Policy::store(ctx,
                          &v_shared[lane],
                          v_fragments[kv_base + lane],
                          v_fragment_offset(lane),
                          kVStoreSite);
        }

        Policy::syncthreads(ctx);

        f16x16_t q_fragment = q_fragments[q_base + subgroup * kWaveSize + lane];
        f16x16_t k_fragment
            = Policy::load(ctx, &k_shared[lane], k_fragment_offset(lane), kKLoadSite);

        f32x8_t qk_acc = zero_f32x8();
        qk_acc         = __builtin_amdgcn_wmma_f32_16x16x32_f16(
            false, q_fragment, false, k_fragment, static_cast<short>(0), qk_acc, false, false);

        f16x16_t p_fragment = qk_accumulator_to_pv_fragment(qk_acc, lane);
        if constexpr(RunPvMma)
        {
            f16x16_t v_fragment
                = Policy::load(ctx, &v_shared[lane], v_fragment_offset(lane), kVLoadSite);

            f32x8_t pv_acc = zero_f32x8();
            pv_acc         = __builtin_amdgcn_wmma_f32_16x16x32_f16(
                false, p_fragment, false, v_fragment, static_cast<short>(0), pv_acc, false, false);

            int value_column = lane & 15;
#pragma unroll
            for(int elem = 0; elem < kAccumulatorElements; ++elem)
            {
                int query_row = gfx1250_wmma_acc_m(lane, elem);
                int query     = subgroup * kQueryTile + query_row;
                out[out_base + query * kValueDim + value_column] = pv_acc[elem];
            }
        }
        else
        {
#pragma unroll
            for(int elem = 0; elem < kOperandElements; ++elem)
            {
                int query_row = lane & 15;
                int key       = gfx1250_wmma_f16_k(lane, elem);
                if(key < kKeyTile)
                {
                    int query                               = subgroup * kQueryTile + query_row;
                    out[out_base + query * kValueDim + key] = static_cast<float>(p_fragment[elem]);
                }
            }
        }
    }

    struct DeviceBuffers
    {
        f16x16_t* q   = nullptr;
        f16x16_t* k   = nullptr;
        f16x16_t* v   = nullptr;
        float*    out = nullptr;

        explicit DeviceBuffers(int blocks)
        {
            HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipMalloc(
                &q, static_cast<std::size_t>(blocks) * kQFragmentsPerBlock * sizeof(f16x16_t)));
            HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipMalloc(
                &k, static_cast<std::size_t>(blocks) * kKVFragmentsPerBlock * sizeof(f16x16_t)));
            HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipMalloc(
                &v, static_cast<std::size_t>(blocks) * kKVFragmentsPerBlock * sizeof(f16x16_t)));
            HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipMalloc(
                &out, static_cast<std::size_t>(blocks) * kOutputPerBlock * sizeof(float)));
        }

        DeviceBuffers(const DeviceBuffers&)            = delete;
        DeviceBuffers& operator=(const DeviceBuffers&) = delete;

        ~DeviceBuffers()
        {
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

            HIP_MOI_REGISTER_BENCH_HIP_CHECK(
                hipMalloc(&device_refs, host_refs.size() * sizeof(host_refs[0])));
            HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipMemcpy(device_refs,
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

    float quantized_input_value(int block, int row, int col, int seed)
    {
        return static_cast<float>(static_cast<half_t>(input_value(block, row, col, seed)));
    }

    void fill_fragments(int                    blocks,
                        std::vector<f16x16_t>* q_fragments,
                        std::vector<f16x16_t>* k_fragments,
                        std::vector<f16x16_t>* v_fragments)
    {
        q_fragments->resize(static_cast<std::size_t>(blocks) * kQFragmentsPerBlock);
        k_fragments->resize(static_cast<std::size_t>(blocks) * kKVFragmentsPerBlock);
        v_fragments->resize(static_cast<std::size_t>(blocks) * kKVFragmentsPerBlock);
        for(int block = 0; block < blocks; ++block)
        {
            std::size_t q_base  = static_cast<std::size_t>(block) * kQFragmentsPerBlock;
            std::size_t kv_base = static_cast<std::size_t>(block) * kKVFragmentsPerBlock;
            for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
            {
                int query_base = subgroup * kQueryTile;
                for(int lane = 0; lane < kWaveSize; ++lane)
                {
                    int query_row = query_base + (lane & 15);
                    for(int elem = 0; elem < kOperandElements; ++elem)
                    {
                        int dim = gfx1250_wmma_f16_k(lane, elem);
                        (*q_fragments)[q_base + subgroup * kWaveSize + lane][elem]
                            = static_cast<half_t>(input_value(block, query_row, dim, /*seed=*/1));
                    }
                }
            }

            for(int lane = 0; lane < kWaveSize; ++lane)
            {
                int key          = lane & 15;
                int value_column = lane & 15;
                for(int elem = 0; elem < kOperandElements; ++elem)
                {
                    int dim = gfx1250_wmma_f16_k(lane, elem);
                    (*k_fragments)[kv_base + lane][elem]
                        = static_cast<half_t>(input_value(block, key, dim, /*seed=*/2));
                    (*v_fragments)[kv_base + lane][elem]
                        = dim < kKeyTile ? static_cast<half_t>(
                                               input_value(block, dim, value_column, /*seed=*/3))
                                         : static_cast<half_t>(0.0f);
                }
            }
        }
    }

    std::vector<float> host_handoff_reference(int blocks)
    {
        std::vector<float> out(static_cast<std::size_t>(blocks) * kOutputPerBlock, 0.0f);
        for(int block = 0; block < blocks; ++block)
        {
            std::size_t base = static_cast<std::size_t>(block) * kOutputPerBlock;
            for(int query = 0; query < kSubgroupCount * kQueryTile; ++query)
            {
                for(int key = 0; key < kKeyTile; ++key)
                {
                    float qk = 0.0f;
                    for(int dim = 0; dim < kHeadDim; ++dim)
                    {
                        float q = quantized_input_value(block, query, dim, /*seed=*/1);
                        float k = quantized_input_value(block, key, dim, /*seed=*/2);
                        qk += q * k;
                    }
                    out[base + query * kValueDim + key]
                        = static_cast<float>(static_cast<half_t>(qk * kQkScale));
                }
            }
        }
        return out;
    }

    std::vector<float> host_pv_reference(int blocks)
    {
        std::vector<float> out(static_cast<std::size_t>(blocks) * kOutputPerBlock, 0.0f);
        for(int block = 0; block < blocks; ++block)
        {
            std::size_t base = static_cast<std::size_t>(block) * kOutputPerBlock;
            for(int query = 0; query < kSubgroupCount * kQueryTile; ++query)
            {
                half_t p[kKeyTile];
                for(int key = 0; key < kKeyTile; ++key)
                {
                    float qk = 0.0f;
                    for(int dim = 0; dim < kHeadDim; ++dim)
                    {
                        float q = quantized_input_value(block, query, dim, /*seed=*/1);
                        float k = quantized_input_value(block, key, dim, /*seed=*/2);
                        qk += q * k;
                    }
                    p[key] = static_cast<half_t>(qk * kQkScale);
                }

                for(int value_column = 0; value_column < kValueDim; ++value_column)
                {
                    float acc = 0.0f;
                    for(int key = 0; key < kKeyTile; ++key)
                    {
                        float value = quantized_input_value(block, key, value_column, /*seed=*/3);
                        acc += static_cast<float>(p[key]) * value;
                    }
                    out[base + query * kValueDim + value_column] = acc;
                }
            }
        }
        return out;
    }

    void copy_inputs(const std::vector<f16x16_t>& q,
                     const std::vector<f16x16_t>& k,
                     const std::vector<f16x16_t>& v,
                     DeviceBuffers*               buffers)
    {
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(
            hipMemcpy(buffers->q, q.data(), q.size() * sizeof(q[0]), hipMemcpyHostToDevice));
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(
            hipMemcpy(buffers->k, k.data(), k.size() * sizeof(k[0]), hipMemcpyHostToDevice));
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(
            hipMemcpy(buffers->v, v.data(), v.size() * sizeof(v[0]), hipMemcpyHostToDevice));
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipMemset(
            buffers->out, 0, q.size() / kQFragmentsPerBlock * kOutputPerBlock * sizeof(float)));
    }

    template <typename Policy, bool RunPvMma>
    void launch_kernel(const DeviceBuffers& buffers, int blocks, Runtime runtime)
    {
        hipLaunchKernelGGL((register_handoff_kernel<Policy, RunPvMma>),
                           dim3(blocks),
                           dim3(kThreadCount),
                           0,
                           0,
                           buffers.q,
                           buffers.k,
                           buffers.v,
                           buffers.out,
                           runtime);
    }

    template <typename Policy, bool RunPvMma>
    double
        run_launches(const DeviceBuffers& buffers, int blocks, StorageSet* storage, int iterations)
    {
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipEventCreate(&start));
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipEventCreate(&stop));
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipEventRecord(start, 0));
        for(int i = 0; i < iterations; ++i)
        {
            Runtime runtime{};
            if(storage)
            {
                runtime = storage->next_runtime();
            }
            launch_kernel<Policy, RunPvMma>(buffers, blocks, runtime);
        }
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipEventRecord(stop, 0));
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipEventDestroy(stop));
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipEventDestroy(start));
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipGetLastError());
        return static_cast<double>(elapsed_ms);
    }

    template <typename Policy, bool RunPvMma>
    void warmup_for(const DeviceBuffers& buffers, int blocks, StorageSet* storage, int warmup_ms)
    {
        int iterations = 1;
        while(true)
        {
            double elapsed = run_launches<Policy, RunPvMma>(buffers, blocks, storage, iterations);
            if(elapsed >= static_cast<double>(warmup_ms))
            {
                return;
            }
            iterations *= 2;
        }
    }

    template <typename Policy, bool RunPvMma>
    double
        benchmark_for(const DeviceBuffers& buffers, int blocks, StorageSet* storage, int measure_ms)
    {
        int iterations = 1;
        while(true)
        {
            double elapsed = run_launches<Policy, RunPvMma>(buffers, blocks, storage, iterations);
            if(elapsed >= static_cast<double>(measure_ms))
            {
                return elapsed / static_cast<double>(iterations);
            }
            iterations *= 2;
        }
    }

    template <typename Policy, bool RunPvMma>
    void benchmark_mode(const char*          name,
                        const DeviceBuffers& buffers,
                        int                  blocks,
                        StorageSet*          storage,
                        int                  warmup_ms,
                        int                  measure_ms)
    {
        warmup_for<Policy, RunPvMma>(buffers, blocks, storage, warmup_ms);
        double latency_ms = benchmark_for<Policy, RunPvMma>(buffers, blocks, storage, measure_ms);
        DisplayLatency display = display_latency(latency_ms);
        std::printf("%-64s %.3g %s\n", name, display.value, display.unit);
    }

    void verify_outputs(const DeviceBuffers& buffers, const std::vector<float>& expected)
    {
        std::vector<float> actual(expected.size(), 0.0f);
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipMemcpy(
            actual.data(), buffers.out, actual.size() * sizeof(actual[0]), hipMemcpyDeviceToHost));
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
    }

    template <typename Policy, bool RunPvMma>
    void verify_mode(const DeviceBuffers&      buffers,
                     int                       blocks,
                     Runtime                   runtime,
                     const std::vector<float>& expected)
    {
        launch_kernel<Policy, RunPvMma>(buffers, blocks, runtime);
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipGetLastError());
        HIP_MOI_REGISTER_BENCH_HIP_CHECK(hipDeviceSynchronize());
        verify_outputs(buffers, expected);
    }
} // namespace

int main()
{
    int blocks = get_int_env("HIP_MOI_GFX1250_WMMA_REGISTER_HANDOFF_BENCH_BLOCKS", kDefaultBlocks);
    int warmup_ms
        = get_int_env("HIP_MOI_GFX1250_WMMA_REGISTER_HANDOFF_BENCH_WARMUP_MS", kDefaultWarmupMs);
    int measure_ms
        = get_int_env("HIP_MOI_GFX1250_WMMA_REGISTER_HANDOFF_BENCH_MEASURE_MS", kDefaultMeasureMs);
    int watchpoints = get_int_env("HIP_MOI_GFX1250_WMMA_REGISTER_HANDOFF_BENCH_WATCHPOINTS",
                                  kDefaultWatchpoints);

    std::vector<f16x16_t> q;
    std::vector<f16x16_t> k;
    std::vector<f16x16_t> v;
    fill_fragments(blocks, &q, &k, &v);

    DeviceBuffers buffers(blocks);
    copy_inputs(q, k, v, &buffers);

    StorageSet context_storage(blocks, watchpoints);
    StorageSet fast_storage(blocks, watchpoints);

    std::vector<float> expected_handoff = host_handoff_reference(blocks);
    std::vector<float> expected_pv      = host_pv_reference(blocks);

    verify_mode<PassThroughPolicy, false>(buffers, blocks, Runtime{}, expected_handoff);
    verify_mode<ContextSampledPolicy, false>(
        buffers, blocks, context_storage.next_runtime(), expected_handoff);
    verify_mode<SampledWatchpointContextPolicy, false>(
        buffers, blocks, fast_storage.next_runtime(), expected_handoff);
    verify_mode<PassThroughPolicy, true>(buffers, blocks, Runtime{}, expected_pv);
    verify_mode<ContextSampledPolicy, true>(
        buffers, blocks, context_storage.next_runtime(), expected_pv);
    verify_mode<SampledWatchpointContextPolicy, true>(
        buffers, blocks, fast_storage.next_runtime(), expected_pv);

    std::printf("hip-moi gfx1250 WMMA register handoff benchmark: "
                "blocks=%d warmup_ms=%d min_ms=%d watchpoints=%d\n",
                blocks,
                warmup_ms,
                measure_ms,
                watchpoints);
    benchmark_mode<PassThroughPolicy, false>("gfx1250-wmma-register-handoff_pass-through",
                                             buffers,
                                             blocks,
                                             nullptr,
                                             warmup_ms,
                                             measure_ms);
    benchmark_mode<ContextSampledPolicy, false>(
        "gfx1250-wmma-register-handoff_context+sampled-watchpoint",
        buffers,
        blocks,
        &context_storage,
        warmup_ms,
        measure_ms);
    benchmark_mode<SampledWatchpointContextPolicy, false>(
        "gfx1250-wmma-register-handoff_sampled-watchpoint-context",
        buffers,
        blocks,
        &fast_storage,
        warmup_ms,
        measure_ms);
    benchmark_mode<PassThroughPolicy, true>("gfx1250-wmma-register-handoff-pv_pass-through",
                                            buffers,
                                            blocks,
                                            nullptr,
                                            warmup_ms,
                                            measure_ms);
    benchmark_mode<ContextSampledPolicy, true>(
        "gfx1250-wmma-register-handoff-pv_context+sampled-watchpoint",
        buffers,
        blocks,
        &context_storage,
        warmup_ms,
        measure_ms);
    benchmark_mode<SampledWatchpointContextPolicy, true>(
        "gfx1250-wmma-register-handoff-pv_sampled-watchpoint-context",
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
