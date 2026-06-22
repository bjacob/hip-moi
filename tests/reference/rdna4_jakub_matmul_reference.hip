// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/reference/rdna4_jakub_matmul_reference.hip
//
// Reference-only RDNA4 WMMA matmul kernels derived from Jakub's
// sanitizer-strategy/rdna4_matmul corpus. This file borrows the production-like
// packed FP16 WMMA/LDS schedule shape, but intentionally does not copy the
// inline sanitizer policies from that harness.
#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#define HIP_MOI_REFERENCE_HIP_ASSERT(expr)                                   \
    do                                                                       \
    {                                                                        \
        hipError_t hip_moi_reference_status = (expr);                        \
        ASSERT_EQ(hip_moi_reference_status, hipSuccess)                      \
            << #expr << ": " << hipGetErrorString(hip_moi_reference_status); \
    } while(false)

namespace
{
    using half_t = _Float16;
    using f16x8  = half_t __attribute__((ext_vector_type(8)));
    using f32x8  = float __attribute__((ext_vector_type(8)));

    constexpr int kWmmaM               = 16;
    constexpr int kWmmaN               = 16;
    constexpr int kWmmaK               = 16;
    constexpr int kWaveSize            = 32;
    constexpr int kWavesPerWorkgroup   = 8;
    constexpr int kThreadsPerWorkgroup = kWaveSize * kWavesPerWorkgroup;
    constexpr int kProductionMTiles    = 16;
    constexpr int kProductionNTiles    = 8;
    constexpr int kProductionKGroup    = 2;
    constexpr int kProductionWaveTileM = 4;
    constexpr int kProductionWaveTileN = 4;

    constexpr int kCheckM = kProductionMTiles * kWmmaM;
    constexpr int kCheckN = kProductionNTiles * kWmmaN;
    constexpr int kCheckK = 128;

    enum class SyncMode
    {
        Correct,
        MissingLoadComputeBarrier,
        MissingComputeLoadBarrier,
        SometimesMissingLoadComputeBarrier,
        SometimesMissingComputeLoadBarrier,
    };

    __host__ __device__ int ceil_div(int x, int y)
    {
        return (x + y - 1) / y;
    }

    __host__ __device__ int host_wmma_ab_lane(int row_or_col, int k)
    {
        return row_or_col + 16 * ((k >> 2) & 1);
    }

    __host__ __device__ int host_wmma_ab_slot(int k)
    {
        int reg = 2 * (k >> 3) + ((k >> 1) & 1);
        return 2 * reg + (k & 1);
    }

    std::size_t packed_ab_offset(int outer_row,
                                 int outer_col,
                                 int outer_cols,
                                 int subtiles,
                                 int subtile,
                                 int inner_row_or_col,
                                 int k)
    {
        int         lane = host_wmma_ab_lane(inner_row_or_col, k);
        int         slot = host_wmma_ab_slot(k);
        std::size_t vec
            = ((static_cast<std::size_t>(outer_row) * outer_cols + outer_col) * subtiles + subtile)
                  * kWaveSize
              + lane;
        return vec * 8 + slot;
    }

    std::size_t packed_c_offset(int M, int N, int MTiles, int NTiles, int m, int n)
    {
        (void)M;
        int         n_outer_count = N / (NTiles * kWmmaN);
        int         m_outer       = m / (MTiles * kWmmaM);
        int         n_outer       = n / (NTiles * kWmmaN);
        int         m_in_block    = m - m_outer * MTiles * kWmmaM;
        int         n_in_block    = n - n_outer * NTiles * kWmmaN;
        int         subtile_m     = m_in_block / kWmmaM;
        int         subtile_n     = n_in_block / kWmmaN;
        int         m16           = m_in_block - subtile_m * kWmmaM;
        int         n16           = n_in_block - subtile_n * kWmmaN;
        int         lane          = n16 + 16 * (m16 >> 3);
        int         elem          = m16 & 7;
        std::size_t vec
            = ((static_cast<std::size_t>(m_outer) * n_outer_count + n_outer) * MTiles * NTiles
               + subtile_m * NTiles + subtile_n)
                  * kWaveSize
              + lane;
        return vec * 8 + elem;
    }

    __host__ __device__ int small_i32_value(int row, int col, int seed)
    {
        int x = row * 131 + col * 17 + seed * 29 + (row ^ (col * 7));
        x %= 5;
        if(x < 0)
        {
            x += 5;
        }
        return x - 2;
    }

    __device__ f32x8 zero_f32x8()
    {
        f32x8 value;
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            value[i] = 0.0f;
        }
        return value;
    }

    template <int      MTiles,
              int      NTiles,
              bool     Pipeline,
              bool     DoubleBuffer,
              SyncMode Sync,
              int      Waves     = kWavesPerWorkgroup,
              int      KGroup    = 1,
              int      WaveTileM = MTiles,
              int      WaveTileN = NTiles / Waves>
    __global__ __launch_bounds__(Waves* kWaveSize) void fp16_wmma_tiled_kernel(
        const half_t* __restrict__ a_scalar,
        const half_t* __restrict__ b_scalar,
        float* __restrict__ c_scalar,
        int M,
        int N,
        int K)
    {
        static_assert(Waves > 0);
        static_assert(KGroup >= 1);
        static_assert(WaveTileM > 0 && WaveTileN > 0);
        static_assert(MTiles % WaveTileM == 0);
        static_assert(NTiles % WaveTileN == 0);
        static_assert((MTiles / WaveTileM) * (NTiles / WaveTileN) == Waves);

        constexpr int kStages           = DoubleBuffer ? 2 : 1;
        constexpr int kTilesPerWave     = WaveTileM;
        constexpr int kNTilesPerWave    = WaveTileN;
        constexpr int kWaveGridM        = MTiles / WaveTileM;
        constexpr int kThreadLoadStride = Waves * kWaveSize;
        constexpr int kAThreadLoads
            = (MTiles * kWaveSize + kThreadLoadStride - 1) / kThreadLoadStride;
        constexpr int kBThreadLoads
            = (NTiles * kWaveSize + kThreadLoadStride - 1) / kThreadLoadStride;

        const auto* A = reinterpret_cast<const f16x8*>(a_scalar);
        const auto* B = reinterpret_cast<const f16x8*>(b_scalar);
        auto*       C = reinterpret_cast<f32x8*>(c_scalar);

        __shared__ f16x8 a_s[kStages][KGroup][MTiles][kWaveSize];
        __shared__ f16x8 b_s[kStages][KGroup][NTiles][kWaveSize];

        int tid           = static_cast<int>(threadIdx.x);
        int wave          = tid / kWaveSize;
        int lane          = tid & (kWaveSize - 1);
        int wave_m        = wave % kWaveGridM;
        int wave_n        = wave / kWaveGridM;
        int m_outer       = static_cast<int>(blockIdx.y);
        int n_outer       = static_cast<int>(blockIdx.x);
        int n_outer_count = N / (NTiles * kWmmaN);
        int k_outer_count = K / kWmmaK;
        (void)M;

        f32x8 acc[kTilesPerWave][kNTilesPerWave];
#pragma unroll
        for(int t = 0; t < kTilesPerWave; ++t)
        {
#pragma unroll
            for(int n = 0; n < kNTilesPerWave; ++n)
            {
                acc[t][n] = zero_f32x8();
            }
        }

        auto a_byte_offset = [](int stage, int k_group, int tile, int lane_id) {
            return static_cast<uint32_t>(
                ((((stage * KGroup + k_group) * (MTiles + NTiles) + tile) * kWaveSize + lane_id)
                 * sizeof(f16x8)));
        };
        auto b_byte_offset = [](int stage, int k_group, int tile, int lane_id) {
            return static_cast<uint32_t>(
                ((((stage * KGroup + k_group) * (MTiles + NTiles) + MTiles + tile) * kWaveSize
                  + lane_id)
                 * sizeof(f16x8)));
        };
        (void)a_byte_offset;
        (void)b_byte_offset;

        auto group_size = [&](int k_group_outer) {
            int remaining = k_outer_count - k_group_outer * KGroup;
            return remaining < KGroup ? remaining : KGroup;
        };

        auto load_stage = [&](int k_group_outer, int stage) {
#pragma unroll
            for(int g = 0; g < KGroup; ++g)
            {
                int k_outer = k_group_outer * KGroup + g;
                if(k_outer >= k_outer_count)
                {
                    continue;
                }
                const int a_base = (m_outer * k_outer_count + k_outer) * MTiles * kWaveSize;
                const int b_base = (n_outer * k_outer_count + k_outer) * NTiles * kWaveSize;
                for(int idx = tid; idx < MTiles * kWaveSize; idx += blockDim.x)
                {
                    int tile                        = idx / kWaveSize;
                    int local_lane                  = idx & (kWaveSize - 1);
                    a_s[stage][g][tile][local_lane] = A[a_base + idx];
                }
                for(int idx = tid; idx < NTiles * kWaveSize; idx += blockDim.x)
                {
                    int tile                        = idx / kWaveSize;
                    int local_lane                  = idx & (kWaveSize - 1);
                    b_s[stage][g][tile][local_lane] = B[b_base + idx];
                }
            }
        };

        struct Fp16StageRegs
        {
            f16x8 a[KGroup][kAThreadLoads];
            f16x8 b[KGroup][kBThreadLoads];
        };

        struct Fp16ComputeRegs
        {
            f16x8 a[KGroup][kTilesPerWave];
            f16x8 b[KGroup][kNTilesPerWave];
        };

        auto global_to_regs = [&](int k_group_outer, Fp16StageRegs& regs) {
#pragma unroll
            for(int g = 0; g < KGroup; ++g)
            {
                int k_outer = k_group_outer * KGroup + g;
                if(k_outer >= k_outer_count)
                {
                    continue;
                }
                const int a_base = (m_outer * k_outer_count + k_outer) * MTiles * kWaveSize;
                const int b_base = (n_outer * k_outer_count + k_outer) * NTiles * kWaveSize;
#pragma unroll
                for(int copy = 0; copy < kAThreadLoads; ++copy)
                {
                    int idx = tid + copy * kThreadLoadStride;
                    if(idx < MTiles * kWaveSize)
                    {
                        regs.a[g][copy] = A[a_base + idx];
                    }
                }
#pragma unroll
                for(int copy = 0; copy < kBThreadLoads; ++copy)
                {
                    int idx = tid + copy * kThreadLoadStride;
                    if(idx < NTiles * kWaveSize)
                    {
                        regs.b[g][copy] = B[b_base + idx];
                    }
                }
            }
        };

        auto regs_to_shared = [&](int k_group_outer, int stage, const Fp16StageRegs& regs) {
#pragma unroll
            for(int g = 0; g < KGroup; ++g)
            {
                int k_outer = k_group_outer * KGroup + g;
                if(k_outer >= k_outer_count)
                {
                    continue;
                }
#pragma unroll
                for(int copy = 0; copy < kAThreadLoads; ++copy)
                {
                    int idx = tid + copy * kThreadLoadStride;
                    if(idx >= MTiles * kWaveSize)
                    {
                        continue;
                    }
                    int tile                        = idx / kWaveSize;
                    int local_lane                  = idx & (kWaveSize - 1);
                    a_s[stage][g][tile][local_lane] = regs.a[g][copy];
                }
#pragma unroll
                for(int copy = 0; copy < kBThreadLoads; ++copy)
                {
                    int idx = tid + copy * kThreadLoadStride;
                    if(idx >= NTiles * kWaveSize)
                    {
                        continue;
                    }
                    int tile                        = idx / kWaveSize;
                    int local_lane                  = idx & (kWaveSize - 1);
                    b_s[stage][g][tile][local_lane] = regs.b[g][copy];
                }
            }
        };

        auto shared_to_compute_regs = [&](int stage, int valid_groups, Fp16ComputeRegs& regs) {
#pragma unroll
            for(int g = 0; g < KGroup; ++g)
            {
                if(g >= valid_groups)
                {
                    continue;
                }
#pragma unroll
                for(int n = 0; n < kNTilesPerWave; ++n)
                {
                    int n_tile   = wave_n * WaveTileN + n;
                    regs.b[g][n] = b_s[stage][g][n_tile][lane];
                }
#pragma unroll
                for(int t = 0; t < kTilesPerWave; ++t)
                {
                    int m_tile   = wave_m * WaveTileM + t;
                    regs.a[g][t] = a_s[stage][g][m_tile][lane];
                }
            }
        };

        auto compute_from_regs = [&](int valid_groups, const Fp16ComputeRegs& regs) {
#pragma unroll
            for(int g = 0; g < KGroup; ++g)
            {
                if(g >= valid_groups)
                {
                    continue;
                }
#pragma unroll
                for(int t = 0; t < kTilesPerWave; ++t)
                {
#pragma unroll
                    for(int n = 0; n < kNTilesPerWave; ++n)
                    {
                        acc[t][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                            regs.a[g][t], regs.b[g][n], acc[t][n]);
                    }
                }
            }
        };

        auto compute_stage = [&](int stage, int valid_groups) {
#pragma unroll
            for(int g = 0; g < KGroup; ++g)
            {
                if(g >= valid_groups)
                {
                    continue;
                }
                f16x8 b[kNTilesPerWave];
#pragma unroll
                for(int n = 0; n < kNTilesPerWave; ++n)
                {
                    int n_tile = wave_n * WaveTileN + n;
                    b[n]       = b_s[stage][g][n_tile][lane];
                }
                if constexpr(Pipeline)
                {
                    int   first_m_tile = wave_m * WaveTileM;
                    f16x8 next_a       = a_s[stage][g][first_m_tile][lane];
#pragma unroll
                    for(int t = 0; t < kTilesPerWave; ++t)
                    {
                        f16x8 a = next_a;
                        if constexpr(kTilesPerWave > 1)
                        {
                            if(t + 1 < kTilesPerWave)
                            {
                                int next_m_tile = wave_m * WaveTileM + t + 1;
                                next_a          = a_s[stage][g][next_m_tile][lane];
                            }
                        }
#pragma unroll
                        for(int n = 0; n < kNTilesPerWave; ++n)
                        {
                            acc[t][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                a, b[n], acc[t][n]);
                        }
                    }
                }
                else
                {
#pragma unroll
                    for(int t = 0; t < kTilesPerWave; ++t)
                    {
                        int   m_tile = wave_m * WaveTileM + t;
                        f16x8 a      = a_s[stage][g][m_tile][lane];
#pragma unroll
                        for(int n = 0; n < kNTilesPerWave; ++n)
                        {
                            acc[t][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                a, b[n], acc[t][n]);
                        }
                    }
                }
            }
        };

        auto skips_conditional_sync = [&](int k_outer) {
            return ((static_cast<int>(blockIdx.x) ^ static_cast<int>(blockIdx.y) ^ k_outer) & 1)
                   != 0;
        };
        auto sync_after_load = [&](int k_outer) {
            if constexpr(Sync == SyncMode::MissingLoadComputeBarrier)
            {
                return;
            }
            else if constexpr(Sync == SyncMode::SometimesMissingLoadComputeBarrier)
            {
                if(!skips_conditional_sync(k_outer))
                {
                    __syncthreads();
                }
            }
            else
            {
                __syncthreads();
            }
        };
        auto sync_after_compute = [&](int k_outer) {
            if constexpr(Sync == SyncMode::MissingComputeLoadBarrier)
            {
                return;
            }
            else if constexpr(Sync == SyncMode::SometimesMissingComputeLoadBarrier)
            {
                if(!skips_conditional_sync(k_outer))
                {
                    __syncthreads();
                }
            }
            else
            {
                __syncthreads();
            }
        };

        int k_group_count = (k_outer_count + KGroup - 1) / KGroup;
        if constexpr(DoubleBuffer)
        {
            Fp16StageRegs stage0_regs;
            global_to_regs(0, stage0_regs);
            regs_to_shared(0, 0, stage0_regs);
            if(k_group_count > 1)
            {
                Fp16StageRegs stage1_regs;
                global_to_regs(1, stage1_regs);
                regs_to_shared(1, 1, stage1_regs);
            }
            sync_after_load(0);
            int k_group_outer = 0;
            for(; k_group_outer + 1 < k_group_count; k_group_outer += 2)
            {
                Fp16StageRegs next_stage0_regs;
                Fp16StageRegs next_stage1_regs;
                bool          has_next_stage0 = k_group_outer + 2 < k_group_count;
                bool          has_next_stage1 = k_group_outer + 3 < k_group_count;
                if(has_next_stage0)
                {
                    global_to_regs(k_group_outer + 2, next_stage0_regs);
                }
                compute_stage(0, group_size(k_group_outer));
                if(has_next_stage0)
                {
                    sync_after_compute(k_group_outer);
                    regs_to_shared(k_group_outer + 2, 0, next_stage0_regs);
                }
                if(has_next_stage1)
                {
                    global_to_regs(k_group_outer + 3, next_stage1_regs);
                }
                compute_stage(1, group_size(k_group_outer + 1));
                if(has_next_stage1)
                {
                    sync_after_compute(k_group_outer + 1);
                    regs_to_shared(k_group_outer + 3, 1, next_stage1_regs);
                    sync_after_load(k_group_outer + 2);
                }
                else if(has_next_stage0)
                {
                    sync_after_load(k_group_outer + 2);
                }
            }
            if(k_group_outer < k_group_count)
            {
                compute_stage(0, group_size(k_group_outer));
            }
        }
        else if constexpr(Pipeline)
        {
            Fp16StageRegs   pending_regs;
            Fp16ComputeRegs compute_regs;
            global_to_regs(0, pending_regs);
            regs_to_shared(0, 0, pending_regs);
            if(k_group_count >= 2)
            {
                Fp16StageRegs next_regs;
                global_to_regs(1, next_regs);
                sync_after_load(0);
                shared_to_compute_regs(0, group_size(0), compute_regs);
                for(int k_group_outer = 0; k_group_outer < k_group_count - 2; ++k_group_outer)
                {
                    sync_after_compute(k_group_outer);
                    regs_to_shared(k_group_outer + 1, 0, next_regs);
                    global_to_regs(k_group_outer + 2, next_regs);
                    compute_from_regs(group_size(k_group_outer), compute_regs);
                    sync_after_load(k_group_outer + 1);
                    shared_to_compute_regs(0, group_size(k_group_outer + 1), compute_regs);
                }
                int penultimate = k_group_count - 2;
                sync_after_compute(penultimate);
                regs_to_shared(penultimate + 1, 0, next_regs);
                compute_from_regs(group_size(penultimate), compute_regs);
                sync_after_load(penultimate + 1);
                shared_to_compute_regs(0, group_size(penultimate + 1), compute_regs);
                compute_from_regs(group_size(penultimate + 1), compute_regs);
            }
            else
            {
                sync_after_load(0);
                shared_to_compute_regs(0, group_size(0), compute_regs);
                compute_from_regs(group_size(0), compute_regs);
            }
        }
        else
        {
            for(int k_group_outer = 0; k_group_outer < k_group_count; ++k_group_outer)
            {
                load_stage(k_group_outer, 0);
                sync_after_load(k_group_outer);
                compute_stage(0, group_size(k_group_outer));
                sync_after_compute(k_group_outer);
            }
        }

        int c_tile_base = (m_outer * n_outer_count + n_outer) * MTiles * NTiles * kWaveSize;
#pragma unroll
        for(int t = 0; t < kTilesPerWave; ++t)
        {
#pragma unroll
            for(int n = 0; n < kNTilesPerWave; ++n)
            {
                int m_tile = wave_m * WaveTileM + t;
                int n_tile = wave_n * WaveTileN + n;
                C[c_tile_base + (m_tile * NTiles + n_tile) * kWaveSize + lane] = acc[t][n];
            }
        }
    }

    template <bool Pipeline, bool DoubleBuffer, SyncMode Sync>
    void launch_prod_16x8(const half_t* A, const half_t* B, float* C, int M, int N, int K)
    {
        dim3 block(kThreadsPerWorkgroup);
        dim3 grid(ceil_div(N, kProductionNTiles * kWmmaN), ceil_div(M, kProductionMTiles * kWmmaM));
        hipLaunchKernelGGL((fp16_wmma_tiled_kernel<kProductionMTiles,
                                                   kProductionNTiles,
                                                   Pipeline,
                                                   DoubleBuffer,
                                                   Sync,
                                                   kWavesPerWorkgroup,
                                                   kProductionKGroup,
                                                   kProductionWaveTileM,
                                                   kProductionWaveTileN>),
                           grid,
                           block,
                           0,
                           0,
                           A,
                           B,
                           C,
                           M,
                           N,
                           K);
    }

    std::vector<half_t> make_fp16_tiled(int rows, int cols, int row_tiles, int seed)
    {
        std::vector<half_t> out(rows * cols);
        int                 outer_rows = rows / (row_tiles * kWmmaM);
        int                 outer_cols = cols / kWmmaK;
        for(int ro = 0; ro < outer_rows; ++ro)
        {
            for(int co = 0; co < outer_cols; ++co)
            {
                for(int subtile = 0; subtile < row_tiles; ++subtile)
                {
                    for(int inner = 0; inner < kWmmaM; ++inner)
                    {
                        int logical_row = ro * row_tiles * kWmmaM + subtile * kWmmaM + inner;
                        for(int k = 0; k < kWmmaK; ++k)
                        {
                            int         logical_col = co * kWmmaK + k;
                            std::size_t off         = packed_ab_offset(
                                ro, co, outer_cols, row_tiles, subtile, inner, k);
                            out[off] = static_cast<half_t>(
                                small_i32_value(logical_row, logical_col, seed));
                        }
                    }
                }
            }
        }
        return out;
    }

    std::vector<float> reference_fp16_tiled(int M, int N, int K, int MTiles, int NTiles)
    {
        std::vector<float> C(static_cast<std::size_t>(M) * N, 0.0f);
        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                int sum = 0;
                for(int k = 0; k < K; ++k)
                {
                    sum += small_i32_value(m, k, /*seed=*/1) * small_i32_value(n, k, /*seed=*/2);
                }
                C[packed_c_offset(M, N, MTiles, NTiles, m, n)] = static_cast<float>(sum);
            }
        }
        return C;
    }

    template <typename T>
    T* gpu_alloc(std::size_t count)
    {
        T*         ptr    = nullptr;
        hipError_t status = hipMalloc(&ptr, count * sizeof(T));
        EXPECT_EQ(status, hipSuccess) << "hipMalloc: " << hipGetErrorString(status);
        return ptr;
    }

    template <typename T>
    void copy_to_gpu(T* dst, const std::vector<T>& src)
    {
        HIP_MOI_REFERENCE_HIP_ASSERT(
            hipMemcpy(dst, src.data(), src.size() * sizeof(T), hipMemcpyHostToDevice));
    }

    struct ReferenceVariant
    {
        const char* name;
        void (*launch)(const half_t*, const half_t*, float*, int, int, int);
    };

    const ReferenceVariant kNoPipelineVariant{
        "NoPipelineProd16x8",
        launch_prod_16x8<false, false, SyncMode::Correct>,
    };
    const ReferenceVariant kPipelinedVariant{
        "PipelinedProd16x8",
        launch_prod_16x8<true, false, SyncMode::Correct>,
    };
    const ReferenceVariant kDoubleBufferedVariant{
        "DoubleBufferedProd16x8",
        launch_prod_16x8<true, true, SyncMode::Correct>,
    };
    const ReferenceVariant kMissingLoadComputeBarrierVariant{
        "MissingLoadComputeBarrierProd16x8",
        launch_prod_16x8<true, false, SyncMode::MissingLoadComputeBarrier>,
    };
    const ReferenceVariant kConditionalMissingLoadComputeBarrierVariant{
        "ConditionalMissingLoadComputeBarrierProd16x8",
        launch_prod_16x8<true, false, SyncMode::SometimesMissingLoadComputeBarrier>,
    };
    const ReferenceVariant kMissingComputeLoadBarrierVariant{
        "MissingComputeLoadBarrierProd16x8",
        launch_prod_16x8<true, true, SyncMode::MissingComputeLoadBarrier>,
    };
    const ReferenceVariant kConditionalMissingComputeLoadBarrierVariant{
        "ConditionalMissingComputeLoadBarrierProd16x8",
        launch_prod_16x8<true, true, SyncMode::SometimesMissingComputeLoadBarrier>,
    };

    class JakubRdna4MatmulReference : public testing::TestWithParam<const ReferenceVariant*>
    {
    };

    TEST_P(JakubRdna4MatmulReference, MatchesHostReference)
    {
        const ReferenceVariant& variant = *GetParam();

        std::vector<half_t> host_a
            = make_fp16_tiled(kCheckM, kCheckK, kProductionMTiles, /*seed=*/1);
        std::vector<half_t> host_b
            = make_fp16_tiled(kCheckN, kCheckK, kProductionNTiles, /*seed=*/2);
        std::vector<float> expected
            = reference_fp16_tiled(kCheckM, kCheckN, kCheckK, kProductionMTiles, kProductionNTiles);
        std::vector<float> got(expected.size(), -1.0f);

        half_t* device_a = gpu_alloc<half_t>(host_a.size());
        half_t* device_b = gpu_alloc<half_t>(host_b.size());
        float*  device_c = gpu_alloc<float>(got.size());
        ASSERT_NE(device_a, nullptr);
        ASSERT_NE(device_b, nullptr);
        ASSERT_NE(device_c, nullptr);

        copy_to_gpu(device_a, host_a);
        copy_to_gpu(device_b, host_b);
        HIP_MOI_REFERENCE_HIP_ASSERT(hipMemset(device_c, 0, got.size() * sizeof(float)));

        variant.launch(device_a, device_b, device_c, kCheckM, kCheckN, kCheckK);
        HIP_MOI_REFERENCE_HIP_ASSERT(hipGetLastError());
        HIP_MOI_REFERENCE_HIP_ASSERT(hipDeviceSynchronize());
        HIP_MOI_REFERENCE_HIP_ASSERT(
            hipMemcpy(got.data(), device_c, got.size() * sizeof(float), hipMemcpyDeviceToHost));

        for(std::size_t i = 0; i < got.size(); ++i)
        {
            ASSERT_EQ(got[i], expected[i]) << "variant=" << variant.name << " index=" << i;
        }

        HIP_MOI_REFERENCE_HIP_ASSERT(hipFree(device_a));
        HIP_MOI_REFERENCE_HIP_ASSERT(hipFree(device_b));
        HIP_MOI_REFERENCE_HIP_ASSERT(hipFree(device_c));
    }

    TEST(JakubRdna4MatmulReferenceNegativeShapes, MissingBarrierVariantsCompileOnly)
    {
        EXPECT_NE(kMissingLoadComputeBarrierVariant.launch, nullptr);
        EXPECT_NE(kConditionalMissingLoadComputeBarrierVariant.launch, nullptr);
        EXPECT_NE(kMissingComputeLoadBarrierVariant.launch, nullptr);
        EXPECT_NE(kConditionalMissingComputeLoadBarrierVariant.launch, nullptr);
    }

    std::string variant_name(const testing::TestParamInfo<const ReferenceVariant*>& info)
    {
        return info.param->name;
    }

    INSTANTIATE_TEST_SUITE_P(SafeFp16Packed,
                             JakubRdna4MatmulReference,
                             testing::Values(&kNoPipelineVariant,
                                             &kPipelinedVariant,
                                             &kDoubleBufferedVariant),
                             variant_name);
} // namespace
