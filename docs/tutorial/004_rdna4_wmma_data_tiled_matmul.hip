// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// docs/tutorial/004_rdna4_wmma_data_tiled_matmul.hip
//
// A real RDNA4 WMMA example using data-tiled A/B/C fragments and hip-moi
// instrumentation for the LDS fragment staging.
#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace
{
    constexpr int kThreadCount        = 32;
    constexpr int kTileK              = 16;
    constexpr int kFragmentElements   = 8;
    constexpr int kPackedTileElements = kThreadCount * kFragmentElements;
    constexpr int kOutputElements     = kPackedTileElements;

    using f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
    using f32x8_t = float __attribute__((ext_vector_type(8)));

    void check_hip(hipError_t status, const char* expression)
    {
        if(status == hipSuccess)
        {
            return;
        }

        std::fprintf(stderr, "%s failed: %s\n", expression, hipGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }

    __host__ __device__ int a_value(int m, int k)
    {
        return ((2 * m + 3 * k) % 5) - 2;
    }

    __host__ __device__ int b_value(int n, int k)
    {
        return ((1 + 3 * n + k) % 5) - 2;
    }

    __host__ __device__ int rdna4_wmma_f16_k(int lane, int idx)
    {
        int reg  = idx >> 1;
        int half = idx & 1;
        return half | ((reg & 1) << 1) | ((lane >> 4) << 2) | ((reg >> 1) << 3);
    }

    __host__ __device__ int rdna4_wmma_acc_m(int lane, int elem)
    {
        return elem + 8 * (lane >> 4);
    }

    __host__ __device__ int data_tiled_fragment_offset(int lane, int elem)
    {
        return lane * kFragmentElements + elem;
    }

    hip_moi::host_context_options context_options()
    {
        hip_moi::host_context_options options;
        options.access_record_capacity = 128;
        options.diagnostic_capacity    = 8;
        options.subgroup_capacity      = 1;
        return options;
    }

    void fill_data_tiled_inputs(_Float16* a, _Float16* b)
    {
        for(int lane = 0; lane < kThreadCount; ++lane)
        {
            int m = lane & 15;
            int n = lane & 15;
            for(int idx = 0; idx < kFragmentElements; ++idx)
            {
                int k      = rdna4_wmma_f16_k(lane, idx);
                int offset = data_tiled_fragment_offset(lane, idx);
                a[offset]  = static_cast<_Float16>(a_value(m, k));
                b[offset]  = static_cast<_Float16>(b_value(n, k));
            }
        }
    }

    float reference_value(int m, int n)
    {
        int acc = 0;
        for(int k = 0; k < kTileK; ++k)
        {
            acc += a_value(m, k) * b_value(n, k);
        }
        return static_cast<float>(acc);
    }

    bool check_reference_output(const float* output, const char* label)
    {
        for(int lane = 0; lane < kThreadCount; ++lane)
        {
            int n = lane & 15;
            for(int elem = 0; elem < kFragmentElements; ++elem)
            {
                int   m        = rdna4_wmma_acc_m(lane, elem);
                float expected = reference_value(m, n);
                float actual   = output[data_tiled_fragment_offset(lane, elem)];
                if(actual != expected)
                {
                    std::fprintf(stderr,
                                 "%s C[%d,%d] mismatch: expected %.1f, got %.1f\n",
                                 label,
                                 m,
                                 n,
                                 expected,
                                 actual);
                    return false;
                }
            }
        }
        return true;
    }

    __device__ f16x8_t load_global_fragment(const _Float16* global_fragments, int lane)
    {
        return reinterpret_cast<const f16x8_t*>(global_fragments)[lane];
    }

    __device__ void store_global_fragment(float* global_fragments, int lane, f32x8_t fragment)
    {
        reinterpret_cast<f32x8_t*>(global_fragments)[lane] = fragment;
    }

    __global__ void plain_data_tiled_wmma_kernel(const _Float16* a_global,
                                                 const _Float16* b_global,
                                                 float*          c_global)
    {
        __shared__ f16x8_t a_shared[kThreadCount];
        __shared__ f16x8_t b_shared[kThreadCount];

        int lane = static_cast<int>(threadIdx.x) & 31;

        a_shared[lane] = load_global_fragment(a_global, lane);
        b_shared[lane] = load_global_fragment(b_global, lane);

        __syncthreads();

        f16x8_t a   = a_shared[lane];
        f16x8_t b   = b_shared[lane];
        f32x8_t acc = {};
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);

        __syncthreads();

        store_global_fragment(c_global, lane, acc);
    }

    __device__ hip_moi::thread_level_context make_context(hip_moi::thread_level_context::storage_ref storage)
    {
        hip_moi::thread_level_context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
            /*subgroup_count=*/1,
        };
        return hip_moi::thread_level_context(storage, cfg);
    }

    __global__ void instrumented_data_tiled_wmma_kernel(const _Float16*             a_global,
                                                        const _Float16*             b_global,
                                                        float*                      c_global,
                                                        hip_moi::thread_level_context::storage_ref storage)
    {
        __shared__ f16x8_t a_shared[kThreadCount];
        __shared__ f16x8_t b_shared[kThreadCount];

        hip_moi::thread_level_context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane = static_cast<int>(threadIdx.x) & 31;

        ctx.lds_store(&a_shared[lane], load_global_fragment(a_global, lane));
        ctx.lds_store(&b_shared[lane], load_global_fragment(b_global, lane));

        ctx.syncthreads();

        f16x8_t a   = ctx.lds_load(&a_shared[lane]);
        f16x8_t b   = ctx.lds_load(&b_shared[lane]);
        f32x8_t acc = {};
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);

        ctx.syncthreads();

        store_global_fragment(c_global, lane, acc);
    }
} // namespace

int main()
{
    _Float16 host_a[kPackedTileElements];
    _Float16 host_b[kPackedTileElements];
    float    plain_c[kOutputElements]       = {};
    float    instrumented_c[kOutputElements] = {};
    fill_data_tiled_inputs(host_a, host_b);

    _Float16* device_a = nullptr;
    _Float16* device_b = nullptr;
    float*    device_plain_c = nullptr;
    float*    device_moi_c   = nullptr;
    check_hip(hipMalloc(&device_a, sizeof(host_a)), "hipMalloc(device_a)");
    check_hip(hipMalloc(&device_b, sizeof(host_b)), "hipMalloc(device_b)");
    check_hip(hipMalloc(&device_plain_c, sizeof(plain_c)), "hipMalloc(device_plain_c)");
    check_hip(hipMalloc(&device_moi_c, sizeof(instrumented_c)), "hipMalloc(device_moi_c)");
    check_hip(hipMemcpy(device_a, host_a, sizeof(host_a), hipMemcpyHostToDevice),
              "hipMemcpy(device_a)");
    check_hip(hipMemcpy(device_b, host_b, sizeof(host_b), hipMemcpyHostToDevice),
              "hipMemcpy(device_b)");

    hipLaunchKernelGGL(plain_data_tiled_wmma_kernel,
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       device_a,
                       device_b,
                       device_plain_c);
    check_hip(hipGetLastError(), "hipGetLastError(plain)");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(plain)");

    hip_moi::host_context moi(context_options());
    hipLaunchKernelGGL(instrumented_data_tiled_wmma_kernel,
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       device_a,
                       device_b,
                       device_moi_c,
                       moi.device_ref());
    check_hip(hipGetLastError(), "hipGetLastError(instrumented)");

    HIP_MOI_CHECK(moi);

    check_hip(hipMemcpy(plain_c, device_plain_c, sizeof(plain_c), hipMemcpyDeviceToHost),
              "hipMemcpy(plain_c)");
    check_hip(hipMemcpy(instrumented_c,
                        device_moi_c,
                        sizeof(instrumented_c),
                        hipMemcpyDeviceToHost),
              "hipMemcpy(instrumented_c)");

    check_hip(hipFree(device_a), "hipFree(device_a)");
    check_hip(hipFree(device_b), "hipFree(device_b)");
    check_hip(hipFree(device_plain_c), "hipFree(device_plain_c)");
    check_hip(hipFree(device_moi_c), "hipFree(device_moi_c)");

    if(!check_reference_output(plain_c, "plain")
       || !check_reference_output(instrumented_c, "instrumented"))
    {
        return EXIT_FAILURE;
    }

    std::printf("hip-moi tutorial 004: plain and instrumented matmul passed\n");
    return EXIT_SUCCESS;
}
