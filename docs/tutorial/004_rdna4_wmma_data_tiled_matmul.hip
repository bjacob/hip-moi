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

    constexpr int kASharedByteOffset = 0;
    constexpr int kBSharedByteOffset = kThreadCount * static_cast<int>(sizeof(f16x8_t));

    void check_hip(hipError_t status, const char* expression)
    {
        if(status == hipSuccess)
        {
            return;
        }

        std::fprintf(stderr, "%s failed: %s\n", expression, hipGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }

    // Small integer inputs make the f16->f32 WMMA result exactly comparable to
    // the host reference, avoiding tolerance-based checks in the tutorial.
    __host__ __device__ int a_value(int m, int k)
    {
        return ((2 * m + 3 * k) % 5) - 2;
    }

    __host__ __device__ int b_value(int n, int k)
    {
        return ((1 + 3 * n + k) % 5) - 2;
    }

    // RDNA4 maps each lane's f16x8 fragment elements to K coordinates in a
    // fixed pattern. The tutorial uses that mapping to pack A/B inputs exactly
    // as the WMMA intrinsic expects them.
    __host__ __device__ int rdna4_wmma_f16_k(int lane, int idx)
    {
        int reg  = idx >> 1;
        int half = idx & 1;
        return half | ((reg & 1) << 1) | ((lane >> 4) << 2) | ((reg >> 1) << 3);
    }

    // Each lane receives eight accumulator elements. This maps a lane-local
    // accumulator element back to the matrix row it represents.
    __host__ __device__ int rdna4_wmma_acc_m(int lane, int elem)
    {
        return elem + 8 * (lane >> 4);
    }

    // Data-tiled layout stores one vector fragment per lane contiguously. Lane
    // i therefore starts at i * kFragmentElements.
    __host__ __device__ int data_tiled_fragment_offset(int lane, int elem)
    {
        return lane * kFragmentElements + elem;
    }

    // The matmul uses one 32-thread subgroup, so hip-moi only needs one
    // subgroup slot. The storage budget is intentionally small for the tutorial.
    hip_moi::host_context_options context_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes     = 64 * 1024;
        options.subgroup_capacity = 1;
        return options;
    }

    // Prepare host inputs already packed in the data-tiled per-lane layout that
    // load_global_fragment reads with one f16x8 vector load.
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

    // The kernel stores C in the same data-tiled per-lane layout, so validation
    // walks lane fragments rather than a conventional row-major matrix.
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

    // One lane owns one packed f16x8 fragment in global memory.
    __device__ f16x8_t load_global_fragment(const _Float16* global_fragments, int lane)
    {
        return reinterpret_cast<const f16x8_t*>(global_fragments)[lane];
    }

    // Store the lane's f32x8 accumulator fragment with the same data-tiled
    // vector layout used by the input fragments.
    __device__ void store_global_fragment(float* global_fragments, int lane, f32x8_t fragment)
    {
        reinterpret_cast<f32x8_t*>(global_fragments)[lane] = fragment;
    }

    // Plain HIP/RDNA4 version: stage data-tiled A/B fragments through LDS,
    // synchronize, then execute one WMMA instruction per lane.
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

    // Create the general diagnostic context inside the kernel. This tutorial
    // uses one subgroup, so it is a correctness smoke test rather than a race
    // diagnostic example.
    __device__ hip_moi::context make_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
            /*subgroup_count=*/1,
        };
        return hip_moi::context(storage, cfg);
    }

    // Instrumented version of the same matmul: only the LDS publication and
    // consumption are wrapped; the WMMA intrinsic and global stores are ordinary
    // kernel code.
    __global__ void instrumented_data_tiled_wmma_kernel(const _Float16*             a_global,
                                                        const _Float16*             b_global,
                                                        float*                      c_global,
                                                        hip_moi::context::storage_ref storage)
    {
        __shared__ f16x8_t a_shared[kThreadCount];
        __shared__ f16x8_t b_shared[kThreadCount];

        hip_moi::context ctx = make_context(storage);
        ctx.init_workgroup();

        int lane = static_cast<int>(threadIdx.x) & 31;

        // The explicit LDS byte offsets distinguish the A and B shared arrays
        // and identify each lane's vector slot within them.
        ctx.lds_store_at(&a_shared[lane],
                         load_global_fragment(a_global, lane),
                         /*lds_byte_offset=*/kASharedByteOffset
                             + lane * static_cast<int>(sizeof(f16x8_t)));
        ctx.lds_store_at(&b_shared[lane],
                         load_global_fragment(b_global, lane),
                         /*lds_byte_offset=*/kBSharedByteOffset
                             + lane * static_cast<int>(sizeof(f16x8_t)));

        ctx.syncthreads();

        // After ctx.syncthreads(), these loads are in the next epoch and should
        // not conflict with the earlier stores.
        f16x8_t a = ctx.lds_load_at(&a_shared[lane],
                                    /*lds_byte_offset=*/kASharedByteOffset
                                        + lane * static_cast<int>(sizeof(f16x8_t)));
        f16x8_t b = ctx.lds_load_at(&b_shared[lane],
                                    /*lds_byte_offset=*/kBSharedByteOffset
                                        + lane * static_cast<int>(sizeof(f16x8_t)));
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

    // host_context owns the device metadata buffer used by the instrumented
    // kernel and is checked immediately after launch.
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
