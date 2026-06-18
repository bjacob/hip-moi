// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/reference/mvp_reference_kernels.hip
//
// Uninstrumented HIP reference kernels for the hip-moi MVP test corpus.
//
#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <string>
#include <vector>

namespace
{

    constexpr int kDefaultThreadCount = 8;
    constexpr int kArrayThreadCount   = 32;
    constexpr int kMaxOutputCount     = 256;
    constexpr int kSentinel           = -1234567;

    struct Pair
    {
        int x;
        int y;
    };

    class ReferenceCase
    {
    public:
        virtual ~ReferenceCase()                     = default;
        virtual const char* name() const             = 0;
        virtual dim3        grid_dim() const         = 0;
        virtual dim3        block_dim() const        = 0;
        virtual void        launch(int* out) const   = 0;
        virtual void        expected(int* out) const = 0;
    };

#define HIP_MOI_REFERENCE_CASE(CaseClass, KernelClass, ThreadCount)                   \
    class CaseClass final : public ReferenceCase                                      \
    {                                                                                 \
    public:                                                                           \
        CaseClass()                                                                   \
            : grid_dim_(1)                                                            \
            , block_dim_(ThreadCount)                                                 \
        {                                                                             \
        }                                                                             \
        const char* name() const override                                             \
        {                                                                             \
            return KernelClass::name();                                               \
        }                                                                             \
        dim3 grid_dim() const override                                                \
        {                                                                             \
            return grid_dim_;                                                         \
        }                                                                             \
        dim3 block_dim() const override                                               \
        {                                                                             \
            return block_dim_;                                                        \
        }                                                                             \
        void launch(int* out) const override                                          \
        {                                                                             \
            hipLaunchKernelGGL(KernelClass::run, grid_dim(), block_dim(), 0, 0, out); \
        }                                                                             \
        void expected(int* out) const override                                        \
        {                                                                             \
            KernelClass::expected(out);                                               \
        }                                                                             \
                                                                                      \
    private:                                                                          \
        dim3 grid_dim_;                                                               \
        dim3 block_dim_;                                                              \
    }

#define HIP_MOI_REFERENCE_GRID_CASE(CaseClass, KernelClass, ThreadCount, GridX, GridY, GridZ) \
    class CaseClass final : public ReferenceCase                                              \
    {                                                                                         \
    public:                                                                                   \
        CaseClass()                                                                           \
            : grid_dim_(GridX, GridY, GridZ)                                                  \
            , block_dim_(ThreadCount)                                                         \
        {                                                                                     \
        }                                                                                     \
        const char* name() const override                                                     \
        {                                                                                     \
            return KernelClass::name();                                                       \
        }                                                                                     \
        dim3 grid_dim() const override                                                        \
        {                                                                                     \
            return grid_dim_;                                                                 \
        }                                                                                     \
        dim3 block_dim() const override                                                       \
        {                                                                                     \
            return block_dim_;                                                                \
        }                                                                                     \
        void launch(int* out) const override                                                  \
        {                                                                                     \
            hipLaunchKernelGGL(KernelClass::run, grid_dim(), block_dim(), 0, 0, out);         \
        }                                                                                     \
        void expected(int* out) const override                                                \
        {                                                                                     \
            KernelClass::expected(out);                                                       \
        }                                                                                     \
                                                                                              \
    private:                                                                                  \
        dim3 grid_dim_;                                                                       \
        dim3 block_dim_;                                                                      \
    }

    __host__ __device__ int wrap_power_of_two(int value, int modulus)
    {
        return value & (modulus - 1);
    }

    __host__ __device__ int matmul_a_value(int tile_id, int m, int k)
    {
        return 1 + tile_id + 2 * m + k;
    }

    __host__ __device__ int matmul_b_value(int tile_id, int n, int k)
    {
        return 2 + tile_id + n + 3 * k;
    }

    __host__ __device__ int matmul_expected_value(int tile_id, int m, int n, int k_size)
    {
        int acc = 0;
        for(int k = 0; k < k_size; ++k)
        {
            acc += matmul_a_value(tile_id, m, k) * matmul_b_value(tile_id, n, k);
        }
        return acc;
    }

    template <int MTile, int NTile, int KTile>
    void fill_matmul_expected(int* expected, int output_offset, int tile_id)
    {
        for(int m = 0; m < MTile; ++m)
        {
            for(int n = 0; n < NTile; ++n)
            {
                expected[output_offset + n + NTile * m]
                    = matmul_expected_value(tile_id, m, n, KTile);
            }
        }
    }

    template <int MTile, int NTile, int KTile, int Threads>
    __device__ void load_matmul_tile(int* a_shared, int* b_shared, int tile_id)
    {
        int tid = threadIdx.x;
        for(int idx = tid; idx < MTile * KTile; idx += Threads)
        {
            int m         = idx / KTile;
            int k         = idx - KTile * m;
            a_shared[idx] = matmul_a_value(tile_id, m, k);
        }
        for(int idx = tid; idx < NTile * KTile; idx += Threads)
        {
            int n         = idx / KTile;
            int k         = idx - KTile * n;
            b_shared[idx] = matmul_b_value(tile_id, n, k);
        }
    }

    template <int MTile, int NTile, int KTile>
    __device__ int matmul_mac_from_lds(const int* a_shared, const int* b_shared, int m, int n)
    {
        int acc = 0;
        for(int k = 0; k < KTile; ++k)
        {
            acc += a_shared[k + KTile * m] * b_shared[k + KTile * n];
        }
        return acc;
    }

    template <int MTile, int NTile, int KTile, int Threads>
    __device__ void run_simple_matmul_body(
        int* out, int output_offset, int tile_id, int* a_shared, int* b_shared)
    {
        int tid = threadIdx.x;
        load_matmul_tile<MTile, NTile, KTile, Threads>(a_shared, b_shared, tile_id);
        __syncthreads();

        if(tid < MTile * NTile)
        {
            int m = tid / NTile;
            int n = tid - NTile * m;
            out[output_offset + tid]
                = matmul_mac_from_lds<MTile, NTile, KTile>(a_shared, b_shared, m, n);
        }
        __syncthreads();
    }

    template <int MTile, int NTile, int KTile, int KOuter, int Threads>
    __device__ void run_chunked_matmul_body(int* out, int* a_shared, int* b_shared)
    {
        int tid = threadIdx.x;
        int acc = 0;
        int m   = tid / NTile;
        int n   = tid - NTile * m;

        for(int k_outer = 0; k_outer < KOuter; ++k_outer)
        {
            int tile_id = k_outer;
            load_matmul_tile<MTile, NTile, KTile, Threads>(a_shared, b_shared, tile_id);
            __syncthreads();
            if(tid < MTile * NTile)
            {
                acc += matmul_mac_from_lds<MTile, NTile, KTile>(a_shared, b_shared, m, n);
            }
            __syncthreads();
        }

        if(tid < MTile * NTile)
        {
            out[tid] = acc;
        }
    }

    template <int MTile, int NTile, int KTile, int KOuter>
    void fill_chunked_matmul_expected(int* expected)
    {
        for(int m = 0; m < MTile; ++m)
        {
            for(int n = 0; n < NTile; ++n)
            {
                int acc = 0;
                for(int k_outer = 0; k_outer < KOuter; ++k_outer)
                {
                    acc += matmul_expected_value(k_outer, m, n, KTile);
                }
                expected[n + NTile * m] = acc;
            }
        }
    }

    template <int MTile, int NTile, int KTile, int Threads>
    __device__ void run_skewed_matmul_body(int* out, int* a_shared, int* b_shared)
    {
        constexpr int AStride = KTile + 1;
        constexpr int BStride = KTile + 1;
        int           tid     = threadIdx.x;

        for(int idx = tid; idx < MTile * KTile; idx += Threads)
        {
            int m                     = idx / KTile;
            int k                     = idx - KTile * m;
            a_shared[k + AStride * m] = matmul_a_value(0, m, k);
        }
        for(int idx = tid; idx < NTile * KTile; idx += Threads)
        {
            int n                     = idx / KTile;
            int k                     = idx - KTile * n;
            b_shared[k + BStride * n] = matmul_b_value(0, n, k);
        }
        __syncthreads();

        if(tid < MTile * NTile)
        {
            int m   = tid / NTile;
            int n   = tid - NTile * m;
            int acc = 0;
            for(int k = 0; k < KTile; ++k)
            {
                acc += a_shared[k + AStride * m] * b_shared[k + BStride * n];
            }
            out[tid] = acc;
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------- //
    // Tiny scalar references.
    // -------------------------------------------------------------------------- //

    struct SameThreadStoreLoad
    {
        static const char* name()
        {
            return "same_thread_store_load";
        }
        [[maybe_unused]] static constexpr int thread_count = kDefaultThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int value;
            if(threadIdx.x == 0)
            {
                value  = 123;
                out[0] = value;
            }
        }
        static void expected(int* out)
        {
            out[0] = 123;
        }
    };

    struct DifferentThreadsNonoverlappingAddresses
    {
        static const char* name()
        {
            return "different_threads_nonoverlapping_addresses";
        }
        [[maybe_unused]] static constexpr int thread_count = kDefaultThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int values[2];
            if(threadIdx.x == 0)
            {
                values[0] = 10;
                out[0]    = values[0];
            }
            if(threadIdx.x == 1)
            {
                values[1] = 20;
                out[1]    = values[1];
            }
        }
        static void expected(int* out)
        {
            out[0] = 10;
            out[1] = 20;
        }
    };

    struct DifferentThreadsReadSameAddress
    {
        static const char* name()
        {
            return "different_threads_read_same_address";
        }
        [[maybe_unused]] static constexpr int thread_count = kDefaultThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int value;
            if(threadIdx.x == 0)
            {
                value = 77;
            }
            __syncthreads();
            if(threadIdx.x < 2)
            {
                out[threadIdx.x] = value;
            }
        }
        static void expected(int* out)
        {
            out[0] = 77;
            out[1] = 77;
        }
    };

    struct AdjacentAddressesSameEpoch
    {
        static const char* name()
        {
            return "adjacent_addresses_same_epoch";
        }
        [[maybe_unused]] static constexpr int thread_count = kDefaultThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int values[2];
            if(threadIdx.x == 0)
            {
                values[0] = 11;
            }
            if(threadIdx.x == 1)
            {
                values[1] = 22;
            }
            __syncthreads();
            if(threadIdx.x < 2)
            {
                out[threadIdx.x] = values[threadIdx.x];
            }
        }
        static void expected(int* out)
        {
            out[0] = 11;
            out[1] = 22;
        }
    };

    struct WriteReadSeparatedBySyncthreads
    {
        static const char* name()
        {
            return "write_read_separated_by_syncthreads";
        }
        [[maybe_unused]] static constexpr int thread_count = kDefaultThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int value;
            if(threadIdx.x == 0)
            {
                value = 321;
            }
            __syncthreads();
            if(threadIdx.x == 1)
            {
                out[0] = value;
            }
        }
        static void expected(int* out)
        {
            out[0] = 321;
        }
    };

    struct WriteWriteSeparatedBySyncthreads
    {
        static const char* name()
        {
            return "write_write_separated_by_syncthreads";
        }
        [[maybe_unused]] static constexpr int thread_count = kDefaultThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int value;
            if(threadIdx.x == 0)
            {
                value = 12;
            }
            __syncthreads();
            if(threadIdx.x == 1)
            {
                value = 34;
            }
            __syncthreads();
            if(threadIdx.x == 2)
            {
                out[0] = value;
            }
        }
        static void expected(int* out)
        {
            out[0] = 34;
        }
    };

    struct MultipleEpochsReuseSameAddress
    {
        static const char* name()
        {
            return "multiple_epochs_reuse_same_address";
        }
        [[maybe_unused]] static constexpr int thread_count = kDefaultThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int value;
            if(threadIdx.x == 0)
            {
                value = 5;
            }
            __syncthreads();
            if(threadIdx.x == 1)
            {
                out[0] = value;
            }
            __syncthreads();
            if(threadIdx.x == 1)
            {
                value = 9;
            }
            __syncthreads();
            if(threadIdx.x == 0)
            {
                out[1] = value;
            }
        }
        static void expected(int* out)
        {
            out[0] = 5;
            out[1] = 9;
        }
    };

    struct TriviallyCopyableStructAccess
    {
        static const char* name()
        {
            return "trivially_copyable_struct_access";
        }
        [[maybe_unused]] static constexpr int thread_count = kDefaultThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ Pair pair;
            if(threadIdx.x == 0)
            {
                pair = Pair{4, 8};
            }
            __syncthreads();
            if(threadIdx.x == 1)
            {
                out[0] = pair.x;
            }
            if(threadIdx.x == 2)
            {
                out[1] = pair.y;
            }
        }
        static void expected(int* out)
        {
            out[0] = 4;
            out[1] = 8;
        }
    };

    // -------------------------------------------------------------------------- //
    // Cooperative array references: all or most threads participate.
    // -------------------------------------------------------------------------- //

    struct ArrayIdentityAllThreads
    {
        static const char* name()
        {
            return "array_identity_all_threads";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid = threadIdx.x;
            data[tid]          = 100 + tid;
            __syncthreads();
            out[tid] = data[tid];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 100 + i;
            }
        }
    };

    struct ArrayNeighborReadAllThreads
    {
        static const char* name()
        {
            return "array_neighbor_read_all_threads";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid = threadIdx.x;
            data[tid]          = tid;
            __syncthreads();
            out[tid] = data[wrap_power_of_two(tid + 1, kArrayThreadCount)];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = wrap_power_of_two(i + 1, thread_count);
            }
        }
    };

    struct ArrayReverseReadAllThreads
    {
        static const char* name()
        {
            return "array_reverse_read_all_threads";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid = threadIdx.x;
            data[tid]          = 10 * tid;
            __syncthreads();
            out[tid] = data[kArrayThreadCount - 1 - tid];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 10 * (thread_count - 1 - i);
            }
        }
    };

    struct ArrayPermutationMultiply3AllThreads
    {
        static const char* name()
        {
            return "array_permutation_multiply3_all_threads";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid                                  = threadIdx.x;
            data[wrap_power_of_two(3 * tid, kArrayThreadCount)] = 200 + tid;
            __syncthreads();
            out[tid] = data[wrap_power_of_two(3 * tid, kArrayThreadCount)];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 200 + i;
            }
        }
    };

    struct ArrayPermutationMultiply5Gather
    {
        static const char* name()
        {
            return "array_permutation_multiply5_gather";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid = threadIdx.x;
            data[tid]          = 2 * tid;
            __syncthreads();
            out[tid] = data[wrap_power_of_two(5 * tid, kArrayThreadCount)];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 2 * wrap_power_of_two(5 * i, thread_count);
            }
        }
    };

    struct ArrayTwoArraysZip
    {
        static const char* name()
        {
            return "array_two_arrays_zip";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int a[kArrayThreadCount];
            __shared__ int b[kArrayThreadCount];
            int            tid = threadIdx.x;
            a[tid]             = tid;
            b[tid]             = 1000 + tid;
            __syncthreads();
            out[2 * tid]     = a[tid];
            out[2 * tid + 1] = b[kArrayThreadCount - 1 - tid];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[2 * i]     = i;
                out[2 * i + 1] = 1000 + thread_count - 1 - i;
            }
        }
    };

    struct ArrayTranspose4x8
    {
        static const char* name()
        {
            return "array_transpose_4x8";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int tile[4][8];
            int            tid = threadIdx.x;
            int            row = tid / 8;
            int            col = tid - 8 * row;
            tile[row][col]     = 10 * row + col;
            __syncthreads();
            out[tid] = tile[tid & 3][tid >> 2];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 10 * (i & 3) + (i >> 2);
            }
        }
    };

    struct ArrayPairwiseSwap
    {
        static const char* name()
        {
            return "array_pairwise_swap";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid = threadIdx.x;
            data[tid]          = tid + 10;
            __syncthreads();
            out[tid] = data[tid ^ 1];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = (i ^ 1) + 10;
            }
        }
    };

    struct ArrayGroupBroadcast
    {
        static const char* name()
        {
            return "array_group_broadcast";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int group_values[8];
            int            tid   = threadIdx.x;
            int            group = tid >> 2;
            int            lane  = tid & 3;
            if(lane == 0)
            {
                group_values[group] = group * 100;
            }
            __syncthreads();
            out[tid] = group_values[group] + lane;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 100 * (i >> 2) + (i & 3);
            }
        }
    };

    struct ArrayRollingWindowSum
    {
        static const char* name()
        {
            return "array_rolling_window_sum";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid = threadIdx.x;
            data[tid]          = tid;
            __syncthreads();
            out[tid] = data[tid] + data[wrap_power_of_two(tid + 1, thread_count)]
                       + data[wrap_power_of_two(tid - 1, thread_count)];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = i + wrap_power_of_two(i + 1, thread_count)
                         + wrap_power_of_two(i - 1, thread_count);
            }
        }
    };

    struct ArrayStructPerThread
    {
        static const char* name()
        {
            return "array_struct_per_thread";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ Pair pairs[kArrayThreadCount];
            int             tid = threadIdx.x;
            pairs[tid]          = Pair{tid + 1, 2 * tid + 3};
            __syncthreads();
            int other = wrap_power_of_two(tid + 7, thread_count);
            out[tid]  = pairs[other].x + pairs[tid].y;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                int other = wrap_power_of_two(i + 7, thread_count);
                out[i]    = other + 1 + 2 * i + 3;
            }
        }
    };

    struct ArrayStridedTwoElementsPerThread
    {
        static const char* name()
        {
            return "array_strided_two_elements_per_thread";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[2 * kArrayThreadCount];
            int            tid = threadIdx.x;
            data[2 * tid]      = tid;
            data[2 * tid + 1]  = 100 + tid;
            __syncthreads();
            int other = wrap_power_of_two(7 * tid, thread_count);
            out[tid]  = data[2 * other + 1];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 100 + wrap_power_of_two(7 * i, thread_count);
            }
        }
    };

    // -------------------------------------------------------------------------- //
    // Looping references.
    // -------------------------------------------------------------------------- //

    struct LoopIdentityEpochs
    {
        static const char* name()
        {
            return "loop_identity_epochs";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid = threadIdx.x;
            int            acc = 0;
            for(int iter = 0; iter < 5; ++iter)
            {
                data[tid] = 100 * iter + tid;
                __syncthreads();
                acc += data[tid];
                __syncthreads();
            }
            out[tid] = acc;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 1000 + 5 * i;
            }
        }
    };

    struct LoopNeighborEpochs
    {
        static const char* name()
        {
            return "loop_neighbor_epochs";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid = threadIdx.x;
            int            acc = 0;
            for(int iter = 0; iter < 5; ++iter)
            {
                data[tid] = 100 * iter + tid;
                __syncthreads();
                acc += data[wrap_power_of_two(tid + 1, thread_count)];
                __syncthreads();
            }
            out[tid] = acc;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 1000 + 5 * wrap_power_of_two(i + 1, thread_count);
            }
        }
    };

    struct LoopReverseEpochs
    {
        static const char* name()
        {
            return "loop_reverse_epochs";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid = threadIdx.x;
            int            acc = 0;
            for(int iter = 0; iter < 5; ++iter)
            {
                data[tid] = 100 * iter + tid;
                __syncthreads();
                acc += data[kArrayThreadCount - 1 - tid];
                __syncthreads();
            }
            out[tid] = acc;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 1000 + 5 * (thread_count - 1 - i);
            }
        }
    };

    struct LoopPermutation3Epochs
    {
        static const char* name()
        {
            return "loop_permutation3_epochs";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid = threadIdx.x;
            int            acc = 0;
            for(int iter = 0; iter < 5; ++iter)
            {
                data[wrap_power_of_two(3 * tid, thread_count)] = 100 * iter + tid;
                __syncthreads();
                acc += data[wrap_power_of_two(3 * tid, thread_count)];
                __syncthreads();
            }
            out[tid] = acc;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 1000 + 5 * i;
            }
        }
    };

    struct LoopPermutation5Gather
    {
        static const char* name()
        {
            return "loop_permutation5_gather";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid = threadIdx.x;
            int            acc = 0;
            for(int iter = 0; iter < 5; ++iter)
            {
                data[tid] = 100 * iter + tid;
                __syncthreads();
                acc += data[wrap_power_of_two(5 * tid, thread_count)];
                __syncthreads();
            }
            out[tid] = acc;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 1000 + 5 * wrap_power_of_two(5 * i, thread_count);
            }
        }
    };

    struct LoopPingPongBuffers
    {
        static const char* name()
        {
            return "loop_ping_pong_buffers";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[2][kArrayThreadCount];
            int            tid = threadIdx.x;
            int            acc = 0;
            for(int iter = 0; iter < 6; ++iter)
            {
                data[iter & 1][tid] = iter + tid;
                __syncthreads();
                acc += data[iter & 1][tid ^ 1];
                __syncthreads();
            }
            out[tid] = acc;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 15 + 6 * (i ^ 1);
            }
        }
    };

    struct LoopRollingWindow
    {
        static const char* name()
        {
            return "loop_rolling_window";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid = threadIdx.x;
            int            acc = 0;
            for(int iter = 0; iter < 4; ++iter)
            {
                data[tid] = iter + tid;
                __syncthreads();
                acc += data[tid] + data[wrap_power_of_two(tid + 1, thread_count)];
                __syncthreads();
            }
            out[tid] = acc;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 12 + 4 * i + 4 * wrap_power_of_two(i + 1, thread_count);
            }
        }
    };

    struct LoopGroupBroadcast
    {
        static const char* name()
        {
            return "loop_group_broadcast";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int group_values[8];
            int            tid   = threadIdx.x;
            int            group = tid >> 2;
            int            lane  = tid & 3;
            int            acc   = 0;
            for(int iter = 0; iter < 4; ++iter)
            {
                if(lane == 0)
                {
                    group_values[group] = 10 * iter + group;
                }
                __syncthreads();
                acc += group_values[group] + lane;
                __syncthreads();
            }
            out[tid] = acc;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 60 + 4 * ((i >> 2) + (i & 3));
            }
        }
    };

    struct LoopTwoPhaseProducerConsumer
    {
        static const char* name()
        {
            return "loop_two_phase_producer_consumer";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int input[kArrayThreadCount];
            __shared__ int output[kArrayThreadCount];
            int            tid = threadIdx.x;
            int            acc = 0;
            for(int iter = 0; iter < 4; ++iter)
            {
                input[tid] = 100 * iter + tid;
                __syncthreads();
                output[tid] = input[wrap_power_of_two(tid + 1, thread_count)] + 1;
                __syncthreads();
                acc += output[tid];
                __syncthreads();
            }
            out[tid] = acc;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 604 + 4 * wrap_power_of_two(i + 1, thread_count);
            }
        }
    };

    struct LoopStructPerThread
    {
        static const char* name()
        {
            return "loop_struct_per_thread";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ Pair pairs[kArrayThreadCount];
            int             tid = threadIdx.x;
            int             acc = 0;
            for(int iter = 0; iter < 4; ++iter)
            {
                pairs[tid] = Pair{iter + tid, 10 * iter + tid};
                __syncthreads();
                int other = wrap_power_of_two(tid + iter, thread_count);
                acc += pairs[other].x + pairs[tid].y;
                __syncthreads();
            }
            out[tid] = acc;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                int acc = 0;
                for(int iter = 0; iter < 4; ++iter)
                {
                    int other = wrap_power_of_two(i + iter, thread_count);
                    acc += iter + other + 10 * iter + i;
                }
                out[i] = acc;
            }
        }
    };

    struct LoopPartialThreadCount16
    {
        static const char* name()
        {
            return "loop_partial_thread_count16";
        }
        [[maybe_unused]] static constexpr int thread_count = 16;
        __global__ static void run(int* out)
        {
            __shared__ int data[16];
            int            tid = threadIdx.x;
            int            acc = 0;
            for(int iter = 0; iter < 7; ++iter)
            {
                data[tid] = iter + 3 * tid;
                __syncthreads();
                acc += data[tid ^ 3];
                __syncthreads();
            }
            out[tid] = acc;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 21 + 21 * (i ^ 3);
            }
        }
    };

    struct LoopTwoDimTile4x8
    {
        static const char* name()
        {
            return "loop_two_dim_tile4x8";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int tile[4][8];
            int            tid = threadIdx.x;
            int            row = tid >> 3;
            int            col = tid & 7;
            int            acc = 0;
            for(int iter = 0; iter < 4; ++iter)
            {
                tile[row][col] = 100 * iter + 10 * row + col;
                __syncthreads();
                acc += tile[col & 3][row + 4 * (col >> 2)];
                __syncthreads();
            }
            out[tid] = acc;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                int row     = i >> 3;
                int col     = i & 7;
                int src_row = col & 3;
                int src_col = row + 4 * (col >> 2);
                out[i]      = 600 + 4 * (10 * src_row + src_col);
            }
        }
    };

    // -------------------------------------------------------------------------- //
    // More complex access patterns.
    // -------------------------------------------------------------------------- //

    struct Tile2DRowMajorCopy4x8
    {
        static const char* name()
        {
            return "tile2d_row_major_copy_4x8";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int tile[4][8];
            int            tid = threadIdx.x;
            int            row = tid / 8;
            int            col = tid - 8 * row;
            tile[row][col]     = 100 * row + col;
            __syncthreads();
            out[tid] = tile[row][col];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 100 * (i / 8) + (i & 7);
            }
        }
    };

    struct Tile2DColumnMajorLayout4x8
    {
        static const char* name()
        {
            return "tile2d_column_major_layout_4x8";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int tile[32];
            int            tid  = threadIdx.x;
            int            row  = tid / 8;
            int            col  = tid - 8 * row;
            tile[row + 4 * col] = 100 * row + col;
            __syncthreads();
            out[tid] = tile[row + 4 * col];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 100 * (i / 8) + (i & 7);
            }
        }
    };

    struct Tile2DTranspose4x8To8x4
    {
        static const char* name()
        {
            return "tile2d_transpose_4x8_to_8x4";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int tile[4][8];
            int            tid = threadIdx.x;
            int            row = tid / 8;
            int            col = tid - 8 * row;
            tile[row][col]     = 10 * row + col;
            __syncthreads();
            int out_row = tid / 4;
            int out_col = tid - 4 * out_row;
            out[tid]    = tile[out_col][out_row];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                int out_row = i / 4;
                int out_col = i - 4 * out_row;
                out[i]      = 10 * out_col + out_row;
            }
        }
    };

    struct Tile2DSkewedRowStride
    {
        static const char* name()
        {
            return "tile2d_skewed_row_stride";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int tile[4 * 9];
            int            tid  = threadIdx.x;
            int            row  = tid / 8;
            int            col  = tid - 8 * row;
            tile[col + 9 * row] = 100 * row + col;
            __syncthreads();
            out[tid] = tile[col + 9 * row];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 100 * (i / 8) + (i & 7);
            }
        }
    };

    struct Tile2DBlocked2x2Layout
    {
        static const char* name()
        {
            return "tile2d_blocked_2x2_layout";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int tile[32];
            int            tid       = threadIdx.x;
            int            row       = tid / 8;
            int            col       = tid - 8 * row;
            int            block_row = row >> 1;
            int            block_col = col >> 1;
            int            lane      = (row & 1) * 2 + (col & 1);
            int            index     = 4 * (block_col + 4 * block_row) + lane;
            tile[index]              = 100 * row + col;
            __syncthreads();
            out[tid] = tile[index];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 100 * (i / 8) + (i & 7);
            }
        }
    };

    struct Tile2DDiagonalGather
    {
        static const char* name()
        {
            return "tile2d_diagonal_gather";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int tile[4][8];
            int            tid = threadIdx.x;
            int            row = tid / 8;
            int            col = tid - 8 * row;
            tile[row][col]     = 10 * row + col;
            __syncthreads();
            out[tid] = tile[row][wrap_power_of_two(col + row, 8)];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                int row = i / 8;
                int col = i - 8 * row;
                out[i]  = 10 * row + wrap_power_of_two(col + row, 8);
            }
        }
    };

    template <int OddConstant>
    struct PseudoRandomOddScatter
    {
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid   = threadIdx.x;
            int            index = wrap_power_of_two(OddConstant * tid, thread_count);
            data[index]          = 1000 + tid;
            __syncthreads();
            out[tid] = data[index];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 1000 + i;
            }
        }
    };

    struct PseudoRandomOdd3Scatter : PseudoRandomOddScatter<3>
    {
        static const char* name()
        {
            return "pseudorandom_odd3_scatter";
        }
    };

    struct PseudoRandomOdd5Scatter : PseudoRandomOddScatter<5>
    {
        static const char* name()
        {
            return "pseudorandom_odd5_scatter";
        }
    };

    struct PseudoRandomOdd7Scatter : PseudoRandomOddScatter<7>
    {
        static const char* name()
        {
            return "pseudorandom_odd7_scatter";
        }
    };

    struct PseudoRandomOdd11Scatter : PseudoRandomOddScatter<11>
    {
        static const char* name()
        {
            return "pseudorandom_odd11_scatter";
        }
    };

    struct PseudoRandomGatherAfterIdentity
    {
        static const char* name()
        {
            return "pseudorandom_gather_after_identity";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[kArrayThreadCount];
            int            tid = threadIdx.x;
            data[tid]          = 10 * tid;
            __syncthreads();
            out[tid] = data[wrap_power_of_two(13 * tid, thread_count)];
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[i] = 10 * wrap_power_of_two(13 * i, thread_count);
            }
        }
    };

    struct VectorizedPairTile
    {
        static const char* name()
        {
            return "vectorized_pair_tile";
        }
        [[maybe_unused]] static constexpr int thread_count = 16;
        __global__ static void run(int* out)
        {
            __shared__ Pair tile[16];
            int             tid = threadIdx.x;
            tile[tid]           = Pair{tid, 100 + tid};
            __syncthreads();
            int other        = wrap_power_of_two(5 * tid, 16);
            out[2 * tid]     = tile[other].x;
            out[2 * tid + 1] = tile[tid].y;
        }
        static void expected(int* out)
        {
            for(int i = 0; i < thread_count; ++i)
            {
                out[2 * i]     = wrap_power_of_two(5 * i, 16);
                out[2 * i + 1] = 100 + i;
            }
        }
    };

    struct Tile2DStripedLoadStore
    {
        static const char* name()
        {
            return "tile2d_striped_load_store";
        }
        [[maybe_unused]] static constexpr int thread_count = 16;
        __global__ static void run(int* out)
        {
            __shared__ int tile[32];
            int            tid = threadIdx.x;
            for(int idx = tid; idx < 32; idx += thread_count)
            {
                tile[idx] = 3 * idx;
            }
            __syncthreads();
            for(int idx = tid; idx < 32; idx += thread_count)
            {
                out[idx] = tile[wrap_power_of_two(idx + 9, 32)];
            }
        }
        static void expected(int* out)
        {
            for(int i = 0; i < 32; ++i)
            {
                out[i] = 3 * wrap_power_of_two(i + 9, 32);
            }
        }
    };

    // -------------------------------------------------------------------------- //
    // Matmul-inspired references.
    // -------------------------------------------------------------------------- //

    struct Matmul2x2K3Simple
    {
        static const char* name()
        {
            return "matmul_2x2_k3_simple";
        }
        [[maybe_unused]] static constexpr int thread_count = 4;
        __global__ static void run(int* out)
        {
            __shared__ int a_shared[2 * 3];
            __shared__ int b_shared[2 * 3];
            run_simple_matmul_body<2, 2, 3, thread_count>(out, 0, 0, a_shared, b_shared);
        }
        static void expected(int* out)
        {
            fill_matmul_expected<2, 2, 3>(out, 0, 0);
        }
    };

    struct Matmul2x4K3Simple
    {
        static const char* name()
        {
            return "matmul_2x4_k3_simple";
        }
        [[maybe_unused]] static constexpr int thread_count = 8;
        __global__ static void run(int* out)
        {
            __shared__ int a_shared[2 * 3];
            __shared__ int b_shared[4 * 3];
            run_simple_matmul_body<2, 4, 3, thread_count>(out, 0, 0, a_shared, b_shared);
        }
        static void expected(int* out)
        {
            fill_matmul_expected<2, 4, 3>(out, 0, 0);
        }
    };

    struct Matmul4x4K4Simple
    {
        static const char* name()
        {
            return "matmul_4x4_k4_simple";
        }
        [[maybe_unused]] static constexpr int thread_count = 16;
        __global__ static void run(int* out)
        {
            __shared__ int a_shared[4 * 4];
            __shared__ int b_shared[4 * 4];
            run_simple_matmul_body<4, 4, 4, thread_count>(out, 0, 0, a_shared, b_shared);
        }
        static void expected(int* out)
        {
            fill_matmul_expected<4, 4, 4>(out, 0, 0);
        }
    };

    struct Matmul4x2K5Simple
    {
        static const char* name()
        {
            return "matmul_4x2_k5_simple";
        }
        [[maybe_unused]] static constexpr int thread_count = 8;
        __global__ static void run(int* out)
        {
            __shared__ int a_shared[4 * 5];
            __shared__ int b_shared[2 * 5];
            run_simple_matmul_body<4, 2, 5, thread_count>(out, 0, 0, a_shared, b_shared);
        }
        static void expected(int* out)
        {
            fill_matmul_expected<4, 2, 5>(out, 0, 0);
        }
    };

    struct Matmul4x8K4Simple
    {
        static const char* name()
        {
            return "matmul_4x8_k4_simple";
        }
        [[maybe_unused]] static constexpr int thread_count = 32;
        __global__ static void run(int* out)
        {
            __shared__ int a_shared[4 * 4];
            __shared__ int b_shared[8 * 4];
            run_simple_matmul_body<4, 8, 4, thread_count>(out, 0, 0, a_shared, b_shared);
        }
        static void expected(int* out)
        {
            fill_matmul_expected<4, 8, 4>(out, 0, 0);
        }
    };

    struct MatmulChunked2x2K2x3
    {
        static const char* name()
        {
            return "matmul_chunked_2x2_k2x3";
        }
        [[maybe_unused]] static constexpr int thread_count = 4;
        __global__ static void run(int* out)
        {
            __shared__ int a_shared[2 * 2];
            __shared__ int b_shared[2 * 2];
            run_chunked_matmul_body<2, 2, 2, 3, thread_count>(out, a_shared, b_shared);
        }
        static void expected(int* out)
        {
            fill_chunked_matmul_expected<2, 2, 2, 3>(out);
        }
    };

    struct MatmulChunked4x4K2x4
    {
        static const char* name()
        {
            return "matmul_chunked_4x4_k2x4";
        }
        [[maybe_unused]] static constexpr int thread_count = 16;
        __global__ static void run(int* out)
        {
            __shared__ int a_shared[4 * 2];
            __shared__ int b_shared[4 * 2];
            run_chunked_matmul_body<4, 4, 2, 4, thread_count>(out, a_shared, b_shared);
        }
        static void expected(int* out)
        {
            fill_chunked_matmul_expected<4, 4, 2, 4>(out);
        }
    };

    struct MatmulDoubleBuffered2x4K2x3
    {
        static const char* name()
        {
            return "matmul_double_buffered_2x4_k2x3";
        }
        [[maybe_unused]] static constexpr int thread_count = 8;
        __global__ static void run(int* out)
        {
            __shared__ int a_shared[2][2 * 2];
            __shared__ int b_shared[2][4 * 2];
            int            tid = threadIdx.x;
            int            acc = 0;
            int            m   = tid / 4;
            int            n   = tid - 4 * m;

            load_matmul_tile<2, 4, 2, thread_count>(a_shared[0], b_shared[0], 0);
            for(int k_outer = 1; k_outer < 3; ++k_outer)
            {
                int next    = k_outer & 1;
                int current = next ^ 1;
                load_matmul_tile<2, 4, 2, thread_count>(a_shared[next], b_shared[next], k_outer);
                __syncthreads();
                if(tid < 8)
                {
                    acc += matmul_mac_from_lds<2, 4, 2>(a_shared[current], b_shared[current], m, n);
                }
                __syncthreads();
            }
            __syncthreads();
            if(tid < 8)
            {
                acc += matmul_mac_from_lds<2, 4, 2>(a_shared[0], b_shared[0], m, n);
                out[tid] = acc;
            }
            __syncthreads();
        }
        static void expected(int* out)
        {
            fill_chunked_matmul_expected<2, 4, 2, 3>(out);
        }
    };

    struct MatmulSkewedLds4x4K4
    {
        static const char* name()
        {
            return "matmul_skewed_lds_4x4_k4";
        }
        [[maybe_unused]] static constexpr int thread_count = 16;
        __global__ static void run(int* out)
        {
            __shared__ int a_shared[4 * 5];
            __shared__ int b_shared[4 * 5];
            run_skewed_matmul_body<4, 4, 4, thread_count>(out, a_shared, b_shared);
        }
        static void expected(int* out)
        {
            fill_matmul_expected<4, 4, 4>(out, 0, 0);
        }
    };

    struct MatmulTwoWorkgroups2x2K3
    {
        static const char* name()
        {
            return "matmul_two_workgroups_2x2_k3";
        }
        [[maybe_unused]] static constexpr int thread_count = 4;
        __global__ static void run(int* out)
        {
            __shared__ int a_shared[2 * 3];
            __shared__ int b_shared[2 * 3];
            int            tile_id = blockIdx.x;
            run_simple_matmul_body<2, 2, 3, thread_count>(
                out, blockIdx.x * 4, tile_id, a_shared, b_shared);
        }
        static void expected(int* out)
        {
            fill_matmul_expected<2, 2, 3>(out, 0, 0);
            fill_matmul_expected<2, 2, 3>(out, 4, 1);
        }
    };

    // -------------------------------------------------------------------------- //
    // Diagnostic-positive and hard references. Compiled but not launched.
    // -------------------------------------------------------------------------- //

    struct SameEpochWriteReadConflictReference
    {
        static const char* name()
        {
            return "same_epoch_write_read_conflict_reference";
        }
        [[maybe_unused]] static constexpr int thread_count = kDefaultThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int value;
            if(threadIdx.x == 0)
            {
                value = 123;
            }
            if(threadIdx.x == 1)
            {
                out[0] = value;
            }
        }
        static void expected(int*) {}
    };

    struct SameEpochWriteWriteConflictReference
    {
        static const char* name()
        {
            return "same_epoch_write_write_conflict_reference";
        }
        [[maybe_unused]] static constexpr int thread_count = kDefaultThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int value;
            if(threadIdx.x == 0)
            {
                value = 123;
            }
            if(threadIdx.x == 1)
            {
                value = 456;
            }
            __syncthreads();
            if(threadIdx.x == 2)
            {
                out[0] = value;
            }
        }
        static void expected(int*) {}
    };

    struct PartiallyOverlappingWriteConflictReference
    {
        static const char* name()
        {
            return "partially_overlapping_write_conflict_reference";
        }
        [[maybe_unused]] static constexpr int thread_count = kDefaultThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ Pair pair;
            if(threadIdx.x == 0)
            {
                pair = Pair{1, 2};
            }
            if(threadIdx.x == 1)
            {
                pair.y = 3;
            }
            __syncthreads();
            if(threadIdx.x == 2)
            {
                out[0] = pair.y;
            }
        }
        static void expected(int*) {}
    };

    struct ArraySameIndexWriteConflictReference
    {
        static const char* name()
        {
            return "array_same_index_write_conflict_reference";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[1];
            data[0] = threadIdx.x;
            __syncthreads();
            if(threadIdx.x == 0)
            {
                out[0] = data[0];
            }
        }
        static void expected(int*) {}
    };

    struct ArraySameIndexReadWriteConflictReference
    {
        static const char* name()
        {
            return "array_same_index_read_write_conflict_reference";
        }
        [[maybe_unused]] static constexpr int thread_count = kArrayThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int data[1];
            if(threadIdx.x == 0)
            {
                data[0] = 1;
            }
            if(threadIdx.x == 1)
            {
                out[0] = data[0];
            }
        }
        static void expected(int*) {}
    };

    struct DivergentBarrierMissingReaderReference
    {
        static const char* name()
        {
            return "divergent_barrier_missing_reader_reference";
        }
        [[maybe_unused]] static constexpr int thread_count = kDefaultThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int value;
            if(threadIdx.x == 0)
            {
                value = 1;
                __syncthreads();
            }
            if(threadIdx.x == 1)
            {
                out[0] = value;
            }
        }
        static void expected(int*) {}
    };

    struct DivergentBarrierMissingWriterReference
    {
        static const char* name()
        {
            return "divergent_barrier_missing_writer_reference";
        }
        [[maybe_unused]] static constexpr int thread_count = kDefaultThreadCount;
        __global__ static void run(int* out)
        {
            __shared__ int value;
            if(threadIdx.x == 0)
            {
                value = 1;
            }
            if(threadIdx.x == 1)
            {
                __syncthreads();
                out[0] = value;
            }
        }
        static void expected(int*) {}
    };

    // -------------------------------------------------------------------------- //
    // Host-side reference case descriptors.
    // -------------------------------------------------------------------------- //

    HIP_MOI_REFERENCE_CASE(SameThreadStoreLoadCase, SameThreadStoreLoad, kDefaultThreadCount);
    HIP_MOI_REFERENCE_CASE(DifferentThreadsNonoverlappingAddressesCase,
                           DifferentThreadsNonoverlappingAddresses,
                           kDefaultThreadCount);
    HIP_MOI_REFERENCE_CASE(DifferentThreadsReadSameAddressCase,
                           DifferentThreadsReadSameAddress,
                           kDefaultThreadCount);
    HIP_MOI_REFERENCE_CASE(AdjacentAddressesSameEpochCase,
                           AdjacentAddressesSameEpoch,
                           kDefaultThreadCount);
    HIP_MOI_REFERENCE_CASE(WriteReadSeparatedBySyncthreadsCase,
                           WriteReadSeparatedBySyncthreads,
                           kDefaultThreadCount);
    HIP_MOI_REFERENCE_CASE(WriteWriteSeparatedBySyncthreadsCase,
                           WriteWriteSeparatedBySyncthreads,
                           kDefaultThreadCount);
    HIP_MOI_REFERENCE_CASE(MultipleEpochsReuseSameAddressCase,
                           MultipleEpochsReuseSameAddress,
                           kDefaultThreadCount);
    HIP_MOI_REFERENCE_CASE(TriviallyCopyableStructAccessCase,
                           TriviallyCopyableStructAccess,
                           kDefaultThreadCount);

    HIP_MOI_REFERENCE_CASE(ArrayIdentityAllThreadsCase, ArrayIdentityAllThreads, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(ArrayNeighborReadAllThreadsCase,
                           ArrayNeighborReadAllThreads,
                           kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(ArrayReverseReadAllThreadsCase,
                           ArrayReverseReadAllThreads,
                           kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(ArrayPermutationMultiply3AllThreadsCase,
                           ArrayPermutationMultiply3AllThreads,
                           kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(ArrayPermutationMultiply5GatherCase,
                           ArrayPermutationMultiply5Gather,
                           kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(ArrayTwoArraysZipCase, ArrayTwoArraysZip, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(ArrayTranspose4x8Case, ArrayTranspose4x8, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(ArrayPairwiseSwapCase, ArrayPairwiseSwap, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(ArrayGroupBroadcastCase, ArrayGroupBroadcast, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(ArrayRollingWindowSumCase, ArrayRollingWindowSum, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(ArrayStructPerThreadCase, ArrayStructPerThread, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(ArrayStridedTwoElementsPerThreadCase,
                           ArrayStridedTwoElementsPerThread,
                           kArrayThreadCount);

    HIP_MOI_REFERENCE_CASE(LoopIdentityEpochsCase, LoopIdentityEpochs, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(LoopNeighborEpochsCase, LoopNeighborEpochs, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(LoopReverseEpochsCase, LoopReverseEpochs, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(LoopPermutation3EpochsCase, LoopPermutation3Epochs, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(LoopPermutation5GatherCase, LoopPermutation5Gather, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(LoopPingPongBuffersCase, LoopPingPongBuffers, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(LoopRollingWindowCase, LoopRollingWindow, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(LoopGroupBroadcastCase, LoopGroupBroadcast, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(LoopTwoPhaseProducerConsumerCase,
                           LoopTwoPhaseProducerConsumer,
                           kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(LoopStructPerThreadCase, LoopStructPerThread, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(LoopPartialThreadCount16Case, LoopPartialThreadCount16, 16);
    HIP_MOI_REFERENCE_CASE(LoopTwoDimTile4x8Case, LoopTwoDimTile4x8, kArrayThreadCount);

    HIP_MOI_REFERENCE_CASE(Tile2DRowMajorCopy4x8Case, Tile2DRowMajorCopy4x8, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(Tile2DColumnMajorLayout4x8Case,
                           Tile2DColumnMajorLayout4x8,
                           kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(Tile2DTranspose4x8To8x4Case, Tile2DTranspose4x8To8x4, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(Tile2DSkewedRowStrideCase, Tile2DSkewedRowStride, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(Tile2DBlocked2x2LayoutCase, Tile2DBlocked2x2Layout, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(Tile2DDiagonalGatherCase, Tile2DDiagonalGather, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(PseudoRandomOdd3ScatterCase, PseudoRandomOdd3Scatter, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(PseudoRandomOdd5ScatterCase, PseudoRandomOdd5Scatter, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(PseudoRandomOdd7ScatterCase, PseudoRandomOdd7Scatter, kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(PseudoRandomOdd11ScatterCase,
                           PseudoRandomOdd11Scatter,
                           kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(PseudoRandomGatherAfterIdentityCase,
                           PseudoRandomGatherAfterIdentity,
                           kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(VectorizedPairTileCase, VectorizedPairTile, 16);
    HIP_MOI_REFERENCE_CASE(Tile2DStripedLoadStoreCase, Tile2DStripedLoadStore, 16);

    HIP_MOI_REFERENCE_CASE(Matmul2x2K3SimpleCase, Matmul2x2K3Simple, 4);
    HIP_MOI_REFERENCE_CASE(Matmul2x4K3SimpleCase, Matmul2x4K3Simple, 8);
    HIP_MOI_REFERENCE_CASE(Matmul4x4K4SimpleCase, Matmul4x4K4Simple, 16);
    HIP_MOI_REFERENCE_CASE(Matmul4x2K5SimpleCase, Matmul4x2K5Simple, 8);
    HIP_MOI_REFERENCE_CASE(Matmul4x8K4SimpleCase, Matmul4x8K4Simple, 32);
    HIP_MOI_REFERENCE_CASE(MatmulChunked2x2K2x3Case, MatmulChunked2x2K2x3, 4);
    HIP_MOI_REFERENCE_CASE(MatmulChunked4x4K2x4Case, MatmulChunked4x4K2x4, 16);
    HIP_MOI_REFERENCE_CASE(MatmulDoubleBuffered2x4K2x3Case, MatmulDoubleBuffered2x4K2x3, 8);
    HIP_MOI_REFERENCE_CASE(MatmulSkewedLds4x4K4Case, MatmulSkewedLds4x4K4, 16);
    HIP_MOI_REFERENCE_GRID_CASE(MatmulTwoWorkgroups2x2K3Case, MatmulTwoWorkgroups2x2K3, 4, 2, 1, 1);

    HIP_MOI_REFERENCE_CASE(SameEpochWriteReadConflictReferenceCase,
                           SameEpochWriteReadConflictReference,
                           kDefaultThreadCount);
    HIP_MOI_REFERENCE_CASE(SameEpochWriteWriteConflictReferenceCase,
                           SameEpochWriteWriteConflictReference,
                           kDefaultThreadCount);
    HIP_MOI_REFERENCE_CASE(PartiallyOverlappingWriteConflictReferenceCase,
                           PartiallyOverlappingWriteConflictReference,
                           kDefaultThreadCount);
    HIP_MOI_REFERENCE_CASE(ArraySameIndexWriteConflictReferenceCase,
                           ArraySameIndexWriteConflictReference,
                           kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(ArraySameIndexReadWriteConflictReferenceCase,
                           ArraySameIndexReadWriteConflictReference,
                           kArrayThreadCount);
    HIP_MOI_REFERENCE_CASE(DivergentBarrierMissingReaderReferenceCase,
                           DivergentBarrierMissingReaderReference,
                           kDefaultThreadCount);
    HIP_MOI_REFERENCE_CASE(DivergentBarrierMissingWriterReferenceCase,
                           DivergentBarrierMissingWriterReference,
                           kDefaultThreadCount);

    bool hip_check(hipError_t status, const char* expr, const char* file, int line)
    {
        if(status == hipSuccess)
        {
            return true;
        }
        std::fprintf(
            stderr, "%s:%d: HIP error from %s: %s\n", file, line, expr, hipGetErrorString(status));
        return false;
    }

#define HIP_MOI_REFERENCE_HIP_CHECK(expr)                 \
    do                                                    \
    {                                                     \
        if(!hip_check((expr), #expr, __FILE__, __LINE__)) \
        {                                                 \
            return false;                                 \
        }                                                 \
    } while(false)

#define HIP_MOI_REFERENCE_CHECK(expr) \
    do                                \
    {                                 \
        if(!(expr))                   \
        {                             \
            return false;             \
        }                             \
    } while(false)

    bool reset_output(int* device_output)
    {
        std::array<int, kMaxOutputCount> initial{};
        initial.fill(kSentinel);
        HIP_MOI_REFERENCE_HIP_CHECK(
            hipMemcpy(device_output, initial.data(), sizeof(initial), hipMemcpyHostToDevice));
        return true;
    }

    bool check_output(const char*                             test_name,
                      int*                                    device_output,
                      const std::array<int, kMaxOutputCount>& expected)
    {
        std::array<int, kMaxOutputCount> actual{};
        HIP_MOI_REFERENCE_HIP_CHECK(
            hipMemcpy(actual.data(), device_output, sizeof(actual), hipMemcpyDeviceToHost));

        bool ok = true;
        for(int i = 0; i < kMaxOutputCount; ++i)
        {
            if(actual[i] != expected[i])
            {
                std::fprintf(stderr,
                             "%s: out[%d] expected %d, got %d\n",
                             test_name,
                             i,
                             expected[i],
                             actual[i]);
                ok = false;
            }
        }
        return ok;
    }

    bool run_case(const ReferenceCase& test_case, int* device_output)
    {
        HIP_MOI_REFERENCE_CHECK(reset_output(device_output));
        test_case.launch(device_output);
        HIP_MOI_REFERENCE_HIP_CHECK(hipGetLastError());
        HIP_MOI_REFERENCE_HIP_CHECK(hipDeviceSynchronize());

        std::array<int, kMaxOutputCount> expected{};
        expected.fill(kSentinel);
        test_case.expected(expected.data());
        return check_output(test_case.name(), device_output, expected);
    }

    void instantiate_compile_only_kernels(int* device_output)
    {
        SameEpochWriteReadConflictReferenceCase        same_epoch_write_read_conflict;
        SameEpochWriteWriteConflictReferenceCase       same_epoch_write_write_conflict;
        PartiallyOverlappingWriteConflictReferenceCase partially_overlapping_write;
        ArraySameIndexWriteConflictReferenceCase       array_same_index_write;
        ArraySameIndexReadWriteConflictReferenceCase   array_same_index_read_write;
        DivergentBarrierMissingReaderReferenceCase     divergent_barrier_missing_reader;
        DivergentBarrierMissingWriterReferenceCase     divergent_barrier_missing_writer;

        const ReferenceCase* compile_only_cases[] = {
            &same_epoch_write_read_conflict,
            &same_epoch_write_write_conflict,
            &partially_overlapping_write,
            &array_same_index_write,
            &array_same_index_read_write,
            &divergent_barrier_missing_reader,
            &divergent_barrier_missing_writer,
        };

        if(false)
        {
            for(const ReferenceCase* test_case : compile_only_cases)
            {
                test_case->launch(device_output);
            }
        }
    }

    bool run_safe_reference_case(const ReferenceCase& test_case)
    {
        int* device_output = nullptr;
        if(!hip_check(hipMalloc(&device_output, kMaxOutputCount * sizeof(int)),
                      "hipMalloc(&device_output, kMaxOutputCount * sizeof(int))",
                      __FILE__,
                      __LINE__))
        {
            return false;
        }

        instantiate_compile_only_kernels(device_output);

        bool ok = run_case(test_case, device_output);
        ok &= hip_check(hipFree(device_output), "hipFree(device_output)", __FILE__, __LINE__);
        return ok;
    }

    const std::vector<const ReferenceCase*>& safe_reference_cases()
    {
        static const SameThreadStoreLoadCase                     same_thread_store_load;
        static const DifferentThreadsNonoverlappingAddressesCase nonoverlapping_addresses;
        static const DifferentThreadsReadSameAddressCase         read_same_address;
        static const AdjacentAddressesSameEpochCase              adjacent_addresses;
        static const WriteReadSeparatedBySyncthreadsCase         write_read_syncthreads;
        static const WriteWriteSeparatedBySyncthreadsCase        write_write_syncthreads;
        static const MultipleEpochsReuseSameAddressCase          multiple_epochs_reuse;
        static const TriviallyCopyableStructAccessCase           struct_access;

        static const ArrayIdentityAllThreadsCase             array_identity;
        static const ArrayNeighborReadAllThreadsCase         array_neighbor;
        static const ArrayReverseReadAllThreadsCase          array_reverse;
        static const ArrayPermutationMultiply3AllThreadsCase array_permutation3;
        static const ArrayPermutationMultiply5GatherCase     array_permutation5;
        static const ArrayTwoArraysZipCase                   array_two_arrays_zip;
        static const ArrayTranspose4x8Case                   array_transpose;
        static const ArrayPairwiseSwapCase                   array_pairwise_swap;
        static const ArrayGroupBroadcastCase                 array_group_broadcast;
        static const ArrayRollingWindowSumCase               array_rolling_window;
        static const ArrayStructPerThreadCase                array_struct;
        static const ArrayStridedTwoElementsPerThreadCase    array_strided_two_elements;

        static const LoopIdentityEpochsCase           loop_identity;
        static const LoopNeighborEpochsCase           loop_neighbor;
        static const LoopReverseEpochsCase            loop_reverse;
        static const LoopPermutation3EpochsCase       loop_permutation3;
        static const LoopPermutation5GatherCase       loop_permutation5;
        static const LoopPingPongBuffersCase          loop_ping_pong;
        static const LoopRollingWindowCase            loop_rolling_window;
        static const LoopGroupBroadcastCase           loop_group_broadcast;
        static const LoopTwoPhaseProducerConsumerCase loop_two_phase;
        static const LoopStructPerThreadCase          loop_struct;
        static const LoopPartialThreadCount16Case     loop_partial16;
        static const LoopTwoDimTile4x8Case            loop_2d_tile;

        static const Tile2DRowMajorCopy4x8Case           tile_row_major;
        static const Tile2DColumnMajorLayout4x8Case      tile_column_major;
        static const Tile2DTranspose4x8To8x4Case         tile_transpose;
        static const Tile2DSkewedRowStrideCase           tile_skewed;
        static const Tile2DBlocked2x2LayoutCase          tile_blocked;
        static const Tile2DDiagonalGatherCase            tile_diagonal;
        static const PseudoRandomOdd3ScatterCase         pseudorandom3;
        static const PseudoRandomOdd5ScatterCase         pseudorandom5;
        static const PseudoRandomOdd7ScatterCase         pseudorandom7;
        static const PseudoRandomOdd11ScatterCase        pseudorandom11;
        static const PseudoRandomGatherAfterIdentityCase pseudorandom_gather;
        static const VectorizedPairTileCase              vectorized_pair;
        static const Tile2DStripedLoadStoreCase          striped_load_store;

        static const Matmul2x2K3SimpleCase           matmul_2x2;
        static const Matmul2x4K3SimpleCase           matmul_2x4;
        static const Matmul4x4K4SimpleCase           matmul_4x4;
        static const Matmul4x2K5SimpleCase           matmul_4x2;
        static const Matmul4x8K4SimpleCase           matmul_4x8;
        static const MatmulChunked2x2K2x3Case        matmul_chunked_2x2;
        static const MatmulChunked4x4K2x4Case        matmul_chunked_4x4;
        static const MatmulDoubleBuffered2x4K2x3Case matmul_double_buffered;
        static const MatmulSkewedLds4x4K4Case        matmul_skewed;
        static const MatmulTwoWorkgroups2x2K3Case    matmul_two_workgroups;

        static const std::vector<const ReferenceCase*> cases = {
            &same_thread_store_load,
            &nonoverlapping_addresses,
            &read_same_address,
            &adjacent_addresses,
            &write_read_syncthreads,
            &write_write_syncthreads,
            &multiple_epochs_reuse,
            &struct_access,

            &array_identity,
            &array_neighbor,
            &array_reverse,
            &array_permutation3,
            &array_permutation5,
            &array_two_arrays_zip,
            &array_transpose,
            &array_pairwise_swap,
            &array_group_broadcast,
            &array_rolling_window,
            &array_struct,
            &array_strided_two_elements,

            &loop_identity,
            &loop_neighbor,
            &loop_reverse,
            &loop_permutation3,
            &loop_permutation5,
            &loop_ping_pong,
            &loop_rolling_window,
            &loop_group_broadcast,
            &loop_two_phase,
            &loop_struct,
            &loop_partial16,
            &loop_2d_tile,

            &tile_row_major,
            &tile_column_major,
            &tile_transpose,
            &tile_skewed,
            &tile_blocked,
            &tile_diagonal,
            &pseudorandom3,
            &pseudorandom5,
            &pseudorandom7,
            &pseudorandom11,
            &pseudorandom_gather,
            &vectorized_pair,
            &striped_load_store,

            &matmul_2x2,
            &matmul_2x4,
            &matmul_4x4,
            &matmul_4x2,
            &matmul_4x8,
            &matmul_chunked_2x2,
            &matmul_chunked_4x4,
            &matmul_double_buffered,
            &matmul_skewed,
            &matmul_two_workgroups,
        };

        return cases;
    }

    bool is_gtest_name_char(char c)
    {
        return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || ('0' <= c && c <= '9')
               || c == '_';
    }

    std::string reference_case_test_name(const ::testing::TestParamInfo<const ReferenceCase*>& info)
    {
        std::string name = info.param->name();
        for(char& c : name)
        {
            if(!is_gtest_name_char(c))
            {
                c = '_';
            }
        }
        if(name.empty() || ('0' <= name.front() && name.front() <= '9'))
        {
            name.insert(name.begin(), '_');
        }
        return name;
    }

    class SafeReferenceCase : public ::testing::TestWithParam<const ReferenceCase*>
    {
    };

} // namespace

TEST_P(SafeReferenceCase, MatchesExpectedOutput)
{
    EXPECT_TRUE(run_safe_reference_case(*GetParam()));
}

INSTANTIATE_TEST_SUITE_P(HipMoiReferenceKernels,
                         SafeReferenceCase,
                         ::testing::ValuesIn(safe_reference_cases()),
                         reference_case_test_name);
