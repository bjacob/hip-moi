// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/028_rdna4_wmma_streamk_arrival_counter_test.hip
//
// Stage 9 atomics test: a small RDNA4 WMMA Stream-K-shaped arrival counter.
// Two subgroups compute two K-slice partials for one C tile, publish their LDS
// partials with a global fetch-add counter, and the last arriving subgroup
// folds both partials.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace
{
    using half_t  = _Float16;
    using f16x8_t = half_t __attribute__((ext_vector_type(8)));
    using f32x8_t = float __attribute__((ext_vector_type(8)));

    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kThreadCount        = kThreadsPerSubgroup * kSubgroupCount;
    constexpr int kFragmentElements   = 8;
    constexpr int kTileM              = 16;
    constexpr int kTileN              = 16;
    constexpr int kKSlice             = 16;
    constexpr int kCElementCount      = kTileM * kTileN;
    constexpr int kInitialOutput      = -1;

    constexpr std::size_t kInputFragmentCount = kSubgroupCount * kThreadsPerSubgroup;
    constexpr std::size_t kInputBytes         = kInputFragmentCount * sizeof(f16x8_t);
    constexpr std::size_t kOutputBytes        = kCElementCount * sizeof(float);

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

    constexpr hip_moi::site_id kAStoreSite(0x1c01);
    constexpr hip_moi::site_id kBStoreSite(0x1c02);
    constexpr hip_moi::site_id kALoadSite(0x1c03);
    constexpr hip_moi::site_id kBLoadSite(0x1c04);
    constexpr hip_moi::site_id kPartialStoreSite(0x1c05);
    constexpr hip_moi::site_id kPartialLoadSite(0x1c06);
    constexpr hip_moi::site_id kCounterSite(0x1c07);

    __host__ __device__ int rdna4_wmma_acc_m(int lane, int elem)
    {
        return elem + 8 * (lane >> 4);
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
        float first_slice  = static_cast<float>(kKSlice);
        float second_slice = static_cast<float>(2 * kKSlice);
        return first_slice + second_slice;
    }

    hip_moi::host_context_options streamk_wmma_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes               = 256 * 1024;
        options.exact_shadow_entry_capacity = kExactShadowEntries;
        options.sampled_watchpoint_capacity = 0;
        options.diagnostic_capacity         = 16;
        options.subgroup_capacity           = kSubgroupCount;
        return options;
    }

    __device__ hip_moi::context make_streamk_wmma_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
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

    template <hip_moi::atomic_memory_order CounterOrder>
    __global__ void wmma_streamk_arrival_counter_kernel(hip_moi::context::storage_ref storage,
                                                        const f16x8_t* __restrict__ a_fragments,
                                                        const f16x8_t* __restrict__ b_fragments,
                                                        int* __restrict__ counter,
                                                        float* __restrict__ out)
    {
        __shared__ f16x8_t a_shared[kSubgroupCount][kThreadsPerSubgroup];
        __shared__ f16x8_t b_shared[kSubgroupCount][kThreadsPerSubgroup];
        __shared__ float   partials[kSubgroupCount][kThreadsPerSubgroup][kFragmentElements];

        hip_moi::context ctx = make_streamk_wmma_context(storage);
        ctx.init_workgroup();

        int tid      = static_cast<int>(threadIdx.x);
        int subgroup = tid / kThreadsPerSubgroup;
        int lane     = tid & (kThreadsPerSubgroup - 1);

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

        int old_value = 0;
        if(lane == 0)
        {
            old_value = ctx.atomic_fetch_add(counter,
                                             1,
                                             CounterOrder,
                                             hip_moi::atomic_memory_scope::agent,
                                             kCounterSite);
        }

        uint32_t old_value_bits
            = __builtin_amdgcn_ds_bpermute(/*byte_offset=*/0,
                                           static_cast<uint32_t>(old_value));
        if(static_cast<int>(old_value_bits) == 1)
        {
#pragma unroll
            for(int elem = 0; elem < kFragmentElements; ++elem)
            {
                float value = 0.0f;
#pragma unroll
                for(int partial_subgroup = 0; partial_subgroup < kSubgroupCount; ++partial_subgroup)
                {
                    value += ctx.lds_load_at(&partials[partial_subgroup][lane][elem],
                                             partial_offset(partial_subgroup, lane, elem),
                                             kPartialLoadSite);
                }
                int row             = rdna4_wmma_acc_m(lane, elem);
                int column          = lane & 15;
                out[row * kTileN + column] = value;
            }
        }
    }

    void fill_fragments(std::vector<f16x8_t>* a_fragments, std::vector<f16x8_t>* b_fragments)
    {
        a_fragments->resize(kInputFragmentCount);
        b_fragments->resize(kInputFragmentCount);
        for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
        {
            for(int lane = 0; lane < kThreadsPerSubgroup; ++lane)
            {
                f16x8_t a;
                f16x8_t b;
#pragma unroll
                for(int elem = 0; elem < kFragmentElements; ++elem)
                {
                    a[elem] = static_cast<half_t>(subgroup + 1);
                    b[elem] = static_cast<half_t>(1);
                }
                (*a_fragments)[subgroup * kThreadsPerSubgroup + lane] = a;
                (*b_fragments)[subgroup * kThreadsPerSubgroup + lane] = b;
            }
        }
    }

    struct device_buffers
    {
        f16x8_t* a       = nullptr;
        f16x8_t* b       = nullptr;
        int*     counter = nullptr;
        float*   out     = nullptr;

        ~device_buffers()
        {
            if(a)
            {
                (void)hipFree(a);
            }
            if(b)
            {
                (void)hipFree(b);
            }
            if(counter)
            {
                (void)hipFree(counter);
            }
            if(out)
            {
                (void)hipFree(out);
            }
        }
    };

    void allocate_and_copy_inputs(const std::vector<f16x8_t>& a_fragments,
                                  const std::vector<f16x8_t>& b_fragments,
                                  device_buffers*             buffers)
    {
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->a, kInputBytes));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->b, kInputBytes));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->counter, sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->out, kOutputBytes));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(buffers->a, a_fragments.data(), kInputBytes, hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(buffers->b, b_fragments.data(), kInputBytes, hipMemcpyHostToDevice));
    }

    void reset_counter_and_output(device_buffers* buffers)
    {
        int   counter = 0;
        std::vector<float> output(kCElementCount, static_cast<float>(kInitialOutput));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(buffers->counter, &counter, sizeof(counter), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(buffers->out, output.data(), kOutputBytes, hipMemcpyHostToDevice));
    }

    template <hip_moi::atomic_memory_order CounterOrder>
    void launch_wmma_streamk_arrival_counter(hip_moi::context::storage_ref storage,
                                             device_buffers*                buffers)
    {
        hipLaunchKernelGGL((wmma_streamk_arrival_counter_kernel<CounterOrder>),
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage,
                           buffers->a,
                           buffers->b,
                           buffers->counter,
                           buffers->out);
    }

    void copy_output(const device_buffers& buffers, std::vector<float>* output)
    {
        output->assign(kCElementCount, 0.0f);
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output->data(), buffers.out, kOutputBytes, hipMemcpyDeviceToHost));
    }

    template <hip_moi::atomic_memory_order CounterOrder>
    void run_wmma_streamk_arrival_counter(hip_moi::host_context* moi,
                                          device_buffers*        buffers)
    {
        reset_counter_and_output(buffers);
        launch_wmma_streamk_arrival_counter<CounterOrder>(moi->launch_ref(), buffers);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());
        std::vector<float> output;
        copy_output(*buffers, &output);
        for(std::size_t i = 0; i < output.size(); ++i)
        {
            EXPECT_EQ(output[i], expected_c_value()) << "mismatch at C element " << i;
        }
    }

    void make_inputs_and_buffers(device_buffers* buffers)
    {
        std::vector<f16x8_t> a_fragments;
        std::vector<f16x8_t> b_fragments;
        fill_fragments(&a_fragments, &b_fragments);
        allocate_and_copy_inputs(a_fragments, b_fragments, buffers);
    }
} // namespace

TEST(HipMoiRdna4WmmaStreamKArrivalCounter, AcqRelFetchAddOrdersWmmaPartials)
{
    hip_moi::host_context moi(streamk_wmma_options());
    device_buffers        buffers;
    ASSERT_NO_FATAL_FAILURE(make_inputs_and_buffers(&buffers));

    ASSERT_NO_FATAL_FAILURE(
        (run_wmma_streamk_arrival_counter<hip_moi::atomic_memory_order::acq_rel>(&moi,
                                                                                 &buffers)));
    EXPECT_TRUE(moi.check(nullptr));
}

TEST(HipMoiRdna4WmmaStreamKArrivalCounter, RelaxedFetchAddDiagnosesWmmaPartialRace)
{
    hip_moi::host_context moi(streamk_wmma_options());
    device_buffers        buffers;
    ASSERT_NO_FATAL_FAILURE(make_inputs_and_buffers(&buffers));

    ASSERT_NO_FATAL_FAILURE(
        (run_wmma_streamk_arrival_counter<hip_moi::atomic_memory_order::relaxed>(&moi,
                                                                                 &buffers)));
    EXPECT_FALSE(moi.check(nullptr));
    EXPECT_GT(moi.last_diagnostic_count(), 0);
}
    __host__ __device__ int rdna4_wmma_acc_m(int lane, int elem)
    {
        return elem + 8 * (lane >> 4);
    }
