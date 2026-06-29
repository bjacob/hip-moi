// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/029_cdna4_mfma_streamk_tree_atomic_or_test.hip
//
// Stage 9 atomics test: a small CDNA4 MFMA Stream-K-tree-shaped atomicOr
// bitmask. Four subgroups compute four K-slice partials for one C tile. The
// first three subgroups publish LDS partials by setting bits in a global
// bitmask, and the final subgroup observes those bits before folding all
// partials.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{
    using half_t  = _Float16;
    using f16x4_t = half_t __attribute__((ext_vector_type(4)));
    using f32x4_t = float __attribute__((ext_vector_type(4)));

    constexpr int kThreadsPerSubgroup = 64;
    constexpr int kSubgroupCount      = 4;
    constexpr int kFinalSubgroup      = kSubgroupCount - 1;
    constexpr int kThreadCount        = kThreadsPerSubgroup * kSubgroupCount;
    constexpr int kFragmentElements   = 4;
    constexpr int kTileM              = 16;
    constexpr int kTileN              = 16;
    constexpr int kKSlice             = 16;
    constexpr int kCElementCount      = kTileM * kTileN;
    constexpr int kInitialOutput      = -1;

    constexpr std::size_t kInputFragmentCount = kSubgroupCount * kThreadsPerSubgroup;
    constexpr std::size_t kInputBytes         = kInputFragmentCount * sizeof(f16x4_t);
    constexpr std::size_t kOutputBytes        = kCElementCount * sizeof(float);

    constexpr uint32_t kASharedByteOffset = 0;
    constexpr uint32_t kBSharedByteOffset
        = kASharedByteOffset
          + static_cast<uint32_t>(kSubgroupCount * kThreadsPerSubgroup * sizeof(f16x4_t));
    constexpr uint32_t kPartialsSharedByteOffset
        = kBSharedByteOffset
          + static_cast<uint32_t>(kSubgroupCount * kThreadsPerSubgroup * sizeof(f16x4_t));
    constexpr int kSharedBytes = static_cast<int>(kPartialsSharedByteOffset
                                                  + kSubgroupCount * kThreadsPerSubgroup
                                                        * kFragmentElements * sizeof(float));
    constexpr int kExactShadowEntries
        = (kSharedBytes + static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes) - 1)
          / static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes);

    constexpr hip_moi::site_id kAStoreSite(0x1e01);
    constexpr hip_moi::site_id kBStoreSite(0x1e02);
    constexpr hip_moi::site_id kALoadSite(0x1e03);
    constexpr hip_moi::site_id kBLoadSite(0x1e04);
    constexpr hip_moi::site_id kPartialStoreSite(0x1e05);
    constexpr hip_moi::site_id kPartialLoadSite(0x1e06);
    constexpr hip_moi::site_id kBitmaskSite(0x1e07);

    __host__ __device__ int cdna4_mfma_acc_m(int lane, int elem)
    {
        return elem + 4 * (lane >> 4);
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
               + static_cast<uint32_t>((subgroup * kThreadsPerSubgroup + lane) * sizeof(f16x4_t));
    }

    __host__ __device__ uint32_t b_fragment_offset(int subgroup, int lane)
    {
        return kBSharedByteOffset
               + static_cast<uint32_t>((subgroup * kThreadsPerSubgroup + lane) * sizeof(f16x4_t));
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

    hip_moi::host_context_options streamk_tree_mfma_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes               = 512 * 1024;
        options.exact_shadow_entry_capacity = kExactShadowEntries;
        options.sampled_watchpoint_capacity = 0;
        options.diagnostic_capacity         = 16;
        options.subgroup_capacity           = kSubgroupCount;
        return options;
    }

    __device__ hip_moi::context
               make_streamk_tree_mfma_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    __device__ f32x4_t zero_f32x4()
    {
        f32x4_t value;
#pragma unroll
        for(int elem = 0; elem < kFragmentElements; ++elem)
        {
            value[elem] = 0.0f;
        }
        return value;
    }

    template <hip_moi::atomic_memory_order ProducerOrder, hip_moi::atomic_memory_order FinalOrder>
    __global__ void mfma_streamk_tree_atomic_or_kernel(hip_moi::context::storage_ref storage,
                                                       const f16x4_t* __restrict__ a_fragments,
                                                       const f16x4_t* __restrict__ b_fragments,
                                                       int* __restrict__ bitmask,
                                                       float* __restrict__ out)
    {
        __shared__ f16x4_t a_shared[kSubgroupCount][kThreadsPerSubgroup];
        __shared__ f16x4_t b_shared[kSubgroupCount][kThreadsPerSubgroup];
        __shared__ float   partials[kSubgroupCount][kThreadsPerSubgroup][kFragmentElements];

        hip_moi::context ctx = make_streamk_tree_mfma_context(storage);
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

        f16x4_t a_fragment = ctx.lds_load_at(
            &a_shared[subgroup][lane], a_fragment_offset(subgroup, lane), kALoadSite);
        f16x4_t b_fragment = ctx.lds_load_at(
            &b_shared[subgroup][lane], b_fragment_offset(subgroup, lane), kBLoadSite);
        f32x4_t acc = zero_f32x4();
        acc         = __builtin_amdgcn_mfma_f32_16x16x16f16(a_fragment, b_fragment, acc, 0, 0, 0);

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
                (void)ctx.atomic_fetch_or(bitmask,
                                          subgroup_bit(subgroup),
                                          ProducerOrder,
                                          hip_moi::atomic_memory_scope::agent,
                                          kBitmaskSite);
            }
        }
        else
        {
            if(lane == 0)
            {
                while((__hip_atomic_load(bitmask, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
                       & producer_bits())
                      != producer_bits())
                {
                }
                old_mask = ctx.atomic_fetch_or(bitmask,
                                               subgroup_bit(subgroup),
                                               FinalOrder,
                                               hip_moi::atomic_memory_scope::agent,
                                               kBitmaskSite);
            }

            uint32_t old_mask_bits
                = __builtin_amdgcn_ds_bpermute(/*byte_offset=*/0, static_cast<uint32_t>(old_mask));
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
                    int row                    = cdna4_mfma_acc_m(lane, elem);
                    int column                 = lane & 15;
                    out[row * kTileN + column] = value;
                }
            }
        }
    }

    void fill_fragments(std::vector<f16x4_t>* a_fragments, std::vector<f16x4_t>* b_fragments)
    {
        a_fragments->resize(kInputFragmentCount);
        b_fragments->resize(kInputFragmentCount);
        for(int subgroup = 0; subgroup < kSubgroupCount; ++subgroup)
        {
            for(int lane = 0; lane < kThreadsPerSubgroup; ++lane)
            {
                f16x4_t a;
                f16x4_t b;
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
        f16x4_t* a       = nullptr;
        f16x4_t* b       = nullptr;
        int*     bitmask = nullptr;
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
            if(bitmask)
            {
                (void)hipFree(bitmask);
            }
            if(out)
            {
                (void)hipFree(out);
            }
        }
    };

    void allocate_and_copy_inputs(const std::vector<f16x4_t>& a_fragments,
                                  const std::vector<f16x4_t>& b_fragments,
                                  device_buffers*             buffers)
    {
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->a, kInputBytes));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->b, kInputBytes));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->bitmask, sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->out, kOutputBytes));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(buffers->a, a_fragments.data(), kInputBytes, hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(buffers->b, b_fragments.data(), kInputBytes, hipMemcpyHostToDevice));
    }

    void reset_bitmask_and_output(device_buffers* buffers)
    {
        int                bitmask = 0;
        std::vector<float> output(kCElementCount, static_cast<float>(kInitialOutput));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(buffers->bitmask, &bitmask, sizeof(bitmask), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(buffers->out, output.data(), kOutputBytes, hipMemcpyHostToDevice));
    }

    template <hip_moi::atomic_memory_order ProducerOrder, hip_moi::atomic_memory_order FinalOrder>
    void launch_mfma_streamk_tree_atomic_or(hip_moi::context::storage_ref storage,
                                            device_buffers*               buffers)
    {
        hipLaunchKernelGGL((mfma_streamk_tree_atomic_or_kernel<ProducerOrder, FinalOrder>),
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           storage,
                           buffers->a,
                           buffers->b,
                           buffers->bitmask,
                           buffers->out);
    }

    void copy_output(const device_buffers& buffers, std::vector<float>* output)
    {
        output->assign(kCElementCount, 0.0f);
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(output->data(), buffers.out, kOutputBytes, hipMemcpyDeviceToHost));
    }

    template <hip_moi::atomic_memory_order ProducerOrder, hip_moi::atomic_memory_order FinalOrder>
    void run_mfma_streamk_tree_atomic_or(hip_moi::host_context* moi, device_buffers* buffers)
    {
        reset_bitmask_and_output(buffers);
        launch_mfma_streamk_tree_atomic_or<ProducerOrder, FinalOrder>(moi->launch_ref(), buffers);
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
        std::vector<f16x4_t> a_fragments;
        std::vector<f16x4_t> b_fragments;
        fill_fragments(&a_fragments, &b_fragments);
        allocate_and_copy_inputs(a_fragments, b_fragments, buffers);
    }
} // namespace

TEST(HipMoiCdna4MfmaStreamKTreeAtomicOr, AcqRelBitmaskOrdersMfmaPartials)
{
    hip_moi::host_context moi(streamk_tree_mfma_options());
    device_buffers        buffers;
    ASSERT_NO_FATAL_FAILURE(make_inputs_and_buffers(&buffers));

    ASSERT_NO_FATAL_FAILURE(
        (run_mfma_streamk_tree_atomic_or<hip_moi::atomic_memory_order::release,
                                         hip_moi::atomic_memory_order::acq_rel>(&moi, &buffers)));
    EXPECT_TRUE(moi.check(nullptr));
}

TEST(HipMoiCdna4MfmaStreamKTreeAtomicOr, RelaxedProducerBitsDiagnoseMfmaPartialRace)
{
    hip_moi::host_context moi(streamk_tree_mfma_options());
    device_buffers        buffers;
    ASSERT_NO_FATAL_FAILURE(make_inputs_and_buffers(&buffers));

    ASSERT_NO_FATAL_FAILURE(
        (run_mfma_streamk_tree_atomic_or<hip_moi::atomic_memory_order::relaxed,
                                         hip_moi::atomic_memory_order::acq_rel>(&moi, &buffers)));
    EXPECT_FALSE(moi.check(nullptr));
    EXPECT_GT(moi.last_diagnostic_count(), 0);
}
