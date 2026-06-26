// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/037_streamk_two_level_reduction_test.hip
//
// Stream-K-style two-level reduction test. Four producer subgroups write LDS
// partials, two pair-reducer subgroups consume those partials after release /
// acquire flag synchronization, and one reducer subgroup consumes the pair
// partials after a second release / acquire flag handoff.
//
#include "test_support.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace
{
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kProducerCount      = 4;
    constexpr int kPairReducerCount   = 2;
    constexpr int kSubgroupCount      = kProducerCount + kPairReducerCount;
    constexpr int kThreadCount        = kThreadsPerSubgroup * kSubgroupCount;
    constexpr int kProducerFlagCount  = kProducerCount;
    constexpr int kPairFlagCount      = kPairReducerCount;
    constexpr int kFlagCount          = kProducerFlagCount + kPairFlagCount;
    constexpr int kPairFlagBase       = kProducerFlagCount;
    constexpr int kFlagReady          = 1;
    constexpr int kInitialOutput      = -1;

    constexpr uint32_t kPartialsByteOffset = 0;
    constexpr uint32_t kPairPartialsByteOffset
        = kPartialsByteOffset
          + static_cast<uint32_t>(kProducerCount * kThreadsPerSubgroup * sizeof(int));
    constexpr int kSharedBytes = static_cast<int>(
        kPairPartialsByteOffset + kPairReducerCount * kThreadsPerSubgroup * sizeof(int));
    constexpr int kExactShadowEntries
        = (kSharedBytes + static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes) - 1)
          / static_cast<int>(hip_moi::detail::exact_shadow::granule_bytes);

    constexpr hip_moi::site_id kPartialStoreSite(0x2501);
    constexpr hip_moi::site_id kPartialLoadSite(0x2502);
    constexpr hip_moi::site_id kPairPartialStoreSite(0x2503);
    constexpr hip_moi::site_id kPairPartialLoadSite(0x2504);
    constexpr hip_moi::site_id kFlagSite(0x2505);

    __host__ __device__ int pair_reducer_subgroup(int pair)
    {
        return kProducerCount + pair;
    }

    __host__ __device__ int producer_flag_index(int producer)
    {
        return producer;
    }

    __host__ __device__ int pair_flag_index(int pair)
    {
        return kPairFlagBase + pair;
    }

    __host__ __device__ int producer_value(int producer, int lane)
    {
        return (producer + 1) * 100 + lane;
    }

    __host__ __device__ int expected_value(int lane)
    {
        int value = 0;
        for(int producer = 0; producer < kProducerCount; ++producer)
        {
            value += producer_value(producer, lane);
        }
        return value;
    }

    __host__ __device__ uint32_t partial_offset(int producer, int lane)
    {
        return kPartialsByteOffset
               + static_cast<uint32_t>((producer * kThreadsPerSubgroup + lane) * sizeof(int));
    }

    __host__ __device__ uint32_t pair_partial_offset(int pair, int lane)
    {
        return kPairPartialsByteOffset
               + static_cast<uint32_t>((pair * kThreadsPerSubgroup + lane) * sizeof(int));
    }

    hip_moi::host_context_options streamk_two_level_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes               = 128 * 1024;
        options.exact_shadow_entry_capacity = kExactShadowEntries;
        options.sampled_watchpoint_capacity = 0;
        options.diagnostic_capacity         = 16;
        options.subgroup_capacity           = kSubgroupCount;
        return options;
    }

    __device__ hip_moi::context
               make_streamk_two_level_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    template <hip_moi::atomic_memory_order ProducerOrder,
              hip_moi::atomic_memory_order PairPublishOrder,
              hip_moi::atomic_memory_order FinalAcquireOrder>
    __global__ void streamk_two_level_reduction_kernel(hip_moi::context::storage_ref storage,
                                                       int*                          flags,
                                                       int*                          out)
    {
        __shared__ int partials[kProducerCount][kThreadsPerSubgroup];
        __shared__ int pair_partials[kPairReducerCount][kThreadsPerSubgroup];

        hip_moi::context ctx = make_streamk_two_level_context(storage);
        ctx.init_workgroup();

        int tid      = static_cast<int>(threadIdx.x);
        int subgroup = tid / kThreadsPerSubgroup;
        int lane     = tid & (kThreadsPerSubgroup - 1);

        if(subgroup < kProducerCount)
        {
            ctx.lds_store_at(&partials[subgroup][lane],
                             producer_value(subgroup, lane),
                             partial_offset(subgroup, lane),
                             kPartialStoreSite);
            if(lane == 0)
            {
                ctx.atomic_store(&flags[producer_flag_index(subgroup)],
                                 kFlagReady,
                                 ProducerOrder,
                                 hip_moi::atomic_memory_scope::agent,
                                 kFlagSite);
            }
        }

        if(subgroup >= kProducerCount)
        {
            int pair = subgroup - kProducerCount;
            if(lane == 0)
            {
                for(int producer = pair * 2; producer < pair * 2 + 2; ++producer)
                {
                    while(ctx.atomic_load(&flags[producer_flag_index(producer)],
                                          hip_moi::atomic_memory_order::acquire,
                                          hip_moi::atomic_memory_scope::agent,
                                          kFlagSite)
                          != kFlagReady)
                    {
                    }
                }
            }

            int first_producer  = pair * 2;
            int second_producer = first_producer + 1;
            int first           = ctx.lds_load_at(&partials[first_producer][lane],
                                        partial_offset(first_producer, lane),
                                        kPartialLoadSite);
            int second          = ctx.lds_load_at(&partials[second_producer][lane],
                                         partial_offset(second_producer, lane),
                                         kPartialLoadSite);
            ctx.lds_store_at(&pair_partials[pair][lane],
                             first + second,
                             pair_partial_offset(pair, lane),
                             kPairPartialStoreSite);

            if(lane == 0)
            {
                ctx.atomic_store(&flags[pair_flag_index(pair)],
                                 kFlagReady,
                                 PairPublishOrder,
                                 hip_moi::atomic_memory_scope::agent,
                                 kFlagSite);
            }
        }

        if(subgroup == pair_reducer_subgroup(/*pair=*/0))
        {
            if(lane == 0)
            {
                for(int pair = 0; pair < kPairReducerCount; ++pair)
                {
                    while(ctx.atomic_load(&flags[pair_flag_index(pair)],
                                          FinalAcquireOrder,
                                          hip_moi::atomic_memory_scope::agent,
                                          kFlagSite)
                          != kFlagReady)
                    {
                    }
                }
            }

            int value0 = ctx.lds_load_at(&pair_partials[0][lane],
                                         pair_partial_offset(/*pair=*/0, lane),
                                         kPairPartialLoadSite);
            int value1 = ctx.lds_load_at(&pair_partials[1][lane],
                                         pair_partial_offset(/*pair=*/1, lane),
                                         kPairPartialLoadSite);
            out[lane]  = value0 + value1;
        }
    }

    struct device_buffers
    {
        int* flags = nullptr;
        int* out   = nullptr;

        device_buffers() = default;

        device_buffers(const device_buffers&)            = delete;
        device_buffers& operator=(const device_buffers&) = delete;

        ~device_buffers()
        {
            (void)hipFree(flags);
            (void)hipFree(out);
        }
    };

    void allocate_buffers(device_buffers* buffers)
    {
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->flags, kFlagCount * sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&buffers->out, kThreadsPerSubgroup * sizeof(int)));
    }

    void reset_buffers(device_buffers* buffers)
    {
        int              flags[kFlagCount] = {};
        std::vector<int> output(kThreadsPerSubgroup, kInitialOutput);
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(buffers->flags, flags, sizeof(flags), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(
            buffers->out, output.data(), output.size() * sizeof(output[0]), hipMemcpyHostToDevice));
    }

    template <hip_moi::atomic_memory_order ProducerOrder,
              hip_moi::atomic_memory_order PairPublishOrder,
              hip_moi::atomic_memory_order FinalAcquireOrder>
    void run_streamk_two_level_reduction(hip_moi::host_context* moi, device_buffers* buffers)
    {
        reset_buffers(buffers);
        hipLaunchKernelGGL((streamk_two_level_reduction_kernel<ProducerOrder,
                                                               PairPublishOrder,
                                                               FinalAcquireOrder>),
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           moi->launch_ref(),
                           buffers->flags,
                           buffers->out);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        std::vector<int> output(kThreadsPerSubgroup, 0);
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(
            output.data(), buffers->out, output.size() * sizeof(output[0]), hipMemcpyDeviceToHost));
        for(int lane = 0; lane < kThreadsPerSubgroup; ++lane)
        {
            EXPECT_EQ(output[static_cast<std::size_t>(lane)], expected_value(lane))
                << "lane " << lane;
        }
    }
} // namespace

TEST(HipMoiStreamKTwoLevelReduction, ReleaseAcquireTwoLevelFlagsOrderPayloads)
{
    hip_moi::host_context moi(streamk_two_level_options());
    device_buffers        buffers;
    ASSERT_NO_FATAL_FAILURE(allocate_buffers(&buffers));

    ASSERT_NO_FATAL_FAILURE(
        (run_streamk_two_level_reduction<hip_moi::atomic_memory_order::release,
                                         hip_moi::atomic_memory_order::release,
                                         hip_moi::atomic_memory_order::acquire>(&moi, &buffers)));
    EXPECT_TRUE(moi.check(nullptr));
}

TEST(HipMoiStreamKTwoLevelReduction, RelaxedProducerFlagsDiagnoseFirstLevelRace)
{
    hip_moi::host_context moi(streamk_two_level_options());
    device_buffers        buffers;
    ASSERT_NO_FATAL_FAILURE(allocate_buffers(&buffers));

    ASSERT_NO_FATAL_FAILURE(
        (run_streamk_two_level_reduction<hip_moi::atomic_memory_order::relaxed,
                                         hip_moi::atomic_memory_order::release,
                                         hip_moi::atomic_memory_order::acquire>(&moi, &buffers)));
    EXPECT_FALSE(moi.check(nullptr));
    EXPECT_GT(moi.last_diagnostic_count(), 0);
}

TEST(HipMoiStreamKTwoLevelReduction, RelaxedPairFlagsDiagnoseSecondLevelRace)
{
    hip_moi::host_context moi(streamk_two_level_options());
    device_buffers        buffers;
    ASSERT_NO_FATAL_FAILURE(allocate_buffers(&buffers));

    ASSERT_NO_FATAL_FAILURE(
        (run_streamk_two_level_reduction<hip_moi::atomic_memory_order::release,
                                         hip_moi::atomic_memory_order::relaxed,
                                         hip_moi::atomic_memory_order::acquire>(&moi, &buffers)));
    EXPECT_FALSE(moi.check(nullptr));
    EXPECT_GT(moi.last_diagnostic_count(), 0);
}
