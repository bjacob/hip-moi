// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/034_atomic_bitwise_happens_before_test.hip
//
// Stage 16 atomics tests: bitwise RMWs beyond atomicOr. These are modeled as
// Stream-K-tree-like bitmask operations where the old value returned by a RMW
// decides whether an LDS payload is safe to read.
//
#include "test_support.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kPayloadValue       = 149;
    constexpr int kInitialOutput      = -1;
    constexpr int kProducerBit        = 1;
    constexpr int kInitialAndMask     = kProducerBit;
    constexpr int kInitialXorMask     = 0;

    enum class bitwise_operation
    {
        fetch_and_clear,
        fetch_xor_toggle,
    };

    hip_moi::host_context_options atomic_bitwise_options()
    {
        hip_moi::host_context_options options;
        options.storage_bytes               = 64 * 1024;
        options.exact_shadow_entry_capacity = 16;
        options.sampled_watchpoint_capacity = 0;
        options.diagnostic_capacity         = 8;
        options.subgroup_capacity           = kSubgroupCount;
        return options;
    }

    __device__ hip_moi::context make_atomic_bitwise_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    template <bitwise_operation Operation, hip_moi::atomic_memory_order ProducerOrder>
    __global__ void atomic_bitwise_orders_lds_payload_kernel(hip_moi::context::storage_ref storage,
                                                             int*                          bitmask,
                                                             int*                          out)
    {
        __shared__ int payload;

        hip_moi::context ctx = make_atomic_bitwise_context(storage);
        ctx.init_workgroup();

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at(&payload,
                             kPayloadValue,
                             /*lds_byte_offset=*/0,
                             HIP_MOI_SITE_ID());
            if constexpr(Operation == bitwise_operation::fetch_and_clear)
            {
                (void)ctx.atomic_fetch_and(bitmask,
                                           ~kProducerBit,
                                           ProducerOrder,
                                           hip_moi::atomic_memory_scope::agent,
                                           HIP_MOI_SITE_ID());
            }
            else
            {
                (void)ctx.atomic_fetch_xor(bitmask,
                                           kProducerBit,
                                           ProducerOrder,
                                           hip_moi::atomic_memory_scope::agent,
                                           HIP_MOI_SITE_ID());
            }
        }

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            int old_mask = 0;
            do
            {
                if constexpr(Operation == bitwise_operation::fetch_and_clear)
                {
                    old_mask = ctx.atomic_fetch_and(bitmask,
                                                    ~0,
                                                    hip_moi::atomic_memory_order::acquire,
                                                    hip_moi::atomic_memory_scope::agent,
                                                    HIP_MOI_SITE_ID());
                }
                else
                {
                    old_mask = ctx.atomic_fetch_xor(bitmask,
                                                    0,
                                                    hip_moi::atomic_memory_order::acquire,
                                                    hip_moi::atomic_memory_scope::agent,
                                                    HIP_MOI_SITE_ID());
                }
            } while((Operation == bitwise_operation::fetch_and_clear)
                        ? ((old_mask & kProducerBit) != 0)
                        : ((old_mask & kProducerBit) == 0));

            out[0] = ctx.lds_load_at(&payload, /*lds_byte_offset=*/0, HIP_MOI_SITE_ID());
        }
    }

    void reset_bitmask_and_output(bitwise_operation operation, int* device_bitmask, int* device_out)
    {
        int bitmask
            = operation == bitwise_operation::fetch_and_clear ? kInitialAndMask : kInitialXorMask;
        int output = kInitialOutput;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_bitmask, &bitmask, sizeof(bitmask), hipMemcpyHostToDevice));
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(device_out, &output, sizeof(output), hipMemcpyHostToDevice));
    }

    template <bitwise_operation Operation, hip_moi::atomic_memory_order ProducerOrder>
    void run_bitwise_case(bool should_be_ordered)
    {
        hip_moi::host_context moi(atomic_bitwise_options());
        int*                  device_bitmask = nullptr;
        int*                  device_output  = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_bitmask, sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, sizeof(int)));
        reset_bitmask_and_output(Operation, device_bitmask, device_output);

        hipLaunchKernelGGL((atomic_bitwise_orders_lds_payload_kernel<Operation, ProducerOrder>),
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           moi.launch_ref(),
                           device_bitmask,
                           device_output);
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int output = 0;
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
        EXPECT_EQ(output, kPayloadValue);
        EXPECT_EQ(moi.check(nullptr), should_be_ordered);
        if(!should_be_ordered)
        {
            EXPECT_GT(moi.last_diagnostic_count(), 0);
        }

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_bitmask));
    }
} // namespace

TEST(HipMoiAtomicBitwiseHappensBefore, ReleaseFetchAndOrdersClearedBitPayload)
{
    run_bitwise_case<bitwise_operation::fetch_and_clear, hip_moi::atomic_memory_order::release>(
        /*should_be_ordered=*/true);
}

TEST(HipMoiAtomicBitwiseHappensBefore, RelaxedFetchAndDoesNotOrderClearedBitPayload)
{
    run_bitwise_case<bitwise_operation::fetch_and_clear, hip_moi::atomic_memory_order::relaxed>(
        /*should_be_ordered=*/false);
}

TEST(HipMoiAtomicBitwiseHappensBefore, ReleaseFetchXorOrdersToggledBitPayload)
{
    run_bitwise_case<bitwise_operation::fetch_xor_toggle, hip_moi::atomic_memory_order::release>(
        /*should_be_ordered=*/true);
}

TEST(HipMoiAtomicBitwiseHappensBefore, RelaxedFetchXorDoesNotOrderToggledBitPayload)
{
    run_bitwise_case<bitwise_operation::fetch_xor_toggle, hip_moi::atomic_memory_order::relaxed>(
        /*should_be_ordered=*/false);
}
