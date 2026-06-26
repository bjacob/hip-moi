// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/035_attention_lds_alias_handoff_test.hip
//
// Source-level model of an attention pipeline LDS alias handoff. One logical
// tile is read from a reusable LDS slot while the next logical tile may be
// prefetched into the same slot. The aliased case needs a full-workgroup
// barrier between the read phase and the write phase; non-aliased slots do not.
#include "test_support.hpp"

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

namespace
{
    constexpr int kThreadCount        = 64;
    constexpr int kThreadsPerSubgroup = 32;
    constexpr int kSubgroupCount      = 2;
    constexpr int kDiagnosticCapacity = 8;
    constexpr int kExactShadowEntries = 2;
    constexpr int kTailKValue         = 17;
    constexpr int kNextVValue         = 29;

    constexpr hip_moi::site_id kTailKStoreSite(0x3501);
    constexpr hip_moi::site_id kTailKLoadSite(0x3502);
    constexpr hip_moi::site_id kNextVStoreSite(0x3503);

    __device__ hip_moi::context make_alias_context(hip_moi::context::storage_ref storage)
    {
        hip_moi::context::config cfg{
            /*thread_count=*/kThreadCount,
            /*threads_per_subgroup=*/kThreadsPerSubgroup,
            /*subgroup_count=*/kSubgroupCount,
        };
        return hip_moi::context(storage, cfg);
    }

    template <bool AliasSlot, bool InsertHandoffBarrier>
    __global__ void attention_alias_handoff_kernel(int* out, hip_moi::context::storage_ref storage)
    {
        __shared__ int slots[2];

        hip_moi::context ctx = make_alias_context(storage);
        ctx.init_workgroup();

        constexpr uint32_t tail_k_offset = 0;
        constexpr uint32_t next_v_offset = AliasSlot ? 0 : sizeof(int);
        int*               next_v_slot   = AliasSlot ? &slots[0] : &slots[1];

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at<hip_moi::backend_kind::exact_shadow>(
                &slots[0], kTailKValue, tail_k_offset, kTailKStoreSite);
        }
        ctx.syncthreads();

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            out[0] = ctx.lds_load_at<hip_moi::backend_kind::exact_shadow>(
                &slots[0], tail_k_offset, kTailKLoadSite);
        }

        if constexpr(InsertHandoffBarrier)
        {
            ctx.syncthreads();
        }

        if(ctx.thread_id() == 0)
        {
            ctx.lds_store_at<hip_moi::backend_kind::exact_shadow>(
                next_v_slot, kNextVValue, next_v_offset, kNextVStoreSite);
        }
        ctx.syncthreads();

        if(ctx.thread_id() == kThreadsPerSubgroup)
        {
            out[1] = ctx.lds_load_at<hip_moi::backend_kind::exact_shadow>(
                next_v_slot, next_v_offset, kTailKLoadSite);
        }
    }

    template <bool AliasSlot, bool InsertHandoffBarrier>
    void run_alias_handoff_case(bool expect_diagnostic, bool expect_stable_tail_value)
    {
        hip_moi::test::device_context_storage storage;
        ASSERT_NO_FATAL_FAILURE(storage.allocate_with_exact_shadow(
            /*diagnostic_capacity=*/kDiagnosticCapacity,
            /*subgroup_capacity=*/kSubgroupCount,
            /*exact_shadow_entry_capacity=*/kExactShadowEntries));

        int* device_output = nullptr;
        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&device_output, 2 * sizeof(int)));
        HIP_MOI_TEST_HIP_ASSERT(hipMemset(device_output, 0, 2 * sizeof(int)));

        hipLaunchKernelGGL((attention_alias_handoff_kernel<AliasSlot, InsertHandoffBarrier>),
                           dim3(1),
                           dim3(kThreadCount),
                           0,
                           0,
                           device_output,
                           storage.ref());
        HIP_MOI_TEST_HIP_ASSERT(hipGetLastError());
        HIP_MOI_TEST_HIP_ASSERT(hipDeviceSynchronize());

        int host_output[2] = {};
        HIP_MOI_TEST_HIP_ASSERT(
            hipMemcpy(host_output, device_output, sizeof(host_output), hipMemcpyDeviceToHost));

        if(expect_stable_tail_value)
        {
            EXPECT_EQ(host_output[0], kTailKValue);
        }
        EXPECT_EQ(host_output[1], kNextVValue);

        if(expect_diagnostic)
        {
            int count = -1;
            HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(
                &count, storage.diagnostic_count_device(), sizeof(count), hipMemcpyDeviceToHost));
            EXPECT_GT(count, 0);
        }
        else
        {
            ASSERT_NO_FATAL_FAILURE(hip_moi::test::expect_diagnostic_count(storage, 0));
        }

        HIP_MOI_TEST_HIP_ASSERT(hipFree(device_output));
    }
} // namespace

TEST(HipMoiAttentionLdsAliasHandoff, MissingBarrierOnAliasedSlotReports)
{
    run_alias_handoff_case</*AliasSlot=*/true, /*InsertHandoffBarrier=*/false>(
        /*expect_diagnostic=*/true,
        /*expect_stable_tail_value=*/false);
}

TEST(HipMoiAttentionLdsAliasHandoff, ConditionalBarrierOnAliasedSlotIsClean)
{
    run_alias_handoff_case</*AliasSlot=*/true, /*InsertHandoffBarrier=*/true>(
        /*expect_diagnostic=*/false,
        /*expect_stable_tail_value=*/true);
}

TEST(HipMoiAttentionLdsAliasHandoff, NonAliasedSlotsDoNotNeedHandoffBarrier)
{
    run_alias_handoff_case</*AliasSlot=*/false, /*InsertHandoffBarrier=*/false>(
        /*expect_diagnostic=*/false,
        /*expect_stable_tail_value=*/true);
}
