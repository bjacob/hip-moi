// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/test_support.hpp
//
// Host-side helpers for hip-moi instrumented GPU tests.
#ifndef HIP_MOI_TESTS_INSTRUMENTED_TEST_SUPPORT_HPP
#define HIP_MOI_TESTS_INSTRUMENTED_TEST_SUPPORT_HPP

#include "hip_moi/hip_moi.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#define HIP_MOI_TEST_HIP_ASSERT(expr)                                   \
    do                                                                  \
    {                                                                   \
        hipError_t hip_moi_test_status = (expr);                        \
        ASSERT_EQ(hip_moi_test_status, hipSuccess)                      \
            << #expr << ": " << hipGetErrorString(hip_moi_test_status); \
    } while(false)

namespace hip_moi::test
{
    __device__ inline context make_single_subgroup_context(context::storage_ref storage)
    {
        context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
            /*subgroup_count=*/1,
        };
        return context(storage, cfg);
    }

    class device_context_storage
    {
    public:
        using diagnostic  = context::diagnostic;
        using storage_ref = context::storage_ref;

        device_context_storage() = default;

        device_context_storage(const device_context_storage&)            = delete;
        device_context_storage& operator=(const device_context_storage&) = delete;

        ~device_context_storage()
        {
            release();
        }

        void allocate_with_exact_shadow(int diagnostic_capacity,
                                        int subgroup_capacity,
                                        int exact_shadow_entry_capacity)
        {
            allocate(diagnostic_capacity,
                     subgroup_capacity,
                     exact_shadow_entry_capacity,
                     /*sampled_watchpoint_capacity=*/0,
                     backend_kind::exact_shadow);
        }

        void allocate_with_sampled_watchpoints(int diagnostic_capacity,
                                               int subgroup_capacity,
                                               int sampled_watchpoint_capacity)
        {
            allocate(diagnostic_capacity,
                     subgroup_capacity,
                     /*exact_shadow_entry_capacity=*/0,
                     sampled_watchpoint_capacity,
                     backend_kind::sampled_watchpoint);
        }

        void allocate(int          diagnostic_capacity,
                      int          subgroup_capacity,
                      int          exact_shadow_entry_capacity,
                      int          sampled_watchpoint_capacity,
                      backend_kind backend = backend_kind::exact_shadow)
        {
            diagnostic_capacity_         = diagnostic_capacity;
            subgroup_capacity_           = subgroup_capacity;
            exact_shadow_entry_capacity_ = exact_shadow_entry_capacity;
            sampled_watchpoint_capacity_ = sampled_watchpoint_capacity;
            backend_                     = backend;

            HIP_MOI_TEST_HIP_ASSERT(
                hipMalloc(&diagnostics_, diagnostic_capacity * sizeof(diagnostic)));
            HIP_MOI_TEST_HIP_ASSERT(
                hipMalloc(&subgroup_states_, subgroup_capacity * sizeof(subgroup_state)));
            HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&diagnostic_count_, sizeof(int)));
            HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&simulated_barrier_arrival_count_, sizeof(int)));
            if(exact_shadow_entry_capacity_ > 0)
            {
                HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&exact_shadow_entries_,
                                                  exact_shadow_entry_capacity_ * sizeof(uint64_t)));
                HIP_MOI_TEST_HIP_ASSERT(hipMemset(
                    exact_shadow_entries_, 0, exact_shadow_entry_capacity_ * sizeof(uint64_t)));
            }
            if(sampled_watchpoint_capacity_ > 0)
            {
                HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&sampled_watchpoints_,
                                                  sampled_watchpoint_capacity_ * sizeof(uint64_t)));
                HIP_MOI_TEST_HIP_ASSERT(hipMemset(
                    sampled_watchpoints_, 0, sampled_watchpoint_capacity_ * sizeof(uint64_t)));
            }
        }

        storage_ref ref() const
        {
            return storage_ref{
                diagnostics_,
                diagnostic_capacity_,
                subgroup_states_,
                subgroup_capacity_,
                diagnostic_count_,
                simulated_barrier_arrival_count_,
                exact_shadow_entries_,
                exact_shadow_entry_capacity_,
                sampled_watchpoints_,
                sampled_watchpoint_capacity_,
                1,
                static_cast<uint32_t>(backend_),
                1,
                1,
                0,
                1,
            };
        }

        diagnostic* diagnostics_device() const
        {
            return diagnostics_;
        }

        int* diagnostic_count_device() const
        {
            return diagnostic_count_;
        }

        int* simulated_barrier_arrival_count_device() const
        {
            return simulated_barrier_arrival_count_;
        }

    private:
        void release()
        {
            if(diagnostics_)
            {
                (void)hipFree(diagnostics_);
                diagnostics_ = nullptr;
            }
            if(subgroup_states_)
            {
                (void)hipFree(subgroup_states_);
                subgroup_states_ = nullptr;
            }
            if(diagnostic_count_)
            {
                (void)hipFree(diagnostic_count_);
                diagnostic_count_ = nullptr;
            }
            if(simulated_barrier_arrival_count_)
            {
                (void)hipFree(simulated_barrier_arrival_count_);
                simulated_barrier_arrival_count_ = nullptr;
            }
            if(exact_shadow_entries_)
            {
                (void)hipFree(exact_shadow_entries_);
                exact_shadow_entries_ = nullptr;
            }
            if(sampled_watchpoints_)
            {
                (void)hipFree(sampled_watchpoints_);
                sampled_watchpoints_ = nullptr;
            }
        }

        diagnostic*     diagnostics_                     = nullptr;
        subgroup_state* subgroup_states_                 = nullptr;
        int*            diagnostic_count_                = nullptr;
        int*            simulated_barrier_arrival_count_ = nullptr;
        uint64_t*       exact_shadow_entries_            = nullptr;
        uint64_t*       sampled_watchpoints_             = nullptr;
        int             diagnostic_capacity_             = 0;
        int             subgroup_capacity_               = 0;
        int             exact_shadow_entry_capacity_     = 0;
        int             sampled_watchpoint_capacity_     = 0;
        backend_kind    backend_                         = backend_kind::exact_shadow;
    };

    template <typename Storage>
    void expect_diagnostic_count(const Storage& storage, int expected_diagnostic_count)
    {
        int diagnostic_count = -1;
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&diagnostic_count,
                                          storage.diagnostic_count_device(),
                                          sizeof(diagnostic_count),
                                          hipMemcpyDeviceToHost));

        EXPECT_EQ(diagnostic_count, expected_diagnostic_count);
    }
} // namespace hip_moi::test

#endif // HIP_MOI_TESTS_INSTRUMENTED_TEST_SUPPORT_HPP
