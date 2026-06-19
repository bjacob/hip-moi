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
    template <typename Context>
    class device_context_storage_for
    {
    public:
        using access_record = typename Context::access_record;
        using diagnostic    = typename Context::diagnostic;
        using storage_ref   = typename Context::storage_ref;

        device_context_storage_for() = default;

        device_context_storage_for(const device_context_storage_for&)            = delete;
        device_context_storage_for& operator=(const device_context_storage_for&) = delete;

        ~device_context_storage_for()
        {
            release();
        }

        void allocate(int access_capacity, int diagnostic_capacity, int subgroup_capacity)
        {
            access_record_capacity_ = access_capacity;
            diagnostic_capacity_    = diagnostic_capacity;
            subgroup_capacity_      = subgroup_capacity;

            HIP_MOI_TEST_HIP_ASSERT(
                hipMalloc(&access_records_, access_capacity * sizeof(access_record)));
            HIP_MOI_TEST_HIP_ASSERT(
                hipMalloc(&diagnostics_, diagnostic_capacity * sizeof(diagnostic)));
            HIP_MOI_TEST_HIP_ASSERT(
                hipMalloc(&subgroup_states_, subgroup_capacity * sizeof(subgroup_state)));
            HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&access_count_, sizeof(int)));
            HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&epoch_access_count_, sizeof(int)));
            HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&diagnostic_count_, sizeof(int)));
        }

        storage_ref ref() const
        {
            return storage_ref{
                access_records_,
                access_record_capacity_,
                diagnostics_,
                diagnostic_capacity_,
                subgroup_states_,
                subgroup_capacity_,
                access_count_,
                epoch_access_count_,
                diagnostic_count_,
            };
        }

        diagnostic* diagnostics_device() const
        {
            return diagnostics_;
        }

        access_record* access_records_device() const
        {
            return access_records_;
        }

        int* access_count_device() const
        {
            return access_count_;
        }

        int* diagnostic_count_device() const
        {
            return diagnostic_count_;
        }

    private:
        void release()
        {
            if(access_records_)
            {
                (void)hipFree(access_records_);
                access_records_ = nullptr;
            }
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
            if(access_count_)
            {
                (void)hipFree(access_count_);
                access_count_ = nullptr;
            }
            if(epoch_access_count_)
            {
                (void)hipFree(epoch_access_count_);
                epoch_access_count_ = nullptr;
            }
            if(diagnostic_count_)
            {
                (void)hipFree(diagnostic_count_);
                diagnostic_count_ = nullptr;
            }
        }

        access_record*  access_records_         = nullptr;
        diagnostic*     diagnostics_            = nullptr;
        subgroup_state* subgroup_states_        = nullptr;
        int*            access_count_           = nullptr;
        int*            epoch_access_count_     = nullptr;
        int*            diagnostic_count_       = nullptr;
        int             access_record_capacity_ = 0;
        int             diagnostic_capacity_    = 0;
        int             subgroup_capacity_      = 0;
    };

    using thread_level_device_context_storage
        = device_context_storage_for<hip_moi::thread_level_context>;
    using device_context_storage = thread_level_device_context_storage;
    using subgroup_level_device_context_storage
        = device_context_storage_for<hip_moi::subgroup_level_context>;
} // namespace hip_moi::test

#endif // HIP_MOI_TESTS_INSTRUMENTED_TEST_SUPPORT_HPP
