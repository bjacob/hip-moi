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
    __device__ Context make_single_subgroup_context(typename Context::storage_ref storage)
    {
        typename Context::config cfg{
            /*thread_count=*/static_cast<int>(blockDim.x),
            /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
            /*subgroup_count=*/1,
        };
        return Context(storage, cfg);
    }

    template <typename Context>
    class device_context_storage_for
    {
    public:
        using access_record = typename Context::access_record;
        using coalesced_access_record =
            typename hip_moi::detail::optional_coalesced_access_record<Context>::type;
        using coalescing_access_record =
            typename hip_moi::detail::optional_coalescing_access_record<Context>::type;
        using coalescing_group_record =
            typename hip_moi::detail::optional_coalescing_group_record<Context>::type;
        using diagnostic    = typename Context::diagnostic;
        using storage_ref   = typename Context::storage_ref;

        static constexpr bool has_coalesced_access_records
            = hip_moi::detail::optional_coalesced_access_record<Context>::available;
        static constexpr bool has_coalescing_access_records
            = hip_moi::detail::optional_coalescing_access_record<Context>::available;
        static constexpr bool has_coalescing_group_records
            = hip_moi::detail::optional_coalescing_group_record<Context>::available;

        device_context_storage_for() = default;

        device_context_storage_for(const device_context_storage_for&)            = delete;
        device_context_storage_for& operator=(const device_context_storage_for&) = delete;

        ~device_context_storage_for()
        {
            release();
        }

        void allocate(int access_capacity, int diagnostic_capacity, int subgroup_capacity)
        {
            allocate(access_capacity,
                     diagnostic_capacity,
                     subgroup_capacity,
                     /*coalescing_access_capacity=*/0,
                     /*coalescing_group_capacity=*/0);
        }

        void allocate(int access_capacity,
                      int diagnostic_capacity,
                      int subgroup_capacity,
                      int coalescing_access_capacity)
        {
            allocate(access_capacity,
                     diagnostic_capacity,
                     subgroup_capacity,
                     coalescing_access_capacity,
                     /*coalescing_group_capacity=*/0);
        }

        void allocate(int access_capacity,
                      int diagnostic_capacity,
                      int subgroup_capacity,
                      int coalescing_access_capacity,
                      int coalescing_group_capacity)
        {
            access_record_capacity_ = access_capacity;
            coalesced_access_record_capacity_ = access_capacity;
            coalescing_access_record_capacity_ = coalescing_access_capacity;
            coalescing_group_record_capacity_ = coalescing_group_capacity;
            diagnostic_capacity_    = diagnostic_capacity;
            subgroup_capacity_      = subgroup_capacity;

            HIP_MOI_TEST_HIP_ASSERT(
                hipMalloc(&access_records_, access_capacity * sizeof(access_record)));
            if constexpr(has_coalesced_access_records)
            {
                HIP_MOI_TEST_HIP_ASSERT(
                    hipMalloc(&coalesced_access_records_,
                              coalesced_access_record_capacity_ * sizeof(coalesced_access_record)));
                HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&coalesced_access_count_, sizeof(int)));
            }
            if constexpr(has_coalescing_access_records)
            {
                HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&coalescing_access_count_, sizeof(int)));
                HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&epoch_coalescing_access_count_, sizeof(int)));
                HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&coalescing_fallback_count_, sizeof(int)));
                if(coalescing_access_record_capacity_ > 0)
                {
                    HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&coalescing_access_records_,
                                                      coalescing_access_record_capacity_
                                                          * sizeof(coalescing_access_record)));
                }
                if constexpr(has_coalescing_group_records)
                {
                    if(coalescing_group_record_capacity_ > 0)
                    {
                        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&coalescing_group_records_,
                                                          coalescing_group_record_capacity_
                                                              * sizeof(coalescing_group_record)));
                        HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&coalescing_group_count_, sizeof(int)));
                    }
                }
            }
            HIP_MOI_TEST_HIP_ASSERT(
                hipMalloc(&diagnostics_, diagnostic_capacity * sizeof(diagnostic)));
            HIP_MOI_TEST_HIP_ASSERT(
                hipMalloc(&subgroup_states_, subgroup_capacity * sizeof(subgroup_state)));
            HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&access_count_, sizeof(int)));
            HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&epoch_access_count_, sizeof(int)));
            HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&diagnostic_count_, sizeof(int)));
            HIP_MOI_TEST_HIP_ASSERT(hipMalloc(&simulated_barrier_arrival_count_, sizeof(int)));
        }

        storage_ref ref() const
        {
            if constexpr(has_coalescing_access_records)
            {
                storage_ref ref{
                    access_records_,
                    access_record_capacity_,
                    diagnostics_,
                    diagnostic_capacity_,
                    subgroup_states_,
                    subgroup_capacity_,
                    access_count_,
                    epoch_access_count_,
                    diagnostic_count_,
                    coalesced_access_records_,
                    coalesced_access_record_capacity_,
                    coalesced_access_count_,
                    coalescing_access_records_,
                    coalescing_access_record_capacity_,
                    coalescing_access_count_,
                    epoch_coalescing_access_count_,
                    coalescing_fallback_count_,
                    coalescing_group_records_,
                    coalescing_group_record_capacity_,
                    coalescing_group_count_,
                };
                ref.simulated_barrier_arrival_count = simulated_barrier_arrival_count_;
                return ref;
            }
            else if constexpr(has_coalesced_access_records)
            {
                storage_ref ref{
                    access_records_,
                    access_record_capacity_,
                    diagnostics_,
                    diagnostic_capacity_,
                    subgroup_states_,
                    subgroup_capacity_,
                    access_count_,
                    epoch_access_count_,
                    diagnostic_count_,
                    coalesced_access_records_,
                    coalesced_access_record_capacity_,
                    coalesced_access_count_,
                };
                ref.simulated_barrier_arrival_count = simulated_barrier_arrival_count_;
                return ref;
            }
            else
            {
                storage_ref ref{
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
                ref.simulated_barrier_arrival_count = simulated_barrier_arrival_count_;
                return ref;
            }
        }

        diagnostic* diagnostics_device() const
        {
            return diagnostics_;
        }

        access_record* access_records_device() const
        {
            return access_records_;
        }

        coalesced_access_record* coalesced_access_records_device() const
        {
            return coalesced_access_records_;
        }

        coalescing_access_record* coalescing_access_records_device() const
        {
            return coalescing_access_records_;
        }

        coalescing_group_record* coalescing_group_records_device() const
        {
            return coalescing_group_records_;
        }

        int* access_count_device() const
        {
            return access_count_;
        }

        int* coalesced_access_count_device() const
        {
            return coalesced_access_count_;
        }

        int* coalescing_access_count_device() const
        {
            return coalescing_access_count_;
        }

        int* coalescing_fallback_count_device() const
        {
            return coalescing_fallback_count_;
        }

        int* coalescing_group_count_device() const
        {
            return coalescing_group_count_;
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
            if(access_records_)
            {
                (void)hipFree(access_records_);
                access_records_ = nullptr;
            }
            if(coalesced_access_records_)
            {
                (void)hipFree(coalesced_access_records_);
                coalesced_access_records_ = nullptr;
            }
            if(coalescing_access_records_)
            {
                (void)hipFree(coalescing_access_records_);
                coalescing_access_records_ = nullptr;
            }
            if(coalescing_group_records_)
            {
                (void)hipFree(coalescing_group_records_);
                coalescing_group_records_ = nullptr;
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
            if(simulated_barrier_arrival_count_)
            {
                (void)hipFree(simulated_barrier_arrival_count_);
                simulated_barrier_arrival_count_ = nullptr;
            }
            if(coalesced_access_count_)
            {
                (void)hipFree(coalesced_access_count_);
                coalesced_access_count_ = nullptr;
            }
            if(coalescing_access_count_)
            {
                (void)hipFree(coalescing_access_count_);
                coalescing_access_count_ = nullptr;
            }
            if(epoch_coalescing_access_count_)
            {
                (void)hipFree(epoch_coalescing_access_count_);
                epoch_coalescing_access_count_ = nullptr;
            }
            if(coalescing_fallback_count_)
            {
                (void)hipFree(coalescing_fallback_count_);
                coalescing_fallback_count_ = nullptr;
            }
            if(coalescing_group_count_)
            {
                (void)hipFree(coalescing_group_count_);
                coalescing_group_count_ = nullptr;
            }
        }

        access_record*           access_records_                   = nullptr;
        coalesced_access_record* coalesced_access_records_         = nullptr;
        coalescing_access_record* coalescing_access_records_         = nullptr;
        coalescing_group_record* coalescing_group_records_         = nullptr;
        diagnostic*              diagnostics_                      = nullptr;
        subgroup_state*          subgroup_states_                  = nullptr;
        int*                     access_count_                     = nullptr;
        int*                     epoch_access_count_               = nullptr;
        int*                     diagnostic_count_                 = nullptr;
        int*                      simulated_barrier_arrival_count_   = nullptr;
        int*                     coalesced_access_count_           = nullptr;
        int*                      coalescing_access_count_           = nullptr;
        int*                      epoch_coalescing_access_count_     = nullptr;
        int*                      coalescing_fallback_count_         = nullptr;
        int*                     coalescing_group_count_           = nullptr;
        int                      access_record_capacity_           = 0;
        int                      coalesced_access_record_capacity_ = 0;
        int                       coalescing_access_record_capacity_ = 0;
        int                      coalescing_group_record_capacity_ = 0;
        int                      diagnostic_capacity_              = 0;
        int                      subgroup_capacity_                = 0;
    };

    using thread_level_device_context_storage
        = device_context_storage_for<hip_moi::thread_level_context>;
    using device_context_storage = thread_level_device_context_storage;
    using subgroup_level_device_context_storage
        = device_context_storage_for<hip_moi::subgroup_level_context>;

    template <typename Storage>
    void expect_metadata_counts(const Storage& storage,
                                int            expected_access_count,
                                int            expected_diagnostic_count)
    {
        int access_count = -1;
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&access_count,
                                          storage.access_count_device(),
                                          sizeof(access_count),
                                          hipMemcpyDeviceToHost));

        int diagnostic_count = -1;
        HIP_MOI_TEST_HIP_ASSERT(hipMemcpy(&diagnostic_count,
                                          storage.diagnostic_count_device(),
                                          sizeof(diagnostic_count),
                                          hipMemcpyDeviceToHost));

        EXPECT_EQ(access_count, expected_access_count);
        EXPECT_EQ(diagnostic_count, expected_diagnostic_count);
    }
} // namespace hip_moi::test

#endif // HIP_MOI_TESTS_INSTRUMENTED_TEST_SUPPORT_HPP
