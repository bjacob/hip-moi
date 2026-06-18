// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// hip_moi/hip_moi.hpp
//
// Public header for the hip-moi HIP memory-ordering instrumentation library.
#ifndef HIP_MOI_HIP_MOI_HPP
#define HIP_MOI_HIP_MOI_HPP

#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

namespace hip_moi
{
    enum class access_kind : uint32_t
    {
        load  = 0,
        store = 1,
    };

    enum class diagnostic_kind : uint32_t
    {
        none            = 0,
        access_conflict = 1,
        metadata_full   = 2,
    };

    struct config
    {
        int thread_count;
        int threads_per_subgroup;
        int subgroup_count;
    };

    struct access_record
    {
        uintptr_t address;
        uint32_t  byte_count;
        uint32_t  thread_id;
        uint32_t  subgroup_id;
        uint32_t  epoch;
        uint32_t  kind;
        uint32_t  valid;
    };

    struct diagnostic
    {
        uint32_t  kind;
        uint32_t  epoch;
        uint32_t  writer_or_first_thread_id;
        uint32_t  reader_or_second_thread_id;
        uintptr_t first_addr;
        uintptr_t second_addr;
        uint32_t  first_size;
        uint32_t  second_size;
    };

    struct subgroup_state
    {
        uint32_t epoch;
    };

    struct context_storage_ref
    {
        access_record*  access_records;
        int             access_record_capacity;
        diagnostic*     diagnostics;
        int             diagnostic_capacity;
        subgroup_state* subgroup_states;
        int             subgroup_capacity;
        int*            access_count;
        int*            diagnostic_count;
    };

    template <int AccessCapacity, int DiagnosticCapacity, int SubgroupCapacity = 1>
    struct static_context_storage
    {
        access_record  access_records[AccessCapacity];
        diagnostic     diagnostics[DiagnosticCapacity];
        subgroup_state subgroup_states[SubgroupCapacity];
        int            access_count;
        int            diagnostic_count;

        __device__ context_storage_ref ref()
        {
            return context_storage_ref{
                access_records,
                AccessCapacity,
                diagnostics,
                DiagnosticCapacity,
                subgroup_states,
                SubgroupCapacity,
                &access_count,
                &diagnostic_count,
            };
        }
    };

    class context
    {
    public:
        __device__ context(context_storage_ref storage, config cfg)
            : storage_(storage)
            , cfg_(cfg)
        {
        }

        __device__ void init_workgroup()
        {
            if(threadIdx.x == 0)
            {
                if(storage_.access_count)
                {
                    *storage_.access_count = 0;
                }
                if(storage_.diagnostic_count)
                {
                    *storage_.diagnostic_count = 0;
                }
                for(int i = 0; storage_.subgroup_states && i < storage_.subgroup_capacity; ++i)
                {
                    storage_.subgroup_states[i].epoch = 0;
                }
            }
            __syncthreads();
        }

        template <typename T>
        __device__ T lds_load(const T* ptr)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::context::lds_load requires a trivially copyable type");
            return *ptr;
        }

        template <typename T>
        __device__ void lds_store(T* ptr, T value)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::context::lds_store requires a trivially copyable type");
            *ptr = value;
        }

        __device__ void syncthreads()
        {
            __syncthreads();
            if(threadIdx.x == 0 && storage_.subgroup_states && storage_.subgroup_capacity > 0)
            {
                ++storage_.subgroup_states[0].epoch;
            }
            __syncthreads();
        }

        __device__ bool has_error() const
        {
            return error_count() != 0;
        }

        __device__ int error_count() const
        {
            return storage_.diagnostic_count ? *storage_.diagnostic_count : 0;
        }

        __device__ config configuration() const
        {
            return cfg_;
        }

    private:
        context_storage_ref storage_;
        config              cfg_;
    };

} // namespace hip_moi

#endif // HIP_MOI_HIP_MOI_HPP
