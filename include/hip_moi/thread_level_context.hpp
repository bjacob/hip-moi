// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// hip_moi/thread_level_context.hpp
//
// Thread-level HIP memory-ordering instrumentation context.
#ifndef HIP_MOI_THREAD_LEVEL_CONTEXT_HPP
#define HIP_MOI_THREAD_LEVEL_CONTEXT_HPP

#include "hip_moi/common.hpp"

#include <type_traits>

namespace hip_moi
{
    class thread_level_context
    {
    public:
        __device__ thread_level_context(context_storage_ref storage, config cfg)
            : storage_(storage)
            , cfg_(cfg)
        {
        }

        __device__ void init_workgroup()
        {
            if(thread_id() == 0)
            {
                if(storage_.access_count)
                {
                    *storage_.access_count = 0;
                }
                if(storage_.epoch_access_count)
                {
                    *storage_.epoch_access_count = 0;
                }
                if(storage_.diagnostic_count)
                {
                    *storage_.diagnostic_count = 0;
                }
                for(int i = 0; storage_.subgroup_states && i < storage_.subgroup_capacity; ++i)
                {
                    storage_.subgroup_states[i].epoch = 0;
                }
                for(int i = 0; storage_.access_records && i < storage_.access_record_capacity; ++i)
                {
                    storage_.access_records[i].valid = 0;
                }
                __threadfence();
            }
            __syncthreads();
        }

        template <typename T>
        __device__ T lds_load(const T* ptr)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::thread_level_context::lds_load requires a trivially copyable "
                          "type");
            record_access(ptr, sizeof(T), access_kind::load);
            return *ptr;
        }

        template <typename T>
        __device__ void lds_store(T* ptr, T value)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::thread_level_context::lds_store requires a trivially "
                          "copyable type");
            record_access(ptr, sizeof(T), access_kind::store);
            *ptr = value;
        }

        __device__ void syncthreads()
        {
            __syncthreads();
            if(thread_id() == 0 && storage_.subgroup_states && storage_.subgroup_capacity > 0)
            {
                if(storage_.epoch_access_count)
                {
                    *storage_.epoch_access_count = 0;
                }
                int subgroup_count = detail::stored_subgroup_count(storage_, cfg_);
                for(int i = 0; i < subgroup_count; ++i)
                {
                    ++storage_.subgroup_states[i].epoch;
                }
                __threadfence();
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

        __device__ uint32_t thread_id() const
        {
            return static_cast<uint32_t>(threadIdx.x
                                         + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
        }

        __device__ uint32_t subgroup_id() const
        {
            if(cfg_.threads_per_subgroup <= 0)
            {
                return 0;
            }

            uint32_t subgroup = thread_id() / static_cast<uint32_t>(cfg_.threads_per_subgroup);
            uint32_t subgroup_count = detail::configured_subgroup_count(cfg_);
            return subgroup < subgroup_count ? subgroup : subgroup_count - 1;
        }

        __device__ uint32_t thread_rank_in_subgroup() const
        {
            if(cfg_.threads_per_subgroup <= 0)
            {
                return thread_id();
            }
            return thread_id() % static_cast<uint32_t>(cfg_.threads_per_subgroup);
        }

    private:
        __device__ bool conflicts_with(const access_record& first,
                                       const access_record& second) const
        {
            return first.valid && first.epoch == second.epoch && first.thread_id != second.thread_id
                   && (detail::is_write(first.kind) || detail::is_write(second.kind))
                   && detail::byte_ranges_overlap(first, second);
        }

        __device__ void record_access(const void* ptr, uint32_t byte_count, access_kind kind)
        {
            if(!storage_.access_records || !storage_.access_count || !storage_.epoch_access_count
               || storage_.access_record_capacity <= 0)
            {
                return;
            }

            uint32_t      subgroup = subgroup_id();
            access_record record{
                reinterpret_cast<uintptr_t>(ptr),
                byte_count,
                thread_id(),
                subgroup,
                detail::current_epoch(storage_, subgroup),
                static_cast<uint32_t>(kind),
                1,
            };

            (void)atomicAdd(storage_.access_count, 1);
            int record_index = atomicAdd(storage_.epoch_access_count, 1);

            if(record_index < storage_.access_record_capacity)
            {
                record.valid                                = 0;
                storage_.access_records[record_index]       = record;
                storage_.access_records[record_index].valid = 0;
                __threadfence();
                storage_.access_records[record_index].valid = 1;
                __threadfence();
            }

            int scan_limit = atomicAdd(storage_.epoch_access_count, 0);
            if(scan_limit > storage_.access_record_capacity)
            {
                scan_limit = storage_.access_record_capacity;
            }
            for(int i = 0; i < scan_limit; ++i)
            {
                if(i == record_index)
                {
                    continue;
                }
                access_record prior = storage_.access_records[i];
                if(conflicts_with(prior, record))
                {
                    detail::emit_conflict(storage_, prior, record);
                }
            }

            if(record_index >= storage_.access_record_capacity)
            {
                detail::emit_metadata_full(storage_, record);
            }
        }

        context_storage_ref storage_;
        config              cfg_;
    };

    using context = thread_level_context;
} // namespace hip_moi

#endif // HIP_MOI_THREAD_LEVEL_CONTEXT_HPP
