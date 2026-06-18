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
        int*            metadata_lock;
    };

    template <int AccessCapacity, int DiagnosticCapacity, int SubgroupCapacity = 1>
    struct static_context_storage
    {
        access_record  access_records[AccessCapacity];
        diagnostic     diagnostics[DiagnosticCapacity];
        subgroup_state subgroup_states[SubgroupCapacity];
        int            access_count;
        int            diagnostic_count;
        int            metadata_lock;

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
                &metadata_lock,
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
                if(storage_.metadata_lock)
                {
                    *storage_.metadata_lock = 0;
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
            record_access(ptr, sizeof(T), access_kind::load);
            return *ptr;
        }

        template <typename T>
        __device__ void lds_store(T* ptr, T value)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::context::lds_store requires a trivially copyable type");
            record_access(ptr, sizeof(T), access_kind::store);
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
        __device__ uint32_t subgroup_id() const
        {
            if(cfg_.threads_per_subgroup <= 0)
            {
                return 0;
            }

            uint32_t subgroup = static_cast<uint32_t>(threadIdx.x / cfg_.threads_per_subgroup);
            if(cfg_.subgroup_count <= 0)
            {
                return 0;
            }

            uint32_t subgroup_count = static_cast<uint32_t>(cfg_.subgroup_count);
            return subgroup < subgroup_count ? subgroup : subgroup_count - 1;
        }

        __device__ uint32_t current_epoch(uint32_t subgroup) const
        {
            if(storage_.subgroup_states
               && subgroup < static_cast<uint32_t>(storage_.subgroup_capacity))
            {
                return storage_.subgroup_states[subgroup].epoch;
            }
            return 0;
        }

        __device__ void lock_metadata() const
        {
            if(!storage_.metadata_lock)
            {
                return;
            }
            while(atomicCAS(storage_.metadata_lock, 0, 1) != 0)
            {
            }
        }

        __device__ void unlock_metadata() const
        {
            if(!storage_.metadata_lock)
            {
                return;
            }
            __threadfence();
            atomicExch(storage_.metadata_lock, 0);
        }

        __device__ static bool byte_ranges_overlap(const access_record& first,
                                                   const access_record& second)
        {
            uintptr_t first_end  = first.address + first.byte_count;
            uintptr_t second_end = second.address + second.byte_count;
            return first.address < second_end && second.address < first_end;
        }

        __device__ static bool is_write(access_kind kind)
        {
            return kind == access_kind::store;
        }

        __device__ static bool is_write(uint32_t kind)
        {
            return is_write(static_cast<access_kind>(kind));
        }

        __device__ void emit_diagnostic(diagnostic diagnostic_record) const
        {
            if(!storage_.diagnostics || !storage_.diagnostic_count
               || storage_.diagnostic_capacity <= 0)
            {
                return;
            }

            int index = *storage_.diagnostic_count;
            if(index < storage_.diagnostic_capacity)
            {
                storage_.diagnostics[index] = diagnostic_record;
            }
            *storage_.diagnostic_count = index + 1;
        }

        __device__ void emit_conflict(const access_record& first, const access_record& second) const
        {
            emit_diagnostic(diagnostic{
                static_cast<uint32_t>(diagnostic_kind::access_conflict),
                second.epoch,
                first.thread_id,
                second.thread_id,
                first.address,
                second.address,
                first.byte_count,
                second.byte_count,
            });
        }

        __device__ void emit_metadata_full(const access_record& record) const
        {
            emit_diagnostic(diagnostic{
                static_cast<uint32_t>(diagnostic_kind::metadata_full),
                record.epoch,
                record.thread_id,
                record.thread_id,
                record.address,
                record.address,
                record.byte_count,
                record.byte_count,
            });
        }

        __device__ bool conflicts_with(const access_record& first,
                                       const access_record& second) const
        {
            return first.valid && first.epoch == second.epoch
                   && first.subgroup_id == second.subgroup_id && first.thread_id != second.thread_id
                   && (is_write(first.kind) || is_write(second.kind))
                   && byte_ranges_overlap(first, second);
        }

        __device__ void record_access(const void* ptr, uint32_t byte_count, access_kind kind)
        {
            if(!storage_.access_records || !storage_.access_count
               || storage_.access_record_capacity <= 0)
            {
                return;
            }

            uint32_t      subgroup = subgroup_id();
            access_record record{
                reinterpret_cast<uintptr_t>(ptr),
                byte_count,
                static_cast<uint32_t>(threadIdx.x),
                subgroup,
                current_epoch(subgroup),
                static_cast<uint32_t>(kind),
                1,
            };

            lock_metadata();

            int access_count = *storage_.access_count;
            for(int i = 0; i < access_count && i < storage_.access_record_capacity; ++i)
            {
                access_record prior = storage_.access_records[i];
                if(conflicts_with(prior, record))
                {
                    emit_conflict(prior, record);
                }
            }

            if(access_count < storage_.access_record_capacity)
            {
                storage_.access_records[access_count] = record;
            }
            else
            {
                emit_metadata_full(record);
            }
            *storage_.access_count = access_count + 1;

            unlock_metadata();
        }

        context_storage_ref storage_;
        config              cfg_;
    };

} // namespace hip_moi

#endif // HIP_MOI_HIP_MOI_HPP
