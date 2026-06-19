// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// hip_moi/common.hpp
//
// Shared public types for the hip-moi instrumentation contexts.
#ifndef HIP_MOI_COMMON_HPP
#define HIP_MOI_COMMON_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

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
        int*            epoch_access_count;
        int*            diagnostic_count;
    };

    template <int AccessCapacity, int DiagnosticCapacity, int SubgroupCapacity = 1>
    struct static_context_storage
    {
        access_record  access_records[AccessCapacity];
        diagnostic     diagnostics[DiagnosticCapacity];
        subgroup_state subgroup_states[SubgroupCapacity];
        int            access_count;
        int            epoch_access_count;
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
                &epoch_access_count,
                &diagnostic_count,
            };
        }
    };

    namespace detail
    {
        __device__ inline uint32_t configured_subgroup_count(config cfg)
        {
            return cfg.subgroup_count > 0 ? static_cast<uint32_t>(cfg.subgroup_count) : 1u;
        }

        __device__ inline int stored_subgroup_count(context_storage_ref storage, config cfg)
        {
            if(!storage.subgroup_states || storage.subgroup_capacity <= 0)
            {
                return 0;
            }

            uint32_t configured_count = configured_subgroup_count(cfg);
            uint32_t storage_capacity = static_cast<uint32_t>(storage.subgroup_capacity);
            return static_cast<int>(configured_count < storage_capacity ? configured_count
                                                                        : storage_capacity);
        }

        __device__ inline uint32_t current_epoch(context_storage_ref storage, uint32_t subgroup)
        {
            if(storage.subgroup_states
               && subgroup < static_cast<uint32_t>(storage.subgroup_capacity))
            {
                return storage.subgroup_states[subgroup].epoch;
            }
            return 0;
        }

        __device__ inline bool byte_ranges_overlap(const access_record& first,
                                                   const access_record& second)
        {
            uintptr_t first_end  = first.address + first.byte_count;
            uintptr_t second_end = second.address + second.byte_count;
            return first.address < second_end && second.address < first_end;
        }

        __device__ inline bool is_write(access_kind kind)
        {
            return kind == access_kind::store;
        }

        __device__ inline bool is_write(uint32_t kind)
        {
            return is_write(static_cast<access_kind>(kind));
        }

        __device__ inline void emit_diagnostic(context_storage_ref storage,
                                               diagnostic          diagnostic_record)
        {
            if(!storage.diagnostics || !storage.diagnostic_count
               || storage.diagnostic_capacity <= 0)
            {
                return;
            }

            int index = atomicAdd(storage.diagnostic_count, 1);
            if(index < storage.diagnostic_capacity)
            {
                storage.diagnostics[index] = diagnostic_record;
            }
        }

        __device__ inline void emit_conflict(context_storage_ref  storage,
                                             const access_record& first,
                                             const access_record& second)
        {
            emit_diagnostic(storage,
                            diagnostic{
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

        __device__ inline void emit_metadata_full(context_storage_ref  storage,
                                                  const access_record& record)
        {
            emit_diagnostic(storage,
                            diagnostic{
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
    } // namespace detail
} // namespace hip_moi

#endif // HIP_MOI_COMMON_HPP
