// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// hip_moi/common.hpp
//
// Shared public types and helper routines for the hip-moi instrumentation
// contexts.
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

    struct subgroup_state
    {
        uint32_t epoch;
    };

    namespace detail
    {
        template <typename Config>
        __device__ inline uint32_t configured_subgroup_count(Config cfg)
        {
            return cfg.subgroup_count > 0 ? static_cast<uint32_t>(cfg.subgroup_count) : 1u;
        }

        template <typename StorageRef, typename Config>
        __device__ inline int stored_subgroup_count(StorageRef storage, Config cfg)
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

        template <typename StorageRef>
        __device__ inline uint32_t current_epoch(StorageRef storage, uint32_t subgroup)
        {
            if(storage.subgroup_states
               && subgroup < static_cast<uint32_t>(storage.subgroup_capacity))
            {
                return storage.subgroup_states[subgroup].epoch;
            }
            return 0;
        }

        template <typename AccessRecord>
        __device__ inline bool byte_ranges_overlap(const AccessRecord& first,
                                                   const AccessRecord& second)
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

        template <typename StorageRef>
        __device__ inline void
            emit_diagnostic(StorageRef                           storage,
                            typename StorageRef::diagnostic_type diagnostic_record)
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
    } // namespace detail
} // namespace hip_moi

#endif // HIP_MOI_COMMON_HPP
