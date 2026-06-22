// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// hip_moi/context.hpp
//
// Subgroup-scoped HIP memory-ordering instrumentation context.
#ifndef HIP_MOI_CONTEXT_HPP
#define HIP_MOI_CONTEXT_HPP

#include "hip_moi/common.hpp"

#include <type_traits>

namespace hip_moi
{
    class context
    {
    public:
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
            uint32_t  subgroup_id;
            uint32_t  epoch;
            uint32_t  kind;
            uint32_t  valid;
            uint64_t  site_id;
        };

        struct coalesced_access_record
        {
            uintptr_t first_address;
            uint32_t  byte_count;
            uint32_t  span_byte_count;
            uint32_t  first_lane;
            uint32_t  subgroup_id;
            uint32_t  epoch;
            uint32_t  kind;
            uint32_t  participant_count;
            uint32_t  valid;
            int64_t   address_stride;
            uint64_t  site_id;
        };

        struct coalescing_access_record
        {
            uintptr_t address;
            uint32_t  byte_count;
            uint32_t  lane;
            uint32_t  subgroup_id;
            uint32_t  epoch;
            uint32_t  kind;
            uint32_t  valid;
            uint64_t  site_id;
        };

        struct coalescing_group_record
        {
            uintptr_t first_lane_address;
            uintptr_t last_lane_address;
            uint64_t  site_id;
            uint64_t  lane_mask;
            uint64_t  repeated_lane_mask;
            uint32_t  byte_count;
            uint32_t  first_lane;
            uint32_t  last_lane;
            uint32_t  subgroup_id;
            uint32_t  epoch;
            uint32_t  kind;
            uint32_t  participant_count;
            uint32_t  valid;
        };

        struct diagnostic
        {
            uint32_t  kind;
            uint32_t  epoch;
            uint32_t  first_subgroup_id;
            uint32_t  second_subgroup_id;
            uintptr_t first_addr;
            uintptr_t second_addr;
            uint32_t  first_size;
            uint32_t  second_size;
            uint64_t  first_site_id;
            uint64_t  second_site_id;
            uint32_t  expected_thread_count = 0;
            uint32_t  observed_thread_count = 0;
        };

        struct storage_ref
        {
            using access_record_type = access_record;
            using diagnostic_type    = diagnostic;

            access_record*            access_records;
            int                       access_record_capacity;
            diagnostic*               diagnostics;
            int                       diagnostic_capacity;
            subgroup_state*           subgroup_states;
            int                       subgroup_capacity;
            int*                      access_count;
            int*                      epoch_access_count;
            int*                      diagnostic_count;
            coalesced_access_record*  coalesced_access_records          = nullptr;
            int                       coalesced_access_record_capacity  = 0;
            int*                      coalesced_access_count            = nullptr;
            coalescing_access_record* coalescing_access_records         = nullptr;
            int                       coalescing_access_record_capacity = 0;
            int*                      coalescing_access_count           = nullptr;
            int*                      epoch_coalescing_access_count     = nullptr;
            int*                      coalescing_fallback_count         = nullptr;
            coalescing_group_record*  coalescing_group_records          = nullptr;
            int                       coalescing_group_record_capacity  = 0;
            int*                      coalescing_group_count            = nullptr;
            int*                      simulated_barrier_arrival_count   = nullptr;
        };

        template <int AccessCapacity,
                  int DiagnosticCapacity,
                  int SubgroupCapacity         = 1,
                  int CoalescedAccessCapacity  = AccessCapacity,
                  int CoalescingAccessCapacity = AccessCapacity,
                  int CoalescingGroupCapacity  = CoalescingAccessCapacity>
        struct static_context_storage
        {
            access_record            access_records[AccessCapacity];
            diagnostic               diagnostics[DiagnosticCapacity];
            subgroup_state           subgroup_states[SubgroupCapacity];
            coalesced_access_record  coalesced_access_records[CoalescedAccessCapacity];
            coalescing_access_record coalescing_access_records[CoalescingAccessCapacity];
            coalescing_group_record  coalescing_group_records[CoalescingGroupCapacity];
            int                      access_count;
            int                      epoch_access_count;
            int                      diagnostic_count;
            int                      coalesced_access_count;
            int                      coalescing_access_count;
            int                      epoch_coalescing_access_count;
            int                      coalescing_fallback_count;
            int                      coalescing_group_count;
            int                      simulated_barrier_arrival_count;

            __device__ storage_ref ref()
            {
                return storage_ref{
                    access_records,
                    AccessCapacity,
                    diagnostics,
                    DiagnosticCapacity,
                    subgroup_states,
                    SubgroupCapacity,
                    &access_count,
                    &epoch_access_count,
                    &diagnostic_count,
                    coalesced_access_records,
                    CoalescedAccessCapacity,
                    &coalesced_access_count,
                    coalescing_access_records,
                    CoalescingAccessCapacity,
                    &coalescing_access_count,
                    &epoch_coalescing_access_count,
                    &coalescing_fallback_count,
                    coalescing_group_records,
                    CoalescingGroupCapacity,
                    &coalescing_group_count,
                    &simulated_barrier_arrival_count,
                };
            }
        };

        __device__ context(storage_ref storage, config cfg)
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
                if(storage_.coalesced_access_count)
                {
                    *storage_.coalesced_access_count = 0;
                }
                if(storage_.coalescing_access_count)
                {
                    *storage_.coalescing_access_count = 0;
                }
                if(storage_.epoch_coalescing_access_count)
                {
                    *storage_.epoch_coalescing_access_count = 0;
                }
                if(storage_.coalescing_fallback_count)
                {
                    *storage_.coalescing_fallback_count = 0;
                }
                if(storage_.coalescing_group_count)
                {
                    *storage_.coalescing_group_count = 0;
                }
                if(storage_.simulated_barrier_arrival_count)
                {
                    *storage_.simulated_barrier_arrival_count = 0;
                }
                for(int i = 0; storage_.subgroup_states && i < storage_.subgroup_capacity; ++i)
                {
                    storage_.subgroup_states[i].epoch = 0;
                }
                for(int i = 0; storage_.access_records && i < storage_.access_record_capacity; ++i)
                {
                    storage_.access_records[i].valid = 0;
                }
                for(int i = 0; storage_.coalesced_access_records
                               && i < storage_.coalesced_access_record_capacity;
                    ++i)
                {
                    storage_.coalesced_access_records[i].valid = 0;
                }
                for(int i = 0; storage_.coalescing_access_records
                               && i < storage_.coalescing_access_record_capacity;
                    ++i)
                {
                    storage_.coalescing_access_records[i].valid = 0;
                }
                for(int i = 0; storage_.coalescing_group_records
                               && i < storage_.coalescing_group_record_capacity;
                    ++i)
                {
                    storage_.coalescing_group_records[i].valid = 0;
                }
                if(detail::configured_subgroup_count(cfg_) > static_cast<uint32_t>(
                       storage_.subgroup_capacity > 0 ? storage_.subgroup_capacity : 0))
                {
                    emit_subgroup_capacity_full();
                }
                __threadfence();
            }
            __syncthreads();
        }

        template <typename T>
        __device__ T lds_load(const T* ptr, site_id site = no_site_id)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::context::lds_load requires a trivially "
                          "copyable type");
            record_access(ptr, sizeof(T), access_kind::load, site);
            return *ptr;
        }

        template <typename T>
        __device__ void lds_store(T* ptr, T value, site_id site = no_site_id)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::context::lds_store requires a trivially "
                          "copyable type");
            record_access(ptr, sizeof(T), access_kind::store, site);
            *ptr = value;
        }

        __device__ void simulate_syncthreads(bool participates, site_id site = no_site_id)
        {
            if(!storage_.simulated_barrier_arrival_count)
            {
                return;
            }

            if(participates)
            {
                (void)atomicAdd(storage_.simulated_barrier_arrival_count, 1);
            }

            __syncthreads();
            if(thread_id() == 0)
            {
                int observed_count = *storage_.simulated_barrier_arrival_count;
                int expected_count = cfg_.thread_count > 0
                                         ? cfg_.thread_count
                                         : static_cast<int>(blockDim.x * blockDim.y * blockDim.z);
                if(observed_count == expected_count)
                {
                    close_current_epoch(/*advance_epochs=*/true);
                }
                else
                {
                    emit_barrier_divergence(expected_count, observed_count, site);
                }
                *storage_.simulated_barrier_arrival_count = 0;
                __threadfence();
            }
            __syncthreads();
        }

        __device__ void syncthreads()
        {
            __syncthreads();
            close_current_epoch(/*advance_epochs=*/true);
            __syncthreads();
        }

        __device__ void close_epoch()
        {
            __syncthreads();
            close_current_epoch(/*advance_epochs=*/true);
            __syncthreads();
        }

        __device__ void finish()
        {
            close_epoch();
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

        __device__ uint32_t lane_in_subgroup() const
        {
            if(cfg_.threads_per_subgroup <= 0)
            {
                return thread_id();
            }
            return thread_id() % static_cast<uint32_t>(cfg_.threads_per_subgroup);
        }

        __device__ uint32_t thread_rank_in_subgroup() const
        {
            return lane_in_subgroup();
        }

    private:
        __device__ bool conflicts_with(const access_record& first,
                                       const access_record& second) const
        {
            return first.valid && first.epoch == second.epoch
                   && first.subgroup_id != second.subgroup_id
                   && (detail::is_write(first.kind) || detail::is_write(second.kind))
                   && detail::byte_ranges_overlap(first, second);
        }

        __device__ int current_epoch_access_scan_limit() const
        {
            if(!storage_.access_records || !storage_.epoch_access_count
               || storage_.access_record_capacity <= 0)
            {
                return 0;
            }

            int scan_limit = *storage_.epoch_access_count;
            return scan_limit > storage_.access_record_capacity ? storage_.access_record_capacity
                                                                : scan_limit;
        }

        __device__ int current_coalesced_access_scan_limit() const
        {
            if(!storage_.coalesced_access_records || !storage_.coalesced_access_count
               || storage_.coalesced_access_record_capacity <= 0)
            {
                return 0;
            }

            int scan_limit = *storage_.coalesced_access_count;
            return scan_limit > storage_.coalesced_access_record_capacity
                       ? storage_.coalesced_access_record_capacity
                       : scan_limit;
        }

        __device__ int current_epoch_coalescing_access_scan_limit() const
        {
            if(!storage_.coalescing_access_records || !storage_.epoch_coalescing_access_count
               || storage_.coalescing_access_record_capacity <= 0)
            {
                return 0;
            }

            int scan_limit = *storage_.epoch_coalescing_access_count;
            return scan_limit > storage_.coalescing_access_record_capacity
                       ? storage_.coalescing_access_record_capacity
                       : scan_limit;
        }

        __device__ access_record make_access_record(const void* ptr,
                                                    uint32_t    byte_count,
                                                    access_kind kind,
                                                    site_id     site) const
        {
            uint32_t subgroup = subgroup_id();
            return access_record{
                reinterpret_cast<uintptr_t>(ptr),
                byte_count,
                subgroup,
                detail::current_epoch(storage_, subgroup),
                static_cast<uint32_t>(kind),
                1,
                site.value(),
            };
        }

        __device__ void
            record_access(const void* ptr, uint32_t byte_count, access_kind kind, site_id site)
        {
            access_record record           = make_access_record(ptr, byte_count, kind, site);
            bool          wants_coalescing = site.allows_coalescing();
            if(wants_coalescing && record_coalescing_access(record))
            {
                return;
            }
            if(wants_coalescing)
            {
                record_coalescing_fallback();
            }

            record_exact_access(record);
        }

        __device__ void record_exact_access(const access_record& record) const
        {
            if(!storage_.access_records || !storage_.access_count || !storage_.epoch_access_count
               || storage_.access_record_capacity <= 0)
            {
                return;
            }

            (void)atomicAdd(storage_.access_count, 1);
            int record_index = atomicAdd(storage_.epoch_access_count, 1);

            if(record_index < storage_.access_record_capacity)
            {
                access_record stored_record                 = record;
                stored_record.valid                         = 0;
                storage_.access_records[record_index]       = stored_record;
                storage_.access_records[record_index].valid = 0;
                __threadfence();
                storage_.access_records[record_index].valid = 1;
                __threadfence();
            }

            if(record_index >= storage_.access_record_capacity)
            {
                emit_metadata_full(record);
            }
        }

        __device__ bool record_coalescing_access(const access_record& record) const
        {
            if(record.site_id == 0 || !storage_.coalescing_access_records
               || !storage_.coalescing_access_count || !storage_.epoch_coalescing_access_count
               || storage_.coalescing_access_record_capacity <= 0)
            {
                return false;
            }

            coalescing_access_record coalescing_record{
                record.address,
                record.byte_count,
                lane_in_subgroup(),
                record.subgroup_id,
                record.epoch,
                record.kind,
                1,
                record.site_id,
            };

            (void)atomicAdd(storage_.coalescing_access_count, 1);
            int coalescing_access_index = atomicAdd(storage_.epoch_coalescing_access_count, 1);
            if(coalescing_access_index >= storage_.coalescing_access_record_capacity)
            {
                return false;
            }

            coalescing_record.valid                                           = 0;
            storage_.coalescing_access_records[coalescing_access_index]       = coalescing_record;
            storage_.coalescing_access_records[coalescing_access_index].valid = 0;
            __threadfence();
            storage_.coalescing_access_records[coalescing_access_index].valid = 1;
            __threadfence();
            return true;
        }

        __device__ void record_coalescing_fallback() const
        {
            if(storage_.coalescing_fallback_count)
            {
                (void)atomicAdd(storage_.coalescing_fallback_count, 1);
            }
        }

        __device__ bool same_coalescing_key(const coalescing_access_record& first,
                                            const coalescing_access_record& second) const
        {
            return first.valid && second.valid && first.site_id != 0
                   && first.site_id == second.site_id && first.epoch == second.epoch
                   && first.subgroup_id == second.subgroup_id && first.kind == second.kind
                   && first.byte_count == second.byte_count;
        }

        __device__ bool same_coalescing_key(const coalescing_group_record&  group,
                                            const coalescing_access_record& record) const
        {
            return group.valid && record.valid && group.site_id != 0
                   && group.site_id == record.site_id && group.epoch == record.epoch
                   && group.subgroup_id == record.subgroup_id && group.kind == record.kind
                   && group.byte_count == record.byte_count;
        }

        __device__ uint64_t mix_coalescing_hash(uint64_t value) const
        {
            value ^= value >> 33;
            value *= 0xff51afd7ed558ccdull;
            value ^= value >> 33;
            value *= 0xc4ceb9fe1a85ec53ull;
            value ^= value >> 33;
            return value;
        }

        __device__ int coalescing_group_hash(const coalescing_access_record& record) const
        {
            uint64_t hash = record.site_id;
            hash
                ^= mix_coalescing_hash(static_cast<uint64_t>(record.epoch) + 0x9e3779b97f4a7c15ull);
            hash ^= mix_coalescing_hash(static_cast<uint64_t>(record.subgroup_id)
                                        + 0xbf58476d1ce4e5b9ull);
            hash ^= mix_coalescing_hash(static_cast<uint64_t>(record.kind) + 0x94d049bb133111ebull);
            hash ^= mix_coalescing_hash(static_cast<uint64_t>(record.byte_count)
                                        + 0x2545f4914f6cdd1dull);
            return static_cast<int>(
                hash % static_cast<uint64_t>(storage_.coalescing_group_record_capacity));
        }

        __device__ bool find_or_insert_coalescing_group(const coalescing_access_record& record,
                                                        int* group_index) const
        {
            int capacity = storage_.coalescing_group_record_capacity;
            int start    = coalescing_group_hash(record);

            for(int probe = 0; probe < capacity; ++probe)
            {
                int index = start + probe;
                if(index >= capacity)
                {
                    index -= capacity;
                }

                if(same_coalescing_key(storage_.coalescing_group_records[index], record))
                {
                    *group_index = index;
                    return true;
                }

                if(!storage_.coalescing_group_records[index].valid)
                {
                    int group_count = *storage_.coalescing_group_count;
                    if(group_count >= capacity)
                    {
                        return false;
                    }

                    initialize_coalescing_group(index, record);
                    ++(*storage_.coalescing_group_count);
                    *group_index = index;
                    return true;
                }
            }

            return false;
        }

        __device__ void initialize_coalescing_group(int                             group_index,
                                                    const coalescing_access_record& record) const
        {
            storage_.coalescing_group_records[group_index] = coalescing_group_record{
                record.address,
                record.address,
                record.site_id,
                0,
                0,
                record.byte_count,
                record.lane,
                record.lane,
                record.subgroup_id,
                record.epoch,
                record.kind,
                0,
                1,
            };
        }

        __device__ bool update_coalescing_group(coalescing_group_record*        group,
                                                const coalescing_access_record& record) const
        {
            if(record.lane >= 64)
            {
                return false;
            }

            uint64_t lane_bit = 1ull << record.lane;
            if(group->lane_mask & lane_bit)
            {
                group->repeated_lane_mask |= lane_bit;
            }
            else
            {
                group->lane_mask |= lane_bit;
            }

            ++group->participant_count;
            if(record.lane < group->first_lane)
            {
                group->first_lane         = record.lane;
                group->first_lane_address = record.address;
            }
            if(record.lane > group->last_lane)
            {
                group->last_lane         = record.lane;
                group->last_lane_address = record.address;
            }
            return true;
        }

        __device__ bool build_coalescing_groups(int scan_limit) const
        {
            if(!storage_.coalescing_group_records || !storage_.coalescing_group_count
               || storage_.coalescing_group_record_capacity <= 0)
            {
                return false;
            }

            *storage_.coalescing_group_count = 0;
            for(int i = 0; i < storage_.coalescing_group_record_capacity; ++i)
            {
                storage_.coalescing_group_records[i].valid = 0;
            }

            for(int i = 0; i < scan_limit; ++i)
            {
                coalescing_access_record record = storage_.coalescing_access_records[i];
                if(!record.valid || record.site_id == 0)
                {
                    continue;
                }

                int group_index = 0;
                if(!find_or_insert_coalescing_group(record, &group_index))
                {
                    return false;
                }

                if(!update_coalescing_group(&storage_.coalescing_group_records[group_index],
                                            record))
                {
                    return false;
                }
            }
            return true;
        }

        __device__ bool has_prior_coalescing_leader(const coalescing_access_record& record,
                                                    int                             record_index,
                                                    int scan_limit) const
        {
            for(int i = 0; i < record_index && i < scan_limit; ++i)
            {
                if(same_coalescing_key(storage_.coalescing_access_records[i], record))
                {
                    return true;
                }
            }
            return false;
        }

        __device__ bool build_coalesced_access_record(const coalescing_group_record& group,
                                                      int                            scan_limit,
                                                      coalesced_access_record*       result) const
        {
            if(!group.valid || group.participant_count < 2 || group.repeated_lane_mask != 0)
            {
                return false;
            }

            uint32_t lane_span = group.last_lane - group.first_lane + 1;
            if(lane_span != group.participant_count)
            {
                return false;
            }

            uint64_t expected_mask
                = lane_span == 64 ? UINT64_MAX : ((1ull << lane_span) - 1ull) << group.first_lane;
            if(group.lane_mask != expected_mask)
            {
                return false;
            }

            bool     descending_address = group.last_lane_address < group.first_lane_address;
            uint64_t absolute_delta
                = descending_address
                      ? static_cast<uint64_t>(group.first_lane_address - group.last_lane_address)
                      : static_cast<uint64_t>(group.last_lane_address - group.first_lane_address);
            int64_t lane_delta = static_cast<int64_t>(group.last_lane - group.first_lane);
            if(lane_delta == 0 || absolute_delta > static_cast<uint64_t>(INT64_MAX)
               || absolute_delta % static_cast<uint64_t>(lane_delta) != 0)
            {
                return false;
            }

            int64_t address_stride = static_cast<int64_t>(absolute_delta) / lane_delta;
            if(descending_address)
            {
                address_stride = -address_stride;
            }
            int64_t stride_magnitude = address_stride < 0 ? -address_stride : address_stride;
            if(stride_magnitude < static_cast<int64_t>(group.byte_count))
            {
                return false;
            }

            for(int i = 0; i < scan_limit; ++i)
            {
                coalescing_access_record candidate = storage_.coalescing_access_records[i];
                if(!same_coalescing_key(group, candidate))
                {
                    continue;
                }

                uint32_t lane_offset = candidate.lane - group.first_lane;
                if(lane_offset != 0
                   && static_cast<uint64_t>(stride_magnitude)
                          > UINT64_MAX / static_cast<uint64_t>(lane_offset))
                {
                    return false;
                }
                uint64_t step
                    = static_cast<uint64_t>(lane_offset) * static_cast<uint64_t>(stride_magnitude);
                uintptr_t expected_address = group.first_lane_address;
                if(address_stride >= 0)
                {
                    if(step > UINTPTR_MAX - expected_address)
                    {
                        return false;
                    }
                    expected_address += static_cast<uintptr_t>(step);
                }
                else
                {
                    if(step > expected_address)
                    {
                        return false;
                    }
                    expected_address -= static_cast<uintptr_t>(step);
                }

                if(candidate.address != expected_address)
                {
                    return false;
                }
            }

            uintptr_t lower_address = group.first_lane_address;
            uintptr_t upper_address = group.last_lane_address;
            if(address_stride < 0)
            {
                lower_address = group.last_lane_address;
                upper_address = group.first_lane_address;
            }

            uint64_t span = static_cast<uint64_t>(upper_address - lower_address)
                            + static_cast<uint64_t>(group.byte_count);
            if(span > UINT32_MAX)
            {
                return false;
            }

            *result = coalesced_access_record{
                group.first_lane_address,
                group.byte_count,
                static_cast<uint32_t>(span),
                group.first_lane,
                group.subgroup_id,
                group.epoch,
                group.kind,
                group.participant_count,
                1,
                address_stride,
                group.site_id,
            };
            return true;
        }

        __device__ bool build_coalesced_access_record(const coalescing_access_record& key,
                                                      int                             scan_limit,
                                                      coalesced_access_record*        result) const
        {
            uint32_t                 min_lane           = 0;
            uint32_t                 max_lane           = 0;
            int                      count              = 0;
            uint64_t                 lane_mask          = 0;
            uint64_t                 repeated_lane_mask = 0;
            bool                     found              = false;
            coalescing_access_record first_lane_record{};
            coalescing_access_record last_lane_record{};

            for(int i = 0; i < scan_limit; ++i)
            {
                coalescing_access_record candidate = storage_.coalescing_access_records[i];
                if(!same_coalescing_key(candidate, key))
                {
                    continue;
                }
                if(candidate.lane >= 64)
                {
                    return false;
                }

                uint64_t lane_bit = 1ull << candidate.lane;
                if(lane_mask & lane_bit)
                {
                    repeated_lane_mask |= lane_bit;
                }
                else
                {
                    lane_mask |= lane_bit;
                }

                ++count;
                if(!found || candidate.lane < min_lane)
                {
                    min_lane          = candidate.lane;
                    first_lane_record = candidate;
                }
                if(!found || candidate.lane > max_lane)
                {
                    max_lane         = candidate.lane;
                    last_lane_record = candidate;
                }
                found = true;
            }

            if(count < 2 || repeated_lane_mask != 0)
            {
                return false;
            }

            uint32_t lane_span = max_lane - min_lane + 1;
            if(lane_span != static_cast<uint32_t>(count))
            {
                return false;
            }

            uint64_t expected_mask
                = lane_span == 64 ? UINT64_MAX : ((1ull << lane_span) - 1ull) << min_lane;
            if(lane_mask != expected_mask)
            {
                return false;
            }

            bool     descending_address = last_lane_record.address < first_lane_record.address;
            uint64_t absolute_delta
                = descending_address
                      ? static_cast<uint64_t>(first_lane_record.address - last_lane_record.address)
                      : static_cast<uint64_t>(last_lane_record.address - first_lane_record.address);
            int64_t lane_delta = static_cast<int64_t>(max_lane - min_lane);
            if(lane_delta == 0 || absolute_delta > static_cast<uint64_t>(INT64_MAX)
               || absolute_delta % static_cast<uint64_t>(lane_delta) != 0)
            {
                return false;
            }

            int64_t address_stride = static_cast<int64_t>(absolute_delta) / lane_delta;
            if(descending_address)
            {
                address_stride = -address_stride;
            }
            int64_t stride_magnitude = address_stride < 0 ? -address_stride : address_stride;
            if(stride_magnitude < static_cast<int64_t>(key.byte_count))
            {
                return false;
            }

            for(int i = 0; i < scan_limit; ++i)
            {
                coalescing_access_record candidate = storage_.coalescing_access_records[i];
                if(!same_coalescing_key(candidate, key))
                {
                    continue;
                }

                uint32_t lane_offset = candidate.lane - min_lane;
                if(lane_offset != 0
                   && static_cast<uint64_t>(stride_magnitude)
                          > UINT64_MAX / static_cast<uint64_t>(lane_offset))
                {
                    return false;
                }
                uint64_t step
                    = static_cast<uint64_t>(lane_offset) * static_cast<uint64_t>(stride_magnitude);
                uintptr_t expected_address = first_lane_record.address;
                if(address_stride >= 0)
                {
                    if(step > UINTPTR_MAX - expected_address)
                    {
                        return false;
                    }
                    expected_address += static_cast<uintptr_t>(step);
                }
                else
                {
                    if(step > expected_address)
                    {
                        return false;
                    }
                    expected_address -= static_cast<uintptr_t>(step);
                }

                if(candidate.address != expected_address)
                {
                    return false;
                }
            }

            uintptr_t lower_address = first_lane_record.address;
            uintptr_t upper_address = last_lane_record.address;
            if(address_stride < 0)
            {
                lower_address = last_lane_record.address;
                upper_address = first_lane_record.address;
            }

            uint64_t span = static_cast<uint64_t>(upper_address - lower_address)
                            + static_cast<uint64_t>(key.byte_count);
            if(span > UINT32_MAX)
            {
                return false;
            }

            *result = coalesced_access_record{
                first_lane_record.address,
                key.byte_count,
                static_cast<uint32_t>(span),
                min_lane,
                key.subgroup_id,
                key.epoch,
                key.kind,
                static_cast<uint32_t>(count),
                1,
                address_stride,
                key.site_id,
            };
            return true;
        }

        __device__ void append_coalesced_access_record(const coalesced_access_record& record) const
        {
            int output_index = *storage_.coalesced_access_count;
            ++(*storage_.coalesced_access_count);
            if(output_index < storage_.coalesced_access_record_capacity)
            {
                storage_.coalesced_access_records[output_index] = record;
            }
        }

        __device__ bool collect_coalesced_access_records_from_groups(int scan_limit) const
        {
            if(!build_coalescing_groups(scan_limit))
            {
                return false;
            }

            int group_count = *storage_.coalescing_group_count;
            int seen_groups = 0;
            for(int i = 0;
                i < storage_.coalescing_group_record_capacity && seen_groups < group_count;
                ++i)
            {
                if(!storage_.coalescing_group_records[i].valid)
                {
                    continue;
                }
                ++seen_groups;

                coalesced_access_record coalesced_record{};
                if(build_coalesced_access_record(
                       storage_.coalescing_group_records[i], scan_limit, &coalesced_record))
                {
                    append_coalesced_access_record(coalesced_record);
                }
            }
            return true;
        }

        __device__ void collect_coalesced_access_records() const
        {
            if(!storage_.coalesced_access_records || !storage_.coalesced_access_count
               || storage_.coalesced_access_record_capacity <= 0
               || !storage_.coalescing_access_records || !storage_.epoch_coalescing_access_count)
            {
                return;
            }

            int scan_limit = *storage_.epoch_coalescing_access_count;
            if(scan_limit > storage_.coalescing_access_record_capacity)
            {
                scan_limit = storage_.coalescing_access_record_capacity;
            }

            if(collect_coalesced_access_records_from_groups(scan_limit))
            {
                return;
            }

            for(int i = 0; i < scan_limit; ++i)
            {
                coalescing_access_record key = storage_.coalescing_access_records[i];
                if(!key.valid || key.site_id == 0
                   || has_prior_coalescing_leader(key, i, scan_limit))
                {
                    continue;
                }

                coalesced_access_record coalesced_record{};
                if(!build_coalesced_access_record(key, scan_limit, &coalesced_record))
                {
                    continue;
                }

                append_coalesced_access_record(coalesced_record);
            }
        }

        __device__ bool coalesced_member_address(const coalesced_access_record& record,
                                                 uint32_t                       participant,
                                                 uintptr_t*                     address) const
        {
            if(participant >= record.participant_count)
            {
                return false;
            }

            uint64_t stride_magnitude = record.address_stride < 0
                                            ? static_cast<uint64_t>(-record.address_stride)
                                            : static_cast<uint64_t>(record.address_stride);
            uint64_t step             = static_cast<uint64_t>(participant) * stride_magnitude;
            if(record.address_stride >= 0)
            {
                if(step > UINTPTR_MAX - record.first_address)
                {
                    return false;
                }
                *address = record.first_address + static_cast<uintptr_t>(step);
                return true;
            }

            if(step > record.first_address)
            {
                return false;
            }
            *address = record.first_address - static_cast<uintptr_t>(step);
            return true;
        }

        __device__ bool
            coalesced_access_overlaps_exact(const coalesced_access_record& coalesced_record,
                                            uintptr_t                      exact_address,
                                            uint32_t                       exact_byte_count) const
        {
            if(!coalesced_record.valid)
            {
                return false;
            }

            for(uint32_t i = 0; i < coalesced_record.participant_count; ++i)
            {
                uintptr_t member_address = 0;
                if(!coalesced_member_address(coalesced_record, i, &member_address))
                {
                    return false;
                }
                if(detail::byte_ranges_overlap(member_address,
                                               coalesced_record.byte_count,
                                               exact_address,
                                               exact_byte_count))
                {
                    return true;
                }
            }
            return false;
        }

        __device__ bool coalesced_accesses_overlap(const coalesced_access_record& first,
                                                   const coalesced_access_record& second) const
        {
            if(!first.valid || !second.valid)
            {
                return false;
            }

            for(uint32_t i = 0; i < first.participant_count; ++i)
            {
                uintptr_t first_member_address = 0;
                if(!coalesced_member_address(first, i, &first_member_address))
                {
                    return false;
                }
                if(coalesced_access_overlaps_exact(second, first_member_address, first.byte_count))
                {
                    return true;
                }
            }
            return false;
        }

        __device__ bool coalesced_conflicts_with(const coalesced_access_record& first,
                                                 const coalesced_access_record& second) const
        {
            return first.valid && second.valid && first.epoch == second.epoch
                   && first.subgroup_id != second.subgroup_id
                   && (detail::is_write(first.kind) || detail::is_write(second.kind))
                   && coalesced_accesses_overlap(first, second);
        }

        __device__ bool coalesced_conflicts_with(const coalesced_access_record& summary,
                                                 const access_record&           exact) const
        {
            return summary.valid && exact.valid && summary.epoch == exact.epoch
                   && summary.subgroup_id != exact.subgroup_id
                   && (detail::is_write(summary.kind) || detail::is_write(exact.kind))
                   && coalesced_access_overlaps_exact(summary, exact.address, exact.byte_count);
        }

        __device__ bool coalesced_conflicts_with(const coalesced_access_record&  summary,
                                                 const coalescing_access_record& exact) const
        {
            return summary.valid && exact.valid && summary.epoch == exact.epoch
                   && summary.subgroup_id != exact.subgroup_id
                   && (detail::is_write(summary.kind) || detail::is_write(exact.kind))
                   && coalesced_access_overlaps_exact(summary, exact.address, exact.byte_count);
        }

        __device__ bool conflicts_with(const access_record&            first,
                                       const coalescing_access_record& second) const
        {
            return first.valid && second.valid && first.epoch == second.epoch
                   && first.subgroup_id != second.subgroup_id
                   && (detail::is_write(first.kind) || detail::is_write(second.kind))
                   && detail::byte_ranges_overlap(
                       first.address, first.byte_count, second.address, second.byte_count);
        }

        __device__ bool conflicts_with(const coalescing_access_record& first,
                                       const coalescing_access_record& second) const
        {
            return first.valid && second.valid && first.epoch == second.epoch
                   && first.subgroup_id != second.subgroup_id
                   && (detail::is_write(first.kind) || detail::is_write(second.kind))
                   && detail::byte_ranges_overlap(
                       first.address, first.byte_count, second.address, second.byte_count);
        }

        __device__ bool
            coalesced_summary_contains_exact_member(const coalesced_access_record& summary,
                                                    const access_record&           exact) const
        {
            if(!summary.valid || !exact.valid || summary.epoch != exact.epoch
               || summary.subgroup_id != exact.subgroup_id || summary.kind != exact.kind
               || summary.byte_count != exact.byte_count || summary.site_id != exact.site_id)
            {
                return false;
            }

            for(uint32_t i = 0; i < summary.participant_count; ++i)
            {
                uintptr_t member_address = 0;
                if(!coalesced_member_address(summary, i, &member_address))
                {
                    return false;
                }
                if(member_address == exact.address)
                {
                    return true;
                }
            }
            return false;
        }

        __device__ bool
            coalesced_summary_contains_exact_member(const coalesced_access_record&  summary,
                                                    const coalescing_access_record& exact) const
        {
            if(!summary.valid || !exact.valid || summary.epoch != exact.epoch
               || summary.subgroup_id != exact.subgroup_id || summary.kind != exact.kind
               || summary.byte_count != exact.byte_count || summary.site_id != exact.site_id)
            {
                return false;
            }

            for(uint32_t i = 0; i < summary.participant_count; ++i)
            {
                uintptr_t member_address = 0;
                if(!coalesced_member_address(summary, i, &member_address))
                {
                    return false;
                }
                if(member_address == exact.address)
                {
                    return true;
                }
            }
            return false;
        }

        __device__ bool record_has_coalesced_summary(const access_record& exact,
                                                     int                  first_summary_index,
                                                     int                  summary_scan_limit) const
        {
            for(int i = first_summary_index; i < summary_scan_limit; ++i)
            {
                if(coalesced_summary_contains_exact_member(storage_.coalesced_access_records[i],
                                                           exact))
                {
                    return true;
                }
            }
            return false;
        }

        __device__ bool record_has_coalesced_summary(const coalescing_access_record& exact,
                                                     int first_summary_index,
                                                     int summary_scan_limit) const
        {
            for(int i = first_summary_index; i < summary_scan_limit; ++i)
            {
                if(coalesced_summary_contains_exact_member(storage_.coalesced_access_records[i],
                                                           exact))
                {
                    return true;
                }
            }
            return false;
        }

        __device__ void emit_deferred_conflicts(int first_summary_index,
                                                int exact_scan_limit,
                                                int coalescing_scan_limit) const
        {
            int summary_scan_limit = current_coalesced_access_scan_limit();
            if(storage_.coalesced_access_records && first_summary_index >= 0)
            {
                for(int i = first_summary_index; i < summary_scan_limit; ++i)
                {
                    coalesced_access_record first = storage_.coalesced_access_records[i];
                    if(!first.valid)
                    {
                        continue;
                    }

                    for(int j = i + 1; j < summary_scan_limit; ++j)
                    {
                        coalesced_access_record second = storage_.coalesced_access_records[j];
                        if(coalesced_conflicts_with(first, second))
                        {
                            emit_conflict(first, second);
                        }
                    }

                    if(storage_.access_records)
                    {
                        for(int j = 0; j < exact_scan_limit; ++j)
                        {
                            access_record exact = storage_.access_records[j];
                            if(record_has_coalesced_summary(
                                   exact, first_summary_index, summary_scan_limit))
                            {
                                continue;
                            }
                            if(coalesced_conflicts_with(first, exact))
                            {
                                emit_conflict(first, exact);
                            }
                        }
                    }

                    if(storage_.coalescing_access_records)
                    {
                        for(int j = 0; j < coalescing_scan_limit; ++j)
                        {
                            coalescing_access_record exact = storage_.coalescing_access_records[j];
                            if(record_has_coalesced_summary(
                                   exact, first_summary_index, summary_scan_limit))
                            {
                                continue;
                            }
                            if(coalesced_conflicts_with(first, exact))
                            {
                                emit_conflict(first, exact);
                            }
                        }
                    }
                }
            }

            if(storage_.access_records)
            {
                for(int i = 0; i < exact_scan_limit; ++i)
                {
                    access_record first = storage_.access_records[i];
                    if(record_has_coalesced_summary(first, first_summary_index, summary_scan_limit))
                    {
                        continue;
                    }
                    for(int j = i + 1; j < exact_scan_limit; ++j)
                    {
                        access_record second = storage_.access_records[j];
                        if(record_has_coalesced_summary(
                               second, first_summary_index, summary_scan_limit))
                        {
                            continue;
                        }
                        if(conflicts_with(first, second))
                        {
                            emit_conflict(first, second);
                        }
                    }

                    if(!storage_.coalescing_access_records)
                    {
                        continue;
                    }

                    for(int j = 0; j < coalescing_scan_limit; ++j)
                    {
                        coalescing_access_record second = storage_.coalescing_access_records[j];
                        if(record_has_coalesced_summary(
                               second, first_summary_index, summary_scan_limit))
                        {
                            continue;
                        }
                        if(conflicts_with(first, second))
                        {
                            emit_conflict(first, second);
                        }
                    }
                }
            }

            if(!storage_.coalescing_access_records)
            {
                return;
            }

            for(int i = 0; i < coalescing_scan_limit; ++i)
            {
                coalescing_access_record first = storage_.coalescing_access_records[i];
                if(record_has_coalesced_summary(first, first_summary_index, summary_scan_limit))
                {
                    continue;
                }
                for(int j = i + 1; j < coalescing_scan_limit; ++j)
                {
                    coalescing_access_record second = storage_.coalescing_access_records[j];
                    if(record_has_coalesced_summary(
                           second, first_summary_index, summary_scan_limit))
                    {
                        continue;
                    }
                    if(conflicts_with(first, second))
                    {
                        emit_conflict(first, second);
                    }
                }
            }
        }

        __device__ void close_current_epoch(bool advance_epochs) const
        {
            if(thread_id() == 0 && storage_.subgroup_states && storage_.subgroup_capacity > 0)
            {
                int exact_scan_limit             = current_epoch_access_scan_limit();
                int coalescing_access_scan_limit = current_epoch_coalescing_access_scan_limit();
                int first_new_summary_index      = current_coalesced_access_scan_limit();
                collect_coalesced_access_records();
                emit_deferred_conflicts(
                    first_new_summary_index, exact_scan_limit, coalescing_access_scan_limit);
                if(storage_.epoch_access_count)
                {
                    *storage_.epoch_access_count = 0;
                }
                if(storage_.epoch_coalescing_access_count)
                {
                    *storage_.epoch_coalescing_access_count = 0;
                }
                if(advance_epochs)
                {
                    int subgroup_count = detail::stored_subgroup_count(storage_, cfg_);
                    for(int i = 0; i < subgroup_count; ++i)
                    {
                        ++storage_.subgroup_states[i].epoch;
                    }
                }
                __threadfence();
            }
        }

        __device__ void emit_conflict(const access_record& first, const access_record& second) const
        {
            detail::emit_diagnostic(storage_,
                                    diagnostic{
                                        static_cast<uint32_t>(diagnostic_kind::access_conflict),
                                        second.epoch,
                                        first.subgroup_id,
                                        second.subgroup_id,
                                        first.address,
                                        second.address,
                                        first.byte_count,
                                        second.byte_count,
                                        first.site_id,
                                        second.site_id,
                                    });
        }

        __device__ void emit_conflict(const coalesced_access_record& first,
                                      const coalesced_access_record& second) const
        {
            detail::emit_diagnostic(storage_,
                                    diagnostic{
                                        static_cast<uint32_t>(diagnostic_kind::access_conflict),
                                        first.epoch,
                                        first.subgroup_id,
                                        second.subgroup_id,
                                        first.first_address,
                                        second.first_address,
                                        first.span_byte_count,
                                        second.span_byte_count,
                                        first.site_id,
                                        second.site_id,
                                    });
        }

        __device__ void emit_conflict(const coalesced_access_record& first,
                                      const access_record&           second) const
        {
            detail::emit_diagnostic(storage_,
                                    diagnostic{
                                        static_cast<uint32_t>(diagnostic_kind::access_conflict),
                                        first.epoch,
                                        first.subgroup_id,
                                        second.subgroup_id,
                                        first.first_address,
                                        second.address,
                                        first.span_byte_count,
                                        second.byte_count,
                                        first.site_id,
                                        second.site_id,
                                    });
        }

        __device__ void emit_conflict(const coalesced_access_record&  first,
                                      const coalescing_access_record& second) const
        {
            detail::emit_diagnostic(storage_,
                                    diagnostic{
                                        static_cast<uint32_t>(diagnostic_kind::access_conflict),
                                        first.epoch,
                                        first.subgroup_id,
                                        second.subgroup_id,
                                        first.first_address,
                                        second.address,
                                        first.span_byte_count,
                                        second.byte_count,
                                        first.site_id,
                                        second.site_id,
                                    });
        }

        __device__ void emit_conflict(const access_record&            first,
                                      const coalescing_access_record& second) const
        {
            detail::emit_diagnostic(storage_,
                                    diagnostic{
                                        static_cast<uint32_t>(diagnostic_kind::access_conflict),
                                        first.epoch,
                                        first.subgroup_id,
                                        second.subgroup_id,
                                        first.address,
                                        second.address,
                                        first.byte_count,
                                        second.byte_count,
                                        first.site_id,
                                        second.site_id,
                                    });
        }

        __device__ void emit_conflict(const coalescing_access_record& first,
                                      const coalescing_access_record& second) const
        {
            detail::emit_diagnostic(storage_,
                                    diagnostic{
                                        static_cast<uint32_t>(diagnostic_kind::access_conflict),
                                        first.epoch,
                                        first.subgroup_id,
                                        second.subgroup_id,
                                        first.address,
                                        second.address,
                                        first.byte_count,
                                        second.byte_count,
                                        first.site_id,
                                        second.site_id,
                                    });
        }

        __device__ void emit_metadata_full(const access_record& record) const
        {
            detail::emit_diagnostic(storage_,
                                    diagnostic{
                                        static_cast<uint32_t>(diagnostic_kind::metadata_full),
                                        record.epoch,
                                        record.subgroup_id,
                                        record.subgroup_id,
                                        record.address,
                                        record.address,
                                        record.byte_count,
                                        record.byte_count,
                                        record.site_id,
                                        record.site_id,
                                    });
        }

        __device__ void
            emit_barrier_divergence(int expected_count, int observed_count, site_id site) const
        {
            detail::emit_diagnostic(storage_,
                                    diagnostic{
                                        static_cast<uint32_t>(diagnostic_kind::barrier_divergence),
                                        detail::current_epoch(storage_, /*subgroup=*/0),
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        site.value(),
                                        site.value(),
                                        static_cast<uint32_t>(expected_count),
                                        static_cast<uint32_t>(observed_count),
                                    });
        }

        __device__ void emit_subgroup_capacity_full() const
        {
            detail::emit_diagnostic(storage_,
                                    diagnostic{
                                        static_cast<uint32_t>(diagnostic_kind::metadata_full),
                                        0,
                                        static_cast<uint32_t>(storage_.subgroup_capacity),
                                        detail::configured_subgroup_count(cfg_),
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                    });
        }

        storage_ref storage_;
        config      cfg_;
    };
} // namespace hip_moi

#endif // HIP_MOI_CONTEXT_HPP
