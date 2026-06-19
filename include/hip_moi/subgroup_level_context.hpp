// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// hip_moi/subgroup_level_context.hpp
//
// Subgroup-level HIP memory-ordering instrumentation context.
#ifndef HIP_MOI_SUBGROUP_LEVEL_CONTEXT_HPP
#define HIP_MOI_SUBGROUP_LEVEL_CONTEXT_HPP

#include "hip_moi/common.hpp"

#include <type_traits>

namespace hip_moi
{
    class subgroup_level_context
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

        struct coalescing_proof_record
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
        };

        struct storage_ref
        {
            using access_record_type = access_record;
            using diagnostic_type    = diagnostic;

            access_record*           access_records;
            int                      access_record_capacity;
            diagnostic*              diagnostics;
            int                      diagnostic_capacity;
            subgroup_state*          subgroup_states;
            int                      subgroup_capacity;
            int*                     access_count;
            int*                     epoch_access_count;
            int*                     diagnostic_count;
            coalesced_access_record* coalesced_access_records         = nullptr;
            int                      coalesced_access_record_capacity = 0;
            int*                     coalesced_access_count           = nullptr;
            coalescing_proof_record* coalescing_proof_records         = nullptr;
            int                      coalescing_proof_record_capacity = 0;
            int*                     coalescing_proof_count           = nullptr;
            int*                     epoch_coalescing_proof_count     = nullptr;
        };

        template <int AccessCapacity,
                  int DiagnosticCapacity,
                  int SubgroupCapacity        = 1,
                  int CoalescedAccessCapacity = AccessCapacity,
                  int CoalescingProofCapacity = AccessCapacity>
        struct static_context_storage
        {
            access_record           access_records[AccessCapacity];
            diagnostic              diagnostics[DiagnosticCapacity];
            subgroup_state          subgroup_states[SubgroupCapacity];
            coalesced_access_record coalesced_access_records[CoalescedAccessCapacity];
            coalescing_proof_record coalescing_proof_records[CoalescingProofCapacity];
            int                     access_count;
            int                     epoch_access_count;
            int                     diagnostic_count;
            int                     coalesced_access_count;
            int                     coalescing_proof_count;
            int                     epoch_coalescing_proof_count;

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
                    coalescing_proof_records,
                    CoalescingProofCapacity,
                    &coalescing_proof_count,
                    &epoch_coalescing_proof_count,
                };
            }
        };

        __device__ subgroup_level_context(storage_ref storage, config cfg)
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
                if(storage_.coalescing_proof_count)
                {
                    *storage_.coalescing_proof_count = 0;
                }
                if(storage_.epoch_coalescing_proof_count)
                {
                    *storage_.epoch_coalescing_proof_count = 0;
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
                for(int i = 0; storage_.coalescing_proof_records
                               && i < storage_.coalescing_proof_record_capacity;
                    ++i)
                {
                    storage_.coalescing_proof_records[i].valid = 0;
                }
                __threadfence();
            }
            __syncthreads();
        }

        template <typename T>
        __device__ T lds_load(const T* ptr, site_id site = no_site_id)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::subgroup_level_context::lds_load requires a trivially "
                          "copyable type");
            record_access(ptr, sizeof(T), access_kind::load, site);
            return *ptr;
        }

        template <typename T>
        __device__ void lds_store(T* ptr, T value, site_id site = no_site_id)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::subgroup_level_context::lds_store requires a trivially "
                          "copyable type");
            record_access(ptr, sizeof(T), access_kind::store, site);
            *ptr = value;
        }

        __device__ void syncthreads()
        {
            __syncthreads();
            if(thread_id() == 0 && storage_.subgroup_states && storage_.subgroup_capacity > 0)
            {
                collect_coalesced_access_records();
                if(storage_.epoch_access_count)
                {
                    *storage_.epoch_access_count = 0;
                }
                if(storage_.epoch_coalescing_proof_count)
                {
                    *storage_.epoch_coalescing_proof_count = 0;
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

        __device__ void
            record_access(const void* ptr, uint32_t byte_count, access_kind kind, site_id site)
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
                subgroup,
                detail::current_epoch(storage_, subgroup),
                static_cast<uint32_t>(kind),
                1,
                site.value(),
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
                record_coalescing_proof(record);
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
                    emit_conflict(prior, record);
                }
            }

            if(record_index >= storage_.access_record_capacity)
            {
                emit_metadata_full(record);
            }
        }

        __device__ void record_coalescing_proof(const access_record& record) const
        {
            if(record.site_id == 0 || !storage_.coalescing_proof_records
               || !storage_.coalescing_proof_count || !storage_.epoch_coalescing_proof_count
               || storage_.coalescing_proof_record_capacity <= 0)
            {
                return;
            }

            coalescing_proof_record proof_record{
                record.address,
                record.byte_count,
                lane_in_subgroup(),
                record.subgroup_id,
                record.epoch,
                record.kind,
                1,
                record.site_id,
            };

            (void)atomicAdd(storage_.coalescing_proof_count, 1);
            int proof_index = atomicAdd(storage_.epoch_coalescing_proof_count, 1);
            if(proof_index < storage_.coalescing_proof_record_capacity)
            {
                proof_record.valid                                   = 0;
                storage_.coalescing_proof_records[proof_index]       = proof_record;
                storage_.coalescing_proof_records[proof_index].valid = 0;
                __threadfence();
                storage_.coalescing_proof_records[proof_index].valid = 1;
                __threadfence();
            }
        }

        __device__ bool same_coalescing_key(const coalescing_proof_record& first,
                                            const coalescing_proof_record& second) const
        {
            return first.valid && second.valid && first.site_id != 0
                   && first.site_id == second.site_id && first.epoch == second.epoch
                   && first.subgroup_id == second.subgroup_id && first.kind == second.kind
                   && first.byte_count == second.byte_count;
        }

        __device__ bool has_prior_coalescing_leader(const coalescing_proof_record& record,
                                                    int                            record_index,
                                                    int                            scan_limit) const
        {
            for(int i = 0; i < record_index && i < scan_limit; ++i)
            {
                if(same_coalescing_key(storage_.coalescing_proof_records[i], record))
                {
                    return true;
                }
            }
            return false;
        }

        __device__ bool find_record_for_lane(const coalescing_proof_record& key,
                                             int                            scan_limit,
                                             uint32_t                       lane,
                                             coalescing_proof_record*       record) const
        {
            bool found = false;
            for(int i = 0; i < scan_limit; ++i)
            {
                coalescing_proof_record candidate = storage_.coalescing_proof_records[i];
                if(same_coalescing_key(candidate, key) && candidate.lane == lane)
                {
                    if(found)
                    {
                        return false;
                    }
                    *record = candidate;
                    found   = true;
                }
            }
            return found;
        }

        __device__ bool build_coalesced_access_record(const coalescing_proof_record& key,
                                                      int                            scan_limit,
                                                      coalesced_access_record*       result) const
        {
            uint32_t min_lane           = key.lane;
            uint32_t max_lane           = key.lane;
            int      count              = 0;
            uint64_t lane_mask          = 0;
            uint64_t repeated_lane_mask = 0;

            for(int i = 0; i < scan_limit; ++i)
            {
                coalescing_proof_record candidate = storage_.coalescing_proof_records[i];
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
                if(candidate.lane < min_lane)
                {
                    min_lane = candidate.lane;
                }
                if(candidate.lane > max_lane)
                {
                    max_lane = candidate.lane;
                }
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

            coalescing_proof_record first_lane_record{};
            coalescing_proof_record last_lane_record{};
            if(!find_record_for_lane(key, scan_limit, min_lane, &first_lane_record)
               || !find_record_for_lane(key, scan_limit, max_lane, &last_lane_record))
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

            for(uint32_t offset = 0; offset < lane_span; ++offset)
            {
                uint32_t                lane = min_lane + offset;
                coalescing_proof_record candidate{};
                if(!find_record_for_lane(key, scan_limit, lane, &candidate))
                {
                    return false;
                }

                uint64_t step
                    = static_cast<uint64_t>(offset) * static_cast<uint64_t>(stride_magnitude);
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

        __device__ void collect_coalesced_access_records() const
        {
            if(!storage_.coalesced_access_records || !storage_.coalesced_access_count
               || storage_.coalesced_access_record_capacity <= 0
               || !storage_.coalescing_proof_records || !storage_.epoch_coalescing_proof_count)
            {
                return;
            }

            int scan_limit = *storage_.epoch_coalescing_proof_count;
            if(scan_limit > storage_.coalescing_proof_record_capacity)
            {
                scan_limit = storage_.coalescing_proof_record_capacity;
            }

            for(int i = 0; i < scan_limit; ++i)
            {
                coalescing_proof_record key = storage_.coalescing_proof_records[i];
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

                int output_index = *storage_.coalesced_access_count;
                ++(*storage_.coalesced_access_count);
                if(output_index < storage_.coalesced_access_record_capacity)
                {
                    storage_.coalesced_access_records[output_index] = coalesced_record;
                }
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

        storage_ref storage_;
        config      cfg_;
    };
} // namespace hip_moi

#endif // HIP_MOI_SUBGROUP_LEVEL_CONTEXT_HPP
