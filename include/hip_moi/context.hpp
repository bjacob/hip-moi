// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// hip_moi/context.hpp
//
// Subgroup-scoped HIP memory-ordering instrumentation context.
#ifndef HIP_MOI_CONTEXT_HPP
#define HIP_MOI_CONTEXT_HPP

#include "hip_moi/common.hpp"
#include "hip_moi/shadow.hpp"

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
            using diagnostic_type = diagnostic;

            diagnostic*     diagnostics                     = nullptr;
            int             diagnostic_capacity             = 0;
            subgroup_state* subgroup_states                 = nullptr;
            int             subgroup_capacity               = 0;
            int*            diagnostic_count                = nullptr;
            int*            simulated_barrier_arrival_count = nullptr;
            uint64_t*       exact_shadow_entries            = nullptr;
            int             exact_shadow_entry_capacity     = 0;
            uint64_t*       sampled_watchpoints             = nullptr;
            int             sampled_watchpoint_capacity     = 0;
            uint64_t        generation                      = 0;
            uint32_t        backend                         = 0;
            uint32_t        sampled_watchpoint_sample_skip      = 1;
            uint32_t        sampled_watchpoint_probe_count      = 1;
            uint32_t        sampled_watchpoint_delay_iters      = 0;
            uint32_t        sampled_watchpoint_report_conflicts = 1;
        };

        template <int DiagnosticCapacity,
                  int SubgroupCapacity,
                  int ExactShadowEntryCapacity  = 0,
                  int SampledWatchpointCapacity = 0>
        struct static_context_storage
        {
            diagnostic     diagnostics[DiagnosticCapacity];
            subgroup_state subgroup_states[SubgroupCapacity];
            int            diagnostic_count;
            int            simulated_barrier_arrival_count;
            uint64_t
                exact_shadow_entries[ExactShadowEntryCapacity > 0 ? ExactShadowEntryCapacity : 1];
            uint64_t
                sampled_watchpoints[SampledWatchpointCapacity > 0 ? SampledWatchpointCapacity : 1];

            __device__ storage_ref ref(backend_kind backend = backend_kind::exact_shadow)
            {
                return storage_ref{
                    diagnostics,
                    DiagnosticCapacity,
                    subgroup_states,
                    SubgroupCapacity,
                    &diagnostic_count,
                    &simulated_barrier_arrival_count,
                    ExactShadowEntryCapacity > 0 ? exact_shadow_entries : nullptr,
                    ExactShadowEntryCapacity,
                    SampledWatchpointCapacity > 0 ? sampled_watchpoints : nullptr,
                    SampledWatchpointCapacity,
                    1,
                    static_cast<uint32_t>(backend),
                    1,
                    1,
                    0,
                    1,
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
                if(storage_.diagnostic_count)
                {
                    *storage_.diagnostic_count = 0;
                }
                if(storage_.simulated_barrier_arrival_count)
                {
                    *storage_.simulated_barrier_arrival_count = 0;
                }
                for(int i = 0; storage_.subgroup_states && i < storage_.subgroup_capacity; ++i)
                {
                    storage_.subgroup_states[i].epoch = 0;
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
        __device__ T lds_load_at(const T* ptr, uint32_t lds_byte_offset, site_id site = no_site_id)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::context::lds_load_at requires a trivially "
                          "copyable type");
            record_access_at(ptr, sizeof(T), access_kind::load, lds_byte_offset, site);
            return *ptr;
        }

        template <backend_kind Backend, typename T>
        __device__ T lds_load_at(const T* ptr, uint32_t lds_byte_offset, site_id site = no_site_id)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::context::lds_load_at requires a trivially "
                          "copyable type");
            record_access_at<Backend>(ptr, sizeof(T), access_kind::load, lds_byte_offset, site);
            return *ptr;
        }

        template <backend_kind Backend, typename SampledPolicy, typename T>
        __device__ T lds_load_at(const T* ptr, uint32_t lds_byte_offset, site_id site = no_site_id)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::context::lds_load_at requires a trivially "
                          "copyable type");
            record_access_at<Backend, SampledPolicy>(
                ptr, sizeof(T), access_kind::load, lds_byte_offset, site);
            return *ptr;
        }

        template <typename T>
        __device__ void
            lds_store_at(T* ptr, T value, uint32_t lds_byte_offset, site_id site = no_site_id)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::context::lds_store_at requires a trivially "
                          "copyable type");
            record_access_at(ptr, sizeof(T), access_kind::store, lds_byte_offset, site);
            *ptr = value;
        }

        template <backend_kind Backend, typename T>
        __device__ void
            lds_store_at(T* ptr, T value, uint32_t lds_byte_offset, site_id site = no_site_id)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::context::lds_store_at requires a trivially "
                          "copyable type");
            record_access_at<Backend>(ptr, sizeof(T), access_kind::store, lds_byte_offset, site);
            *ptr = value;
        }

        template <backend_kind Backend, typename SampledPolicy, typename T>
        __device__ void
            lds_store_at(T* ptr, T value, uint32_t lds_byte_offset, site_id site = no_site_id)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::context::lds_store_at requires a trivially "
                          "copyable type");
            record_access_at<Backend, SampledPolicy>(
                ptr, sizeof(T), access_kind::store, lds_byte_offset, site);
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

        // Online backends report conflicts at access time. finish() remains as a
        // compatibility alias for explicit epoch advancement in tests and
        // experiments; ordinary kernels do not need a final flush.
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

    private:
        __device__ backend_kind selected_backend() const
        {
            return static_cast<backend_kind>(storage_.backend);
        }

        __device__ void record_access_at(const void* ptr,
                                         uint32_t    byte_count,
                                         access_kind kind,
                                         uint32_t    lds_byte_offset,
                                         site_id     site) const
        {
            if(byte_count == 0)
            {
                return;
            }

            if(selected_backend() == backend_kind::sampled_watchpoint)
            {
                record_sampled_watchpoint_access(ptr, byte_count, kind, lds_byte_offset, site);
                return;
            }

            record_exact_shadow_access(ptr, byte_count, kind, lds_byte_offset, site);
        }

        template <backend_kind Backend>
        __device__ void record_access_at(const void* ptr,
                                         uint32_t    byte_count,
                                         access_kind kind,
                                         uint32_t    lds_byte_offset,
                                         site_id     site) const
        {
            if(byte_count == 0)
            {
                return;
            }

            if constexpr(Backend == backend_kind::sampled_watchpoint)
            {
                record_sampled_watchpoint_access(ptr, byte_count, kind, lds_byte_offset, site);
            }
            else
            {
                record_exact_shadow_access(ptr, byte_count, kind, lds_byte_offset, site);
            }
        }

        template <backend_kind Backend, typename SampledPolicy>
        __device__ void record_access_at(const void* ptr,
                                         uint32_t    byte_count,
                                         access_kind kind,
                                         uint32_t    lds_byte_offset,
                                         site_id     site) const
        {
            if(byte_count == 0)
            {
                return;
            }

            if constexpr(Backend == backend_kind::sampled_watchpoint)
            {
                record_sampled_watchpoint_access<SampledPolicy>(
                    ptr, byte_count, kind, lds_byte_offset, site);
            }
            else
            {
                record_exact_shadow_access(ptr, byte_count, kind, lds_byte_offset, site);
            }
        }

        __device__ void record_exact_shadow_access(const void* ptr,
                                                   uint32_t    byte_count,
                                                   access_kind kind,
                                                   uint32_t    lds_byte_offset,
                                                   site_id     site) const
        {
            if(!storage_.exact_shadow_entries || storage_.exact_shadow_entry_capacity <= 0)
            {
                emit_shadow_metadata_full(ptr, byte_count, lds_byte_offset, site);
                return;
            }

            uint64_t last_byte_offset
                = static_cast<uint64_t>(lds_byte_offset) + static_cast<uint64_t>(byte_count) - 1u;
            uint32_t first_cell = lds_byte_offset >> detail::exact_shadow::granule_shift;
            uint64_t last_cell  = last_byte_offset >> detail::exact_shadow::granule_shift;
            if(last_cell >= static_cast<uint64_t>(storage_.exact_shadow_entry_capacity))
            {
                emit_shadow_metadata_full(ptr, byte_count, lds_byte_offset, site);
                return;
            }

            uint32_t subgroup = subgroup_id();
            uint32_t epoch    = detail::current_epoch(storage_, subgroup);
            uint64_t packed
                = detail::pack_exact_shadow_entry(detail::shadow_kind_from_access_kind(kind),
                                                  subgroup,
                                                  epoch,
                                                  static_cast<uint32_t>(storage_.generation),
                                                  site.value());

            detail::exact_shadow_entry current = detail::decode_exact_shadow_entry(packed);
            for(uint32_t cell = first_cell; cell <= static_cast<uint32_t>(last_cell); ++cell)
            {
                uint64_t prior_value
                    = atomic_exchange_shadow_entry(&storage_.exact_shadow_entries[cell], packed);
                detail::exact_shadow_entry prior = detail::decode_exact_shadow_entry(prior_value);
                if(detail::exact_shadow_entries_conflict(current, prior))
                {
                    emit_shadow_conflict(prior, current, byte_count, lds_byte_offset, cell);
                }
            }
        }

        __device__ void record_sampled_watchpoint_access(const void* ptr,
                                                         uint32_t    byte_count,
                                                         access_kind kind,
                                                         uint32_t    lds_byte_offset,
                                                         site_id     site) const
        {
            if(!storage_.sampled_watchpoints || storage_.sampled_watchpoint_capacity <= 0)
            {
                emit_shadow_metadata_full(ptr, byte_count, lds_byte_offset, site);
                return;
            }

            uint32_t selection_seed = sampled_selection_seed(site);
            if(!sampled_should_publish(selection_seed))
            {
                return;
            }

            uint64_t last_byte_offset
                = static_cast<uint64_t>(lds_byte_offset) + static_cast<uint64_t>(byte_count) - 1u;
            uint32_t start_cell = lds_byte_offset >> detail::sampled_watchpoint::granule_shift;
            uint64_t end_cell
                = (last_byte_offset >> detail::sampled_watchpoint::granule_shift) + 1u;
            if(start_cell > detail::sampled_watchpoint::max_start
               || end_cell > static_cast<uint64_t>(detail::sampled_watchpoint::max_start) + 1u)
            {
                emit_shadow_metadata_full(ptr, byte_count, lds_byte_offset, site);
                return;
            }

            while(static_cast<uint64_t>(start_cell) < end_cell)
            {
                uint64_t remaining = end_cell - static_cast<uint64_t>(start_cell);
                uint32_t chunk     = remaining > detail::sampled_watchpoint::max_count
                                         ? detail::sampled_watchpoint::max_count
                                         : static_cast<uint32_t>(remaining);
                record_sampled_watchpoint_range(
                    kind, start_cell, chunk, lds_byte_offset, byte_count, site, selection_seed);
                start_cell += chunk;
            }
        }

        template <typename SampledPolicy>
        __device__ void record_sampled_watchpoint_access(const void* ptr,
                                                         uint32_t    byte_count,
                                                         access_kind kind,
                                                         uint32_t    lds_byte_offset,
                                                         site_id     site) const
        {
            if(!storage_.sampled_watchpoints || storage_.sampled_watchpoint_capacity <= 0)
            {
                emit_shadow_metadata_full(ptr, byte_count, lds_byte_offset, site);
                return;
            }

            uint32_t selection_seed = sampled_selection_seed(site);
            if(!sampled_should_publish<SampledPolicy>(selection_seed))
            {
                return;
            }

            uint64_t last_byte_offset
                = static_cast<uint64_t>(lds_byte_offset) + static_cast<uint64_t>(byte_count) - 1u;
            uint32_t start_cell = lds_byte_offset >> detail::sampled_watchpoint::granule_shift;
            uint64_t end_cell
                = (last_byte_offset >> detail::sampled_watchpoint::granule_shift) + 1u;
            if(start_cell > detail::sampled_watchpoint::max_start
               || end_cell > static_cast<uint64_t>(detail::sampled_watchpoint::max_start) + 1u)
            {
                emit_shadow_metadata_full(ptr, byte_count, lds_byte_offset, site);
                return;
            }

            while(static_cast<uint64_t>(start_cell) < end_cell)
            {
                uint64_t remaining = end_cell - static_cast<uint64_t>(start_cell);
                uint32_t chunk     = remaining > detail::sampled_watchpoint::max_count
                                         ? detail::sampled_watchpoint::max_count
                                         : static_cast<uint32_t>(remaining);
                record_sampled_watchpoint_range<SampledPolicy>(
                    kind, start_cell, chunk, lds_byte_offset, byte_count, site, selection_seed);
                start_cell += chunk;
            }
        }

        __device__ uint32_t flat_workgroup_id() const
        {
            return static_cast<uint32_t>(
                blockIdx.x
                + gridDim.x * (blockIdx.y + gridDim.y * static_cast<uint32_t>(blockIdx.z)));
        }

        __device__ uint32_t sampled_site_seed(site_id site) const
        {
            return static_cast<uint32_t>(site.value() ^ (site.value() >> 32));
        }

        __device__ uint32_t sampled_selection_seed(site_id site) const
        {
            uint32_t seed = static_cast<uint32_t>(storage_.generation) * 0x9e3779b9u;
            seed ^= (flat_workgroup_id() + 1u) * 0x85ebca6bu;
            seed ^= (subgroup_id() + 1u) * 0xc2b2ae35u;
            seed ^= sampled_site_seed(site) * 0x27d4eb2du;
            return detail::mix32(seed);
        }

        __device__ bool sampled_should_publish(uint32_t selection_seed) const
        {
            uint32_t subgroup_threads = cfg_.threads_per_subgroup > 0
                                            ? static_cast<uint32_t>(cfg_.threads_per_subgroup)
                                            : 1u;
            uint32_t skip             = storage_.sampled_watchpoint_sample_skip == 0
                                            ? 1u
                                            : storage_.sampled_watchpoint_sample_skip;
            if(skip > 1u)
            {
                bool     power_of_two = (skip & (skip - 1u)) == 0;
                uint32_t remainder
                    = power_of_two ? (selection_seed & (skip - 1u)) : (selection_seed % skip);
                if(remainder != 0)
                {
                    return false;
                }
            }

            uint32_t selected_lane = map_sampled_index(selection_seed >> 16, subgroup_threads);
            return lane_in_subgroup() == selected_lane;
        }

        template <typename SampledPolicy>
        __device__ bool sampled_should_publish(uint32_t selection_seed) const
        {
            uint32_t subgroup_threads = cfg_.threads_per_subgroup > 0
                                            ? static_cast<uint32_t>(cfg_.threads_per_subgroup)
                                            : 1u;
            if constexpr(SampledPolicy::sample_skip > 1)
            {
                if constexpr((SampledPolicy::sample_skip & (SampledPolicy::sample_skip - 1u)) == 0)
                {
                    if((selection_seed & (SampledPolicy::sample_skip - 1u)) != 0)
                    {
                        return false;
                    }
                }
                else if(selection_seed % SampledPolicy::sample_skip != 0)
                {
                    return false;
                }
            }

            uint32_t selected_lane = map_sampled_index(selection_seed >> 16, subgroup_threads);
            return lane_in_subgroup() == selected_lane;
        }

        __device__ void record_sampled_watchpoint_range(access_kind kind,
                                                        uint32_t    start_cell,
                                                        uint32_t    cell_count,
                                                        uint32_t    lds_byte_offset,
                                                        uint32_t    byte_count,
                                                        site_id     site,
                                                        uint32_t    selection_seed) const
        {
            uint32_t subgroup = subgroup_id();
            uint32_t epoch    = detail::current_epoch(storage_, subgroup);
            uint64_t packed
                = detail::pack_sampled_watchpoint_entry(detail::shadow_kind_from_access_kind(kind),
                                                        subgroup,
                                                        epoch,
                                                        static_cast<uint32_t>(storage_.generation),
                                                        start_cell,
                                                        cell_count);
            uint32_t                         slot = sampled_watchpoint_slot(start_cell, epoch);
            detail::sampled_watchpoint_entry current
                = detail::decode_sampled_watchpoint_entry(packed);
            uint64_t prior_value
                = atomic_exchange_shadow_entry(&storage_.sampled_watchpoints[slot], packed);
            if(storage_.sampled_watchpoint_report_conflicts)
            {
                detail::sampled_watchpoint_entry prior
                    = detail::decode_sampled_watchpoint_entry(prior_value);
                if(detail::sampled_watchpoints_conflict(current, prior))
                {
                    emit_sampled_watchpoint_conflict(
                        prior, current, lds_byte_offset, byte_count, site);
                }
                scan_sampled_watchpoint_conflicts(
                    current, slot, start_cell, lds_byte_offset, byte_count, site, selection_seed);
            }
            sampled_delay();
        }

        template <typename SampledPolicy>
        __device__ void record_sampled_watchpoint_range(access_kind kind,
                                                        uint32_t    start_cell,
                                                        uint32_t    cell_count,
                                                        uint32_t    lds_byte_offset,
                                                        uint32_t    byte_count,
                                                        site_id     site,
                                                        uint32_t    selection_seed) const
        {
            uint32_t subgroup = subgroup_id();
            uint32_t epoch    = detail::current_epoch(storage_, subgroup);
            uint64_t packed
                = detail::pack_sampled_watchpoint_entry(detail::shadow_kind_from_access_kind(kind),
                                                        subgroup,
                                                        epoch,
                                                        static_cast<uint32_t>(storage_.generation),
                                                        start_cell,
                                                        cell_count);
            uint32_t slot = sampled_watchpoint_slot(start_cell, epoch);
            if constexpr(SampledPolicy::report_conflicts)
            {
                detail::sampled_watchpoint_entry current
                    = detail::decode_sampled_watchpoint_entry(packed);
                uint64_t prior_value
                    = atomic_exchange_shadow_entry(&storage_.sampled_watchpoints[slot], packed);
                detail::sampled_watchpoint_entry prior
                    = detail::decode_sampled_watchpoint_entry(prior_value);
                if(detail::sampled_watchpoints_conflict(current, prior))
                {
                    emit_sampled_watchpoint_conflict(
                        prior, current, lds_byte_offset, byte_count, site);
                }
                scan_sampled_watchpoint_conflicts<SampledPolicy>(
                    current, slot, start_cell, lds_byte_offset, byte_count, site, selection_seed);
            }
            else
            {
                (void)atomic_exchange_shadow_entry(&storage_.sampled_watchpoints[slot], packed);
            }
            sampled_delay<SampledPolicy>();
        }

        __device__ uint32_t map_sampled_index(uint32_t value, uint32_t count) const
        {
            if(count == 0)
            {
                return 0;
            }
            bool power_of_two = (count & (count - 1u)) == 0;
            return power_of_two ? (value & (count - 1u)) : (value % count);
        }

        __device__ uint32_t sampled_watchpoint_slot(uint32_t start_cell, uint32_t epoch) const
        {
            uint32_t seed = epoch * 0x85ebca6bu;
            seed ^= static_cast<uint32_t>(storage_.generation) * 0xc2b2ae35u;
            seed ^= start_cell * 0x165667b1u;
            return map_sampled_index(detail::mix32(seed),
                                     static_cast<uint32_t>(storage_.sampled_watchpoint_capacity));
        }

        __device__ uint32_t sampled_probe_step(uint32_t selection_seed, uint32_t start_cell) const
        {
            uint32_t capacity = static_cast<uint32_t>(storage_.sampled_watchpoint_capacity);
            if(capacity <= 1)
            {
                return 0;
            }
            uint32_t seed = selection_seed ^ (start_cell * 0x94d049bbu);
            uint32_t step = detail::mix32(seed) | 1u;
            return map_sampled_index(step, capacity);
        }

        __device__ void
            scan_sampled_watchpoint_conflicts(const detail::sampled_watchpoint_entry& current,
                                              uint32_t                                slot,
                                              uint32_t                                start_cell,
                                              uint32_t lds_byte_offset,
                                              uint32_t byte_count,
                                              site_id  site,
                                              uint32_t selection_seed) const
        {
            uint32_t capacity = static_cast<uint32_t>(storage_.sampled_watchpoint_capacity);
            uint32_t probes   = storage_.sampled_watchpoint_probe_count == 0
                                    ? capacity
                                    : storage_.sampled_watchpoint_probe_count;
            if(probes > capacity)
            {
                probes = capacity;
            }
            uint32_t step = sampled_probe_step(selection_seed, start_cell);
            for(uint32_t probe = 0; probe < probes; ++probe)
            {
                uint32_t index = map_sampled_index(slot + probe * step, capacity);
                uint64_t prior_value
                    = *reinterpret_cast<volatile uint64_t*>(&storage_.sampled_watchpoints[index]);
                detail::sampled_watchpoint_entry prior
                    = detail::decode_sampled_watchpoint_entry(prior_value);
                if(detail::sampled_watchpoints_conflict(current, prior))
                {
                    emit_sampled_watchpoint_conflict(
                        prior, current, lds_byte_offset, byte_count, site);
                }
            }
        }

        template <typename SampledPolicy>
        __device__ void
            scan_sampled_watchpoint_conflicts(const detail::sampled_watchpoint_entry& current,
                                              uint32_t                                slot,
                                              uint32_t                                start_cell,
                                              uint32_t lds_byte_offset,
                                              uint32_t byte_count,
                                              site_id  site,
                                              uint32_t selection_seed) const
        {
            uint32_t capacity = static_cast<uint32_t>(storage_.sampled_watchpoint_capacity);
            uint32_t probes
                = SampledPolicy::probe_count == 0 ? capacity : SampledPolicy::probe_count;
            if(probes > capacity)
            {
                probes = capacity;
            }
            uint32_t step = sampled_probe_step(selection_seed, start_cell);
            for(uint32_t probe = 0; probe < probes; ++probe)
            {
                uint32_t index = map_sampled_index(slot + probe * step, capacity);
                uint64_t prior_value
                    = *reinterpret_cast<volatile uint64_t*>(&storage_.sampled_watchpoints[index]);
                detail::sampled_watchpoint_entry prior
                    = detail::decode_sampled_watchpoint_entry(prior_value);
                if(detail::sampled_watchpoints_conflict(current, prior))
                {
                    emit_sampled_watchpoint_conflict(
                        prior, current, lds_byte_offset, byte_count, site);
                }
            }
        }

        __device__ void sampled_delay() const
        {
#pragma unroll 1
            for(uint32_t i = 0; i < storage_.sampled_watchpoint_delay_iters; ++i)
            {
                asm volatile("s_nop 0" ::: "memory");
            }
        }

        template <typename SampledPolicy>
        __device__ void sampled_delay() const
        {
            if constexpr(SampledPolicy::delay_iters != 0)
            {
#pragma unroll 1
                for(uint32_t i = 0; i < SampledPolicy::delay_iters; ++i)
                {
                    asm volatile("s_nop 0" ::: "memory");
                }
            }
        }

        __device__ uint64_t atomic_exchange_shadow_entry(uint64_t* address, uint64_t value) const
        {
            return static_cast<uint64_t>(atomicExch(reinterpret_cast<unsigned long long*>(address),
                                                    static_cast<unsigned long long>(value)));
        }

        __device__ void close_current_epoch(bool advance_epochs) const
        {
            if(thread_id() == 0 && storage_.subgroup_states && storage_.subgroup_capacity > 0)
            {
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

        __device__ void emit_shadow_conflict(const detail::exact_shadow_entry& prior,
                                             const detail::exact_shadow_entry& current,
                                             uint32_t                          current_byte_count,
                                             uint32_t                          current_lds_offset,
                                             uint32_t                          cell) const
        {
            uint32_t prior_lds_offset = cell << detail::exact_shadow::granule_shift;
            detail::emit_diagnostic(storage_,
                                    diagnostic{
                                        static_cast<uint32_t>(diagnostic_kind::access_conflict),
                                        current.epoch,
                                        prior.owner_id,
                                        current.owner_id,
                                        static_cast<uintptr_t>(prior_lds_offset),
                                        static_cast<uintptr_t>(current_lds_offset),
                                        detail::exact_shadow::granule_bytes,
                                        current_byte_count,
                                        prior.site_id,
                                        current.site_id,
                                    });
        }

        __device__ void
            emit_sampled_watchpoint_conflict(const detail::sampled_watchpoint_entry& prior,
                                             const detail::sampled_watchpoint_entry& current,
                                             uint32_t current_lds_offset,
                                             uint32_t current_byte_count,
                                             site_id  current_site) const
        {
            uint32_t prior_lds_offset = prior.start_cell
                                        << detail::sampled_watchpoint::granule_shift;
            uint32_t prior_byte_count = prior.cell_count
                                        << detail::sampled_watchpoint::granule_shift;
            detail::emit_diagnostic(storage_,
                                    diagnostic{
                                        static_cast<uint32_t>(diagnostic_kind::access_conflict),
                                        current.epoch,
                                        prior.owner_id,
                                        current.owner_id,
                                        static_cast<uintptr_t>(prior_lds_offset),
                                        static_cast<uintptr_t>(current_lds_offset),
                                        prior_byte_count,
                                        current_byte_count,
                                        0,
                                        current_site.value(),
                                    });
        }

        __device__ void emit_shadow_metadata_full(const void* ptr,
                                                  uint32_t    byte_count,
                                                  uint32_t    lds_byte_offset,
                                                  site_id     site) const
        {
            uint32_t subgroup = subgroup_id();
            detail::emit_diagnostic(storage_,
                                    diagnostic{
                                        static_cast<uint32_t>(diagnostic_kind::metadata_full),
                                        detail::current_epoch(storage_, subgroup),
                                        subgroup,
                                        subgroup,
                                        reinterpret_cast<uintptr_t>(ptr),
                                        static_cast<uintptr_t>(lds_byte_offset),
                                        byte_count,
                                        byte_count,
                                        site.value(),
                                        site.value(),
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
