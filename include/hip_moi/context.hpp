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

        struct atomic_object_record
        {
            uint64_t  generation;
            uintptr_t address;
            uint32_t  releasing_subgroup_id;
            uint32_t  releasing_epoch;
            uint64_t  release_site_id;
        };

        struct atomic_address_cache_record
        {
            uint64_t  generation;
            uintptr_t address;
            uint64_t  producer_mask;
        };

        struct storage_ref
        {
            using diagnostic_type = diagnostic;

            diagnostic*           diagnostics                         = nullptr;
            int                   diagnostic_capacity                 = 0;
            subgroup_state*       subgroup_states                     = nullptr;
            int                   subgroup_capacity                   = 0;
            int*                  diagnostic_count                    = nullptr;
            int*                  simulated_barrier_arrival_count     = nullptr;
            uint64_t*             exact_shadow_entries                = nullptr;
            int                   exact_shadow_entry_capacity         = 0;
            uint64_t*             sampled_watchpoints                 = nullptr;
            int                   sampled_watchpoint_capacity         = 0;
            uint64_t              generation                          = 0;
            uint32_t              backend                             = 0;
            uint32_t              sampled_watchpoint_sample_skip      = 1;
            uint32_t              sampled_watchpoint_probe_count      = 1;
            uint32_t              sampled_watchpoint_delay_iters      = 0;
            uint32_t              sampled_watchpoint_report_conflicts = 1;
            atomic_object_record* atomic_objects                      = nullptr;
            int                   atomic_object_capacity              = 0;
            uint32_t*             acquired_subgroup_epoch_tokens      = nullptr;
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
                    nullptr,
                    0,
                    nullptr,
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
                int acquired_epoch_count = storage_.subgroup_capacity * storage_.subgroup_capacity;
                for(int i = 0; storage_.acquired_subgroup_epoch_tokens && i < acquired_epoch_count;
                    ++i)
                {
                    storage_.acquired_subgroup_epoch_tokens[i] = 0;
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

        template <typename T>
        __device__ T atomic_load(T*                  ptr,
                                 atomic_memory_order order,
                                 atomic_memory_scope scope,
                                 site_id             site = no_site_id) const
        {
            static_assert(detail::is_supported_atomic_value<T>::value,
                          "hip_moi::context::atomic_load currently supports "
                          "4-byte and 8-byte integral values");
            T value = detail::atomic_load_dispatch(ptr, order, scope);
            if(detail::atomic_order_has_acquire(order))
            {
                record_atomic_acquire(ptr);
            }
            else
            {
                (void)site;
            }
            remember_observed_atomic(ptr);
            return value;
        }

        template <typename T>
        __device__ void atomic_store(T*                  ptr,
                                     T                   value,
                                     atomic_memory_order order,
                                     atomic_memory_scope scope,
                                     site_id             site = no_site_id) const
        {
            static_assert(detail::is_supported_atomic_value<T>::value,
                          "hip_moi::context::atomic_store currently supports "
                          "4-byte and 8-byte integral values");
            if(detail::atomic_order_has_release(order))
            {
                record_atomic_release(ptr, sizeof(T), site);
            }
            else
            {
                consume_pending_release_fence(ptr, static_cast<uint32_t>(sizeof(T)), site);
            }
            detail::atomic_store_dispatch(ptr, value, order, scope);
        }

        template <typename T>
        __device__ T atomic_fetch_add(T*                  ptr,
                                      T                   value,
                                      atomic_memory_order order,
                                      atomic_memory_scope scope,
                                      site_id             site = no_site_id) const
        {
            static_assert(detail::is_supported_atomic_value<T>::value,
                          "hip_moi::context::atomic_fetch_add currently supports "
                          "4-byte and 8-byte integral values");
            T old_value = detail::atomic_fetch_add_dispatch(ptr, value, order, scope);
            if(detail::atomic_order_has_release(order))
            {
                record_atomic_rmw_release(ptr, sizeof(T), site);
            }
            else
            {
                consume_pending_release_fence(ptr, static_cast<uint32_t>(sizeof(T)), site);
            }
            if(detail::atomic_order_has_acquire(order))
            {
                // RMW release metadata is recorded after the hardware atomic,
                // so an acquiring RMW can observe the new value before the
                // producer has published its metadata.
                record_atomic_acquire(ptr, kAtomicRmwAcquireRetryCount);
            }
            remember_observed_atomic(ptr);
            return old_value;
        }

        template <typename T>
        __device__ T atomic_fetch_or(T*                  ptr,
                                     T                   value,
                                     atomic_memory_order order,
                                     atomic_memory_scope scope,
                                     site_id             site = no_site_id) const
        {
            static_assert(detail::is_supported_atomic_value<T>::value,
                          "hip_moi::context::atomic_fetch_or currently supports "
                          "4-byte and 8-byte integral values");
            T old_value = detail::atomic_fetch_or_dispatch(ptr, value, order, scope);
            if(detail::atomic_order_has_release(order))
            {
                record_atomic_rmw_release(ptr, sizeof(T), site);
            }
            else
            {
                consume_pending_release_fence(ptr, static_cast<uint32_t>(sizeof(T)), site);
            }
            if(detail::atomic_order_has_acquire(order))
            {
                // See atomic_fetch_add: acquiring RMWs may need to wait for
                // release metadata from the immediately preceding RMW.
                record_atomic_acquire(ptr, kAtomicRmwAcquireRetryCount);
            }
            remember_observed_atomic(ptr);
            return old_value;
        }

        template <typename T>
        __device__ T atomic_exchange(T*                  ptr,
                                     T                   value,
                                     atomic_memory_order order,
                                     atomic_memory_scope scope,
                                     site_id             site = no_site_id) const
        {
            static_assert(detail::is_supported_atomic_value<T>::value,
                          "hip_moi::context::atomic_exchange currently supports "
                          "4-byte and 8-byte integral values");
            T old_value = detail::atomic_exchange_dispatch(ptr, value, order, scope);
            if(detail::atomic_order_has_release(order))
            {
                record_atomic_rmw_release(ptr, sizeof(T), site);
            }
            else
            {
                consume_pending_release_fence(ptr, static_cast<uint32_t>(sizeof(T)), site);
            }
            if(detail::atomic_order_has_acquire(order))
            {
                record_atomic_acquire(ptr, kAtomicRmwAcquireRetryCount);
            }
            remember_observed_atomic(ptr);
            return old_value;
        }

        template <typename T>
        __device__ bool atomic_compare_exchange_strong(T*                  ptr,
                                                       T*                  expected,
                                                       T                   desired,
                                                       atomic_memory_order success_order,
                                                       atomic_memory_order failure_order,
                                                       atomic_memory_scope scope,
                                                       site_id             site = no_site_id) const
        {
            static_assert(detail::is_supported_atomic_value<T>::value,
                          "hip_moi::context::atomic_compare_exchange_strong "
                          "currently supports 4-byte and 8-byte integral values");
            bool exchanged = detail::atomic_compare_exchange_strong_dispatch(
                ptr, expected, desired, success_order, failure_order, scope);
            if(exchanged)
            {
                if(detail::atomic_order_has_release(success_order))
                {
                    record_atomic_rmw_release(ptr, sizeof(T), site);
                }
                else
                {
                    consume_pending_release_fence(ptr, static_cast<uint32_t>(sizeof(T)), site);
                }
                if(detail::atomic_order_has_acquire(success_order))
                {
                    record_atomic_acquire(ptr, kAtomicRmwAcquireRetryCount);
                }
            }
            else if(detail::atomic_order_has_acquire(failure_order))
            {
                record_atomic_acquire(ptr, kAtomicRmwAcquireRetryCount);
            }
            else
            {
                (void)site;
            }
            remember_observed_atomic(ptr);
            return exchanged;
        }

        __device__ void atomic_fence(atomic_memory_order order,
                                     atomic_memory_scope scope,
                                     site_id             site = no_site_id) const
        {
            if(detail::atomic_order_has_release(order))
            {
                pending_release_fence_epoch_token_
                    = detail::current_epoch(storage_, subgroup_id()) + 1u;
                pending_release_fence_site_id_ = site.value();
            }
            detail::atomic_fence_dispatch(order, scope);
            if(detail::atomic_order_has_acquire(order) && last_observed_atomic_valid_)
            {
                record_atomic_acquire(reinterpret_cast<const void*>(last_observed_atomic_address_),
                                      kAtomicRmwAcquireRetryCount);
            }
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
        static constexpr uint64_t kAtomicObjectClaimedGeneration = ~uint64_t{0};
        static constexpr int      kAtomicRmwAcquireRetryCount    = 4;

        __device__ backend_kind selected_backend() const
        {
            return static_cast<backend_kind>(storage_.backend);
        }

        __device__ uint32_t atomic_record_seed(uintptr_t address, uint32_t producer) const
        {
            uint32_t seed
                = static_cast<uint32_t>(address >> 2) ^ static_cast<uint32_t>(address >> 34);
            seed ^= producer * 0x9e3779b9u;
            return seed;
        }

        __device__ uint32_t atomic_address_cache_index(uintptr_t address) const
        {
            uint32_t seed
                = static_cast<uint32_t>(address >> 2) ^ static_cast<uint32_t>(address >> 34);
            return detail::mix32(seed)
                   & (static_cast<uint32_t>(storage_.atomic_object_capacity) - 1u);
        }

        __device__ atomic_object_record* find_or_claim_atomic_object_record(uintptr_t address,
                                                                            uint32_t  byte_count,
                                                                            uint32_t  producer,
                                                                            site_id   site) const
        {
            if(!storage_.atomic_objects || storage_.atomic_object_capacity <= 0)
            {
                emit_atomic_metadata_full(address, byte_count, site);
                return nullptr;
            }

            uint64_t generation = storage_.generation;
            uint32_t capacity   = static_cast<uint32_t>(storage_.atomic_object_capacity);
            uint32_t start = detail::mix32(atomic_record_seed(address, producer)) & (capacity - 1u);
            for(int attempt = 0; attempt < 4; ++attempt)
            {
                bool saw_claimed_slot = false;
                for(uint32_t probe = 0; probe < capacity; ++probe)
                {
                    uint32_t index = start + probe;
                    if(index >= capacity)
                    {
                        index -= capacity;
                    }
                    atomic_object_record* record = &storage_.atomic_objects[index];
                    uint64_t              slot_generation
                        = *reinterpret_cast<volatile uint64_t*>(&record->generation);
                    if(slot_generation == generation)
                    {
                        uintptr_t slot_address
                            = *reinterpret_cast<volatile uintptr_t*>(&record->address);
                        uint32_t slot_producer
                            = *reinterpret_cast<volatile uint32_t*>(&record->releasing_subgroup_id);
                        if(slot_address == address && slot_producer == producer)
                        {
                            return record;
                        }
                        continue;
                    }

                    if(slot_generation == kAtomicObjectClaimedGeneration)
                    {
                        saw_claimed_slot = true;
                        continue;
                    }

                    unsigned long long prior = atomicCAS(
                        reinterpret_cast<unsigned long long*>(&record->generation),
                        static_cast<unsigned long long>(slot_generation),
                        static_cast<unsigned long long>(kAtomicObjectClaimedGeneration));
                    if(prior == static_cast<unsigned long long>(slot_generation))
                    {
                        record->address               = address;
                        record->releasing_subgroup_id = producer;
                        record->releasing_epoch       = 0;
                        record->release_site_id       = 0;
                        return record;
                    }
                    saw_claimed_slot = true;
                }

                if(!saw_claimed_slot)
                {
                    break;
                }
            }

            emit_atomic_metadata_full(address, byte_count, site);
            return nullptr;
        }

        __device__ atomic_object_record* find_atomic_object_record(uintptr_t address,
                                                                   uint32_t  producer) const
        {
            if(!storage_.atomic_objects || storage_.atomic_object_capacity <= 0)
            {
                return nullptr;
            }

            uint64_t generation = storage_.generation;
            uint32_t capacity   = static_cast<uint32_t>(storage_.atomic_object_capacity);
            uint32_t start = detail::mix32(atomic_record_seed(address, producer)) & (capacity - 1u);
            for(uint32_t probe = 0; probe < capacity; ++probe)
            {
                uint32_t index = start + probe;
                if(index >= capacity)
                {
                    index -= capacity;
                }

                atomic_object_record* record = &storage_.atomic_objects[index];
                uint64_t              slot_generation
                    = *reinterpret_cast<volatile uint64_t*>(&record->generation);
                if(slot_generation == kAtomicObjectClaimedGeneration)
                {
                    continue;
                }
                // Records are never deleted within a generation. A stale slot
                // therefore terminates this address's open-addressing probe
                // chain unless another thread is actively claiming it.
                if(slot_generation != generation)
                {
                    return nullptr;
                }

                uintptr_t slot_address = *reinterpret_cast<volatile uintptr_t*>(&record->address);
                uint32_t  slot_producer
                    = *reinterpret_cast<volatile uint32_t*>(&record->releasing_subgroup_id);
                if(slot_address == address && slot_producer == producer)
                {
                    return record;
                }
            }
            return nullptr;
        }

        __device__ void
            record_atomic_release(const void* ptr, uint32_t byte_count, site_id site) const
        {
            record_atomic_release_at_epoch_impl</*UseAddressCache=*/false>(
                ptr, byte_count, detail::current_epoch(storage_, subgroup_id()), site);
        }

        __device__ void
            record_atomic_rmw_release(const void* ptr, uint32_t byte_count, site_id site) const
        {
            if(should_cache_atomic_rmw_release())
            {
                record_atomic_release_at_epoch_impl</*UseAddressCache=*/true>(
                    ptr, byte_count, detail::current_epoch(storage_, subgroup_id()), site);
                return;
            }
            record_atomic_release(ptr, byte_count, site);
        }

        __device__ void record_atomic_release_at_epoch(const void* ptr,
                                                       uint32_t    byte_count,
                                                       uint32_t    releasing_epoch,
                                                       site_id     site) const
        {
            record_atomic_release_at_epoch_impl</*UseAddressCache=*/false>(
                ptr, byte_count, releasing_epoch, site);
        }

        template <bool UseAddressCache>
        __device__ void record_atomic_release_at_epoch_impl(const void* ptr,
                                                            uint32_t    byte_count,
                                                            uint32_t    releasing_epoch,
                                                            site_id     site) const
        {
            uintptr_t             address = reinterpret_cast<uintptr_t>(ptr);
            uint32_t              producer = subgroup_id();
            atomic_object_record* record
                = find_or_claim_atomic_object_record(address, byte_count, producer, site);
            if(!record)
            {
                return;
            }

            record->address               = address;
            record->releasing_subgroup_id = producer;
            record->release_site_id       = site.value();
            (void)atomicMax(&record->releasing_epoch, releasing_epoch);
            if constexpr(UseAddressCache)
            {
                cache_atomic_release(address, producer);
            }
            __threadfence();
            record->generation = storage_.generation;
        }

        __device__ bool should_cache_atomic_rmw_release() const
        {
            return detail::configured_subgroup_count(cfg_) > 2u;
        }

        __device__ void cache_atomic_release(uintptr_t address, uint32_t producer) const
        {
            atomic_address_cache_record* cache = atomic_address_cache_records();
            if(!cache || producer >= 64)
            {
                return;
            }

            atomic_address_cache_record* record     = &cache[atomic_address_cache_index(address)];
            uint64_t                     generation = storage_.generation;
            for(int attempt = 0; attempt < kAtomicRmwAcquireRetryCount; ++attempt)
            {
                uint64_t slot_generation
                    = *reinterpret_cast<volatile uint64_t*>(&record->generation);
                if(slot_generation == generation)
                {
                    uintptr_t slot_address
                        = *reinterpret_cast<volatile uintptr_t*>(&record->address);
                    if(slot_address == address)
                    {
                        (void)atomicOr(
                            reinterpret_cast<unsigned long long*>(&record->producer_mask),
                            static_cast<unsigned long long>(uint64_t{1} << producer));
                    }
                    return;
                }

                if(slot_generation == kAtomicObjectClaimedGeneration)
                {
                    atomic_acquire_retry_delay();
                    continue;
                }

                unsigned long long prior
                    = atomicCAS(reinterpret_cast<unsigned long long*>(&record->generation),
                                static_cast<unsigned long long>(slot_generation),
                                static_cast<unsigned long long>(kAtomicObjectClaimedGeneration));
                if(prior != static_cast<unsigned long long>(slot_generation))
                {
                    atomic_acquire_retry_delay();
                    continue;
                }

                record->address       = address;
                record->producer_mask = uint64_t{1} << producer;
                __threadfence();
                record->generation = generation;
                return;
            }
        }

        __device__ void remember_observed_atomic(const void* ptr) const
        {
            last_observed_atomic_address_ = reinterpret_cast<uintptr_t>(ptr);
            last_observed_atomic_valid_   = true;
        }

        __device__ void
            consume_pending_release_fence(const void* ptr, uint32_t byte_count, site_id site) const
        {
            if(pending_release_fence_epoch_token_ == 0)
            {
                return;
            }

            uint64_t release_site = pending_release_fence_site_id_ != 0
                                        ? pending_release_fence_site_id_
                                        : site.value();
            record_atomic_release_at_epoch(
                ptr, byte_count, pending_release_fence_epoch_token_ - 1u, site_id(release_site));
            pending_release_fence_epoch_token_ = 0;
            pending_release_fence_site_id_     = 0;
        }

        __device__ bool try_record_atomic_acquire(const void* ptr) const
        {
            if(!storage_.acquired_subgroup_epoch_tokens || storage_.subgroup_capacity <= 0)
            {
                return true;
            }

            uint32_t consumer = subgroup_id();
            uint32_t capacity = static_cast<uint32_t>(storage_.subgroup_capacity);
            if(consumer >= capacity)
            {
                return true;
            }

            uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
            bool      cache_hit = false;
            bool      found
                = try_record_atomic_acquire_from_cache(address, consumer, capacity, &cache_hit);
            if(cache_hit)
            {
                return found;
            }

            return try_record_atomic_acquire_from_table(address, consumer, capacity);
        }

        __device__ bool try_record_atomic_acquire_from_cache(uintptr_t address,
                                                             uint32_t  consumer,
                                                             uint32_t  capacity,
                                                             bool*     cache_hit) const
        {
            *cache_hit                         = false;
            atomic_address_cache_record* cache = atomic_address_cache_records();
            if(!cache)
            {
                return false;
            }

            atomic_address_cache_record* cache_record = &cache[atomic_address_cache_index(address)];
            uint64_t                     slot_generation
                = *reinterpret_cast<volatile uint64_t*>(&cache_record->generation);
            if(slot_generation != storage_.generation
               || slot_generation == kAtomicObjectClaimedGeneration)
            {
                return false;
            }

            uintptr_t slot_address = *reinterpret_cast<volatile uintptr_t*>(&cache_record->address);
            if(slot_address != address)
            {
                return false;
            }

            *cache_hit = true;
            uint64_t producer_mask
                = *reinterpret_cast<volatile uint64_t*>(&cache_record->producer_mask);
            bool      found   = false;
            for(uint32_t producer = 0; producer < capacity; ++producer)
            {
                if(producer == consumer)
                {
                    continue;
                }
                if(producer >= 64 || ((producer_mask & (uint64_t{1} << producer)) == 0))
                {
                    continue;
                }

                found |= try_import_atomic_release_from_producer(
                    address, consumer, capacity, producer);
            }
            return found;
        }

        __device__ bool try_record_atomic_acquire_from_table(uintptr_t address,
                                                             uint32_t  consumer,
                                                             uint32_t  capacity) const
        {
            bool found = false;
            for(uint32_t producer = 0; producer < capacity; ++producer)
            {
                if(producer == consumer)
                {
                    continue;
                }

                found |= try_import_atomic_release_from_producer(
                    address, consumer, capacity, producer);
            }
            return found;
        }

        __device__ bool try_import_atomic_release_from_producer(uintptr_t address,
                                                                uint32_t  consumer,
                                                                uint32_t  capacity,
                                                                uint32_t  producer) const
        {
            atomic_object_record* record = find_atomic_object_record(address, producer);
            if(!record)
            {
                return false;
            }

            uint32_t released_epoch
                = *reinterpret_cast<volatile uint32_t*>(&record->releasing_epoch);
            uint32_t token = released_epoch + 1u;
            uint32_t index = consumer * capacity + producer;
            (void)atomicMax(&storage_.acquired_subgroup_epoch_tokens[index], token);
            return true;
        }

        __device__ atomic_address_cache_record* atomic_address_cache_records() const
        {
            if(!storage_.atomic_objects || storage_.atomic_object_capacity <= 0)
            {
                return nullptr;
            }
            return reinterpret_cast<atomic_address_cache_record*>(
                storage_.atomic_objects + storage_.atomic_object_capacity);
        }

        __device__ void atomic_acquire_retry_delay() const
        {
            for(int delay = 0; delay < 16; ++delay)
            {
                __builtin_amdgcn_s_sleep(1);
            }
        }

        __device__ void record_atomic_acquire(const void* ptr, int retry_count = 1) const
        {
            for(int attempt = 0; attempt < retry_count; ++attempt)
            {
                if(try_record_atomic_acquire(ptr))
                {
                    return;
                }
                if(attempt + 1 < retry_count)
                {
                    atomic_acquire_retry_delay();
                }
            }
        }

        __device__ bool
            current_access_acquired_prior_epoch(const detail::exact_shadow_entry& prior,
                                                const detail::exact_shadow_entry& current) const
        {
            if(!storage_.acquired_subgroup_epoch_tokens || storage_.subgroup_capacity <= 0)
            {
                return false;
            }

            uint32_t capacity = static_cast<uint32_t>(storage_.subgroup_capacity);
            if(current.owner_id >= capacity || prior.owner_id >= capacity)
            {
                return false;
            }

            uint32_t index = current.owner_id * capacity + prior.owner_id;
            uint32_t token = *reinterpret_cast<volatile uint32_t*>(
                &storage_.acquired_subgroup_epoch_tokens[index]);
            return token >= prior.epoch + 1u;
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
                if(detail::exact_shadow_entries_conflict(current, prior)
                   && !current_access_acquired_prior_epoch(prior, current))
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
            emit_atomic_metadata_full(uintptr_t address, uint32_t byte_count, site_id site) const
        {
            uint32_t subgroup = subgroup_id();
            detail::emit_diagnostic(storage_,
                                    diagnostic{
                                        static_cast<uint32_t>(diagnostic_kind::metadata_full),
                                        detail::current_epoch(storage_, subgroup),
                                        subgroup,
                                        subgroup,
                                        address,
                                        address,
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

        mutable uintptr_t last_observed_atomic_address_      = 0;
        mutable bool      last_observed_atomic_valid_        = false;
        mutable uint32_t  pending_release_fence_epoch_token_ = 0;
        mutable uint64_t  pending_release_fence_site_id_     = 0;
    };
} // namespace hip_moi

#endif // HIP_MOI_CONTEXT_HPP
