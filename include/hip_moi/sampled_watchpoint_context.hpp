// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// hip_moi/sampled_watchpoint_context.hpp
//
// Minimal sampled-watchpoint device context for publish-only hot paths.
#ifndef HIP_MOI_SAMPLED_WATCHPOINT_CONTEXT_HPP
#define HIP_MOI_SAMPLED_WATCHPOINT_CONTEXT_HPP

#include "hip_moi/common.hpp"
#include "hip_moi/context.hpp"
#include "hip_moi/shadow.hpp"

#include <type_traits>

namespace hip_moi
{
    class sampled_watchpoint_context
    {
    public:
        struct config
        {
            int threads_per_subgroup;
        };

        struct storage_ref
        {
            uint64_t* sampled_watchpoints         = nullptr;
            int       sampled_watchpoint_capacity = 0;
            uint64_t  generation                  = 0;
        };

        __device__ sampled_watchpoint_context(storage_ref storage, config cfg)
            : storage_(storage)
            , cfg_(cfg)
            , epoch_(0)
        {
        }

        __device__ void init_workgroup()
        {
            epoch_ = 0;
            __syncthreads();
        }

        __device__ void syncthreads()
        {
            __syncthreads();
            ++epoch_;
            __syncthreads();
        }

        template <typename SampledPolicy, typename T>
        __device__ T lds_load_at(const T* ptr, uint32_t lds_byte_offset, site_id site = no_site_id)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::sampled_watchpoint_context::lds_load_at "
                          "requires a trivially copyable type");
            // Scalar LDS scratch is the current high-pressure fast-path case.
            // Keep vector accesses on the runtime-shaped path to avoid
            // regressing vector-heavy matmul kernels.
            if constexpr(sizeof(T) <= detail::sampled_watchpoint::granule_bytes)
            {
                record_access_at<SampledPolicy, sizeof(T)>(
                    access_kind::load, lds_byte_offset, site);
            }
            else
            {
                record_access_at<SampledPolicy>(
                    sizeof(T), access_kind::load, lds_byte_offset, site);
            }
            return *ptr;
        }

        template <typename SampledPolicy, typename T>
        __device__ void
            lds_store_at(T* ptr, T value, uint32_t lds_byte_offset, site_id site = no_site_id)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::sampled_watchpoint_context::lds_store_at "
                          "requires a trivially copyable type");
            // Match lds_load_at: scalar accesses get compile-time range
            // simplification, vector accesses keep the older code shape.
            if constexpr(sizeof(T) <= detail::sampled_watchpoint::granule_bytes)
            {
                record_access_at<SampledPolicy, sizeof(T)>(
                    access_kind::store, lds_byte_offset, site);
            }
            else
            {
                record_access_at<SampledPolicy>(
                    sizeof(T), access_kind::store, lds_byte_offset, site);
            }
            *ptr = value;
        }

    private:
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
            return subgroup;
        }

        __device__ uint32_t lane_in_subgroup() const
        {
            if(cfg_.threads_per_subgroup <= 0)
            {
                return thread_id();
            }
            return thread_id() % static_cast<uint32_t>(cfg_.threads_per_subgroup);
        }

        __device__ uint32_t flat_workgroup_id() const
        {
            return static_cast<uint32_t>(
                blockIdx.x
                + gridDim.x * (blockIdx.y + gridDim.y * static_cast<uint32_t>(blockIdx.z)));
        }

        __device__ uint32_t current_epoch() const
        {
            return epoch_;
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

        __device__ uint32_t map_sampled_index(uint32_t value, uint32_t count) const
        {
            if(count == 0)
            {
                return 0;
            }
            bool power_of_two = (count & (count - 1u)) == 0;
            return power_of_two ? (value & (count - 1u)) : (value % count);
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

        template <typename SampledPolicy>
        __device__ uint32_t sampled_watchpoint_slot(uint32_t start_cell, uint32_t epoch) const
        {
            if constexpr(SampledPolicy::static_watchpoint_capacity == 1)
            {
                return 0;
            }

            uint32_t seed = epoch * 0x85ebca6bu;
            seed ^= static_cast<uint32_t>(storage_.generation) * 0xc2b2ae35u;
            seed ^= start_cell * 0x165667b1u;
            if constexpr(SampledPolicy::static_watchpoint_capacity > 1)
            {
                return map_sampled_index(detail::mix32(seed),
                                         SampledPolicy::static_watchpoint_capacity);
            }
            else
            {
                return map_sampled_index(
                    detail::mix32(seed),
                    static_cast<uint32_t>(storage_.sampled_watchpoint_capacity));
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

        template <typename SampledPolicy>
        __device__ bool sampled_storage_available() const
        {
            if constexpr(SampledPolicy::static_watchpoint_capacity > 0)
            {
                // Static capacity is an opt-in promise from the policy, so the
                // hot path only needs to guard the storage pointer.
                return storage_.sampled_watchpoints != nullptr;
            }
            else
            {
                return storage_.sampled_watchpoints && storage_.sampled_watchpoint_capacity > 0;
            }
        }

        template <uint32_t ByteCount>
        static __host__ __device__ constexpr uint32_t max_spanned_sampled_cells()
        {
            return (ByteCount + 2u * detail::sampled_watchpoint::granule_bytes - 2u)
                   >> detail::sampled_watchpoint::granule_shift;
        }

        template <uint32_t ByteCount>
        static __host__ __device__ constexpr uint32_t max_valid_sampled_start_byte()
        {
            return ((detail::sampled_watchpoint::max_start + 1u)
                    << detail::sampled_watchpoint::granule_shift)
                   - ByteCount;
        }

        template <uint32_t ByteCount>
        static __host__ __device__ constexpr bool fits_sampled_encoding()
        {
            return ByteCount > 0
                   && ByteCount <= ((detail::sampled_watchpoint::max_start + 1u)
                                    << detail::sampled_watchpoint::granule_shift);
        }

        template <typename SampledPolicy>
        __device__ void record_access_at(uint32_t    byte_count,
                                         access_kind kind,
                                         uint32_t    lds_byte_offset,
                                         site_id     site) const
        {
            static_assert(!SampledPolicy::report_conflicts,
                          "hip_moi::sampled_watchpoint_context currently supports only "
                          "publish-only sampled policies");
            if(byte_count == 0 || !sampled_storage_available<SampledPolicy>())
            {
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
                return;
            }

            while(static_cast<uint64_t>(start_cell) < end_cell)
            {
                uint64_t remaining = end_cell - static_cast<uint64_t>(start_cell);
                uint32_t chunk     = remaining > detail::sampled_watchpoint::max_count
                                         ? detail::sampled_watchpoint::max_count
                                         : static_cast<uint32_t>(remaining);
                record_range<SampledPolicy>(kind, start_cell, chunk);
                start_cell += chunk;
            }
        }

        template <typename SampledPolicy, uint32_t ByteCount>
        __device__ void
            record_access_at(access_kind kind, uint32_t lds_byte_offset, site_id site) const
        {
            static_assert(!SampledPolicy::report_conflicts,
                          "hip_moi::sampled_watchpoint_context currently supports only "
                          "publish-only sampled policies");
            static_assert(ByteCount > 0,
                          "hip_moi::sampled_watchpoint_context requires non-empty accesses");
            if(!sampled_storage_available<SampledPolicy>())
            {
                return;
            }

            uint32_t selection_seed = sampled_selection_seed(site);
            if(!sampled_should_publish<SampledPolicy>(selection_seed))
            {
                return;
            }

            if constexpr(!fits_sampled_encoding<ByteCount>())
            {
                return;
            }
            else if constexpr(max_spanned_sampled_cells<ByteCount>()
                              <= detail::sampled_watchpoint::max_count)
            {
                if(lds_byte_offset > max_valid_sampled_start_byte<ByteCount>())
                {
                    return;
                }

                uint32_t start_cell = lds_byte_offset >> detail::sampled_watchpoint::granule_shift;
                uint32_t cell_count
                    = ((lds_byte_offset & (detail::sampled_watchpoint::granule_bytes - 1u))
                       + ByteCount + detail::sampled_watchpoint::granule_bytes - 1u)
                      >> detail::sampled_watchpoint::granule_shift;
                record_range<SampledPolicy>(kind, start_cell, cell_count);
            }
            else
            {
                uint64_t last_byte_offset = static_cast<uint64_t>(lds_byte_offset)
                                            + static_cast<uint64_t>(ByteCount) - 1u;
                uint32_t start_cell = lds_byte_offset >> detail::sampled_watchpoint::granule_shift;
                uint64_t end_cell
                    = (last_byte_offset >> detail::sampled_watchpoint::granule_shift) + 1u;
                if(start_cell > detail::sampled_watchpoint::max_start
                   || end_cell > static_cast<uint64_t>(detail::sampled_watchpoint::max_start) + 1u)
                {
                    return;
                }

                while(static_cast<uint64_t>(start_cell) < end_cell)
                {
                    uint64_t remaining = end_cell - static_cast<uint64_t>(start_cell);
                    uint32_t chunk     = remaining > detail::sampled_watchpoint::max_count
                                             ? detail::sampled_watchpoint::max_count
                                             : static_cast<uint32_t>(remaining);
                    record_range<SampledPolicy>(kind, start_cell, chunk);
                    start_cell += chunk;
                }
            }
        }

        template <typename SampledPolicy>
        __device__ void
            record_range(access_kind kind, uint32_t start_cell, uint32_t cell_count) const
        {
            uint32_t subgroup = subgroup_id();
            uint32_t epoch    = current_epoch();
            uint64_t packed
                = detail::pack_sampled_watchpoint_entry(detail::shadow_kind_from_access_kind(kind),
                                                        subgroup,
                                                        epoch,
                                                        static_cast<uint32_t>(storage_.generation),
                                                        start_cell,
                                                        cell_count);
            uint32_t slot = sampled_watchpoint_slot<SampledPolicy>(start_cell, epoch);
            (void)atomicExch(
                reinterpret_cast<unsigned long long*>(&storage_.sampled_watchpoints[slot]),
                static_cast<unsigned long long>(packed));
            sampled_delay<SampledPolicy>();
        }

        storage_ref storage_;
        config      cfg_;
        // Per-thread copy of the workgroup epoch; avoids a global load at each
        // sampled access site in the publish-only fast path.
        uint32_t epoch_;
    };

    __device__ inline sampled_watchpoint_context::storage_ref
        make_sampled_watchpoint_storage_ref(context::storage_ref storage)
    {
        return sampled_watchpoint_context::storage_ref{
            /*sampled_watchpoints=*/storage.sampled_watchpoints,
            /*sampled_watchpoint_capacity=*/storage.sampled_watchpoint_capacity,
            /*generation=*/storage.generation,
        };
    }

    __device__ inline sampled_watchpoint_context
        make_sampled_watchpoint_context(context::storage_ref               storage,
                                        sampled_watchpoint_context::config cfg)
    {
        return sampled_watchpoint_context(make_sampled_watchpoint_storage_ref(storage), cfg);
    }
} // namespace hip_moi

#endif // HIP_MOI_SAMPLED_WATCHPOINT_CONTEXT_HPP
