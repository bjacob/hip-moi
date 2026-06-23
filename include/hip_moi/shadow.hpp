// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// hip_moi/shadow.hpp
//
// Compact shadow/watchpoint metadata helpers for Loom-like hip-moi backends.
#ifndef HIP_MOI_SHADOW_HPP
#define HIP_MOI_SHADOW_HPP

#include "hip_moi/common.hpp"

#include <cstdint>

namespace hip_moi
{
    enum class shadow_access_kind : uint32_t
    {
        empty      = 0,
        read       = 1,
        write      = 2,
        read_write = 3,
        atomic     = 4,
    };

    namespace detail
    {
        __host__ __device__ constexpr uint64_t low_bit_mask(uint32_t bits)
        {
            return bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1u);
        }

        __host__ __device__ constexpr uint64_t bit_field_mask(uint32_t shift, uint32_t bits)
        {
            return low_bit_mask(bits) << shift;
        }

        __host__ __device__ constexpr shadow_access_kind
            shadow_kind_from_access_kind(access_kind kind)
        {
            return kind == access_kind::store ? shadow_access_kind::write
                                              : shadow_access_kind::read;
        }

        __host__ __device__ constexpr bool shadow_kind_is_empty(shadow_access_kind kind)
        {
            return kind == shadow_access_kind::empty;
        }

        __host__ __device__ constexpr bool shadow_kind_conflicts(shadow_access_kind current,
                                                                 shadow_access_kind prior)
        {
            if(shadow_kind_is_empty(current) || shadow_kind_is_empty(prior))
            {
                return false;
            }
            if(current == shadow_access_kind::read)
            {
                return prior != shadow_access_kind::read;
            }
            if(current == shadow_access_kind::atomic)
            {
                return prior != shadow_access_kind::atomic;
            }
            return true;
        }

        namespace exact_shadow
        {
            inline constexpr uint32_t granule_shift = 2;
            inline constexpr uint32_t granule_bytes = 1u << granule_shift;

            inline constexpr uint32_t access_kind_shift = 0;
            inline constexpr uint32_t access_kind_bits  = 3;
            inline constexpr uint32_t owner_shift       = 3;
            inline constexpr uint32_t owner_bits        = 10;
            inline constexpr uint32_t epoch_shift       = 13;
            inline constexpr uint32_t epoch_bits        = 10;
            inline constexpr uint32_t generation_shift  = 23;
            inline constexpr uint32_t generation_bits   = 20;
            inline constexpr uint32_t site_shift        = 43;
            inline constexpr uint32_t site_bits         = 21;

            inline constexpr uint64_t access_kind_mask
                = bit_field_mask(access_kind_shift, access_kind_bits);
            inline constexpr uint64_t owner_mask = bit_field_mask(owner_shift, owner_bits);
            inline constexpr uint64_t epoch_mask = bit_field_mask(epoch_shift, epoch_bits);
            inline constexpr uint64_t generation_mask
                = bit_field_mask(generation_shift, generation_bits);
            inline constexpr uint64_t site_mask             = bit_field_mask(site_shift, site_bits);
            inline constexpr uint64_t epoch_generation_mask = epoch_mask | generation_mask;

            inline constexpr uint32_t max_owner      = (1u << owner_bits) - 1u;
            inline constexpr uint32_t max_epoch      = (1u << epoch_bits) - 1u;
            inline constexpr uint32_t max_generation = (1u << generation_bits) - 1u;
            inline constexpr uint32_t max_site       = (1u << site_bits) - 1u;
        } // namespace exact_shadow

        struct exact_shadow_entry
        {
            shadow_access_kind kind;
            uint32_t           owner_id;
            uint32_t           epoch;
            uint32_t           generation;
            uint32_t           site_id;
        };

        __host__ __device__ constexpr uint64_t pack_exact_shadow_entry(shadow_access_kind kind,
                                                                       uint32_t           owner_id,
                                                                       uint32_t           epoch,
                                                                       uint32_t generation,
                                                                       uint64_t site_id)
        {
            return ((static_cast<uint64_t>(kind) & low_bit_mask(exact_shadow::access_kind_bits))
                    << exact_shadow::access_kind_shift)
                   | ((static_cast<uint64_t>(owner_id) & low_bit_mask(exact_shadow::owner_bits))
                      << exact_shadow::owner_shift)
                   | ((static_cast<uint64_t>(epoch) & low_bit_mask(exact_shadow::epoch_bits))
                      << exact_shadow::epoch_shift)
                   | ((static_cast<uint64_t>(generation)
                       & low_bit_mask(exact_shadow::generation_bits))
                      << exact_shadow::generation_shift)
                   | ((site_id & low_bit_mask(exact_shadow::site_bits))
                      << exact_shadow::site_shift);
        }

        __host__ __device__ constexpr exact_shadow_entry decode_exact_shadow_entry(uint64_t value)
        {
            return exact_shadow_entry{
                static_cast<shadow_access_kind>((value & exact_shadow::access_kind_mask)
                                                >> exact_shadow::access_kind_shift),
                static_cast<uint32_t>((value & exact_shadow::owner_mask)
                                      >> exact_shadow::owner_shift),
                static_cast<uint32_t>((value & exact_shadow::epoch_mask)
                                      >> exact_shadow::epoch_shift),
                static_cast<uint32_t>((value & exact_shadow::generation_mask)
                                      >> exact_shadow::generation_shift),
                static_cast<uint32_t>((value & exact_shadow::site_mask)
                                      >> exact_shadow::site_shift),
            };
        }

        __host__ __device__ constexpr bool
            exact_shadow_entries_conflict(const exact_shadow_entry& current,
                                          const exact_shadow_entry& prior)
        {
            return !shadow_kind_is_empty(prior.kind) && current.epoch == prior.epoch
                   && current.generation == prior.generation && current.owner_id != prior.owner_id
                   && shadow_kind_conflicts(current.kind, prior.kind);
        }

        namespace sampled_watchpoint
        {
            inline constexpr uint32_t granule_shift = 2;
            inline constexpr uint32_t granule_bytes = 1u << granule_shift;

            inline constexpr uint32_t valid_shift       = 0;
            inline constexpr uint32_t consumed_shift    = 1;
            inline constexpr uint32_t access_kind_shift = 2;
            inline constexpr uint32_t access_kind_bits  = 3;
            inline constexpr uint32_t owner_shift       = 5;
            inline constexpr uint32_t owner_bits        = 10;
            inline constexpr uint32_t epoch_shift       = 15;
            inline constexpr uint32_t epoch_bits        = 10;
            inline constexpr uint32_t generation_shift  = 25;
            inline constexpr uint32_t generation_bits   = 20;
            inline constexpr uint32_t start_shift       = 45;
            inline constexpr uint32_t start_bits        = 14;
            inline constexpr uint32_t count_shift       = 59;
            inline constexpr uint32_t count_bits        = 5;

            inline constexpr uint64_t valid_mask    = uint64_t{1} << valid_shift;
            inline constexpr uint64_t consumed_mask = uint64_t{1} << consumed_shift;
            inline constexpr uint64_t access_kind_mask
                = bit_field_mask(access_kind_shift, access_kind_bits);
            inline constexpr uint64_t owner_mask = bit_field_mask(owner_shift, owner_bits);
            inline constexpr uint64_t epoch_mask = bit_field_mask(epoch_shift, epoch_bits);
            inline constexpr uint64_t generation_mask
                = bit_field_mask(generation_shift, generation_bits);
            inline constexpr uint64_t start_mask = bit_field_mask(start_shift, start_bits);
            inline constexpr uint64_t count_mask = bit_field_mask(count_shift, count_bits);
            inline constexpr uint64_t epoch_generation_mask = epoch_mask | generation_mask;

            inline constexpr uint32_t max_owner      = (1u << owner_bits) - 1u;
            inline constexpr uint32_t max_epoch      = (1u << epoch_bits) - 1u;
            inline constexpr uint32_t max_generation = (1u << generation_bits) - 1u;
            inline constexpr uint32_t max_start      = (1u << start_bits) - 1u;
            inline constexpr uint32_t max_count      = 1u << count_bits;
        } // namespace sampled_watchpoint

        struct sampled_watchpoint_entry
        {
            bool               valid;
            bool               consumed;
            shadow_access_kind kind;
            uint32_t           owner_id;
            uint32_t           epoch;
            uint32_t           generation;
            uint32_t           start_cell;
            uint32_t           cell_count;
        };

        __host__ __device__ constexpr uint32_t encode_sampled_cell_count(uint32_t cell_count)
        {
            return cell_count == 0 ? 0 : cell_count - 1u;
        }

        __host__ __device__ constexpr uint64_t
            pack_sampled_watchpoint_entry(shadow_access_kind kind,
                                          uint32_t           owner_id,
                                          uint32_t           epoch,
                                          uint32_t           generation,
                                          uint32_t           start_cell,
                                          uint32_t           cell_count,
                                          bool               consumed = false)
        {
            return sampled_watchpoint::valid_mask
                   | (consumed ? sampled_watchpoint::consumed_mask : uint64_t{0})
                   | ((static_cast<uint64_t>(kind)
                       & low_bit_mask(sampled_watchpoint::access_kind_bits))
                      << sampled_watchpoint::access_kind_shift)
                   | ((static_cast<uint64_t>(owner_id)
                       & low_bit_mask(sampled_watchpoint::owner_bits))
                      << sampled_watchpoint::owner_shift)
                   | ((static_cast<uint64_t>(epoch) & low_bit_mask(sampled_watchpoint::epoch_bits))
                      << sampled_watchpoint::epoch_shift)
                   | ((static_cast<uint64_t>(generation)
                       & low_bit_mask(sampled_watchpoint::generation_bits))
                      << sampled_watchpoint::generation_shift)
                   | ((static_cast<uint64_t>(start_cell)
                       & low_bit_mask(sampled_watchpoint::start_bits))
                      << sampled_watchpoint::start_shift)
                   | ((static_cast<uint64_t>(encode_sampled_cell_count(cell_count))
                       & low_bit_mask(sampled_watchpoint::count_bits))
                      << sampled_watchpoint::count_shift);
        }

        __host__ __device__ constexpr sampled_watchpoint_entry
            decode_sampled_watchpoint_entry(uint64_t value)
        {
            return sampled_watchpoint_entry{
                (value & sampled_watchpoint::valid_mask) != 0,
                (value & sampled_watchpoint::consumed_mask) != 0,
                static_cast<shadow_access_kind>((value & sampled_watchpoint::access_kind_mask)
                                                >> sampled_watchpoint::access_kind_shift),
                static_cast<uint32_t>((value & sampled_watchpoint::owner_mask)
                                      >> sampled_watchpoint::owner_shift),
                static_cast<uint32_t>((value & sampled_watchpoint::epoch_mask)
                                      >> sampled_watchpoint::epoch_shift),
                static_cast<uint32_t>((value & sampled_watchpoint::generation_mask)
                                      >> sampled_watchpoint::generation_shift),
                static_cast<uint32_t>((value & sampled_watchpoint::start_mask)
                                      >> sampled_watchpoint::start_shift),
                static_cast<uint32_t>(
                    ((value & sampled_watchpoint::count_mask) >> sampled_watchpoint::count_shift)
                    + 1u),
            };
        }

        __host__ __device__ constexpr uint32_t range_end_cell(uint32_t start_cell,
                                                              uint32_t cell_count)
        {
            return start_cell + cell_count;
        }

        __host__ __device__ constexpr bool cell_ranges_overlap(uint32_t first_start,
                                                               uint32_t first_count,
                                                               uint32_t second_start,
                                                               uint32_t second_count)
        {
            return first_start < range_end_cell(second_start, second_count)
                   && second_start < range_end_cell(first_start, first_count);
        }

        __host__ __device__ constexpr bool
            sampled_watchpoints_conflict(const sampled_watchpoint_entry& current,
                                         const sampled_watchpoint_entry& prior)
        {
            return current.valid && prior.valid && !prior.consumed && current.epoch == prior.epoch
                   && current.generation == prior.generation && current.owner_id != prior.owner_id
                   && shadow_kind_conflicts(current.kind, prior.kind)
                   && cell_ranges_overlap(
                       current.start_cell, current.cell_count, prior.start_cell, prior.cell_count);
        }
    } // namespace detail
} // namespace hip_moi

#endif // HIP_MOI_SHADOW_HPP
