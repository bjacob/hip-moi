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
#include <type_traits>

namespace hip_moi
{
    class site_id
    {
    public:
        __host__ __device__ constexpr site_id() = default;

        explicit __host__ __device__ constexpr site_id(uint64_t value)
            : value_(value)
        {
        }

        __host__ __device__ constexpr uint64_t value() const
        {
            return value_;
        }

        __host__ __device__ constexpr bool allows_coalescing() const
        {
            return value_ != 0;
        }

    private:
        uint64_t value_ = 0;
    };

    inline constexpr site_id no_site_id{};

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
        template <typename Context, typename = void>
        struct optional_coalescing_access_record
        {
            using type                      = unsigned char;
            static constexpr bool available = false;
        };

        template <typename Context>
        struct optional_coalescing_access_record<
            Context,
            std::void_t<typename Context::coalescing_access_record>>
        {
            using type                      = typename Context::coalescing_access_record;
            static constexpr bool available = true;
        };

        template <typename Context, typename = void>
        struct optional_coalescing_group_record
        {
            using type                      = unsigned char;
            static constexpr bool available = false;
        };

        template <typename Context>
        struct optional_coalescing_group_record<
            Context,
            std::void_t<typename Context::coalescing_group_record>>
        {
            using type                      = typename Context::coalescing_group_record;
            static constexpr bool available = true;
        };

        __host__ __device__ constexpr uint64_t fnv1a64(const char* text)
        {
            uint64_t hash = 14695981039346656037ull;
            while(*text)
            {
                hash ^= static_cast<unsigned char>(*text);
                hash *= 1099511628211ull;
                ++text;
            }
            return hash;
        }

        __host__ __device__ constexpr uint64_t mix64(uint64_t value)
        {
            value ^= value >> 30;
            value *= 0xbf58476d1ce4e5b9ull;
            value ^= value >> 27;
            value *= 0x94d049bb133111ebull;
            value ^= value >> 31;
            return value;
        }

        __host__ __device__ constexpr uint64_t
            make_site_id(const char* file, uint32_t line, uint32_t column, uint32_t counter)
        {
            uint64_t value = fnv1a64(file);
            value ^= static_cast<uint64_t>(line) * 0x9e3779b97f4a7c15ull;
            value ^= static_cast<uint64_t>(column) << 32;
            value ^= static_cast<uint64_t>(counter);

            value = mix64(value);
            return value == 0 ? 1 : value;
        }

        template <uint64_t Id>
        struct site_id_constant
        {
            static constexpr uint64_t value = Id;
        };

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

        __host__ __device__ inline bool byte_ranges_overlap(uintptr_t first_address,
                                                            uint32_t  first_byte_count,
                                                            uintptr_t second_address,
                                                            uint32_t  second_byte_count)
        {
            uintptr_t first_end  = first_byte_count > UINTPTR_MAX - first_address
                                       ? UINTPTR_MAX
                                       : first_address + first_byte_count;
            uintptr_t second_end = second_byte_count > UINTPTR_MAX - second_address
                                       ? UINTPTR_MAX
                                       : second_address + second_byte_count;
            return first_address < second_end && second_address < first_end;
        }

        template <typename AccessRecord>
        __device__ inline bool byte_ranges_overlap(const AccessRecord& first,
                                                   const AccessRecord& second)
        {
            return byte_ranges_overlap(
                first.address, first.byte_count, second.address, second.byte_count);
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

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if __has_builtin(__builtin_LINE)
#define HIP_MOI_DETAIL_LINE() __builtin_LINE()
#else
#define HIP_MOI_DETAIL_LINE() __LINE__
#endif

#if __has_builtin(__builtin_COLUMN)
#define HIP_MOI_DETAIL_COLUMN() __builtin_COLUMN()
#else
#define HIP_MOI_DETAIL_COLUMN() 0
#endif

#define HIP_MOI_SITE_ID()                                                    \
    ::hip_moi::site_id(                                                      \
        ::hip_moi::detail::site_id_constant<::hip_moi::detail::make_site_id( \
            __FILE__, HIP_MOI_DETAIL_LINE(), HIP_MOI_DETAIL_COLUMN(), __COUNTER__)>::value)

#endif // HIP_MOI_COMMON_HPP
