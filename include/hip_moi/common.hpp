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
        none               = 0,
        access_conflict    = 1,
        metadata_full      = 2,
        barrier_divergence = 3,
    };

    enum class backend_kind : uint32_t
    {
        exact_shadow       = 0,
        sampled_watchpoint = 1,
    };

    enum class atomic_memory_order : uint32_t
    {
        relaxed = 0,
        acquire = 1,
        release = 2,
        acq_rel = 3,
        seq_cst = 4,
    };

    enum class atomic_memory_scope : uint32_t
    {
        workgroup = 0,
        agent     = 1,
        system    = 2,
    };

    template <uint32_t SampleSkip               = 1,
              uint32_t ProbeCount               = 1,
              uint32_t DelayIters               = 0,
              bool     ReportConflicts          = true,
              uint32_t StaticWatchpointCapacity = 0>
    struct sampled_watchpoint_policy
    {
        static constexpr uint32_t sample_skip                = SampleSkip;
        static constexpr uint32_t probe_count                = ProbeCount;
        static constexpr uint32_t delay_iters                = DelayIters;
        static constexpr bool     report_conflicts           = ReportConflicts;
        static constexpr uint32_t static_watchpoint_capacity = StaticWatchpointCapacity;
    };

    struct subgroup_state
    {
        uint32_t epoch;
    };

    namespace detail
    {
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

        __host__ __device__ constexpr uint32_t mix32(uint32_t value)
        {
            value ^= value >> 16;
            value *= 0x7feb352du;
            value ^= value >> 15;
            value *= 0x846ca68bu;
            value ^= value >> 16;
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

        template <typename T>
        struct is_supported_atomic_value
        {
            static constexpr bool value
                = std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8);
        };

        template <int Scope, typename T>
        __device__ inline T atomic_load_with_scope(T* ptr, atomic_memory_order order)
        {
            switch(order)
            {
            case atomic_memory_order::relaxed:
                return __hip_atomic_load(ptr, __ATOMIC_RELAXED, Scope);
            case atomic_memory_order::acquire:
                return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, Scope);
            case atomic_memory_order::seq_cst:
                return __hip_atomic_load(ptr, __ATOMIC_SEQ_CST, Scope);
            case atomic_memory_order::release:
            case atomic_memory_order::acq_rel:
                return __hip_atomic_load(ptr, __ATOMIC_SEQ_CST, Scope);
            }
            return __hip_atomic_load(ptr, __ATOMIC_SEQ_CST, Scope);
        }

        template <int Scope, typename T>
        __device__ inline void atomic_store_with_scope(T* ptr, T value, atomic_memory_order order)
        {
            switch(order)
            {
            case atomic_memory_order::relaxed:
                __hip_atomic_store(ptr, value, __ATOMIC_RELAXED, Scope);
                return;
            case atomic_memory_order::release:
                __hip_atomic_store(ptr, value, __ATOMIC_RELEASE, Scope);
                return;
            case atomic_memory_order::seq_cst:
                __hip_atomic_store(ptr, value, __ATOMIC_SEQ_CST, Scope);
                return;
            case atomic_memory_order::acquire:
            case atomic_memory_order::acq_rel:
                __hip_atomic_store(ptr, value, __ATOMIC_SEQ_CST, Scope);
                return;
            }
            __hip_atomic_store(ptr, value, __ATOMIC_SEQ_CST, Scope);
        }

        template <int Scope, typename T>
        __device__ inline T atomic_fetch_add_with_scope(T* ptr, T value, atomic_memory_order order)
        {
            switch(order)
            {
            case atomic_memory_order::relaxed:
                return __hip_atomic_fetch_add(ptr, value, __ATOMIC_RELAXED, Scope);
            case atomic_memory_order::acquire:
                return __hip_atomic_fetch_add(ptr, value, __ATOMIC_ACQUIRE, Scope);
            case atomic_memory_order::release:
                return __hip_atomic_fetch_add(ptr, value, __ATOMIC_RELEASE, Scope);
            case atomic_memory_order::acq_rel:
                return __hip_atomic_fetch_add(ptr, value, __ATOMIC_ACQ_REL, Scope);
            case atomic_memory_order::seq_cst:
                return __hip_atomic_fetch_add(ptr, value, __ATOMIC_SEQ_CST, Scope);
            }
            return __hip_atomic_fetch_add(ptr, value, __ATOMIC_SEQ_CST, Scope);
        }

        template <int Scope, typename T>
        __device__ inline T atomic_fetch_or_with_scope(T* ptr, T value, atomic_memory_order order)
        {
            switch(order)
            {
            case atomic_memory_order::relaxed:
                return __hip_atomic_fetch_or(ptr, value, __ATOMIC_RELAXED, Scope);
            case atomic_memory_order::acquire:
                return __hip_atomic_fetch_or(ptr, value, __ATOMIC_ACQUIRE, Scope);
            case atomic_memory_order::release:
                return __hip_atomic_fetch_or(ptr, value, __ATOMIC_RELEASE, Scope);
            case atomic_memory_order::acq_rel:
                return __hip_atomic_fetch_or(ptr, value, __ATOMIC_ACQ_REL, Scope);
            case atomic_memory_order::seq_cst:
                return __hip_atomic_fetch_or(ptr, value, __ATOMIC_SEQ_CST, Scope);
            }
            return __hip_atomic_fetch_or(ptr, value, __ATOMIC_SEQ_CST, Scope);
        }

        template <typename T>
        __device__ inline T
            atomic_load_dispatch(T* ptr, atomic_memory_order order, atomic_memory_scope scope)
        {
            switch(scope)
            {
            case atomic_memory_scope::workgroup:
                return atomic_load_with_scope<__HIP_MEMORY_SCOPE_WORKGROUP>(ptr, order);
            case atomic_memory_scope::agent:
                return atomic_load_with_scope<__HIP_MEMORY_SCOPE_AGENT>(ptr, order);
            case atomic_memory_scope::system:
                return atomic_load_with_scope<__HIP_MEMORY_SCOPE_SYSTEM>(ptr, order);
            }
            return atomic_load_with_scope<__HIP_MEMORY_SCOPE_AGENT>(ptr, order);
        }

        template <typename T>
        __device__ inline void atomic_store_dispatch(T*                  ptr,
                                                     T                   value,
                                                     atomic_memory_order order,
                                                     atomic_memory_scope scope)
        {
            switch(scope)
            {
            case atomic_memory_scope::workgroup:
                atomic_store_with_scope<__HIP_MEMORY_SCOPE_WORKGROUP>(ptr, value, order);
                return;
            case atomic_memory_scope::agent:
                atomic_store_with_scope<__HIP_MEMORY_SCOPE_AGENT>(ptr, value, order);
                return;
            case atomic_memory_scope::system:
                atomic_store_with_scope<__HIP_MEMORY_SCOPE_SYSTEM>(ptr, value, order);
                return;
            }
            atomic_store_with_scope<__HIP_MEMORY_SCOPE_AGENT>(ptr, value, order);
        }

        template <typename T>
        __device__ inline T atomic_fetch_add_dispatch(T*                  ptr,
                                                      T                   value,
                                                      atomic_memory_order order,
                                                      atomic_memory_scope scope)
        {
            switch(scope)
            {
            case atomic_memory_scope::workgroup:
                return atomic_fetch_add_with_scope<__HIP_MEMORY_SCOPE_WORKGROUP>(ptr, value, order);
            case atomic_memory_scope::agent:
                return atomic_fetch_add_with_scope<__HIP_MEMORY_SCOPE_AGENT>(ptr, value, order);
            case atomic_memory_scope::system:
                return atomic_fetch_add_with_scope<__HIP_MEMORY_SCOPE_SYSTEM>(ptr, value, order);
            }
            return atomic_fetch_add_with_scope<__HIP_MEMORY_SCOPE_AGENT>(ptr, value, order);
        }

        template <typename T>
        __device__ inline T atomic_fetch_or_dispatch(T*                  ptr,
                                                     T                   value,
                                                     atomic_memory_order order,
                                                     atomic_memory_scope scope)
        {
            switch(scope)
            {
            case atomic_memory_scope::workgroup:
                return atomic_fetch_or_with_scope<__HIP_MEMORY_SCOPE_WORKGROUP>(ptr, value, order);
            case atomic_memory_scope::agent:
                return atomic_fetch_or_with_scope<__HIP_MEMORY_SCOPE_AGENT>(ptr, value, order);
            case atomic_memory_scope::system:
                return atomic_fetch_or_with_scope<__HIP_MEMORY_SCOPE_SYSTEM>(ptr, value, order);
            }
            return atomic_fetch_or_with_scope<__HIP_MEMORY_SCOPE_AGENT>(ptr, value, order);
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
