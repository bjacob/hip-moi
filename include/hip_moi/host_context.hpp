// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// hip_moi/host_context.hpp
//
// Host-side ownership and reporting helpers for hip-moi metadata.
#ifndef HIP_MOI_HOST_CONTEXT_HPP
#define HIP_MOI_HOST_CONTEXT_HPP

#include "hip_moi/subgroup_level_context.hpp"
#include "hip_moi/thread_level_context.hpp"

#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace hip_moi
{
    inline constexpr std::size_t default_host_context_storage_bytes = 16u * 1024u * 1024u;

    struct host_context_options
    {
        // Byte budget for host-owned device metadata storage. The owning host
        // context partitions this into the typed buffers needed by its mode.
        std::size_t storage_bytes = default_host_context_storage_bytes;

        // Optional typed-capacity overrides. Negative means "derive from the
        // byte budget"; zero disables optional coalescing buffers.
        int access_record_capacity            = -1;
        int coalesced_access_record_capacity  = -1;
        int coalescing_access_record_capacity = -1;
        int coalescing_group_record_capacity  = -1;
        int diagnostic_capacity               = -1;

        // Number of subgroups represented in this context.
        int subgroup_capacity = 64;

        bool destructor_reports = true;
        bool destructor_aborts  = true;

        std::FILE* diagnostic_stream = stderr;
    };

    namespace detail
    {
        struct thread_level_host_context_traits
        {
            using device_context = thread_level_context;

            static const char* name()
            {
                return "host_context";
            }

            static void report_diagnostic(std::FILE*                              stream,
                                          int                                     index,
                                          const thread_level_context::diagnostic& diagnostic_record,
                                          const char*                             kind_name)
            {
                if(static_cast<diagnostic_kind>(diagnostic_record.kind)
                   == diagnostic_kind::barrier_divergence)
                {
                    std::fprintf(stream,
                                 "hip-moi diagnostic %d: kind=%s epoch=%u "
                                 "expected_threads=%u observed_threads=%u "
                                 "site_id=0x%llx\n",
                                 index,
                                 kind_name,
                                 diagnostic_record.epoch,
                                 diagnostic_record.expected_thread_count,
                                 diagnostic_record.observed_thread_count,
                                 static_cast<unsigned long long>(diagnostic_record.first_site_id));
                    return;
                }

                std::fprintf(stream,
                             "hip-moi diagnostic %d: kind=%s epoch=%u "
                             "first_thread=%u second_thread=%u "
                             "first_addr=0x%llx second_addr=0x%llx "
                             "first_size=%u second_size=%u "
                             "first_site_id=0x%llx second_site_id=0x%llx\n",
                             index,
                             kind_name,
                             diagnostic_record.epoch,
                             diagnostic_record.first_thread_id,
                             diagnostic_record.second_thread_id,
                             static_cast<unsigned long long>(diagnostic_record.first_addr),
                             static_cast<unsigned long long>(diagnostic_record.second_addr),
                             diagnostic_record.first_size,
                             diagnostic_record.second_size,
                             static_cast<unsigned long long>(diagnostic_record.first_site_id),
                             static_cast<unsigned long long>(diagnostic_record.second_site_id));
            }
        };

        struct subgroup_level_host_context_traits
        {
            using device_context = subgroup_level_context;

            static const char* name()
            {
                return "subgroup_level_host_context";
            }

            static void
                report_diagnostic(std::FILE*                                stream,
                                  int                                       index,
                                  const subgroup_level_context::diagnostic& diagnostic_record,
                                  const char*                               kind_name)
            {
                if(static_cast<diagnostic_kind>(diagnostic_record.kind)
                   == diagnostic_kind::barrier_divergence)
                {
                    std::fprintf(stream,
                                 "hip-moi diagnostic %d: kind=%s epoch=%u "
                                 "expected_threads=%u observed_threads=%u "
                                 "site_id=0x%llx\n",
                                 index,
                                 kind_name,
                                 diagnostic_record.epoch,
                                 diagnostic_record.expected_thread_count,
                                 diagnostic_record.observed_thread_count,
                                 static_cast<unsigned long long>(diagnostic_record.first_site_id));
                    return;
                }

                std::fprintf(stream,
                             "hip-moi diagnostic %d: kind=%s epoch=%u "
                             "first_subgroup=%u second_subgroup=%u "
                             "first_addr=0x%llx second_addr=0x%llx "
                             "first_size=%u second_size=%u "
                             "first_site_id=0x%llx second_site_id=0x%llx\n",
                             index,
                             kind_name,
                             diagnostic_record.epoch,
                             diagnostic_record.first_subgroup_id,
                             diagnostic_record.second_subgroup_id,
                             static_cast<unsigned long long>(diagnostic_record.first_addr),
                             static_cast<unsigned long long>(diagnostic_record.second_addr),
                             diagnostic_record.first_size,
                             diagnostic_record.second_size,
                             static_cast<unsigned long long>(diagnostic_record.first_site_id),
                             static_cast<unsigned long long>(diagnostic_record.second_site_id));
            }
        };

        template <typename Traits>
        class host_context_impl
        {
        public:
            using device_context = typename Traits::device_context;
            using access_record  = typename device_context::access_record;
            using coalesced_access_record =
                typename optional_coalesced_access_record<device_context>::type;
            using coalescing_access_record =
                typename optional_coalescing_access_record<device_context>::type;
            using coalescing_group_record =
                typename optional_coalescing_group_record<device_context>::type;
            using diagnostic     = typename device_context::diagnostic;
            using storage_ref    = typename device_context::storage_ref;

            static constexpr bool has_coalesced_access_records
                = optional_coalesced_access_record<device_context>::available;
            static constexpr bool has_coalescing_access_records
                = optional_coalescing_access_record<device_context>::available;
            static constexpr bool has_coalescing_group_records
                = optional_coalescing_group_record<device_context>::available;

            explicit host_context_impl(host_context_options options = host_context_options{})
                : options_(options)
                , destructor_reports_(options.destructor_reports)
                , destructor_aborts_(options.destructor_aborts)
                , diagnostic_stream_(options.diagnostic_stream)
            {
                validate_options_or_abort();
                allocate_or_abort();
                clear_device_metadata_or_abort();
            }

            host_context_impl(const host_context_impl&)            = delete;
            host_context_impl& operator=(const host_context_impl&) = delete;

            ~host_context_impl()
            {
                if(!diagnostics_consumed_ && (destructor_reports_ || destructor_aborts_))
                {
                    std::FILE* stream = destructor_reports_ ? diagnostic_stream_ : nullptr;
                    int        count  = consume_diagnostics(stream);
                    if(count != 0 && destructor_aborts_)
                    {
                        if(stream)
                        {
                            std::fprintf(stream,
                                         "hip-moi: unconsumed diagnostics in %s "
                                         "destructor; aborting\n",
                                         Traits::name());
                            std::fflush(stream);
                        }
                        std::abort();
                    }
                }
                release();
            }

            storage_ref device_ref()
            {
                diagnostics_consumed_ = false;
                if constexpr(has_coalescing_access_records)
                {
                    storage_ref ref{
                        access_records_,
                        access_record_capacity_,
                        diagnostics_,
                        diagnostic_capacity_,
                        subgroup_states_,
                        subgroup_capacity_,
                        access_count_,
                        epoch_access_count_,
                        diagnostic_count_,
                        coalesced_access_records_,
                        coalesced_access_record_capacity_,
                        coalesced_access_count_,
                        coalescing_access_records_,
                        coalescing_access_record_capacity_,
                        coalescing_access_count_,
                        epoch_coalescing_access_count_,
                        coalescing_fallback_count_,
                        coalescing_group_records_,
                        coalescing_group_record_capacity_,
                        coalescing_group_count_,
                    };
                    ref.simulated_barrier_arrival_count = simulated_barrier_arrival_count_;
                    return ref;
                }
                else if constexpr(has_coalesced_access_records)
                {
                    storage_ref ref{
                        access_records_,
                        access_record_capacity_,
                        diagnostics_,
                        diagnostic_capacity_,
                        subgroup_states_,
                        subgroup_capacity_,
                        access_count_,
                        epoch_access_count_,
                        diagnostic_count_,
                        coalesced_access_records_,
                        coalesced_access_record_capacity_,
                        coalesced_access_count_,
                    };
                    ref.simulated_barrier_arrival_count = simulated_barrier_arrival_count_;
                    return ref;
                }
                else
                {
                    storage_ref ref{
                        access_records_,
                        access_record_capacity_,
                        diagnostics_,
                        diagnostic_capacity_,
                        subgroup_states_,
                        subgroup_capacity_,
                        access_count_,
                        epoch_access_count_,
                        diagnostic_count_,
                    };
                    ref.simulated_barrier_arrival_count = simulated_barrier_arrival_count_;
                    return ref;
                }
            }

            storage_ref ref()
            {
                return device_ref();
            }

            bool check(std::FILE* stream = stderr)
            {
                return consume_diagnostics(stream) == 0;
            }

            int consume_diagnostics(std::FILE* stream = stderr)
            {
                diagnostics_consumed_ = true;

                hipError_t status = hipDeviceSynchronize();
                if(status != hipSuccess)
                {
                    report_hip_error(stream, "hipDeviceSynchronize", status);
                    last_diagnostic_count_ = -1;
                    return -1;
                }

                int diagnostic_count = 0;
                status               = hipMemcpy(&diagnostic_count,
                                   diagnostic_count_,
                                   sizeof(diagnostic_count),
                                   hipMemcpyDeviceToHost);
                if(status != hipSuccess)
                {
                    report_hip_error(stream, "hipMemcpy(diagnostic_count)", status);
                    last_diagnostic_count_ = -1;
                    return -1;
                }

                last_diagnostic_count_ = diagnostic_count;
                if(diagnostic_count <= 0)
                {
                    return diagnostic_count;
                }

                int stored_count = diagnostic_count;
                if(stored_count > diagnostic_capacity_)
                {
                    stored_count = diagnostic_capacity_;
                }

                diagnostic* host_diagnostics = new diagnostic[stored_count];
                status                       = hipMemcpy(host_diagnostics,
                                   diagnostics_,
                                   stored_count * sizeof(diagnostic),
                                   hipMemcpyDeviceToHost);
                if(status != hipSuccess)
                {
                    delete[] host_diagnostics;
                    report_hip_error(stream, "hipMemcpy(diagnostics)", status);
                    last_diagnostic_count_ = -1;
                    return -1;
                }

                report_diagnostics(stream, host_diagnostics, stored_count, diagnostic_count);
                delete[] host_diagnostics;
                return diagnostic_count;
            }

            int last_diagnostic_count() const
            {
                return last_diagnostic_count_;
            }

            std::size_t storage_bytes() const
            {
                return storage_bytes_;
            }

            std::size_t layout_bytes() const
            {
                return layout_bytes_;
            }

            int access_record_capacity() const
            {
                return access_record_capacity_;
            }

            int coalesced_access_record_capacity() const
            {
                return coalesced_access_record_capacity_;
            }

            int coalescing_access_record_capacity() const
            {
                return coalescing_access_record_capacity_;
            }

            int coalescing_group_record_capacity() const
            {
                return coalescing_group_record_capacity_;
            }

            int diagnostic_capacity() const
            {
                return diagnostic_capacity_;
            }

            int subgroup_capacity() const
            {
                return subgroup_capacity_;
            }

            void set_destructor_reporting_enabled(bool enabled)
            {
                destructor_reports_ = enabled;
            }

            void set_destructor_abort_enabled(bool enabled)
            {
                destructor_aborts_ = enabled;
            }

            void disable_destructor_reporting()
            {
                set_destructor_reporting_enabled(false);
            }

            void disable_destructor_abort()
            {
                set_destructor_abort_enabled(false);
            }

            void disable_destructor_check()
            {
                disable_destructor_reporting();
                disable_destructor_abort();
            }

        private:
            static const char* diagnostic_kind_name(uint32_t kind)
            {
                switch(static_cast<diagnostic_kind>(kind))
                {
                case diagnostic_kind::none:
                    return "none";
                case diagnostic_kind::access_conflict:
                    return "access_conflict";
                case diagnostic_kind::metadata_full:
                    return "metadata_full";
                case diagnostic_kind::barrier_divergence:
                    return "barrier_divergence";
                }
                return "unknown";
            }

            static void
                report_hip_error(std::FILE* stream, const char* operation, hipError_t status)
            {
                if(!stream)
                {
                    return;
                }
                std::fprintf(
                    stream, "hip-moi: %s failed: %s\n", operation, hipGetErrorString(status));
                std::fflush(stream);
            }

            static void report_diagnostics(std::FILE*        stream,
                                           const diagnostic* diagnostics,
                                           int               stored_count,
                                           int               diagnostic_count)
            {
                if(!stream)
                {
                    return;
                }

                std::fprintf(stream, "hip-moi: %d diagnostic(s)\n", diagnostic_count);
                for(int i = 0; i < stored_count; ++i)
                {
                    const diagnostic& diagnostic_record = diagnostics[i];
                    Traits::report_diagnostic(
                        stream, i, diagnostic_record, diagnostic_kind_name(diagnostic_record.kind));
                }
                if(stored_count < diagnostic_count)
                {
                    std::fprintf(stream,
                                 "hip-moi: diagnostic buffer stored %d of %d diagnostic(s)\n",
                                 stored_count,
                                 diagnostic_count);
                }
                std::fflush(stream);
            }

            static bool is_auto_capacity(int capacity)
            {
                return capacity < 0;
            }

            static std::size_t align_up(std::size_t offset, std::size_t alignment)
            {
                std::size_t remainder = offset % alignment;
                return remainder == 0 ? offset : offset + (alignment - remainder);
            }

            static int checked_capacity(std::size_t capacity, const char* name)
            {
                if(capacity > static_cast<std::size_t>(INT_MAX))
                {
                    std::fprintf(stderr,
                                 "hip-moi: %s auto-capacity does not fit in int (%llu)\n",
                                 name,
                                 static_cast<unsigned long long>(capacity));
                    std::fflush(stderr);
                    std::abort();
                }
                return static_cast<int>(capacity);
            }

            template <typename T>
            static void append_slice_size(std::size_t* offset, int capacity)
            {
                if(capacity <= 0)
                {
                    return;
                }

                *offset = align_up(*offset, alignof(T));
                *offset += static_cast<std::size_t>(capacity) * sizeof(T);
            }

            template <typename T>
            static void
                assign_slice(unsigned char* base, int capacity, std::size_t* offset, T** ptr)
            {
                if(capacity <= 0)
                {
                    *ptr = nullptr;
                    return;
                }

                *offset = align_up(*offset, alignof(T));
                *ptr    = reinterpret_cast<T*>(base + *offset);
                *offset += static_cast<std::size_t>(capacity) * sizeof(T);
            }

            void validate_options_or_abort() const
            {
                if(options_.storage_bytes == 0 || options_.access_record_capacity < -1
                   || options_.coalesced_access_record_capacity < -1
                   || options_.coalescing_access_record_capacity < -1
                   || options_.coalescing_group_record_capacity < -1
                   || options_.diagnostic_capacity < -1 || options_.subgroup_capacity <= 0)
                {
                    std::fprintf(stderr,
                                 "hip-moi: %s storage options are invalid "
                                 "(bytes=%llu access=%d coalesced_access=%d "
                                 "coalescing_access=%d group=%d diagnostics=%d subgroups=%d)\n",
                                 Traits::name(),
                                 static_cast<unsigned long long>(options_.storage_bytes),
                                 options_.access_record_capacity,
                                 options_.coalesced_access_record_capacity,
                                 options_.coalescing_access_record_capacity,
                                 options_.coalescing_group_record_capacity,
                                 options_.diagnostic_capacity,
                                 options_.subgroup_capacity);
                    std::fflush(stderr);
                    std::abort();
                }

                if(options_.access_record_capacity == 0 || options_.diagnostic_capacity == 0)
                {
                    std::fprintf(stderr,
                                 "hip-moi: %s access and diagnostic capacities must be "
                                 "positive, or -1 for auto (access=%d diagnostics=%d)\n",
                                 Traits::name(),
                                 options_.access_record_capacity,
                                 options_.diagnostic_capacity);
                    std::fflush(stderr);
                    std::abort();
                }
            }

            int auto_diagnostic_capacity() const
            {
                std::size_t diagnostic_bytes = storage_bytes_ / 64;
                if(diagnostic_bytes < sizeof(diagnostic))
                {
                    diagnostic_bytes = sizeof(diagnostic);
                }

                constexpr std::size_t kMaxAutoDiagnosticBytes = 256u * 1024u;
                if(diagnostic_bytes > kMaxAutoDiagnosticBytes)
                {
                    diagnostic_bytes = kMaxAutoDiagnosticBytes;
                }

                std::size_t capacity = diagnostic_bytes / sizeof(diagnostic);
                if(capacity == 0)
                {
                    capacity = 1;
                }
                return checked_capacity(capacity, "diagnostic");
            }

            std::size_t layout_byte_count() const
            {
                std::size_t offset = 0;
                append_slice_size<access_record>(&offset, access_record_capacity_);
                if constexpr(has_coalesced_access_records)
                {
                    append_slice_size<coalesced_access_record>(&offset,
                                                               coalesced_access_record_capacity_);
                    append_slice_size<int>(&offset, coalesced_access_record_capacity_ > 0 ? 1 : 0);
                }
                if constexpr(has_coalescing_access_records)
                {
                    append_slice_size<coalescing_access_record>(&offset,
                                                                coalescing_access_record_capacity_);
                    append_slice_size<int>(&offset, 3);
                    if constexpr(has_coalescing_group_records)
                    {
                        append_slice_size<coalescing_group_record>(
                            &offset, coalescing_group_record_capacity_);
                        append_slice_size<int>(&offset,
                                               coalescing_group_record_capacity_ > 0 ? 1 : 0);
                    }
                }
                append_slice_size<diagnostic>(&offset, diagnostic_capacity_);
                append_slice_size<subgroup_state>(&offset, subgroup_capacity_);
                append_slice_size<int>(&offset, 4);
                return offset;
            }

            void assign_layout_slices()
            {
                std::size_t offset = 0;
                assign_slice<access_record>(
                    device_storage_, access_record_capacity_, &offset, &access_records_);
                if constexpr(has_coalesced_access_records)
                {
                    assign_slice<coalesced_access_record>(device_storage_,
                                                          coalesced_access_record_capacity_,
                                                          &offset,
                                                          &coalesced_access_records_);
                    assign_slice<int>(device_storage_,
                                      coalesced_access_record_capacity_ > 0 ? 1 : 0,
                                      &offset,
                                      &coalesced_access_count_);
                }
                if constexpr(has_coalescing_access_records)
                {
                    assign_slice<coalescing_access_record>(device_storage_,
                                                           coalescing_access_record_capacity_,
                                                           &offset,
                                                           &coalescing_access_records_);
                    assign_slice<int>(device_storage_, 1, &offset, &coalescing_access_count_);
                    assign_slice<int>(device_storage_, 1, &offset, &epoch_coalescing_access_count_);
                    assign_slice<int>(device_storage_, 1, &offset, &coalescing_fallback_count_);
                    if constexpr(has_coalescing_group_records)
                    {
                        assign_slice<coalescing_group_record>(device_storage_,
                                                              coalescing_group_record_capacity_,
                                                              &offset,
                                                              &coalescing_group_records_);
                        assign_slice<int>(device_storage_,
                                          coalescing_group_record_capacity_ > 0 ? 1 : 0,
                                          &offset,
                                          &coalescing_group_count_);
                    }
                }
                assign_slice<diagnostic>(
                    device_storage_, diagnostic_capacity_, &offset, &diagnostics_);
                assign_slice<subgroup_state>(
                    device_storage_, subgroup_capacity_, &offset, &subgroup_states_);
                assign_slice<int>(device_storage_, 1, &offset, &access_count_);
                assign_slice<int>(device_storage_, 1, &offset, &epoch_access_count_);
                assign_slice<int>(device_storage_, 1, &offset, &diagnostic_count_);
                assign_slice<int>(device_storage_, 1, &offset, &simulated_barrier_arrival_count_);
                layout_bytes_ = offset;
            }

            int capacity_from_bytes(std::size_t byte_count,
                                    std::size_t element_size,
                                    const char* name) const
            {
                return checked_capacity(byte_count / element_size, name);
            }

            bool shrink_one_auto_capacity()
            {
                if constexpr(has_coalescing_group_records)
                {
                    if(is_auto_capacity(options_.coalescing_group_record_capacity)
                       && coalescing_group_record_capacity_ > 0)
                    {
                        --coalescing_group_record_capacity_;
                        return true;
                    }
                }
                if constexpr(has_coalesced_access_records)
                {
                    if(is_auto_capacity(options_.coalesced_access_record_capacity)
                       && coalesced_access_record_capacity_ > 0)
                    {
                        --coalesced_access_record_capacity_;
                        return true;
                    }
                }
                if constexpr(has_coalescing_access_records)
                {
                    if(is_auto_capacity(options_.coalescing_access_record_capacity)
                       && coalescing_access_record_capacity_ > 0)
                    {
                        --coalescing_access_record_capacity_;
                        return true;
                    }
                }
                if(is_auto_capacity(options_.access_record_capacity) && access_record_capacity_ > 1)
                {
                    --access_record_capacity_;
                    return true;
                }
                return false;
            }

            void compute_layout_or_abort()
            {
                storage_bytes_       = options_.storage_bytes;
                subgroup_capacity_   = options_.subgroup_capacity;
                diagnostic_capacity_ = is_auto_capacity(options_.diagnostic_capacity)
                                           ? auto_diagnostic_capacity()
                                           : options_.diagnostic_capacity;

                access_record_capacity_ = is_auto_capacity(options_.access_record_capacity)
                                              ? 0
                                              : options_.access_record_capacity;

                if constexpr(has_coalesced_access_records)
                {
                    coalesced_access_record_capacity_
                        = is_auto_capacity(options_.coalesced_access_record_capacity)
                              ? 0
                              : options_.coalesced_access_record_capacity;
                }
                if constexpr(has_coalescing_access_records)
                {
                    coalescing_access_record_capacity_
                        = is_auto_capacity(options_.coalescing_access_record_capacity)
                              ? 0
                              : options_.coalescing_access_record_capacity;
                }
                if constexpr(has_coalescing_group_records)
                {
                    coalescing_group_record_capacity_
                        = is_auto_capacity(options_.coalescing_group_record_capacity)
                              ? 0
                              : options_.coalescing_group_record_capacity;
                }

                std::size_t fixed_bytes = layout_byte_count();
                if(fixed_bytes > storage_bytes_)
                {
                    report_storage_too_small_and_abort(fixed_bytes);
                }

                std::size_t remaining_bytes = storage_bytes_ - fixed_bytes;
                int         total_weight    = 0;
                if(is_auto_capacity(options_.access_record_capacity))
                {
                    total_weight += 4;
                }
                if constexpr(has_coalescing_access_records)
                {
                    if(is_auto_capacity(options_.coalescing_access_record_capacity))
                    {
                        total_weight += 8;
                    }
                }
                if constexpr(has_coalesced_access_records)
                {
                    if(is_auto_capacity(options_.coalesced_access_record_capacity))
                    {
                        total_weight += 1;
                    }
                }
                if constexpr(has_coalescing_group_records)
                {
                    if(is_auto_capacity(options_.coalescing_group_record_capacity))
                    {
                        total_weight += 1;
                    }
                }

                if(total_weight > 0)
                {
                    if(is_auto_capacity(options_.access_record_capacity))
                    {
                        access_record_capacity_ = capacity_from_bytes(
                            remaining_bytes * 4 / static_cast<std::size_t>(total_weight),
                            sizeof(access_record),
                            "access");
                    }
                    if constexpr(has_coalescing_access_records)
                    {
                        if(is_auto_capacity(options_.coalescing_access_record_capacity))
                        {
                            coalescing_access_record_capacity_ = capacity_from_bytes(
                                remaining_bytes * 8 / static_cast<std::size_t>(total_weight),
                                sizeof(coalescing_access_record),
                                "coalescing_access");
                        }
                    }
                    if constexpr(has_coalesced_access_records)
                    {
                        if(is_auto_capacity(options_.coalesced_access_record_capacity))
                        {
                            coalesced_access_record_capacity_ = capacity_from_bytes(
                                remaining_bytes / static_cast<std::size_t>(total_weight),
                                sizeof(coalesced_access_record),
                                "coalesced_access");
                        }
                    }
                    if constexpr(has_coalescing_group_records)
                    {
                        if(is_auto_capacity(options_.coalescing_group_record_capacity))
                        {
                            coalescing_group_record_capacity_ = capacity_from_bytes(
                                remaining_bytes / static_cast<std::size_t>(total_weight),
                                sizeof(coalescing_group_record),
                                "coalescing_group");
                        }
                    }
                }

                while(layout_byte_count() > storage_bytes_ && shrink_one_auto_capacity())
                {
                }

                layout_bytes_ = layout_byte_count();
                if(layout_bytes_ > storage_bytes_)
                {
                    report_storage_too_small_and_abort(layout_bytes_);
                }
                if(access_record_capacity_ <= 0 || diagnostic_capacity_ <= 0)
                {
                    std::fprintf(stderr,
                                 "hip-moi: %s storage_bytes=%llu is too small for positive "
                                 "access and diagnostic capacities (access=%d diagnostics=%d)\n",
                                 Traits::name(),
                                 static_cast<unsigned long long>(storage_bytes_),
                                 access_record_capacity_,
                                 diagnostic_capacity_);
                    std::fflush(stderr);
                    std::abort();
                }
            }

            void report_storage_too_small_and_abort(std::size_t required_bytes) const
            {
                std::fprintf(stderr,
                             "hip-moi: %s storage_bytes=%llu is too small; "
                             "layout requires at least %llu bytes "
                             "(access=%d coalesced_access=%d coalescing_access=%d "
                             "coalescing_group=%d diagnostics=%d subgroups=%d)\n",
                             Traits::name(),
                             static_cast<unsigned long long>(storage_bytes_),
                             static_cast<unsigned long long>(required_bytes),
                             access_record_capacity_,
                             coalesced_access_record_capacity_,
                             coalescing_access_record_capacity_,
                             coalescing_group_record_capacity_,
                             diagnostic_capacity_,
                             subgroup_capacity_);
                std::fflush(stderr);
                std::abort();
            }

            void allocate_or_abort()
            {
                compute_layout_or_abort();

                void*      allocated = nullptr;
                hipError_t status    = hipMalloc(&allocated, storage_bytes_);
                if(status != hipSuccess)
                {
                    std::fprintf(stderr,
                                 "hip-moi: hipMalloc(%s storage, %llu bytes) failed: %s\n",
                                 Traits::name(),
                                 static_cast<unsigned long long>(storage_bytes_),
                                 hipGetErrorString(status));
                    std::fflush(stderr);
                    std::abort();
                }
                device_storage_ = static_cast<unsigned char*>(allocated);
                assign_layout_slices();
            }

            void clear_device_metadata_or_abort()
            {
                hip_memset_or_abort(access_records_,
                                    0,
                                    access_record_capacity_ * sizeof(access_record),
                                    "access_records");
                if constexpr(has_coalesced_access_records)
                {
                    if(coalesced_access_record_capacity_ > 0)
                    {
                        hip_memset_or_abort(coalesced_access_records_,
                                            0,
                                            coalesced_access_record_capacity_
                                                * sizeof(coalesced_access_record),
                                            "coalesced_access_records");
                        hip_memset_or_abort(
                            coalesced_access_count_, 0, sizeof(int), "coalesced_access_count");
                    }
                }
                if constexpr(has_coalescing_access_records)
                {
                    hip_memset_or_abort(
                        coalescing_access_count_, 0, sizeof(int), "coalescing_access_count");
                    hip_memset_or_abort(epoch_coalescing_access_count_,
                                        0,
                                        sizeof(int),
                                        "epoch_coalescing_access_count");
                    hip_memset_or_abort(
                        coalescing_fallback_count_, 0, sizeof(int), "coalescing_fallback_count");
                    if(coalescing_access_record_capacity_ > 0)
                    {
                        hip_memset_or_abort(coalescing_access_records_,
                                            0,
                                            coalescing_access_record_capacity_
                                                * sizeof(coalescing_access_record),
                                            "coalescing_access_records");
                    }
                    if constexpr(has_coalescing_group_records)
                    {
                        if(coalescing_group_record_capacity_ > 0)
                        {
                            hip_memset_or_abort(coalescing_group_records_,
                                                0,
                                                coalescing_group_record_capacity_
                                                    * sizeof(coalescing_group_record),
                                                "coalescing_group_records");
                            hip_memset_or_abort(
                                coalescing_group_count_, 0, sizeof(int), "coalescing_group_count");
                        }
                    }
                }
                hip_memset_or_abort(
                    diagnostics_, 0, diagnostic_capacity_ * sizeof(diagnostic), "diagnostics");
                hip_memset_or_abort(subgroup_states_,
                                    0,
                                    subgroup_capacity_ * sizeof(subgroup_state),
                                    "subgroup_states");
                hip_memset_or_abort(access_count_, 0, sizeof(int), "access_count");
                hip_memset_or_abort(epoch_access_count_, 0, sizeof(int), "epoch_access_count");
                hip_memset_or_abort(diagnostic_count_, 0, sizeof(int), "diagnostic_count");
                hip_memset_or_abort(simulated_barrier_arrival_count_,
                                    0,
                                    sizeof(int),
                                    "simulated_barrier_arrival_count");
            }

            static void
                hip_memset_or_abort(void* ptr, int value, size_t byte_count, const char* name)
            {
                if(byte_count == 0)
                {
                    return;
                }

                hipError_t status = hipMemset(ptr, value, byte_count);
                if(status != hipSuccess)
                {
                    std::fprintf(stderr,
                                 "hip-moi: hipMemset(%s, %llu bytes) failed: %s\n",
                                 name,
                                 static_cast<unsigned long long>(byte_count),
                                 hipGetErrorString(status));
                    std::fflush(stderr);
                    std::abort();
                }
            }

            void release()
            {
                if(device_storage_)
                {
                    (void)hipFree(device_storage_);
                    device_storage_ = nullptr;
                }
                access_records_                = nullptr;
                diagnostics_                   = nullptr;
                coalesced_access_records_      = nullptr;
                coalescing_access_records_     = nullptr;
                coalescing_group_records_      = nullptr;
                subgroup_states_               = nullptr;
                access_count_                  = nullptr;
                epoch_access_count_            = nullptr;
                diagnostic_count_              = nullptr;
                simulated_barrier_arrival_count_ = nullptr;
                coalesced_access_count_        = nullptr;
                coalescing_access_count_       = nullptr;
                epoch_coalescing_access_count_ = nullptr;
                coalescing_fallback_count_     = nullptr;
                coalescing_group_count_        = nullptr;
            }

            host_context_options options_;
            bool                 diagnostics_consumed_  = true;
            bool                 destructor_reports_    = true;
            bool                 destructor_aborts_     = true;
            std::FILE*           diagnostic_stream_     = stderr;
            int                  last_diagnostic_count_ = 0;
            std::size_t               storage_bytes_                     = 0;
            std::size_t               layout_bytes_                      = 0;
            int                       access_record_capacity_            = 0;
            int                       coalesced_access_record_capacity_  = 0;
            int                       coalescing_access_record_capacity_ = 0;
            int                       coalescing_group_record_capacity_  = 0;
            int                       diagnostic_capacity_               = 0;
            int                       subgroup_capacity_                 = 0;
            unsigned char*            device_storage_                    = nullptr;
            access_record*       access_records_        = nullptr;
            coalesced_access_record* coalesced_access_records_ = nullptr;
            coalescing_access_record* coalescing_access_records_     = nullptr;
            coalescing_group_record* coalescing_group_records_     = nullptr;
            diagnostic*          diagnostics_           = nullptr;
            subgroup_state*      subgroup_states_       = nullptr;
            int*                 access_count_          = nullptr;
            int*                 epoch_access_count_    = nullptr;
            int*                 diagnostic_count_      = nullptr;
            int*                      simulated_barrier_arrival_count_   = nullptr;
            int*                     coalesced_access_count_   = nullptr;
            int*                      coalescing_access_count_       = nullptr;
            int*                      epoch_coalescing_access_count_ = nullptr;
            int*                      coalescing_fallback_count_     = nullptr;
            int*                     coalescing_group_count_       = nullptr;
        };
    } // namespace detail

    using thread_level_host_context
        = detail::host_context_impl<detail::thread_level_host_context_traits>;
    using subgroup_level_host_context
        = detail::host_context_impl<detail::subgroup_level_host_context_traits>;
    using host_context = thread_level_host_context;

    template <typename HostContext>
    inline void check_or_abort(HostContext& context, const char* file, int line)
    {
        if(context.check(stderr))
        {
            return;
        }

        std::fprintf(stderr, "hip-moi: HIP_MOI_CHECK failed at %s:%d\n", file, line);
        std::fflush(stderr);
        std::abort();
    }
} // namespace hip_moi

#define HIP_MOI_CHECK(context) ::hip_moi::check_or_abort((context), __FILE__, __LINE__)

#endif // HIP_MOI_HOST_CONTEXT_HPP
