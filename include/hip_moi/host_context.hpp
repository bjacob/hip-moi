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

#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace hip_moi
{
    struct host_context_options
    {
        int access_record_capacity = 1024;
        int coalesced_access_record_capacity = 1024;
        int coalescing_access_record_capacity = 0;
        int coalescing_group_record_capacity = 0;
        int diagnostic_capacity    = 64;
        int subgroup_capacity      = 1;

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
                std::fprintf(stream,
                             "hip-moi diagnostic %d: kind=%s epoch=%u "
                             "first_thread=%u second_thread=%u "
                             "first_addr=0x%llx second_addr=0x%llx "
                             "first_size=%u second_size=%u "
                             "first_site=0x%llx second_site=0x%llx\n",
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
                std::fprintf(stream,
                             "hip-moi diagnostic %d: kind=%s epoch=%u "
                             "first_subgroup=%u second_subgroup=%u "
                             "first_addr=0x%llx second_addr=0x%llx "
                             "first_size=%u second_size=%u "
                             "first_site=0x%llx second_site=0x%llx\n",
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
                    return storage_ref{
                        access_records_,
                        options_.access_record_capacity,
                        diagnostics_,
                        options_.diagnostic_capacity,
                        subgroup_states_,
                        options_.subgroup_capacity,
                        access_count_,
                        epoch_access_count_,
                        diagnostic_count_,
                        coalesced_access_records_,
                        options_.coalesced_access_record_capacity,
                        coalesced_access_count_,
                        coalescing_access_records_,
                        options_.coalescing_access_record_capacity,
                        coalescing_access_count_,
                        epoch_coalescing_access_count_,
                        coalescing_fallback_count_,
                        coalescing_group_records_,
                        options_.coalescing_group_record_capacity,
                        coalescing_group_count_,
                    };
                }
                else if constexpr(has_coalesced_access_records)
                {
                    return storage_ref{
                        access_records_,
                        options_.access_record_capacity,
                        diagnostics_,
                        options_.diagnostic_capacity,
                        subgroup_states_,
                        options_.subgroup_capacity,
                        access_count_,
                        epoch_access_count_,
                        diagnostic_count_,
                        coalesced_access_records_,
                        options_.coalesced_access_record_capacity,
                        coalesced_access_count_,
                    };
                }
                else
                {
                    return storage_ref{
                        access_records_,
                        options_.access_record_capacity,
                        diagnostics_,
                        options_.diagnostic_capacity,
                        subgroup_states_,
                        options_.subgroup_capacity,
                        access_count_,
                        epoch_access_count_,
                        diagnostic_count_,
                    };
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
                if(stored_count > options_.diagnostic_capacity)
                {
                    stored_count = options_.diagnostic_capacity;
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

            void validate_options_or_abort() const
            {
                if(options_.access_record_capacity <= 0
                   || options_.coalescing_access_record_capacity < 0
                   || options_.coalescing_group_record_capacity < 0
                   || options_.diagnostic_capacity <= 0 || options_.subgroup_capacity <= 0)
                {
                    std::fprintf(stderr,
                                 "hip-moi: %s capacities must all be positive "
                                 "except coalescing-access/group capacities, which may be zero "
                                 "(access=%d coalesced_access=%d coalescing_access=%d group=%d "
                                 "diagnostics=%d subgroups=%d)\n",
                                 Traits::name(),
                                 options_.access_record_capacity,
                                 options_.coalesced_access_record_capacity,
                                 options_.coalescing_access_record_capacity,
                                 options_.coalescing_group_record_capacity,
                                 options_.diagnostic_capacity,
                                 options_.subgroup_capacity);
                    std::fflush(stderr);
                    std::abort();
                }
                if constexpr(has_coalesced_access_records)
                {
                    if(options_.coalesced_access_record_capacity <= 0)
                    {
                        std::fprintf(stderr,
                                     "hip-moi: %s coalesced access capacity must be positive "
                                     "(coalesced_access=%d)\n",
                                     Traits::name(),
                                     options_.coalesced_access_record_capacity);
                        std::fflush(stderr);
                        std::abort();
                    }
                }
            }

            void allocate_or_abort()
            {
                hip_allocate_or_abort(&access_records_,
                                      options_.access_record_capacity * sizeof(access_record),
                                      "access_records");
                if constexpr(has_coalesced_access_records)
                {
                    hip_allocate_or_abort(&coalesced_access_records_,
                                          options_.coalesced_access_record_capacity
                                              * sizeof(coalesced_access_record),
                                          "coalesced_access_records");
                    hip_allocate_or_abort(
                        &coalesced_access_count_, sizeof(int), "coalesced_access_count");
                }
                if constexpr(has_coalescing_access_records)
                {
                    hip_allocate_or_abort(
                        &coalescing_access_count_, sizeof(int), "coalescing_access_count");
                    hip_allocate_or_abort(&epoch_coalescing_access_count_,
                                          sizeof(int),
                                          "epoch_coalescing_access_count");
                    hip_allocate_or_abort(
                        &coalescing_fallback_count_, sizeof(int), "coalescing_fallback_count");
                    if(options_.coalescing_access_record_capacity > 0)
                    {
                        hip_allocate_or_abort(&coalescing_access_records_,
                                              options_.coalescing_access_record_capacity
                                                  * sizeof(coalescing_access_record),
                                              "coalescing_access_records");
                    }
                    if constexpr(has_coalescing_group_records)
                    {
                        if(options_.coalescing_group_record_capacity > 0)
                        {
                            hip_allocate_or_abort(&coalescing_group_records_,
                                                  options_.coalescing_group_record_capacity
                                                      * sizeof(coalescing_group_record),
                                                  "coalescing_group_records");
                            hip_allocate_or_abort(
                                &coalescing_group_count_, sizeof(int), "coalescing_group_count");
                        }
                    }
                }
                hip_allocate_or_abort(&diagnostics_,
                                      options_.diagnostic_capacity * sizeof(diagnostic),
                                      "diagnostics");
                hip_allocate_or_abort(&subgroup_states_,
                                      options_.subgroup_capacity * sizeof(subgroup_state),
                                      "subgroup_states");
                hip_allocate_or_abort(&access_count_, sizeof(int), "access_count");
                hip_allocate_or_abort(&epoch_access_count_, sizeof(int), "epoch_access_count");
                hip_allocate_or_abort(&diagnostic_count_, sizeof(int), "diagnostic_count");
            }

            void clear_device_metadata_or_abort()
            {
                hip_memset_or_abort(access_records_,
                                    0,
                                    options_.access_record_capacity * sizeof(access_record),
                                    "access_records");
                if constexpr(has_coalesced_access_records)
                {
                    hip_memset_or_abort(coalesced_access_records_,
                                        0,
                                        options_.coalesced_access_record_capacity
                                            * sizeof(coalesced_access_record),
                                        "coalesced_access_records");
                    hip_memset_or_abort(
                        coalesced_access_count_, 0, sizeof(int), "coalesced_access_count");
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
                    if(options_.coalescing_access_record_capacity > 0)
                    {
                        hip_memset_or_abort(coalescing_access_records_,
                                            0,
                                            options_.coalescing_access_record_capacity
                                                * sizeof(coalescing_access_record),
                                            "coalescing_access_records");
                    }
                    if constexpr(has_coalescing_group_records)
                    {
                        if(options_.coalescing_group_record_capacity > 0)
                        {
                            hip_memset_or_abort(coalescing_group_records_,
                                                0,
                                                options_.coalescing_group_record_capacity
                                                    * sizeof(coalescing_group_record),
                                                "coalescing_group_records");
                            hip_memset_or_abort(
                                coalescing_group_count_, 0, sizeof(int), "coalescing_group_count");
                        }
                    }
                }
                hip_memset_or_abort(diagnostics_,
                                    0,
                                    options_.diagnostic_capacity * sizeof(diagnostic),
                                    "diagnostics");
                hip_memset_or_abort(subgroup_states_,
                                    0,
                                    options_.subgroup_capacity * sizeof(subgroup_state),
                                    "subgroup_states");
                hip_memset_or_abort(access_count_, 0, sizeof(int), "access_count");
                hip_memset_or_abort(epoch_access_count_, 0, sizeof(int), "epoch_access_count");
                hip_memset_or_abort(diagnostic_count_, 0, sizeof(int), "diagnostic_count");
            }

            template <typename T>
            static void hip_allocate_or_abort(T** ptr, size_t byte_count, const char* name)
            {
                void*      allocated = nullptr;
                hipError_t status    = hipMalloc(&allocated, byte_count);
                if(status != hipSuccess)
                {
                    std::fprintf(stderr,
                                 "hip-moi: hipMalloc(%s, %llu bytes) failed: %s\n",
                                 name,
                                 static_cast<unsigned long long>(byte_count),
                                 hipGetErrorString(status));
                    std::fflush(stderr);
                    std::abort();
                }
                *ptr = static_cast<T*>(allocated);
            }

            static void
                hip_memset_or_abort(void* ptr, int value, size_t byte_count, const char* name)
            {
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
                if(access_records_)
                {
                    (void)hipFree(access_records_);
                    access_records_ = nullptr;
                }
                if(diagnostics_)
                {
                    (void)hipFree(diagnostics_);
                    diagnostics_ = nullptr;
                }
                if(coalesced_access_records_)
                {
                    (void)hipFree(coalesced_access_records_);
                    coalesced_access_records_ = nullptr;
                }
                if(coalescing_access_records_)
                {
                    (void)hipFree(coalescing_access_records_);
                    coalescing_access_records_ = nullptr;
                }
                if(coalescing_group_records_)
                {
                    (void)hipFree(coalescing_group_records_);
                    coalescing_group_records_ = nullptr;
                }
                if(subgroup_states_)
                {
                    (void)hipFree(subgroup_states_);
                    subgroup_states_ = nullptr;
                }
                if(access_count_)
                {
                    (void)hipFree(access_count_);
                    access_count_ = nullptr;
                }
                if(epoch_access_count_)
                {
                    (void)hipFree(epoch_access_count_);
                    epoch_access_count_ = nullptr;
                }
                if(diagnostic_count_)
                {
                    (void)hipFree(diagnostic_count_);
                    diagnostic_count_ = nullptr;
                }
                if(coalesced_access_count_)
                {
                    (void)hipFree(coalesced_access_count_);
                    coalesced_access_count_ = nullptr;
                }
                if(coalescing_access_count_)
                {
                    (void)hipFree(coalescing_access_count_);
                    coalescing_access_count_ = nullptr;
                }
                if(epoch_coalescing_access_count_)
                {
                    (void)hipFree(epoch_coalescing_access_count_);
                    epoch_coalescing_access_count_ = nullptr;
                }
                if(coalescing_fallback_count_)
                {
                    (void)hipFree(coalescing_fallback_count_);
                    coalescing_fallback_count_ = nullptr;
                }
                if(coalescing_group_count_)
                {
                    (void)hipFree(coalescing_group_count_);
                    coalescing_group_count_ = nullptr;
                }
            }

            host_context_options options_;
            bool                 diagnostics_consumed_  = true;
            bool                 destructor_reports_    = true;
            bool                 destructor_aborts_     = true;
            std::FILE*           diagnostic_stream_     = stderr;
            int                  last_diagnostic_count_ = 0;
            access_record*       access_records_        = nullptr;
            coalesced_access_record* coalesced_access_records_ = nullptr;
            coalescing_access_record* coalescing_access_records_     = nullptr;
            coalescing_group_record* coalescing_group_records_     = nullptr;
            diagnostic*          diagnostics_           = nullptr;
            subgroup_state*      subgroup_states_       = nullptr;
            int*                 access_count_          = nullptr;
            int*                 epoch_access_count_    = nullptr;
            int*                 diagnostic_count_      = nullptr;
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
