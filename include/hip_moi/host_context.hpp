// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// hip_moi/host_context.hpp
//
// Host-side ownership and reporting helpers for hip-moi metadata.
#ifndef HIP_MOI_HOST_CONTEXT_HPP
#define HIP_MOI_HOST_CONTEXT_HPP

#include "hip_moi/context.hpp"

#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace hip_moi
{
    inline constexpr std::size_t default_host_context_storage_bytes = 16u * 1024u * 1024u;

    struct host_context_options
    {
        // Byte budget for host-owned device metadata storage.
        std::size_t storage_bytes = default_host_context_storage_bytes;

        // Negative means "derive from storage_bytes". A positive value fixes
        // the capacity. Zero disables that storage class.
        int exact_shadow_entry_capacity = -1;
        int sampled_watchpoint_capacity = -1;
        int diagnostic_capacity         = -1;

        // Sampled-watchpoint tuning. A skip of 1 samples every selected static
        // site/subgroup instance; larger values randomly thin those instances.
        // A probe count of 0 scans the whole watchpoint table.
        uint32_t sampled_watchpoint_sample_skip = 1;
        uint32_t sampled_watchpoint_probe_count = 1;
        uint32_t sampled_watchpoint_delay_iters = 0;
        bool     sampled_watchpoint_reports     = true;

        // Number of subgroups represented in this context.
        int subgroup_capacity = 64;

        backend_kind backend = backend_kind::exact_shadow;

        bool destructor_reports = true;
        bool destructor_aborts  = true;

        std::FILE* diagnostic_stream = stderr;
    };

    inline host_context_options make_exact_shadow_options()
    {
        host_context_options options;
        options.backend = backend_kind::exact_shadow;
        return options;
    }

    inline host_context_options make_sampled_watchpoint_reporting_options()
    {
        host_context_options options;
        options.backend                     = backend_kind::sampled_watchpoint;
        options.sampled_watchpoint_reports  = true;
        options.exact_shadow_entry_capacity = 0;
        options.sampled_watchpoint_capacity = -1;
        return options;
    }

    inline host_context_options make_sampled_watchpoint_publish_only_options()
    {
        host_context_options options       = make_sampled_watchpoint_reporting_options();
        options.sampled_watchpoint_reports = false;
        return options;
    }

    inline host_context_options make_one_watchpoint_publish_only_options()
    {
        host_context_options options           = make_sampled_watchpoint_publish_only_options();
        options.sampled_watchpoint_capacity    = 1;
        options.sampled_watchpoint_sample_skip = 32;
        options.sampled_watchpoint_probe_count = 1;
        options.sampled_watchpoint_delay_iters = 32;
        return options;
    }

    class host_context
    {
    public:
        using device_context = context;
        using diagnostic     = context::diagnostic;
        using storage_ref    = context::storage_ref;

        explicit host_context(host_context_options options = host_context_options{})
            : options_(options)
            , destructor_reports_(options.destructor_reports)
            , destructor_aborts_(options.destructor_aborts)
            , diagnostic_stream_(options.diagnostic_stream)
        {
            validate_options_or_abort();
            allocate_or_abort();
            clear_device_metadata_or_abort();
        }

        host_context(const host_context&)            = delete;
        host_context& operator=(const host_context&) = delete;

        ~host_context()
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
                                     "hip-moi: unconsumed diagnostics in host_context "
                                     "destructor; aborting\n");
                        std::fflush(stream);
                    }
                    std::abort();
                }
            }
            release();
        }

        storage_ref launch_ref()
        {
            diagnostics_consumed_ = false;
            uint64_t generation   = next_generation();
            return storage_ref{
                diagnostics_,
                diagnostic_capacity_,
                subgroup_states_,
                subgroup_capacity_,
                diagnostic_count_,
                simulated_barrier_arrival_count_,
                exact_shadow_entries_,
                exact_shadow_entry_capacity_,
                sampled_watchpoints_,
                sampled_watchpoint_capacity_,
                generation,
                static_cast<uint32_t>(options_.backend),
                options_.sampled_watchpoint_sample_skip,
                options_.sampled_watchpoint_probe_count,
                options_.sampled_watchpoint_delay_iters,
                options_.sampled_watchpoint_reports ? 1u : 0u,
                atomic_objects_,
                atomic_object_capacity_,
                acquired_subgroup_epoch_tokens_,
            };
        }

        storage_ref device_ref()
        {
            return launch_ref();
        }

        storage_ref ref()
        {
            return launch_ref();
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

        int exact_shadow_entry_capacity() const
        {
            return exact_shadow_entry_capacity_;
        }

        int sampled_watchpoint_capacity() const
        {
            return sampled_watchpoint_capacity_;
        }

        int diagnostic_capacity() const
        {
            return diagnostic_capacity_;
        }

        int subgroup_capacity() const
        {
            return subgroup_capacity_;
        }

        int atomic_object_capacity() const
        {
            return atomic_object_capacity_;
        }

        backend_kind backend() const
        {
            return options_.backend;
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
        static void assign_slice(unsigned char* base, int capacity, std::size_t* offset, T** ptr)
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

        std::size_t acquired_subgroup_epoch_token_count() const
        {
            return static_cast<std::size_t>(subgroup_capacity_)
                   * static_cast<std::size_t>(subgroup_capacity_);
        }

        uint64_t next_generation()
        {
            ++generation_;
            if(generation_ == 0)
            {
                generation_ = 1;
            }
            return generation_;
        }

        void validate_options_or_abort() const
        {
            if(options_.storage_bytes == 0 || options_.exact_shadow_entry_capacity < -1
               || options_.sampled_watchpoint_capacity < -1 || options_.diagnostic_capacity < -1
               || options_.subgroup_capacity <= 0)
            {
                std::fprintf(stderr,
                             "hip-moi: host_context storage options are invalid "
                             "(bytes=%llu exact_shadow=%d sampled_watchpoints=%d "
                             "diagnostics=%d subgroups=%d)\n",
                             static_cast<unsigned long long>(options_.storage_bytes),
                             options_.exact_shadow_entry_capacity,
                             options_.sampled_watchpoint_capacity,
                             options_.diagnostic_capacity,
                             options_.subgroup_capacity);
                std::fflush(stderr);
                std::abort();
            }

            if(options_.diagnostic_capacity == 0)
            {
                std::fprintf(stderr,
                             "hip-moi: host_context diagnostic capacity must be positive, "
                             "or -1 for auto\n");
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

        int auto_atomic_object_capacity() const
        {
            std::size_t atomic_bytes = storage_bytes_ / 64;
            if(atomic_bytes < sizeof(context::atomic_object_record))
            {
                atomic_bytes = sizeof(context::atomic_object_record);
            }

            constexpr std::size_t kMaxAutoAtomicObjectBytes = 256u * 1024u;
            if(atomic_bytes > kMaxAutoAtomicObjectBytes)
            {
                atomic_bytes = kMaxAutoAtomicObjectBytes;
            }

            std::size_t capacity = atomic_bytes / sizeof(context::atomic_object_record);
            if(capacity == 0)
            {
                capacity = 1;
            }
            return checked_capacity(capacity, "atomic_object");
        }

        std::size_t layout_byte_count() const
        {
            std::size_t offset = 0;
            append_slice_size<diagnostic>(&offset, diagnostic_capacity_);
            append_slice_size<subgroup_state>(&offset, subgroup_capacity_);
            append_slice_size<int>(&offset, 2);
            append_slice_size<context::atomic_object_record>(&offset, atomic_object_capacity_);
            append_slice_size<uint32_t>(
                &offset, checked_capacity(acquired_subgroup_epoch_token_count(), "acquired_epoch"));
            append_slice_size<uint64_t>(&offset, exact_shadow_entry_capacity_);
            append_slice_size<uint64_t>(&offset, sampled_watchpoint_capacity_);
            return offset;
        }

        void assign_layout_slices()
        {
            std::size_t offset = 0;
            assign_slice<diagnostic>(device_storage_, diagnostic_capacity_, &offset, &diagnostics_);
            assign_slice<subgroup_state>(
                device_storage_, subgroup_capacity_, &offset, &subgroup_states_);
            assign_slice<int>(device_storage_, 1, &offset, &diagnostic_count_);
            assign_slice<int>(device_storage_, 1, &offset, &simulated_barrier_arrival_count_);
            assign_slice<context::atomic_object_record>(
                device_storage_, atomic_object_capacity_, &offset, &atomic_objects_);
            assign_slice<uint32_t>(
                device_storage_,
                checked_capacity(acquired_subgroup_epoch_token_count(), "acquired_epoch"),
                &offset,
                &acquired_subgroup_epoch_tokens_);
            assign_slice<uint64_t>(
                device_storage_, exact_shadow_entry_capacity_, &offset, &exact_shadow_entries_);
            assign_slice<uint64_t>(
                device_storage_, sampled_watchpoint_capacity_, &offset, &sampled_watchpoints_);
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
            if(is_auto_capacity(options_.sampled_watchpoint_capacity)
               && sampled_watchpoint_capacity_ > 0)
            {
                --sampled_watchpoint_capacity_;
                return true;
            }
            if(is_auto_capacity(options_.exact_shadow_entry_capacity)
               && exact_shadow_entry_capacity_ > 0)
            {
                --exact_shadow_entry_capacity_;
                return true;
            }
            return false;
        }

        void compute_layout_or_abort()
        {
            storage_bytes_          = options_.storage_bytes;
            subgroup_capacity_      = options_.subgroup_capacity;
            atomic_object_capacity_ = auto_atomic_object_capacity();
            diagnostic_capacity_    = is_auto_capacity(options_.diagnostic_capacity)
                                          ? auto_diagnostic_capacity()
                                          : options_.diagnostic_capacity;

            exact_shadow_entry_capacity_ = is_auto_capacity(options_.exact_shadow_entry_capacity)
                                               ? 0
                                               : options_.exact_shadow_entry_capacity;
            sampled_watchpoint_capacity_ = is_auto_capacity(options_.sampled_watchpoint_capacity)
                                               ? 0
                                               : options_.sampled_watchpoint_capacity;

            std::size_t fixed_bytes = layout_byte_count();
            if(fixed_bytes > storage_bytes_)
            {
                report_storage_too_small_and_abort(fixed_bytes);
            }

            std::size_t remaining_bytes = storage_bytes_ - fixed_bytes;
            if(options_.backend == backend_kind::sampled_watchpoint)
            {
                if(is_auto_capacity(options_.sampled_watchpoint_capacity))
                {
                    sampled_watchpoint_capacity_ = capacity_from_bytes(
                        remaining_bytes, sizeof(uint64_t), "sampled_watchpoint");
                }
            }
            else if(is_auto_capacity(options_.exact_shadow_entry_capacity))
            {
                exact_shadow_entry_capacity_
                    = capacity_from_bytes(remaining_bytes, sizeof(uint64_t), "exact_shadow");
            }

            while(layout_byte_count() > storage_bytes_ && shrink_one_auto_capacity())
            {
            }

            layout_bytes_ = layout_byte_count();
            if(layout_bytes_ > storage_bytes_)
            {
                report_storage_too_small_and_abort(layout_bytes_);
            }
            if(diagnostic_capacity_ <= 0)
            {
                std::fprintf(stderr,
                             "hip-moi: host_context storage_bytes=%llu is too small for "
                             "a positive diagnostic capacity\n",
                             static_cast<unsigned long long>(storage_bytes_));
                std::fflush(stderr);
                std::abort();
            }
            if(options_.backend == backend_kind::sampled_watchpoint
               && sampled_watchpoint_capacity_ <= 0)
            {
                std::fprintf(stderr,
                             "hip-moi: host_context sampled_watchpoint capacity must be "
                             "positive for sampled backend\n");
                std::fflush(stderr);
                std::abort();
            }
            if(options_.backend == backend_kind::exact_shadow && exact_shadow_entry_capacity_ <= 0)
            {
                std::fprintf(stderr,
                             "hip-moi: host_context exact_shadow capacity must be positive "
                             "for exact backend\n");
                std::fflush(stderr);
                std::abort();
            }
        }

        void report_storage_too_small_and_abort(std::size_t required_bytes) const
        {
            std::fprintf(stderr,
                         "hip-moi: host_context storage_bytes=%llu is too small; "
                         "layout requires at least %llu bytes "
                         "(exact_shadow=%d sampled_watchpoints=%d diagnostics=%d "
                         "subgroups=%d atomic_objects=%d)\n",
                         static_cast<unsigned long long>(storage_bytes_),
                         static_cast<unsigned long long>(required_bytes),
                         exact_shadow_entry_capacity_,
                         sampled_watchpoint_capacity_,
                         diagnostic_capacity_,
                         subgroup_capacity_,
                         atomic_object_capacity_);
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
                             "hip-moi: hipMalloc(host_context storage, %llu bytes) failed: %s\n",
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
            hip_memset_or_abort(
                diagnostics_, 0, diagnostic_capacity_ * sizeof(diagnostic), "diagnostics");
            hip_memset_or_abort(subgroup_states_,
                                0,
                                subgroup_capacity_ * sizeof(subgroup_state),
                                "subgroup_states");
            hip_memset_or_abort(diagnostic_count_, 0, sizeof(int), "diagnostic_count");
            hip_memset_or_abort(simulated_barrier_arrival_count_,
                                0,
                                sizeof(int),
                                "simulated_barrier_arrival_count");
            hip_memset_or_abort(atomic_objects_,
                                0,
                                atomic_object_capacity_ * sizeof(context::atomic_object_record),
                                "atomic_objects");
            hip_memset_or_abort(acquired_subgroup_epoch_tokens_,
                                0,
                                acquired_subgroup_epoch_token_count() * sizeof(uint32_t),
                                "acquired_subgroup_epoch_tokens");
            hip_memset_or_abort(exact_shadow_entries_,
                                0,
                                exact_shadow_entry_capacity_ * sizeof(uint64_t),
                                "exact_shadow_entries");
            hip_memset_or_abort(sampled_watchpoints_,
                                0,
                                sampled_watchpoint_capacity_ * sizeof(uint64_t),
                                "sampled_watchpoints");
        }

        static void hip_memset_or_abort(void* ptr, int value, size_t byte_count, const char* name)
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

        static void report_hip_error(std::FILE* stream, const char* operation, hipError_t status)
        {
            if(!stream)
            {
                return;
            }
            std::fprintf(stream, "hip-moi: %s failed: %s\n", operation, hipGetErrorString(status));
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
                report_diagnostic(
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

        static void report_diagnostic(std::FILE*        stream,
                                      int               index,
                                      const diagnostic& diagnostic_record,
                                      const char*       kind_name)
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

        void release()
        {
            if(device_storage_)
            {
                (void)hipFree(device_storage_);
                device_storage_ = nullptr;
            }
            diagnostics_                     = nullptr;
            subgroup_states_                 = nullptr;
            diagnostic_count_                = nullptr;
            simulated_barrier_arrival_count_ = nullptr;
            exact_shadow_entries_            = nullptr;
            sampled_watchpoints_             = nullptr;
            atomic_objects_                  = nullptr;
            acquired_subgroup_epoch_tokens_  = nullptr;
        }

        host_context_options           options_;
        bool                           diagnostics_consumed_            = true;
        bool                           destructor_reports_              = true;
        bool                           destructor_aborts_               = true;
        std::FILE*                     diagnostic_stream_               = stderr;
        int                            last_diagnostic_count_           = 0;
        std::size_t                    storage_bytes_                   = 0;
        std::size_t                    layout_bytes_                    = 0;
        int                            exact_shadow_entry_capacity_     = 0;
        int                            sampled_watchpoint_capacity_     = 0;
        int                            diagnostic_capacity_             = 0;
        int                            subgroup_capacity_               = 0;
        int                            atomic_object_capacity_          = 0;
        uint64_t                       generation_                      = 1;
        unsigned char*                 device_storage_                  = nullptr;
        diagnostic*                    diagnostics_                     = nullptr;
        subgroup_state*                subgroup_states_                 = nullptr;
        int*                           diagnostic_count_                = nullptr;
        int*                           simulated_barrier_arrival_count_ = nullptr;
        uint64_t*                      exact_shadow_entries_            = nullptr;
        uint64_t*                      sampled_watchpoints_             = nullptr;
        context::atomic_object_record* atomic_objects_                  = nullptr;
        uint32_t*                      acquired_subgroup_epoch_tokens_  = nullptr;
    };

    inline void check_or_abort(host_context& context, const char* file, int line)
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
