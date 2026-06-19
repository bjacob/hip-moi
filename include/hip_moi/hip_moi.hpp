// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// hip_moi/hip_moi.hpp
//
// Public header for the hip-moi HIP memory-ordering instrumentation library.
#ifndef HIP_MOI_HIP_MOI_HPP
#define HIP_MOI_HIP_MOI_HPP

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace hip_moi
{
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
        uint32_t  thread_id;
        uint32_t  subgroup_id;
        uint32_t  epoch;
        uint32_t  kind;
        uint32_t  valid;
    };

    struct diagnostic
    {
        uint32_t  kind;
        uint32_t  epoch;
        uint32_t  writer_or_first_thread_id;
        uint32_t  reader_or_second_thread_id;
        uintptr_t first_addr;
        uintptr_t second_addr;
        uint32_t  first_size;
        uint32_t  second_size;
    };

    struct subgroup_state
    {
        uint32_t epoch;
    };

    struct context_storage_ref
    {
        access_record*  access_records;
        int             access_record_capacity;
        diagnostic*     diagnostics;
        int             diagnostic_capacity;
        subgroup_state* subgroup_states;
        int             subgroup_capacity;
        int*            access_count;
        int*            epoch_access_count;
        int*            diagnostic_count;
    };

    template <int AccessCapacity, int DiagnosticCapacity, int SubgroupCapacity = 1>
    struct static_context_storage
    {
        access_record  access_records[AccessCapacity];
        diagnostic     diagnostics[DiagnosticCapacity];
        subgroup_state subgroup_states[SubgroupCapacity];
        int            access_count;
        int            epoch_access_count;
        int            diagnostic_count;

        __device__ context_storage_ref ref()
        {
            return context_storage_ref{
                access_records,
                AccessCapacity,
                diagnostics,
                DiagnosticCapacity,
                subgroup_states,
                SubgroupCapacity,
                &access_count,
                &epoch_access_count,
                &diagnostic_count,
            };
        }
    };

    class context
    {
    public:
        __device__ context(context_storage_ref storage, config cfg)
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
                for(int i = 0; storage_.subgroup_states && i < storage_.subgroup_capacity; ++i)
                {
                    storage_.subgroup_states[i].epoch = 0;
                }
                for(int i = 0; storage_.access_records && i < storage_.access_record_capacity; ++i)
                {
                    storage_.access_records[i].valid = 0;
                }
                __threadfence();
            }
            __syncthreads();
        }

        template <typename T>
        __device__ T lds_load(const T* ptr)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::context::lds_load requires a trivially copyable type");
            record_access(ptr, sizeof(T), access_kind::load);
            return *ptr;
        }

        template <typename T>
        __device__ void lds_store(T* ptr, T value)
        {
            static_assert(std::is_trivially_copyable<T>::value,
                          "hip_moi::context::lds_store requires a trivially copyable type");
            record_access(ptr, sizeof(T), access_kind::store);
            *ptr = value;
        }

        __device__ void syncthreads()
        {
            __syncthreads();
            if(thread_id() == 0 && storage_.subgroup_states && storage_.subgroup_capacity > 0)
            {
                if(storage_.epoch_access_count)
                {
                    *storage_.epoch_access_count = 0;
                }
                int subgroup_count = stored_subgroup_count();
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
            uint32_t subgroup_count = configured_subgroup_count();
            return subgroup < subgroup_count ? subgroup : subgroup_count - 1;
        }

        __device__ uint32_t thread_rank_in_subgroup() const
        {
            if(cfg_.threads_per_subgroup <= 0)
            {
                return thread_id();
            }
            return thread_id() % static_cast<uint32_t>(cfg_.threads_per_subgroup);
        }

    private:
        __device__ uint32_t configured_subgroup_count() const
        {
            return cfg_.subgroup_count > 0 ? static_cast<uint32_t>(cfg_.subgroup_count) : 1u;
        }

        __device__ int stored_subgroup_count() const
        {
            if(!storage_.subgroup_states || storage_.subgroup_capacity <= 0)
            {
                return 0;
            }

            uint32_t configured_count = configured_subgroup_count();
            uint32_t storage_capacity = static_cast<uint32_t>(storage_.subgroup_capacity);
            return static_cast<int>(configured_count < storage_capacity ? configured_count
                                                                        : storage_capacity);
        }

        __device__ uint32_t current_epoch(uint32_t subgroup) const
        {
            if(storage_.subgroup_states
               && subgroup < static_cast<uint32_t>(storage_.subgroup_capacity))
            {
                return storage_.subgroup_states[subgroup].epoch;
            }
            return 0;
        }

        __device__ static bool byte_ranges_overlap(const access_record& first,
                                                   const access_record& second)
        {
            uintptr_t first_end  = first.address + first.byte_count;
            uintptr_t second_end = second.address + second.byte_count;
            return first.address < second_end && second.address < first_end;
        }

        __device__ static bool is_write(access_kind kind)
        {
            return kind == access_kind::store;
        }

        __device__ static bool is_write(uint32_t kind)
        {
            return is_write(static_cast<access_kind>(kind));
        }

        __device__ void emit_diagnostic(diagnostic diagnostic_record) const
        {
            if(!storage_.diagnostics || !storage_.diagnostic_count
               || storage_.diagnostic_capacity <= 0)
            {
                return;
            }

            int index = atomicAdd(storage_.diagnostic_count, 1);
            if(index < storage_.diagnostic_capacity)
            {
                storage_.diagnostics[index] = diagnostic_record;
            }
        }

        __device__ void emit_conflict(const access_record& first, const access_record& second) const
        {
            emit_diagnostic(diagnostic{
                static_cast<uint32_t>(diagnostic_kind::access_conflict),
                second.epoch,
                first.thread_id,
                second.thread_id,
                first.address,
                second.address,
                first.byte_count,
                second.byte_count,
            });
        }

        __device__ void emit_metadata_full(const access_record& record) const
        {
            emit_diagnostic(diagnostic{
                static_cast<uint32_t>(diagnostic_kind::metadata_full),
                record.epoch,
                record.thread_id,
                record.thread_id,
                record.address,
                record.address,
                record.byte_count,
                record.byte_count,
            });
        }

        __device__ bool conflicts_with(const access_record& first,
                                       const access_record& second) const
        {
            return first.valid && first.epoch == second.epoch && first.thread_id != second.thread_id
                   && (is_write(first.kind) || is_write(second.kind))
                   && byte_ranges_overlap(first, second);
        }

        __device__ void record_access(const void* ptr, uint32_t byte_count, access_kind kind)
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
                thread_id(),
                subgroup,
                current_epoch(subgroup),
                static_cast<uint32_t>(kind),
                1,
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

        context_storage_ref storage_;
        config              cfg_;
    };

    struct host_context_options
    {
        int access_record_capacity = 1024;
        int diagnostic_capacity    = 64;
        int subgroup_capacity      = 1;

        bool destructor_reports = true;
        bool destructor_aborts  = true;

        std::FILE* diagnostic_stream = stderr;
    };

    class host_context
    {
    public:
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

        context_storage_ref device_ref()
        {
            diagnostics_consumed_ = false;
            return context_storage_ref{
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

        context_storage_ref ref()
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
                std::fprintf(stream,
                             "hip-moi diagnostic %d: kind=%s epoch=%u "
                             "first_thread=%u second_thread=%u "
                             "first_addr=0x%llx second_addr=0x%llx "
                             "first_size=%u second_size=%u\n",
                             i,
                             diagnostic_kind_name(diagnostic_record.kind),
                             diagnostic_record.epoch,
                             diagnostic_record.writer_or_first_thread_id,
                             diagnostic_record.reader_or_second_thread_id,
                             static_cast<unsigned long long>(diagnostic_record.first_addr),
                             static_cast<unsigned long long>(diagnostic_record.second_addr),
                             diagnostic_record.first_size,
                             diagnostic_record.second_size);
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
            if(options_.access_record_capacity <= 0 || options_.diagnostic_capacity <= 0
               || options_.subgroup_capacity <= 0)
            {
                std::fprintf(stderr,
                             "hip-moi: host_context capacities must all be positive "
                             "(access=%d diagnostics=%d subgroups=%d)\n",
                             options_.access_record_capacity,
                             options_.diagnostic_capacity,
                             options_.subgroup_capacity);
                std::fflush(stderr);
                std::abort();
            }
        }

        void allocate_or_abort()
        {
            hip_allocate_or_abort(&access_records_,
                                  options_.access_record_capacity * sizeof(access_record),
                                  "access_records");
            hip_allocate_or_abort(
                &diagnostics_, options_.diagnostic_capacity * sizeof(diagnostic), "diagnostics");
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
            hip_memset_or_abort(
                diagnostics_, 0, options_.diagnostic_capacity * sizeof(diagnostic), "diagnostics");
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

        static void hip_memset_or_abort(void* ptr, int value, size_t byte_count, const char* name)
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
        }

        host_context_options options_;
        bool                 diagnostics_consumed_  = true;
        bool                 destructor_reports_    = true;
        bool                 destructor_aborts_     = true;
        std::FILE*           diagnostic_stream_     = stderr;
        int                  last_diagnostic_count_ = 0;

        access_record*  access_records_   = nullptr;
        diagnostic*     diagnostics_      = nullptr;
        subgroup_state* subgroup_states_  = nullptr;
        int*            access_count_     = nullptr;
        int*            epoch_access_count_ = nullptr;
        int*            diagnostic_count_ = nullptr;
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

#endif // HIP_MOI_HIP_MOI_HPP
