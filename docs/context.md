<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Context Allocation

hip-moi has two context layers:

* `hip_moi::context` is the non-owning device-side view used inside a kernel.
* `hip_moi::host_context` owns detector metadata storage in device global memory
  and hands a `hip_moi::context::storage_ref` view to kernels.

The host-owned path is the default user-facing path. It is intended for kernels
where LDS is already scarce, so hip-moi metadata should not compete with the
program's own `__shared__` allocations.

There is also a narrower `hip_moi::sampled_watchpoint_context`. It is a
publish-only sampled fast view for benchmark-sensitive kernels that use
full-workgroup barriers. It deliberately does not carry the full diagnostic
state of `hip_moi::context`. Treat it as a specialized performance mode, not as
the general semantic model.

For the precise metadata layout and access-time algorithms, see
[`instrumentation_model.md`](instrumentation_model.md). This document focuses on
allocation, construction, and user-facing context choices.

Benchmark rows named `context + sampled_watchpoint` use the general
`hip_moi::context` object with its sampled backend selected. Rows named
`sampled_watchpoint_context` use the narrower class directly. The latter keeps
less state live in the hot kernel and is the faster publish-only benchmark path
in the current architecture-specific benchmark results.

The distinction matters for future scope. `hip_moi::context` should remain the
home for correctness-first diagnostics, storage saturation handling, reporting,
and eventual synchronization models beyond full-workgroup barriers. The sampled
fast view may use stronger assumptions, such as local-only epoch tracking, as
long as those assumptions remain explicit and scoped.

## Choosing A Context

Use the general `hip_moi::context` when you want diagnostics:

* exact shadow diagnostics,
* sampled watchpoint diagnostics in reporting mode,
* metadata saturation diagnostics,
* destructor or `HIP_MOI_CHECK` host reporting,
* source-level release/acquire atomics and full-workgroup barriers.

Use `hip_moi::sampled_watchpoint_context` only for the current narrow fast path:

* sampled watchpoint publication only,
* fixed compile-time sampled policy,
* full-workgroup barriers through `ctx.syncthreads()` or `ctx.barrier()`,
* source-instrumented kernels where the LDS byte offset is already known,
* benchmark rows that should resemble the Jakub-Sampled-Loom publish-only path.

The fast view does not report conflicts. It publishes sampled watchpoint
metadata at selected sites and intentionally omits the cold diagnostic state
that makes `hip_moi::context` useful as a sanitizer-facing API.

## Byte Budget

The primary host option is a byte budget:

```c++
hip_moi::host_context_options options;
options.storage_bytes = 16 * 1024 * 1024;
options.subgroup_capacity = 64;

hip_moi::host_context moi(options);
```

`storage_bytes` defaults to 16 MiB. The host context partitions that budget into
diagnostics, subgroup epoch state, counters, atomic-object metadata, the
internal RMW address cache, and the selected shadow backend's metadata.

This keeps the ordinary API stable as the internal metadata layout changes. A
user should not normally need to decide how many records of each internal type
to allocate.

Atomic-object metadata is currently an internal `hip_moi::context` facility.
The host context derives its capacity from `storage_bytes`: larger global
storage budgets can track more distinct atomic objects before emitting
`metadata_full`, while small test contexts naturally get a smaller bounded
table. The same capacity also sizes the internal direct-mapped RMW address
cache. There is no separate user-facing atomic-object or RMW-cache capacity
knob in the ordinary API.

Release/acquire atomics also use an internal acquired-epoch matrix whose size is
derived from `subgroup_capacity`. This matrix records which producer subgroup
epochs each consumer subgroup has acquired, and the exact-shadow backend uses
it to suppress LDS conflicts that are ordered by a tracked release/acquire
handoff.

## Backend Storage

The current device context has two backends:

* `backend_kind::exact_shadow` stores one packed shadow entry per LDS shadow
  granule. It is the default and gives deterministic conflict diagnostics for
  instrumented cross-subgroup accesses whose byte offsets fit in the shadow.
* `backend_kind::sampled_watchpoint` stores a smaller watchpoint table and
  samples one lane per subgroup/site/generation. It is designed as the
  lower-overhead Loom-inspired benchmark path, at the cost of intentionally
  sampling rather than observing every lane.

For the selected backend, a negative capacity means "derive this capacity from
`storage_bytes`." A positive value fixes that capacity. Zero disables the
backend storage, which is useful only for saturation tests; a host context
rejects zero capacity for the backend it is asked to run.

```c++
hip_moi::host_context_options options =
    hip_moi::make_sampled_watchpoint_reporting_options();
options.sampled_watchpoint_capacity = -1; // fill remaining byte budget
```

Sampled watchpoints also expose policy knobs:

```c++
options.sampled_watchpoint_sample_skip = 32;
options.sampled_watchpoint_probe_count = 1;
options.sampled_watchpoint_delay_iters = 32;
options.sampled_watchpoint_reports = false;
```

`sample_skip` thins static site/subgroup instances after the deterministic
selection seed has been mixed; `1` means no thinning. Reporting mode always
checks the watchpoint entry displaced by the current publish. `probe_count`
controls how many additional watchpoint slots are scanned; `0` means scan the
whole table. `delay_iters` is a benchmark knob matching the Jakub-Sampled-Loom
prototype. `sampled_watchpoint_reports=false` makes the sampled backend publish
watchpoints without scanning for conflicts, which is useful for apples-to-apples
publish-only benchmark rows.

For hot kernels whose sampled policy is intentionally fixed at compile time,
the access API can take a policy type:

```c++
using sampled_publish_only = hip_moi::sampled_watchpoint_policy<
    /*SampleSkip=*/32,
    /*ProbeCount=*/1,
    /*DelayIters=*/32,
    /*ReportConflicts=*/false,
    /*StaticWatchpointCapacity=*/1>;

ctx.lds_store_at<hip_moi::backend_kind::sampled_watchpoint, sampled_publish_only>(
    ptr, value, /*lds_byte_offset=*/offset, site);
```

This is the low-overhead path for benchmark/source-instrumented code that does
not need runtime policy tuning. If the policy must vary between launches, keep
using `host_context_options` and the ordinary `lds_*_at<backend_kind::...>`
overloads.

The final policy argument is optional and defaults to `0`, meaning that the
watchpoint capacity is read from runtime context storage. Passing `1` tells the
sampled hot-path view that the table has exactly one watchpoint entry, so slot
selection folds to zero instead of running the generic slot-mixing code. That is
useful for fair publish-only benchmark rows that intentionally match Jakub's
one-watchpoint Jakub-Sampled-Loom configuration.

`hip_moi::sampled_watchpoint_context` currently supports only publish-only
sampled policies. It tracks its epoch locally across instrumented
full-workgroup barriers and does not update a global epoch word. That is a
benchmark fast-path trade-off, suitable for the current sampled publish-only
matmul rows, not a model for atomics or finer-grained synchronization.

## General Context Example

This is the shape to use when the host should be able to consume diagnostics:

```c++
using sampled_reporting_policy = hip_moi::sampled_watchpoint_policy<
    /*SampleSkip=*/1,
    /*ProbeCount=*/0,
    /*DelayIters=*/0,
    /*ReportConflicts=*/true>;

__device__ hip_moi::context
make_context(hip_moi::context::storage_ref storage)
{
    hip_moi::context::config cfg{
        /*thread_count=*/static_cast<int>(blockDim.x),
        /*threads_per_subgroup=*/32,
        /*subgroup_count=*/(static_cast<int>(blockDim.x) + 31) / 32,
    };
    return hip_moi::context(storage, cfg);
}

__global__ void diagnostic_kernel(hip_moi::context::storage_ref storage,
                                  int* out)
{
    __shared__ int values[64];

    hip_moi::context ctx = make_context(storage);
    ctx.init_workgroup();

    int index = static_cast<int>(threadIdx.x);
    ctx.lds_store_at<hip_moi::backend_kind::sampled_watchpoint,
                     sampled_reporting_policy>(
        &values[index],
        index,
        /*lds_byte_offset=*/index * static_cast<int>(sizeof(int)),
        HIP_MOI_SITE_ID());

    ctx.syncthreads();

    int loaded = ctx.lds_load_at<hip_moi::backend_kind::sampled_watchpoint,
                                 sampled_reporting_policy>(
        &values[index],
        /*lds_byte_offset=*/index * static_cast<int>(sizeof(int)),
        HIP_MOI_SITE_ID());
    if(index == 0)
    {
        out[0] = loaded;
    }
}
```

Code that mirrors lower-level native workgroup synchronization can spell the
same epoch boundary as:

```c++
ctx.release_fence(hip_moi::atomic_memory_scope::workgroup, HIP_MOI_SITE_ID());
ctx.barrier(HIP_MOI_SITE_ID());
ctx.acquire_fence(hip_moi::atomic_memory_scope::workgroup, HIP_MOI_SITE_ID());
```

`ctx.barrier()` is the full-workgroup epoch boundary. The release/acquire fence
wrappers emit native fences and keep this spelling close to the underlying HIP
or Clang builtins. Fences without `ctx.barrier()` do not advance hip-moi's
epoch.

Host setup:

```c++
hip_moi::host_context_options options =
    hip_moi::make_sampled_watchpoint_reporting_options();
options.sampled_watchpoint_sample_skip = 1;
options.sampled_watchpoint_probe_count = 0;
options.sampled_watchpoint_delay_iters = 0;
options.sampled_watchpoint_reports = true;

hip_moi::host_context moi(options);
diagnostic_kernel<<<dim3(1), dim3(64)>>>(moi.launch_ref(), out);
HIP_MOI_CHECK(moi);
```

For a publish-only general-context row, set
`options.sampled_watchpoint_reports = false` or use a policy with
`ReportConflicts=false`. That is still not the same as
`sampled_watchpoint_context`: the full `hip_moi::context` object remains live in
the kernel.

## Fast View Example

The fast view is constructed from the same host-allocated storage, but it uses a
smaller device-side view:

```c++
using publish_only_policy = hip_moi::sampled_watchpoint_policy<
    /*SampleSkip=*/32,
    /*ProbeCount=*/1,
    /*DelayIters=*/32,
    /*ReportConflicts=*/false>;

__device__ hip_moi::sampled_watchpoint_context
make_fast_context(hip_moi::context::storage_ref storage)
{
    hip_moi::sampled_watchpoint_context::config cfg{
        /*threads_per_subgroup=*/32,
    };
    return hip_moi::make_sampled_watchpoint_context(storage, cfg);
}

__global__ void publish_only_kernel(hip_moi::context::storage_ref storage,
                                    int* out)
{
    __shared__ int values[64];

    hip_moi::sampled_watchpoint_context ctx = make_fast_context(storage);
    ctx.init_workgroup();

    int index = static_cast<int>(threadIdx.x);
    uint32_t offset = static_cast<uint32_t>(index * sizeof(int));
    ctx.lds_store_at<publish_only_policy>(
        &values[index], index, /*lds_byte_offset=*/offset, HIP_MOI_SITE_ID());

    ctx.syncthreads();

    int loaded = ctx.lds_load_at<publish_only_policy>(
        &values[index], /*lds_byte_offset=*/offset, HIP_MOI_SITE_ID());
    if(index == 0)
    {
        out[0] = loaded;
    }
}
```

Host setup still uses `hip_moi::host_context` to allocate storage:

```c++
hip_moi::host_context_options options =
    hip_moi::make_sampled_watchpoint_publish_only_options();

hip_moi::host_context moi(options);
publish_only_kernel<<<dim3(1), dim3(64)>>>(moi.launch_ref(), out);
```

There is intentionally no `HIP_MOI_CHECK(moi)` in this example. The fast view is
publish-only, so a successful run means "the kernel ran with sampled metadata
publication enabled," not "the program had no races."

The adapter hides the fast storage view. That gives the implementation room to
keep trimming `sampled_watchpoint_context` without asking users to manually copy
its internal fields.

## One Watchpoint Fast Row

For benchmark rows that intentionally match the one-watchpoint Jakub-Sampled-Loom
configuration, make both the host capacity and the policy explicit:

```c++
using one_watchpoint_policy = hip_moi::sampled_watchpoint_policy<
    /*SampleSkip=*/32,
    /*ProbeCount=*/1,
    /*DelayIters=*/32,
    /*ReportConflicts=*/false,
    /*StaticWatchpointCapacity=*/1>;

hip_moi::host_context_options options =
    hip_moi::make_one_watchpoint_publish_only_options();
```

Only use `StaticWatchpointCapacity=1` when the storage really has exactly one
watchpoint entry. It lets the compiler fold slot selection to zero. If the
capacity is launch-configurable, leave that final policy argument at its
default `0`.

## Multi-Workgroup Storage

A `hip_moi::context::storage_ref` is one detector storage view. For a
multi-workgroup kernel, give each workgroup its own storage view rather than
having every block share the same watchpoint table.

One simple host pattern is to create one `host_context` per workgroup and copy
their device refs to a device array:

```c++
std::vector<std::unique_ptr<hip_moi::host_context>> contexts;
std::vector<hip_moi::context::storage_ref> refs;

for(int i = 0; i < workgroup_count; ++i)
{
    auto context = std::make_unique<hip_moi::host_context>(options);
    refs.push_back(context->launch_ref());
    contexts.push_back(std::move(context));
}

hip_moi::context::storage_ref* refs_device = nullptr;
hipMalloc(&refs_device, refs.size() * sizeof(refs[0]));
hipMemcpy(refs_device,
          refs.data(),
          refs.size() * sizeof(refs[0]),
          hipMemcpyHostToDevice);

kernel<<<grid, block>>>(refs_device, static_cast<int>(refs.size()), out);
```

The kernel selects the ref for its workgroup, then constructs either the general
context or the fast view:

```c++
__device__ uint32_t flat_workgroup_id()
{
    return static_cast<uint32_t>(
        blockIdx.x
        + gridDim.x * (blockIdx.y + gridDim.y * static_cast<uint32_t>(blockIdx.z)));
}

__global__ void kernel(hip_moi::context::storage_ref* refs,
                       int                            ref_count,
                       int*                           out)
{
    uint32_t workgroup = flat_workgroup_id();
    if(workgroup >= static_cast<uint32_t>(ref_count))
    {
        return;
    }

    hip_moi::sampled_watchpoint_context ctx = make_fast_context(refs[workgroup]);
    ctx.init_workgroup();
    // Instrument LDS accesses with ctx.lds_load_at<policy>() and
    // ctx.lds_store_at<policy>().
}
```

This is intentionally a little verbose. It makes generation, storage ownership,
and per-workgroup isolation explicit, which is what the benchmark-sensitive
path currently needs.

## Inspecting The Layout

Host contexts expose the computed layout:

```c++
std::size_t allocated = moi.storage_bytes();
std::size_t used = moi.layout_bytes();
int diagnostics = moi.diagnostic_capacity();
int subgroups = moi.subgroup_capacity();
int atomic_objects = moi.atomic_object_capacity();
int exact_entries = moi.exact_shadow_entry_capacity();
int watchpoints = moi.sampled_watchpoint_capacity();
```

These values are meant for tuning and tests. Kernels still receive only the
non-owning `storage_ref`.

## Manual Storage

The public `hip_moi::context::storage_ref` type remains an explicit typed view.
Tests, generated experiments, or specialized users can still allocate storage
themselves and pass it to kernels without using a host context.

`hip_moi::context::static_context_storage` also exists for fixed-size storage.
That path is useful when a caller deliberately wants static storage, including
possible LDS-backed experiments, but it is not the default recommendation for
real kernels that already pressure LDS capacity.

## Saturation

Storage saturation should degrade into diagnostics, not silent corruption.

Start with the default 16 MiB host-owned budget unless there is a reason not to.
Increase `storage_bytes` when host reports mention `metadata_full` or diagnostic
buffer truncation. Reduce it only when the allocator's reported `layout_bytes()`
and computed capacities show comfortable headroom for the kernels being
diagnosed.

If a shadow table cannot represent an access, hip-moi emits a `metadata_full`
diagnostic when possible. If a release-style atomic operation cannot find or
claim an atomic-object metadata slot, hip-moi also emits `metadata_full`, naming
the atomic object's address, byte size, and source site. If the diagnostic
buffer itself fills, hip-moi keeps counting total diagnostics but only stores
the first `diagnostic_capacity` records for host-side reporting.

## Subgroup Capacity

`subgroup_capacity` is still a semantic host option, not a byte-only detail. The
host context cannot infer how many subgroups a future kernel launch will pass in
its device-side `config`, so users must size this field for the largest
instrumented workgroup shape they will use with that host context.

If a kernel config names more subgroups than the host context allocated, hip-moi
emits a `metadata_full` diagnostic during `ctx.init_workgroup()`. In
diagnostics, `first_subgroup` is the allocated subgroup capacity and
`second_subgroup` is the configured subgroup count.

## Current Implementation

The owning host context currently allocates one HIP global-memory block and
carves it into aligned typed slices. The device-side context receives a plain
`storage_ref`; kernels do not know whether the storage came from one
byte-budgeted host allocation, a set of custom device allocations, or static
storage.
