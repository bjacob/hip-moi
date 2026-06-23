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

The distinction matters for future scope. `hip_moi::context` should remain the
home for correctness-first diagnostics, storage saturation handling, reporting,
and eventual synchronization models beyond full-workgroup barriers. The sampled
fast view may use stronger assumptions, such as local-only epoch tracking, as
long as those assumptions remain explicit and scoped.

## Byte Budget

The primary host option is a byte budget:

```c++
hip_moi::host_context_options options;
options.storage_bytes = 16 * 1024 * 1024;
options.subgroup_capacity = 64;

hip_moi::host_context moi(options);
```

`storage_bytes` defaults to 16 MiB. The host context partitions that budget into
diagnostics, subgroup epoch state, counters, and the selected shadow backend's
metadata.

This keeps the ordinary API stable as the internal metadata layout changes. A
user should not normally need to decide how many records of each internal type
to allocate.

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
hip_moi::host_context_options options;
options.backend = hip_moi::backend_kind::sampled_watchpoint;
options.sampled_watchpoint_capacity = -1; // fill remaining byte budget
options.exact_shadow_entry_capacity = 0;  // no exact-shadow table
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
whole table. `delay_iters` is a benchmark knob matching Jakub's sampled Loom
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
one-watchpoint sampled Loom configuration.

`hip_moi::sampled_watchpoint_context` currently supports only publish-only
sampled policies. It tracks its epoch locally across instrumented
full-workgroup barriers and does not update a global epoch word. That is a
benchmark fast-path trade-off, suitable for the current sampled publish-only
matmul rows, not a model for atomics or finer-grained synchronization.

## Inspecting The Layout

Host contexts expose the computed layout:

```c++
std::size_t allocated = moi.storage_bytes();
std::size_t used = moi.layout_bytes();
int diagnostics = moi.diagnostic_capacity();
int subgroups = moi.subgroup_capacity();
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
diagnostic when possible. If the diagnostic buffer itself fills, hip-moi keeps
counting total diagnostics but only stores the first `diagnostic_capacity`
records for host-side reporting.

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
