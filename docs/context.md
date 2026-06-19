<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Context Allocation

hip-moi has two context layers:

* A device context, such as `hip_moi::thread_level_context` or
  `hip_moi::subgroup_level_context`, is a non-owning view used inside a kernel.
* A host context, such as `hip_moi::host_context` or
  `hip_moi::subgroup_level_host_context`, owns detector metadata storage in
  device global memory and hands a `storage_ref` view to kernels.

The host-owned path is the default user-facing path. It is intended for kernels
where LDS is already scarce, so hip-moi metadata should not compete with the
program's own `__shared__` allocations.

## The Context Allocation Plan

The primary host option is a byte budget:

```c++
hip_moi::host_context_options options;
options.storage_bytes = 16 * 1024 * 1024;
options.subgroup_capacity = 64;

hip_moi::subgroup_level_host_context moi(options);
```

`storage_bytes` defaults to 16 MiB. The host context partitions that budget into
the typed buffers required by the selected mode: exact access records,
diagnostics, subgroup epoch state, counters, and, for subgroup-level mode,
optional coalescing metadata.

This keeps the ordinary API stable as the internal metadata layout changes. A
user should not normally need to decide how many records of each internal type
to allocate.

## Capacity Overrides

`host_context_options` still has typed-capacity fields:

```c++
options.access_record_capacity = 1024;
options.diagnostic_capacity = 64;
```

These are advanced overrides and test controls. A negative value means "derive
this capacity from `storage_bytes`." A positive value fixes that capacity. For
optional subgroup-level coalescing buffers, zero disables that buffer.

The access-record and diagnostic capacities cannot be zero: exact access logging
and diagnostics are the correctness anchor for both instrumentation modes.

## Manual Storage

The public `storage_ref` types remain explicit typed views. This is intentional.
Tests, generated experiments, or specialized users can still allocate storage
themselves and pass it to kernels without using a host context.

Each device context also has a `static_context_storage` helper for fixed-size
storage. That path is useful when a caller deliberately wants static storage,
including possible LDS-backed experiments, but it is not the default
recommendation for real kernels that already pressure LDS capacity.

## Saturation

Storage saturation should degrade into diagnostics or conservative fallback, not
silent corruption.

Exact access-record overflow emits a `metadata_full` diagnostic when possible.
If the diagnostic buffer itself fills, hip-moi keeps counting total diagnostics
but only stores the first `diagnostic_capacity` records for host-side reporting.

Subgroup-level coalescing is optional. If coalescing access storage is absent or
full, opted-in accesses fall back to exact access records and increment
`coalescing_fallback_count` when that counter exists. If coalescing group
scratch is absent or too small, epoch-close summary construction falls back to a
slower scan over the coalescing access log.

The bad saturation case is still possible: if coalescing falls back to exact
records and exact access storage is also full, hip-moi may emit only
`metadata_full` diagnostics and may miss some specific conflicts. The intended
user response is to increase `storage_bytes` or narrow instrumentation scope.

## Current Implementation

The owning host contexts currently allocate one HIP global-memory block and
carve it into aligned typed slices. The device-side contexts still receive a
plain `storage_ref`; kernels do not know whether the storage came from one
byte-budgeted host allocation, a set of custom device allocations, or static
storage.
