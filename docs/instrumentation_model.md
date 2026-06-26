<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Instrumentation Model

This document describes what hip-moi currently records and when that metadata is
used. It is intentionally implementation-facing: the goal is to make the
prototype easy to compare with Loom, Jakub's HIP implementation of Loom-like
ideas, and proposed compiler-rt GPU ThreadSanitizer designs.

## Scope

hip-moi is a HIP source-level instrumentation library. It instruments LDS
accesses only when the user, or a source-rewriting tool, replaces the raw LDS
access with a hip-moi API call. It does not interpose arbitrary memory
operations.

The active detector is subgroup-scoped:

* a **workgroup** is a HIP block;
* a **subgroup** is the instrumentation owner, currently configured as a
  fixed number of threads such as 32;
* a **lane** is a thread's index inside its subgroup;
* an **epoch** is the logical interval between instrumented full-workgroup
  barriers.

The current diagnostic condition is a same-epoch LDS conflict between different
subgroups in the same workgroup, unless `hip_moi::context` can prove that the
accesses are ordered by the supported atomic release/acquire model. hip-moi
intentionally does not diagnose conflicts wholly inside one subgroup.

The current access API requires the caller to supply the LDS byte offset:

```c++
ctx.lds_store_at(ptr, value, /*lds_byte_offset=*/offset, HIP_MOI_SITE_ID());
value = ctx.lds_load_at(ptr, /*lds_byte_offset=*/offset, HIP_MOI_SITE_ID());
```

The pointer is used to perform the actual load or store. The explicit
`lds_byte_offset` is what the metadata backends record and compare.

## Public Contexts

`hip_moi::host_context` owns metadata storage in device global memory. Each
kernel launch receives a `hip_moi::context::storage_ref` by value. Calling
`host_context::launch_ref()` also advances a host-side **generation** counter,
which separates metadata from different launches.

`hip_moi::context` is the general device-side diagnostic context. It can run
the exact shadow backend or the sampled watchpoint backend. It carries
diagnostic pointers, capacities, subgroup epoch state, backend selection, and
runtime sampled-watchpoint options.

`hip_moi::sampled_watchpoint_context` is a narrow publish-only fast path. It
only carries:

* a pointer to sampled watchpoint storage,
* the watchpoint capacity,
* the launch generation,
* the number of threads per subgroup,
* a per-thread local epoch counter.

It does not carry diagnostic storage, runtime backend selection, global subgroup
epoch storage, saturation diagnostics, or reporting code. That is why it can
compile much smaller than `hip_moi::context + backend_kind::sampled_watchpoint`.
The cost is that it cannot diagnose races.

## Site Ids

`HIP_MOI_SITE_ID()` creates a `hip_moi::site_id`. The current implementation
hashes:

* `__FILE__`,
* `__builtin_LINE()` when available, otherwise `__LINE__`,
* `__builtin_COLUMN()` when available, otherwise zero,
* `__COUNTER__`.

The result is a nonzero 64-bit value computed through constexpr hashing. Site
ids serve two purposes:

* diagnostics can report which instrumented source sites participated;
* sampled watchpoint selection uses the site id when choosing whether this
  subgroup/site instance is sampled and which lane publishes.

The default `site_id{0}` is allowed, but call sites that all use zero become
indistinguishable for diagnostics and sampling.

## Stored Metadata

The host context lays out one device allocation into slices:

| Slice | Stored type | Purpose |
| --- | --- | --- |
| diagnostics | `context::diagnostic[]` | Host-consumed reports. |
| subgroup states | `subgroup_state[]` | Per-subgroup epoch for `hip_moi::context`. |
| counters | two `int` values | Diagnostic count and simulated-barrier arrivals. |
| exact shadow | `uint64_t[]` | One exact-shadow entry per 4-byte LDS cell. |
| sampled watchpoints | `uint64_t[]` | Software watchpoint records. |
| atomic objects | `context::atomic_object_record[]` | Address-scoped release metadata, represented as one row per atomic address and producer subgroup. |
| atomic address cache | `context::atomic_address_cache_record[]` | Direct-mapped producer-mask cache for multi-subgroup release-capable RMWs. This is a prefilter, not the authoritative epoch store. |
| acquired epoch tokens | `uint32_t[]` | Pairwise ordering evidence created by atomic acquire operations. These are not additional epoch counters. |

The default host storage budget is 16 MiB. `host_context_options` can either set
explicit capacities or let the implementation derive capacities from that byte
budget.

### Diagnostic Record

`context::diagnostic` contains:

| Field | Meaning |
| --- | --- |
| `kind` | `access_conflict`, `metadata_full`, or `barrier_divergence`. |
| `epoch` | Epoch in which the condition was observed. |
| `first_subgroup_id`, `second_subgroup_id` | Subgroups participating in an access conflict. |
| `first_addr`, `second_addr` | Reported LDS offsets for access conflicts, or auxiliary values for other diagnostics. |
| `first_size`, `second_size` | Reported byte sizes. |
| `first_site_id`, `second_site_id` | Source site ids when available. |
| `expected_thread_count`, `observed_thread_count` | Used for simulated barrier divergence diagnostics. |

`HIP_MOI_CHECK(moi)` synchronizes the device, copies diagnostics to the host,
prints them to `stderr`, and aborts on any diagnostic. The `host_context`
destructor performs the same reporting/abort behavior by default if diagnostics
were not consumed explicitly.

### Exact Shadow Entry

The exact shadow backend uses one 64-bit word per 4-byte LDS cell. The packed
entry records:

| Field | Bits | Meaning |
| --- | ---: | --- |
| access kind | 3 | empty, read, write, read-write, or atomic representation. Current public LDS APIs use read/write. |
| owner | 10 | Subgroup id. |
| epoch | 10 | Per-subgroup epoch. |
| generation | 20 | Host launch generation. |
| site | 21 | Low bits of the site id. |

The exact shadow backend is exact at dword-cell granularity, not byte
granularity. Sub-dword accesses that occupy different bytes in the same 4-byte
cell can therefore collide in the shadow representation.

### Sampled Watchpoint Entry

A sampled watchpoint is a software metadata record in a global table. It is not
a hardware watchpoint.

The sampled backend also uses one 64-bit word per watchpoint slot. The packed
entry records:

| Field | Bits | Meaning |
| --- | ---: | --- |
| valid | 1 | Whether the entry contains a record. |
| consumed | 1 | Reserved by the representation; conflicts ignore consumed prior entries. |
| access kind | 3 | read or write for current public LDS APIs. |
| owner | 10 | Subgroup id. |
| epoch | 10 | Epoch. |
| generation | 20 | Host launch generation. |
| start cell | 14 | First 4-byte LDS cell in the recorded range. |
| cell count | 5 | Encoded count minus one, representing 1 to 32 cells. |

A watchpoint therefore represents one contiguous LDS dword-cell range. Larger
ranges are split into multiple records.

## Exact Shadow Access Algorithm

For `hip_moi::context` with `backend_kind::exact_shadow`, each instrumented
access does the following:

1. Return immediately for zero-byte accesses.
2. Convert the LDS byte range into one or more 4-byte shadow cells.
3. If the range does not fit in the exact shadow table, emit a
   `metadata_full` diagnostic.
4. Read the current subgroup id and that subgroup's current epoch.
5. Pack the current access kind, subgroup id, epoch, generation, and site id
   into a 64-bit shadow entry.
6. For each covered shadow cell, atomically exchange the cell with the current
   entry.
7. Decode the previous entry and check the raw conflict predicate.
8. If the raw predicate is true, check whether an atomic acquire has ordered
   the current subgroup after the prior subgroup's recorded epoch.
9. Emit an `access_conflict` diagnostic if the raw predicate is true and no
   acquired epoch token suppresses it.

The raw exact shadow conflict predicate is:

* the prior entry is not empty;
* current and prior entries have the same epoch;
* current and prior entries have the same generation;
* current and prior entries have different subgroup owners;
* at least one of the two entries is a write.

A raw conflict is suppressed when the current subgroup has acquired ordering
evidence covering the prior subgroup's recorded epoch. The table entry is
indexed as:

```text
acquired_epoch_token[consumer_subgroup][producer_subgroup]
```

A token value of `N + 1` means that the consumer subgroup has acquired
producer epoch `N`. Zero means "no acquired epoch", so representing the
acquired value as `epoch + 1` lets epoch zero participate in atomic ordering.
This table does not create nested epochs. It is only a pairwise fact used to
answer this question during conflict checking: "has the current subgroup
acquired the prior subgroup's recorded epoch?"

This is an immediate-checking algorithm: the check happens at the same
instrumented access that publishes the new metadata.

## Atomic Synchronization Model

Each subgroup has one epoch counter for LDS access records. Instrumented
full-workgroup barriers advance those counters. Atomic operations do not
introduce another epoch hierarchy; they add pairwise ordering evidence:

```text
acquired_epoch_token[consumer_subgroup][producer_subgroup]
```

The table answers whether a consumer subgroup has acquired a producer
subgroup's epoch through a supported release/acquire atomic edge. It is not a
clock that advances on every atomic operation, and it does not change either
subgroup's current epoch.

The represented synchronizes-with shape is an address-scoped, TSan-like
release/acquire synchronization object:

1. A release operation records the releasing subgroup's current epoch in the
   atomic-object table.
2. The release record is keyed by the atomic object's address, the releasing
   subgroup, and the launch generation.
3. An acquire operation performs the user atomic operation first.
4. The acquire imports every release record for that atomic address, except
   records produced by the acquiring subgroup itself.
5. For each imported producer record, hip-moi updates the pairwise
   acquired-epoch token for `(consumer subgroup, producer subgroup)`.
6. Later LDS conflict checks consult that token table before reporting a
   same-epoch cross-subgroup conflict. This applies to exact-shadow diagnostics
   and to sampled-watchpoint diagnostics when reporting is enabled in the
   general `hip_moi::context`.

The current public atomic operations are:

```c++
value = ctx.atomic_load(ptr, order, scope, HIP_MOI_SITE_ID());
ctx.atomic_store(ptr, value, order, scope, HIP_MOI_SITE_ID());
old = ctx.atomic_fetch_add(ptr, delta, order, scope, HIP_MOI_SITE_ID());
old = ctx.atomic_fetch_or(ptr, bits, order, scope, HIP_MOI_SITE_ID());
old = ctx.atomic_fetch_and(ptr, bits, order, scope, HIP_MOI_SITE_ID());
old = ctx.atomic_fetch_xor(ptr, bits, order, scope, HIP_MOI_SITE_ID());
old = ctx.atomic_exchange(ptr, value, order, scope, HIP_MOI_SITE_ID());
exchanged = ctx.atomic_compare_exchange_strong(
    ptr, &expected, desired, success_order, failure_order, scope, HIP_MOI_SITE_ID());
ctx.atomic_fence(order, scope, HIP_MOI_SITE_ID());
```

The API uses hip-moi memory-order and memory-scope enums. The implementation
lowers those operations to HIP/Clang atomic builtins and separately records the
ordering metadata described here.

For exchange and fetch-style RMW operations, release-capable orders publish a
release record for the atomic address and acquire-capable orders import release
records for that address. For compare-exchange, the success order applies only
when the operation actually exchanges the value. A failed compare-exchange does
not publish a release record, but an acquire-capable failure order is modeled as
an acquire load and imports release records for the atomic address.

This adapts the epoch idea by adding an ordering predicate, not by adding more
epochs. A subgroup can be in epoch zero and still have acquired another
subgroup's epoch zero through an atomic flag. Conversely, two subgroups can
both be in the same barrier epoch and still race if no acquire operation has
imported release metadata that would order the payload accesses.

`context::atomic_object_record` stores:

| Field | Meaning |
| --- | --- |
| `generation` | Host launch generation that owns the record. |
| `address` | Atomic object address. The object may be in global memory or LDS. |
| `releasing_subgroup_id` | Subgroup that performed the release. |
| `releasing_epoch` | That subgroup's epoch at the release point. |
| `release_site_id` | Source site id for the release operation. |

Multiple releases from the same producer subgroup to the same atomic address
join by keeping the maximum released producer epoch. Multiple producer
subgroups using the same atomic address occupy separate metadata rows. An
acquire of that address imports all producer rows for that address into the
consumer subgroup's acquired-epoch tokens.

`context::atomic_address_cache_record` stores:

| Field | Meaning |
| --- | --- |
| `generation` | Host launch generation that owns the cache slot. |
| `address` | Atomic object address for the direct-mapped slot. |
| `producer_mask` | Bit mask of producer subgroups that have published generic release records for this address. |

The cache is used only for release-capable RMW operations in workgroups with
more than two subgroups. Release stores and fence-published relaxed stores use
the generic table directly. The cache storage is laid out immediately after
the generic atomic-object table and is derived from that pointer on device, so
`storage_ref` does not carry a separate cache pointer.

The cache never replaces `atomic_object_record`. It lets an acquire skip
generic probes for producer subgroups that have not published to the atomic
address. If the direct slot is stale, claimed, empty, or holds a different
address, the acquire falls back to the generic per-producer scan.

The scalar value stored by a release store, returned by an acquire load, or
returned/produced by a read-modify-write operation is not part of the current
synchronization metadata key. This deliberately reduces VGPR pressure and
matches the DBI-oriented direction where the atomic object address is the main
sync-object identity. The trade-off is precision: an acquire of an atomic
address can import release facts from releases through the same address that it
did not actually read from. That over-approximation can hide a real LDS race,
which is a false negative. It should not create false positives because release
facts are joined, not overwritten.

Address+value keying remains a possible future precision refinement for
protocols where the observed value cheaply identifies useful synchronization
state, such as monotonic counters or bitmasks. It is not the current default.
Even address+value keying is not a true memory-model reads-from identity for
repeated same-address same-value release stores.

A compact address+value mode could hash `(atomic address, scalar value)` into a
single metadata word instead of storing both fields as full tags. That can be a
reasonable precision/performance trade-off for value-distinguishing protocols:
it may reduce false negatives compared with address-only while keeping the
metadata row small. It is not a strict improvement. Hash collisions can create
extra synchronization imports and therefore false negatives, and the
instrumentation still has to keep or recompute the scalar atomic value,
normalize 32-bit and 64-bit cases, and hash the pair on the hot path. The
current address-only model avoids that value-liveness and codegen cost.

Fence-only synchronization is not modeled. The implemented fence support is
the standard paired-atomic shape:

* a release fence records the current subgroup epoch as a pending release;
* the next relaxed atomic store, exchange, or successful read-modify-write
  operation consumes that pending release and publishes the atomic address with
  the fence's epoch;
* a relaxed atomic load or RMW remembers the address it observed;
* a later acquire fence consumes that remembered address and imports release
  records for that address.

A failed compare-exchange is not a release-side publication point for a pending
release fence because it does not modify the atomic object. It can still be the
observation consumed by a later acquire fence.

The support matrix for source-level synchronization operations is:

| Source-level spelling through `hip_moi::context` | Diagnostic model |
| --- | --- |
| `ctx.syncthreads()` | Full-workgroup barrier and epoch boundary. |
| `ctx.release_fence(workgroup); ctx.barrier(); ctx.acquire_fence(workgroup);` | Lower-level full-workgroup barrier spelling; equivalent to `ctx.syncthreads()` for diagnostics. |
| `ctx.release_fence()` or `ctx.acquire_fence()` without `ctx.barrier()` | Emits the native workgroup fence but does not order LDS diagnostics by itself. |
| Release store plus acquire load on one atomic address | Imports producer epochs for that address. |
| Release/acquire `fetch_add`, `fetch_or`, `fetch_and`, `fetch_xor`, or `exchange` | RMW-style publication and observation on one atomic address. |
| Successful compare-exchange with release/acquire success order | Modeled as an RMW on the success path. |
| Failed compare-exchange with acquire-capable failure order | Modeled as an acquire load, not as a release. |
| Release atomic fence before relaxed store/exchange/successful RMW, paired with relaxed load/RMW before acquire atomic fence | Imports producer epochs for the observed atomic address. |
| Relaxed atomics without release/acquire orders or paired fences | Execute the user atomic operation but do not order LDS diagnostics. |

The diagnostic payload remains LDS access. Global atomics are supported as
synchronization operations because real kernels often use global flags,
counters, or bitmasks to order work. Ordinary global loads and stores are not
diagnosed as race payloads by hip-moi.

Atomic metadata saturation is reported as `metadata_full`. Missing acquire
metadata does not suppress an LDS conflict, so the failure mode is conservative
for diagnostics: hip-moi may report a conflict it could not prove ordered, but
it should not hide an unordered LDS conflict because atomic metadata was absent.

`hip_moi::sampled_watchpoint_context` does not implement this atomics model. It
is a publish-only sampled metadata path, not a diagnostic sanitizer mode.

## Sampled Watchpoint Access Algorithm

For the sampled backend, hip-moi first decides whether the current lane should
publish metadata.

The selection seed is mixed from:

* launch generation,
* flat workgroup id,
* subgroup id,
* site id.

`sample_skip` thins subgroup/site instances after this seed is computed. A
`sample_skip` of 1 means no thinning. A larger value samples only the instances
whose mixed seed passes the skip test. For a sampled subgroup/site instance,
one selected lane publishes. Other lanes do not touch the watchpoint table for
that access.

For a publishing lane:

1. Convert the LDS byte range into a 4-byte cell range.
2. Split the range into chunks of at most 32 cells.
3. Pack each chunk into a sampled watchpoint entry.
4. Choose a watchpoint slot from the start cell, epoch, generation, and capacity.
5. Atomically exchange the selected slot with the current entry.

If reporting is enabled in `hip_moi::context`, the access also checks for
conflicts:

1. Decode the overwritten prior entry and compare it with the current entry.
2. Probe additional watchpoint slots according to `probe_count`.
3. For each conflicting prior entry, ask whether the current subgroup has
   acquired the prior subgroup's recorded epoch through supported atomic
   synchronization.
4. Emit an `access_conflict` diagnostic only when no such acquired-epoch token
   exists.

The sampled watchpoint conflict predicate is:

* both entries are valid;
* the prior entry is not marked consumed;
* current and prior entries have the same epoch;
* current and prior entries have the same generation;
* current and prior entries have different subgroup owners;
* at least one of the two entries is a write;
* the recorded dword-cell ranges overlap.

That predicate describes a potential conflict. Reporting then applies the same
atomic-synchronization suppression as exact shadow: if
`acquired_epoch_token[current.owner][prior.owner]` is at least
`prior.epoch + 1`, the current access is considered ordered after the prior
entry and no diagnostic is emitted for that pair.

If reporting is disabled, the sampled backend publishes metadata but does not
consume it to diagnose races.

## Publish-Only Fast Path

`hip_moi::sampled_watchpoint_context` is publish-only by construction. It
accepts only sampled policies with `ReportConflicts=false`.

It is faster than `hip_moi::context + backend_kind::sampled_watchpoint` because
it removes work and live state:

| Area | General `context + sampled_watchpoint` | `sampled_watchpoint_context` |
| --- | --- | --- |
| Diagnostics | Carries diagnostic pointers and emit paths. | No diagnostic state. |
| Epochs | Reads and updates subgroup epoch storage in global memory. | Keeps a local per-thread epoch counter. |
| Backend selection | Can select backend at runtime or compile time. | Only sampled watchpoints. |
| Policy | Runtime or compile-time policy. | Compile-time publish-only policy. |
| Missing metadata | Can emit `metadata_full`. | Drops the publish if storage or encoding is unavailable. |
| Conflict checking | Optional reporting mode. | Not supported. |

This fast path is useful for measuring the overhead of Loom-style sampled
metadata publication and for studying generated code. It is not a sanitizer
mode: it cannot tell a user that a race happened.

## Synchronization Model

`ctx.syncthreads()` performs a real `__syncthreads()`, advances the current
epoch, and performs another `__syncthreads()`. For the general context, thread
0 advances stored subgroup epochs. For `sampled_watchpoint_context`, each
thread increments its local epoch.

The lower-level workgroup-barrier spelling is also supported:

```c++
ctx.release_fence(hip_moi::atomic_memory_scope::workgroup, HIP_MOI_SITE_ID());
ctx.barrier(HIP_MOI_SITE_ID());
ctx.acquire_fence(hip_moi::atomic_memory_scope::workgroup, HIP_MOI_SITE_ID());
```

`ctx.release_fence()` and `ctx.acquire_fence()` emit native fences but do not
interact with the address-scoped atomic-fence state. `ctx.barrier()` performs
the full-workgroup barrier and advances the epoch, so the sequence above is
equivalent to `ctx.syncthreads()` for hip-moi diagnostics. Fences without
`ctx.barrier()` do not advance the epoch and do not order LDS diagnostics.

`hip_moi::context` also models release/acquire synchronization through
instrumented atomic operations, as described in
[Atomic Synchronization Model](#atomic-synchronization-model). Atomic ordering
does not advance the barrier epoch. It records producer epochs observed by
consumer subgroups and lets the LDS conflict predicate suppress conflicts that
are ordered by those acquire observations.

Fence-only modeling is intentionally out of scope. Workgroup fences need
`ctx.barrier()` to form a full-workgroup epoch boundary. Atomic fences need a
paired atomic operation, as described above, to create address-scoped ordering
tokens.

Subgroup-local barriers are not modeled as synchronization operations.

## What The Benchmarks Measure

Benchmark rows use these names consistently:

| Row | Meaning |
| --- | --- |
| `pass-through` | Same kernel shape with no instrumentation. |
| `Jakub-Sampled-Loom` | Local comparison implementation extracted from Jakub's HIP prototype. |
| `context + sampled_watchpoint` | General hip-moi context with sampled backend selected. |
| `sampled_watchpoint_context` | Narrow hip-moi publish-only fast path. |
| `exact shadow` | Precise shadow-memory checking path, used only in the tiny matmul wave-scaling benchmark. |

Latency is not the only metric. For this project, generated-code pressure is
part of the result: VGPR count, spills, private segment size, LDS use, and code
size are all relevant because production kernels are already close to resource
limits.

The benchmark suite and current RDNA4 numbers are documented in
[`benchmarks/README.md`](../benchmarks/README.md).
