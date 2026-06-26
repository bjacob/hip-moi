<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Atomics Model

This document is the delivery-facing description of hip-moi's current atomics
support. It explains what the source-level API models, what metadata is
recorded, when that metadata is used, what precision is intentionally lost, and
what the measured cost currently means.

The staged build history lives in [`atomics_plan.md`](atomics_plan.md). The
kernel corpus that motivated the stages lives in
[`atomics_corpus.md`](atomics_corpus.md). Fast-path notes live in
[`atomics_fast_paths.md`](atomics_fast_paths.md).

## Scope

Atomics are synchronization operations in hip-moi. They are not the race
payload. The race payload remains instrumented LDS loads and stores.

This distinction is the central scope boundary:

* hip-moi diagnoses LDS conflicts between different subgroups in one workgroup;
* source-level atomic operations may be in LDS or global memory;
* global atomics are supported because real kernels use global flags, counters,
  and bitmasks to order LDS or other payload work;
* ordinary global loads and stores are not diagnosed as payload races.

hip-moi only sees operations written through its API. A raw `atomicAdd`, a raw
`__hip_atomic_*` call, a raw Clang builtin, or inline assembly is invisible to
the source-level detector unless a user or source rewriter replaces it with the
corresponding hip-moi call.

## Public API

The current atomics API is implemented on `hip_moi::context`:

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

The API uses `hip_moi::atomic_memory_order` and
`hip_moi::atomic_memory_scope`. The supported orders are `relaxed`, `acquire`,
`release`, `acq_rel`, and `seq_cst`. The supported scopes are `workgroup`,
`agent`, and `system`.

The implementation performs the user atomic operation with HIP/Clang atomic
builtins, then separately records or imports hip-moi ordering metadata. The
metadata is only used by later LDS conflict checks.

## Epochs And Acquired Tokens

Each subgroup has one barrier epoch. `ctx.syncthreads()` and the lower-level
`ctx.release_fence(scope); ctx.barrier(); ctx.acquire_fence(scope);` spelling
advance that epoch. Atomic operations do not create a second epoch hierarchy.

Atomics add pairwise ordering evidence:

```text
acquired_epoch_token[consumer_subgroup][producer_subgroup]
```

A token value of `N + 1` means that the consumer subgroup has acquired producer
epoch `N`. Zero means that no producer epoch has been acquired. The `+1`
encoding lets epoch zero be represented.

During LDS conflict checking, hip-moi first applies the raw conflict predicate:
same launch generation, same barrier epoch, different subgroup owners, and at
least one write. If that raw predicate is true, the detector asks one more
question:

```text
Has the current subgroup acquired the prior subgroup's recorded epoch?
```

If yes, the LDS conflict is treated as ordered and no diagnostic is emitted. If
no, hip-moi emits an `access_conflict` diagnostic.

## Release Records

Release-capable atomic operations publish one metadata fact:

```text
(atomic address, producer subgroup, producer epoch, launch generation, release site)
```

The fact is stored in `context::atomic_object_record`. The atomic object may be
in global memory or LDS. Multiple producer subgroups using the same atomic
address occupy separate rows. Multiple releases by the same producer subgroup
to the same address join by keeping the maximum released epoch.

Acquire-capable atomic operations perform the user atomic first, then import
producer rows for that atomic address. For each imported row, hip-moi updates
the pairwise acquired token for the acquiring subgroup and the releasing
subgroup.

This model is address-scoped. The scalar value stored, loaded, or returned by
the atomic operation is not part of the metadata key.

## Operation Semantics

The following table summarizes the modeled source-level synchronization:

| Source-level operation through `hip_moi::context` | Metadata effect |
| --- | --- |
| Release store | Publishes a release record for the atomic address. |
| Acquire load | Imports release records for the atomic address. |
| `fetch_add`, `fetch_or`, `fetch_and`, `fetch_xor`, or `exchange` with release-capable order | Performs the RMW and publishes a release record. |
| `fetch_add`, `fetch_or`, `fetch_and`, `fetch_xor`, or `exchange` with acquire-capable order | Performs the RMW and imports release records. |
| Successful compare-exchange | Uses the success order; may publish and/or import. |
| Failed compare-exchange | Does not publish; acquire-capable failure order imports like an acquire load. |
| `seq_cst` load/store/RMW | Modeled through the corresponding acquire and/or release effects. |
| Relaxed atomic without paired fences | Performs the user atomic but does not order LDS diagnostics. |

Acquire-capable RMW and compare-exchange paths retry metadata import briefly.
The reason is that the hardware atomic can become visible before the releasing
thread has finished publishing hip-moi's metadata for the same source-level
operation. The retry is bounded; if metadata is still missing, hip-moi leaves
the LDS conflict diagnosable.

## Fences Paired With Atomics

hip-moi supports the standard fence-plus-atomic pattern:

```c++
ctx.atomic_fence(hip_moi::atomic_memory_order::release, scope, HIP_MOI_SITE_ID());
ctx.atomic_store(flag, value, hip_moi::atomic_memory_order::relaxed, scope, HIP_MOI_SITE_ID());

observed = ctx.atomic_load(flag, hip_moi::atomic_memory_order::relaxed, scope, HIP_MOI_SITE_ID());
ctx.atomic_fence(hip_moi::atomic_memory_order::acquire, scope, HIP_MOI_SITE_ID());
```

The release fence records the current subgroup epoch as pending state inside
the device context. The next relaxed atomic store, exchange, or successful RMW
consumes that pending state and publishes a release record for the atomic
address. A relaxed atomic load or RMW remembers the atomic address it observed.
A later acquire fence imports release records for that remembered address.

A failed compare-exchange cannot publish a pending release fence because it
does not modify the atomic object. It can still be the acquire-side observation
consumed by a following acquire fence.

Naked atomic fences are not treated as inter-subgroup synchronization. They
emit the native fence, but without a paired atomic operation there is no
address-scoped object through which hip-moi can build a synchronizes-with edge.

Do not confuse `ctx.atomic_fence` with the lower-level full-workgroup barrier
spelling:

```c++
ctx.release_fence(scope, HIP_MOI_SITE_ID());
ctx.barrier(HIP_MOI_SITE_ID());
ctx.acquire_fence(scope, HIP_MOI_SITE_ID());
```

That sequence is equivalent to `ctx.syncthreads()` for hip-moi diagnostics
because `ctx.barrier()` advances the full-workgroup epoch. The release/acquire
fences in that sequence are workgroup-barrier fences, not address-scoped atomic
fence state.

## Address-Only Precision Trade-Off

The current join key is the atomic address, not `(address, value)`.

This is intentional. Address-only metadata avoids keeping the scalar atomic
value live in the instrumentation path, avoids normalizing value width, and is
a natural shape for future dynamic binary instrumentation where the atomic
object address is visible at instruction level. It also keeps the current
metadata rows compact.

The trade-off is false negatives. If multiple release operations publish
through the same atomic address, an acquire imports all producer records for
that address even when the source-level acquire did not actually read from each
release. That extra import can suppress a real unordered LDS conflict. It
should not create false positives, because release records are joined and
missing metadata leaves conflicts diagnosable.

Address+value keying remains a future precision experiment. A compact version
could hash `(atomic address, scalar value)` into one word. That may improve
protocols where values distinguish states, such as monotonic counters or
bitmasks. It is not a free win: the value must remain live or be recomputed,
the width must be normalized, hashing costs instructions, and hash collisions
can still create false negatives.

## Fast Path

The implemented atomics fast path is deliberately narrow. Release-capable RMWs
in workgroups with more than two subgroups populate a direct-mapped
address-scoped producer-mask cache:

```text
(atomic address, launch generation) -> producer subgroup mask
```

The cache is not authoritative. It only tells acquire operations which producer
subgroups are worth probing in the generic `atomic_object_record` table. If the
cache slot is stale, empty, claimed, or holds a different address, hip-moi falls
back to the generic per-producer table scan.

Release stores, fence-published relaxed stores, and two-subgroup RMWs stay on
the generic path. Measurements showed that the cache did not pay for itself on
those shapes. The main current win is the four-subgroup RDNA4 WMMA Stream-K
`atomicOr` tree row.

## Sampled Watchpoints And Atomics

`hip_moi::sampled_watchpoint_context` does not implement atomics-aware
diagnostics. It is a publish-only fast path and cannot report races.

The general `hip_moi::context` can run a sampled-watchpoint backend with
reporting enabled. In that mode, sampled conflict checks consult the same
acquired epoch tokens as exact shadow. This proves that sampled diagnostics can
respect supported atomic synchronization, but the current benchmark row is not
a fast path: it intentionally scans the watchpoint table and is much slower
than the publish-only context.

Sampling LDS payload observations is compatible with atomics only if
synchronization metadata remains exhaustive. Sampling or dropping release and
acquire imports can create false positives because the detector would miss an
ordering edge that actually exists in the user program.

## Current Coverage

Current source-level coverage includes:

* release/acquire load/store;
* `fetch_add`, `fetch_or`, `fetch_and`, and `fetch_xor`;
* `exchange`;
* successful and failed `compare_exchange_strong`;
* `seq_cst` sanity coverage;
* release/acquire fences paired with relaxed store, load, exchange, successful
  compare-exchange, failed compare-exchange, and RMW operations;
* Stream-K-shaped flag, arrival-counter, and bitmask-tree integration rows.

Current intentional gaps:

* raw atomics not written through the hip-moi API;
* ordinary global load/store race payloads;
* source-level min/max atomics, until a real source-level LDS synchronization
  protocol motivates them;
* address+value precision;
* sampled publish-only atomics diagnostics;
* DBI-level raw instruction semantics.

## Current Measurements

The refreshed RDNA4 atomics rows are all spill-free in the current
`hip_moi::context` implementation. The small two-subgroup microbenchmarks
mostly land in the 6.7 to 9.0 microsecond range through `context`, versus about
3 microseconds pass-through. The Stream-K-shaped integration rows are more
expensive:

| Benchmark key | Pass-through | `context` | Context resources |
| --- | ---: | ---: | --- |
| `atomic-hb-lds-handoff` | 3.11 µs | 8.99 µs | 4 B LDS, 55 SGPR, 23 VGPR, no spills |
| `atomic-rmw-arrival-counter` | 3.21 µs | 8.49 µs | 8 B LDS, 57 SGPR, 23 VGPR, no spills |
| `atomic-fence-handoff` | 3.10 µs | 6.83 µs | 4 B LDS, 57 SGPR, 23 VGPR, no spills |
| `atomic-and-bitmask-handoff` | 2.93 µs | 8.99 µs | 4 B LDS, 59 SGPR, 23 VGPR, no spills |
| `atomic-xor-bitmask-handoff` | 3.08 µs | 8.85 µs | 4 B LDS, 59 SGPR, 23 VGPR, no spills |
| `streamk-flag-fixup` | 3.20 µs | 12.5 µs | 12 B LDS, 82 SGPR, 25 VGPR, no spills |
| `streamk-two-tile-flag-fixup` | 3.13 µs | 12.5 µs | 16 B LDS, 93 SGPR, 60 VGPR, no spills |
| `rdna4-wmma-streamk-arrival-counter` | 3.37 µs | 25.8 µs | 4096 B LDS, 73 SGPR, 51 VGPR, no spills |
| `rdna4-wmma-streamk-tree-atomic-or` | 3.62 µs | 42.7 µs | 8192 B LDS, 82 SGPR, 52 VGPR, no spills |

The key interpretation is that VGPR pressure remains controlled and there are
no spills in the current atomics rows. The remaining overhead is mostly
metadata protocol cost: generic table probes, bounded acquire retries,
exact-shadow LDS instrumentation, and global metadata traffic.

Full benchmark context is in [`benchmarks/README.md`](../benchmarks/README.md).
