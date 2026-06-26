<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Atomics Fast-Path Notes

This note records the current fast-path decision point for atomics in
`hip_moi::context`. It is intentionally separate from
[`atomics.md`](atomics.md) and [`atomics_plan.md`](atomics_plan.md): the model
defines current semantics, the plan tracks implementation history and future
work, and this note explains what the Stream-K-shaped measurements say about
overhead and specialization.

## Delivery Decision Record

The current atomics fast-path position is:

| Question | Current decision | Reason |
| --- | --- | --- |
| Should atomics use `sampled_watchpoint_context`? | No. | That context is publish-only and cannot report races. Atomics need exhaustive synchronization metadata so the detector does not miss real ordering edges and create false positives. |
| Should release/acquire metadata stay address-scoped? | Yes, for now. | Address-only metadata avoids keeping scalar atomic values live, keeps the source-level prototype closer to future DBI, and has measured spill-free codegen. Address+value remains a precision experiment, not the default. |
| Should the generic atomic-object table be removed? | No. | It is the authoritative correctness fallback, carries source-site information, and supports dynamic atomic patterns that do not fit the direct cache. |
| Should we keep tuning the generic acquire loop? | Not without new evidence. | Stage 17 rejected the two obvious local trims: conditional acquired-token publication and a special two-subgroup direct lookup. |
| What is the next plausible speedup? | A protocol-aware or DBI-informed path. | Remaining cost is dominated by metadata protocol work and global metadata traffic, not VGPR spills. |

For Loom and RFC discussion, the key message is that source-level atomics
support is semantically useful and already spill-free, but the generic
address-scoped metadata protocol is not the performance endpoint. It is a
reference point for what information must be represented before a lower-level
or protocol-specialized implementation can safely remove work.

## Current Evidence

The generic atomics implementation is semantically useful now:

* release/acquire flag handoffs suppress ordered LDS payload diagnostics;
* `fetch_add` arrival counters suppress ordered LDS payload diagnostics;
* `atomicOr` bitmask handoffs suppress ordered LDS payload diagnostics;
* release/acquire fences paired with relaxed atomics suppress ordered LDS
  payload diagnostics;
* deliberately relaxed or otherwise broken variants still diagnose.

The most relevant RDNA4 rows after the Stage 17 performance audit are:

| Benchmark key | Shape | Pass-through | `context` | `context` resources |
| --- | --- | ---: | ---: | --- |
| `streamk-flag-fixup` | one owner subgroup, two helper flags | 3.22 µs | 12.8 µs | 12 B LDS, 84 SGPR, 26 VGPR, no spills |
| `streamk-two-tile-flag-fixup` | two independent owner/helper tile fixups | 3.07 µs | 12.4 µs | 16 B LDS, 94 SGPR, 61 VGPR, no spills |
| `rdna4-wmma-streamk-arrival-counter` | two WMMA K-slice partials, `fetch_add` arrival counter | 3.40 µs | 25.5 µs | 4096 B LDS, 75 SGPR, 51 VGPR, no spills |
| `rdna4-wmma-streamk-tree-atomic-or` | four WMMA K-slice partials, `atomicOr` bitmask tree | 3.66 µs | 45.2 µs | 8192 B LDS, 84 SGPR, 52 VGPR, no spills |

The latency and resource signals now point in the same direction: the current
rows are spill-free, but the generic address-scoped metadata path is expensive
enough to dominate the Stream-K-shaped rows. The cost is primarily from the
generic metadata protocol and from exact-shadow LDS instrumentation in the
final fold.

Stage 17 also ruled out two narrower local shortcuts. Skipping the
acquired-token `atomicMax` after a volatile read found a new-enough token
looked like an easy way to reduce global atomic traffic, but it produced false
diagnostics in the four-subgroup `atomicOr` tree. The acquired-token write is a
publication point used by later LDS conflict checks. A special direct lookup
for two-subgroup handoffs looked like an easy way to remove a loop, but it
regressed the shared-context `atomic-flag-handoff` row from roughly 43 µs to
roughly 270 µs. Those failures make the current recommendation stricter:
future speedups should be protocol-aware, not generic loop shaving.

## Why The Generic Path Costs More

For a releasing RMW such as `ctx.atomic_fetch_add`, the current path:

1. executes the user atomic operation;
2. finds or claims an address-scoped atomic-object metadata slot for
   `(atomic address, producer subgroup)`;
3. joins the releasing subgroup epoch into that slot;
4. uses a thread fence before making the slot visible for the current launch
   generation.

For an acquiring RMW, the current path:

1. executes the user atomic operation and obtains the old value;
2. looks up producer metadata slots for the atomic address;
3. retries because an acquiring RMW can observe the hardware atomic before the
   releasing thread has published hip-moi metadata;
4. writes acquired producer epochs into the per-consumer/per-producer
   acquired-epoch matrix. The old value remains available to the user kernel,
   but hip-moi does not use it for the current metadata key.

For each instrumented LDS access, the exact-shadow path:

1. maps the LDS byte range to one or more 4-byte cells;
2. atomically exchanges each shadow cell with the current access metadata;
3. checks whether the prior entry conflicts;
4. queries the acquired-epoch matrix before deciding whether to diagnose.

The address-scoped key is a TSan-like approximation of the memory-model
reads-from relation. It is intentionally coarse: an acquire imports releases
through the same atomic address even if the acquire did not read from each of
those releases. That can suppress real LDS conflicts, so it can create false
negatives. It should not create false positives as long as release metadata is
joined rather than overwritten.

That is the right conservative structure for the current DBI-oriented
prototype. Address+value keying remains a possible future precision refinement
when realistic protocols justify keeping observed or released values live. A
compact version could hash `(atomic address, scalar value)` into a word-sized
key, which may reduce address-only false negatives for monotonic counters or
bitmasks. It would still introduce a probabilistic collision source and value
liveness/codegen cost, so it must be measured rather than assumed to be a free
upgrade.

## What A Safe Fast Path Must Preserve

A fast path must preserve the source-level contract:

* synchronization comes from HIP/LLVM atomic memory order, not from assumed
  hardware lockstep;
* ordinary global loads and stores are not diagnostic payloads;
* LDS accesses remain the diagnostic payload;
* the detector may use compact metadata, but it must still suppress only those
  LDS conflicts ordered by a tracked or deliberately over-approximated
  synchronizes-with edge;
* metadata collisions or unsupported dynamic shapes must fail conservatively,
  either by falling back to the generic path or by leaving the conflict
  diagnosable.

The fast path should not silently treat every atomic RMW as a workgroup barrier.
The current acceptable over-approximation is scoped to the atomic object
address.

## Implemented Stage 7: Direct-Mapped RMW Address Cache

The implemented fast path is an internal cache behind release-capable RMW APIs
such as `ctx.atomic_fetch_add`, `ctx.atomic_fetch_or`, `ctx.atomic_fetch_and`,
and `ctx.atomic_fetch_xor`. It is deliberately narrower than the first sketch:
only release-capable RMWs in workgroups with more than two subgroups populate
the cache.

The cache is keyed by a compact hash of:

* atomic object address;
* launch generation.

Each slot would store:

* address tag;
* producer subgroup bit mask;
* generation.

The generic `atomic_object_record` table remains authoritative for epochs and
source sites. The cache is only a prefilter: on release, hip-moi adds the
producer subgroup bit before publishing the corresponding generic record; on
acquire, hip-moi checks the direct slot first and probes only the producer bits
present in that slot. If the slot is absent, stale, claimed, or holds a
different address, acquire falls back to the generic per-producer table scan.

This preserves the public API and can improve the common case where a benchmark
uses one or a few counters with predictable values and more than two subgroups.
It does not require users to annotate Stream-K protocols. It preserves the
current address-scoped semantics because the generic table remains the
authoritative record; it does not add address+value precision.

Two design details matter for overhead:

* release stores and fence-published relaxed stores do not use this cache;
* the cache storage is laid out immediately after the generic atomic-object
  table and is derived from that pointer on device, so `storage_ref` did not
  grow another launch argument.

The first broader cache attempt applied to every release operation and carried
an explicit cache pointer in `storage_ref`; that raised SGPR pressure in rows
that did not benefit. The committed version avoids that broader tax. The
remaining trade-off is visible in the RDNA4 resource rows: the four-subgroup
`atomicOr` tree gets a clear latency win, but SGPR pressure still rises from
the pre-Stage-7 78 SGPRs to 84 SGPRs. VGPRs, spills, and private segment size
do not change.

## Candidate 2: Explicit Stream-K Counter Slots

A stronger but more intrusive option is a new opt-in API that lets the user
name a small counter slot:

```c++
old = ctx.streamk_fetch_add(counter_slot, ptr, delta, order, scope, site);
old = ctx.streamk_fetch_or(counter_slot, ptr, bits, order, scope, site);
```

The slot would avoid generic address/probe logic on the common path. It could
store a tiny value-to-epoch record only if a later address+value precision mode
is justified for a known Stream-K protocol.

This is likely faster, but it changes the source instrumentation contract. It
asks the user or source rewriter to identify Stream-K counters explicitly. That
is a bigger departure from “meet HIP code where it is,” so it should not be the
first implementation unless the direct-mapped cache fails.

## Candidate 3: DBI-Level Specialization

At the future rocjitsu/DBI level, the instrumentation can see machine
instructions, register values, and hardware execution structure. A hardware
model may justify a much narrower Stream-K counter representation.

That does not directly justify changing `hip_moi::context`. hip-moi remains a
source-level HIP/LLVM prototype. DBI-oriented conclusions should be recorded as
separate notes, not smuggled into the HIP memory model.

## Current Recommendation

Treat the Stage 7 cache as the end of the current source-level atomics fast
path. It is useful enough to keep for multi-subgroup RMW protocols, but it is
not a general solution to atomics overhead. The next major optimization should
come from a new corpus-driven protocol shape or from the future DBI track, not
from adding address+value keying by default.

Do not remove the generic table. It remains the correctness fallback, the
source-site carrier for diagnostics, and the right representation for dynamic
atomic patterns that do not fit the direct cache.
