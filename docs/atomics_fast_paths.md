<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Atomics Fast-Path Notes

This note records the current fast-path decision point for atomics in
`hip_moi::context`. It is intentionally separate from
[`atomics_plan.md`](atomics_plan.md): the plan says which semantic stages to
build, while this note explains what the first Stream-K-shaped measurements say
about overhead and specialization.

## Current Evidence

The generic atomics implementation is semantically useful now:

* release/acquire flag handoffs suppress ordered LDS payload diagnostics;
* `fetch_add` arrival counters suppress ordered LDS payload diagnostics;
* `atomicOr` bitmask handoffs suppress ordered LDS payload diagnostics;
* release/acquire fences paired with relaxed atomics suppress ordered LDS
  payload diagnostics;
* deliberately relaxed or otherwise broken variants still diagnose.

The most relevant RDNA4 rows are:

| Benchmark key | Shape | Pass-through | `context` |
| --- | --- | ---: | ---: |
| `streamk-flag-fixup` | one owner subgroup, two helper flags | 3.35 µs | 13.3 µs |
| `streamk-two-tile-flag-fixup` | two independent owner/helper tile fixups | 3.19 µs | 12.7 µs |
| `rdna4-wmma-streamk-arrival-counter` | two WMMA K-slice partials, `fetch_add` arrival counter | 3.49 µs | 27.6 µs |

The resource counts for these rows should be refreshed after the
address-scoped metadata change before making VGPR or SGPR claims. The latency
signal is already clear enough to motivate the next fast-path question: the
cost is primarily from the generic metadata protocol and from exact-shadow LDS
instrumentation in the final fold.

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

## Candidate 1: Direct-Mapped RMW Metadata Cache

The least disruptive implementation path is an internal cache behind the
existing `ctx.atomic_fetch_add` and `ctx.atomic_fetch_or` APIs.

The cache would be keyed by a compact hash of:

* atomic object address;
* producer subgroup for release slots;
* launch generation.

Each slot would store:

* address tag;
* releasing subgroup;
* releasing epoch;
* generation.

On release, the fast path writes the direct slot if it is empty or already
matches the same key. On collision, it falls back to the existing generic
open-addressed atomic-object table. On acquire, it checks the direct slot first
and falls back to the generic lookup on miss.

This preserves the public API and can improve the common case where a benchmark
uses one or a few counters with predictable values. It does not require users
to annotate Stream-K protocols. It preserves the current address-scoped
semantics because the generic path remains the fallback; it does not add
address+value precision.

The risk is live state and code size: checking both the fast slot and the
generic fallback can increase register pressure unless the fast path is
carefully factored.

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

The next code experiment should be Candidate 1: add a very small direct-mapped
RMW metadata cache behind the existing API, with fallback to the current
generic table on any mismatch or collision.

Measure it first on:

* `atomic-rmw-arrival-counter`;
* `atomic-or-bitmask-handoff`;
* `streamk-flag-fixup`;
* `streamk-two-tile-flag-fixup`;
* `rdna4-wmma-streamk-arrival-counter`.

Do not remove the generic table. It remains the correctness fallback and the
right representation for dynamic atomic patterns that do not fit the cache.
