<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Atomics Support Plan

This document is the staged implementation plan for adding user-kernel atomics
to hip-moi. The corpus inventory lives in
[`atomics_corpus.md`](atomics_corpus.md). The fast-path design notes live in
[`atomics_fast_paths.md`](atomics_fast_paths.md). This document says what to
build, in which order, and what each stage must prove before the next one
starts.

The plan is RocJITsu-first. Use
`/home/benoit/workspace/rocjitsu-test-corpus` as the default source for
examples. Use `/home/benoit/workspace/hip-matmul/matmul_rdna4.hip` only for
Stream-K shapes that go beyond what the RocJITsu corpus currently provides,
especially RDNA4 WMMA Stream-K arrival counters and Stream-K-tree bitmasks.

## Current Status

This table must be updated at the end of each atomics session. A stage is not
complete merely because code exists; it is complete only when its tests,
benchmarks, documentation, and diligence notes have landed.

| Stage | Status | Current note |
| --- | --- | --- |
| 0. Freeze the corpus map | Complete | `atomics_corpus.md` identifies RocJITsu-first seeds and limits `matmul_rdna4.hip` to missing Stream-K variants. |
| 1. Reference kernels before instrumentation | Complete | `tests/reference/atomic_reference_kernels.hip` adds RocJITsu-derived safe and compile-only atomics source shapes. |
| 2. Public atomic API skeleton | Complete | `hip_moi::context` has pass-through atomic load/store/fetch-add/fetch-or wrappers, `019_atomic_api_test.hip` verifies device behavior, and `019_atomic_flag_handoff_benchmark.hip` records wrapper codegen/latency. |
| 3. Atomic object metadata | Complete | `context::atomic_object_record` is allocated from the host byte budget, release-style atomics populate a bounded generation-separated table, and `020_atomic_metadata_test.hip` / `020_atomic_metadata_benchmark.hip` cover saturation, generation reuse, and codegen cost. |
| 4. Happens-before for LDS payload handoffs | Complete | `021_atomic_happens_before_test.hip` proves release/acquire suppresses an ordered LDS handoff while relaxed publication still diagnoses; `021_atomic_happens_before_benchmark.hip` records the first semantic cost baseline. |
| 5. Release/acquire fast path | Complete | Byte-budget-derived atomic-object capacities round up to powers of two so probe starts use a mask, and acquire lookups stop at the first stale slot in an open-addressing chain. |
| 6. RMW atomics | Complete | `023_atomic_rmw_happens_before_test.hip` covers release/acquire `fetch_add` handoff and a two-RMW `acq_rel` `fetch_add` chain. `024_atomic_or_bitmask_happens_before_test.hip` covers old-value-dependent `atomicOr` bitmask control flow. Both have matching benchmarks and RDNA4 resource notes. |
| 7. RMW fast paths | Complete | Multi-subgroup release-capable RMWs now populate a direct-mapped address-scoped producer-mask cache with generic-table fallback. It improves the four-subgroup WMMA `atomicOr` tree row while leaving two-subgroup/store/fence paths on the generic table. |
| 8. Fences paired with atomics | Complete for first standard shape | `025_atomic_fence_happens_before_test.hip` and matching benchmark cover release-fence-before-relaxed-store paired with relaxed-load-before-acquire-fence. Relaxed-without-fences still diagnoses. Relaxed RMW followed by fences, as seen in some matmul helpers, remains a separate source-model analysis item. |
| 9. Stream-K integration tests | Complete for pre-optimization corpus | `026_streamk_flag_protocol_test.hip`, `027_streamk_two_tile_flag_protocol_test.hip`, `028_rdna4_wmma_streamk_arrival_counter_test.hip`, and `029_rdna4_wmma_streamk_tree_atomic_or_test.hip` add Stream-K-shaped flag, ownership, RDNA4 WMMA arrival-counter, and RDNA4 WMMA bitmask-tree rows. |
| 10. DBI-oriented atomic instruction seeds | Complete | `dbi_atomic_seeds.md` records HipKittens buffer atomics, Stream-K `atomicAdd`/`atomicOr`, hip-stream-k locks, llama count-equal, hip-fpsan atomics, and a Tensile buffer-cmpswap audit signal as DBI seeds separate from source-level HIP diagnostics. |
| 11. Exchange and compare-exchange source shapes | Complete | `030_atomic_exchange_compare_exchange_test.hip` and matching benchmark cover release/acquire exchange, successful acquire compare-exchange lock acquisition, and failed acquire compare-exchange as an acquire load. |
| 12. Fences paired with relaxed RMWs | Complete | `031_atomic_fence_rmw_happens_before_test.hip` and matching benchmark cover release-fence-before-relaxed-fetch-add paired with relaxed-fetch-add-before-acquire-fence. Fence-only synchronization remains out of scope. |
| 13. Fences paired with extended relaxed atomics | Complete | `033_atomic_fence_extended_test.hip` and matching benchmark cover release/acquire fences paired with relaxed exchange, successful relaxed compare-exchange, failed relaxed compare-exchange, and a `seq_cst` load/store sanity row. |
| 14. General-context atomics optimization | Complete | `hip_moi::context` now skips the address-cache probe for one- and two-subgroup acquire imports, where the RMW producer-mask cache is not populated. Synchronization metadata remains exhaustive; VGPRs and spills are unchanged; several atomics rows reduce SGPR pressure and some improve latency. |
| 15. Sampled-reporting atomics correctness | Complete | The sampled-watchpoint reporting backend in the general `context` now consults acquired-epoch tokens before emitting conflicts. `008_sampled_watchpoint_backend_test.hip` covers release/acquire suppression and relaxed diagnostics; `021_atomic_happens_before_benchmark.hip` adds a sampled-reporting row. |

Current semantic trade-off: the atomic-object table is address-scoped. A
release records `(atomic address, producer subgroup, generation)` and an
acquire imports all producer records for that atomic address. This is the
TSan-like direction we want for DBI because it avoids keeping the observed or
released scalar value live in the instrumentation path. The trade-off is
precision: releases through the same address that the acquire did not actually
read from can still be imported, causing false negatives. Address+value keying
is kept as a possible future precision refinement, not the current default.

That future refinement does not have to mean storing a full address tag and a
full value tag. A compact variant could hash `(atomic address, scalar value)`
into a word-sized key and may be better than address-only for protocols where
values distinguish synchronization states. It is still a trade-off, not a free
strict improvement: hash collisions can add unwanted synchronization imports,
and the instrumentation must keep or recompute the atomic scalar value,
normalize its width, and hash it on the hot path. Any address+value experiment
should therefore measure VGPR pressure and generated code before being treated
as a win.

## Stage Completion Checklist

For each implementation stage, complete the following before moving on:

1. Add at least one new instrumented test that exercises the capability added
   in that stage. Reference-only Stage 1 is the sole exception; it must add
   concrete uninstrumented tests that the following implementation stages will
   instrument.
2. Add a matching benchmark for the same source shape. The benchmark must
   include at least:
   * a pass-through mode that performs the same user work without hip-moi
     diagnostics;
   * a `hip_moi::context` mode that exercises the new diagnostic path;
   * any applicable hip-moi fast path once such a fast path exists.
3. Update `benchmarks/README.md` in the same commit so the new benchmark is in
   the catalog, the mode table, the resource-pressure table when applicable,
   and the current RDNA4 results table.
4. Perform generated-code and performance diligence before declaring the stage
   complete. At minimum, inspect resource usage and generated code for the hot
   path, compare benchmark results against pass-through, and record the
   interpretation in the benchmark README or a linked note.
5. Update the Current Status table above, including whether the stage is
   complete, in progress, or blocked.
6. Commit at least once for each completed stage. Finer-grained commits are
   allowed and encouraged when they make review easier, but a stage must not be
   marked complete without a commit that records that completed state.

The intent is to avoid accumulating semantic features that work only as tests
but have obviously poor generated code, unmeasured cost, or undocumented
performance trade-offs.

## Scope

Atomics belong first in `hip_moi::context`. They require a real
synchronization model and deterministic diagnostics. They do not belong in
`hip_moi::sampled_watchpoint_context` until the diagnostic model exists and a
separate publish-only fast path is justified.

## Atomics Optimization Roadmap

The optimization priority is the general `hip_moi::context` class. It is the
only current class that implements deterministic atomics-aware diagnostics.
`hip_moi::sampled_watchpoint_context` remains a publish-only sampled LDS
metadata path; it is not an atomics-aware sanitizer mode.

The staged optimization direction is:

1. Optimize atomics inside the general `context` first. Prefer changes that
   reduce table probes, global metadata traffic, live state, SGPR/VGPR
   pressure, or retry work without changing the public API.
2. Keep synchronization metadata exhaustive. Sampling may thin LDS payload
   observation, but atomics, fences, and acquire imports should not be sampled
   by default because missing a synchronization edge can create false
   positives.
3. Sampled diagnostics with atomics now have a first correctness path: sampled
   conflict reporting in the general `context` consults the same
   acquired-epoch tokens used by exact-shadow diagnostics. This is separate
   from the publish-only `sampled_watchpoint_context`, which still does not
   diagnose races.
4. Consider a separate atomics fast context only if measurements show that the
   general `context` carries irreducible overhead that cannot be optimized away
   locally. Do not create a second class merely to mirror
   `sampled_watchpoint_context`.

Stage 14 starts with a narrow in-place optimization: the RMW producer-mask
cache is populated only for workgroups with more than two subgroups. For one-
and two-subgroup workgroups, acquire imports can therefore skip the cache probe
and go directly to the generic atomic-object table. This preserves the same
address-scoped release/acquire model and keeps synchronization metadata
exhaustive.

The first supported memory-ordering rule should be release/acquire
synchronization through an atomic object:

* a release operation publishes the producer's prior instrumented payload
  accesses;
* an acquire operation synchronizes only when it observes the release operation
  or its release sequence;
* payload accesses ordered by that synchronizes-with edge should not be
  diagnosed as races;
* payload accesses not ordered by such an edge should still diagnose normally.

The diagnostic payload remains LDS access. That is the central scope boundary
for this plan:

* instrumented LDS loads and stores are the accesses whose races hip-moi
  diagnoses;
* atomic operations are synchronization operations, and may be in LDS or global
  memory;
* global atomic support is required because real kernels often use global flags
  or counters to order work;
* ordinary global loads and stores are not diagnostic payloads in this plan.

Fence-only modeling remains out of scope. A fence can matter once it is paired
with an atomic or another operation that can create a synchronizes-with edge,
but a naked fence is not a useful first implementation target.

The first realistic Stream-K seed in RocJITsu uses global atomics:

* helpers write global partial accumulators;
* helpers publish a global flag with release semantics;
* owners acquire-load that flag;
* owners read the global partial accumulators.

hip-moi should use that seed for the synchronization protocol, not as a reason
to become a global-memory race detector. The first diagnostic tests should
adapt the protocol so a global release/acquire flag orders LDS payload accesses
inside one workgroup. Actual Stream-K global partial buffers remain useful
corpus material and later DBI motivation, but they are outside the first
source-level race payload.

## Stage 0: Freeze The Corpus Map

Goal: keep the implementation anchored in concrete kernels.

Inputs:

* RocJITsu `corpus/fuzz-targets/third_party/hip-stream-k`;
* RocJITsu `corpus/fuzz-targets/cases/hip-stream-k`;
* RocJITsu llama.cpp `count-equal.cu` for a tiny global `atomicAdd`;
* RocJITsu llama.cpp `allreduce.cu` as a fence-plus-non-atomic counterexample;
* RocJITsu HipKittens inline `buffer_atomic_pk_add_bf16` for later DBI work;
* `matmul_rdna4.hip` only when RocJITsu lacks the next Stream-K variant.

Deliverables:

* keep `docs/atomics_corpus.md` current;
* record each imported kernel's source path and the reason it was selected;
* mark each source as one of: reference-only, instrumented diagnostic target,
  benchmark target, or DBI instruction seed.

Exit criteria:

* the next reference test to add is unambiguous;
* no `matmul_rdna4.hip` extraction is planned while a RocJITsu source covers
  the same semantic step.

## Stage 1: Reference Kernels Before Instrumentation

Goal: add uninstrumented HIP kernels that make the atomics shapes concrete and
keep them compiling.

Add `tests/reference/atomic_reference_kernels.hip` with host-side oracles where
the uninstrumented program is defined. Suggested initial cases:

| Case | Corpus source | Purpose |
| --- | --- | --- |
| `global_atomic_add_counter` | RocJITsu llama.cpp `count-equal.cu` | Tiny global RMW operation with an easy oracle. |
| `release_acquire_flag_handoff` | RocJITsu hip-stream-k `device_locks.hpp` | Minimal producer writes LDS payload, release-stores global flag, consumer acquire-loads flag, consumer reads LDS payload. |
| `two_helper_flag_handoff` | RocJITsu `simple_streamk` / `two_tile_streamk` | Same atomic flag protocol with more than one helper-like participant so ownership and flag indexing are concrete. |
| `plain_flag_handoff_compile_only` | Derived from hip-stream-k | Negative shape: non-atomic publication. Compile only, because launching it relies on undefined behavior or may hang. |
| `relaxed_flag_handoff_compile_only` | Derived from hip-stream-k | Negative shape: atomic flag without release/acquire ordering. Compile only until hip-moi diagnostics exist. |

Keep these kernels simple. They should preserve the synchronization structure,
not the full GEMM math or global partial-buffer payload. The first
Stream-K-like references should use small integer LDS payloads and a tiny
number of subgroups in one workgroup.

Exit criteria:

* reference tests compile and safe cases run under CTest;
* compile-only negative shapes are present for later instrumented diagnostics;
* the reference README links to the atomics plan;
* the Current Status table marks Stage 1 complete and Stage 2 next. The
  instrumented-test and benchmark obligations begin with Stage 2, where the
  first atomics API exists.

## Stage 2: Public Atomic API Skeleton

Goal: define the source-level API before the detector grows storage.

Add atomic methods to `hip_moi::context` only. The first spelling should mirror
the operations we need from RocJITsu:

```c++
value = ctx.atomic_load(ptr, order, scope, HIP_MOI_SITE_ID());
ctx.atomic_store(ptr, value, order, scope, HIP_MOI_SITE_ID());
old = ctx.atomic_fetch_add(ptr, delta, order, scope, HIP_MOI_SITE_ID());
old = ctx.atomic_fetch_or(ptr, bits, order, scope, HIP_MOI_SITE_ID());
```

Use explicit hip-moi enums for memory order and scope rather than passing raw
integer builtin constants through the public API. The implementation can lower
to HIP/Clang builtins internally.

Do not add global payload access methods in this stage. The payload API remains
the existing LDS API:

```c++
ctx.lds_store_at(ptr, value, /*lds_byte_offset=*/offset, HIP_MOI_SITE_ID());
value = ctx.lds_load_at(ptr, /*lds_byte_offset=*/offset, HIP_MOI_SITE_ID());
```

The new atomic methods may operate on global or LDS atomic objects. Their job
is to create synchronization metadata that can order LDS payload accesses.

Exit criteria:

* pass-through atomic wrappers compile on device;
* tests verify that the wrappers preserve the uninstrumented kernel results;
* no synchronization diagnostics are claimed yet.

Status: complete. The Stage 2 benchmark shows the wrapper path compiling to the
same RDNA4 resource profile as pass-through for the atomic flag handoff row: 4 B
LDS, 3 VGPRs, 10 SGPRs, no private segment, and no spills. This is a skeleton
API stage only. It deliberately does not suppress LDS diagnostics through
release/acquire synchronization.

## Stage 3: Atomic Object Metadata

Goal: record enough state for release/acquire handoff without yet modeling all
RMW corner cases.

Add a metadata table keyed by atomic object address. For each release record,
record at least:

* address key;
* releasing participant;
* releasing participant clock;
* site id for diagnostics;
* generation.

Here, a participant should be the unit that hip-moi currently diagnoses:
subgroup within a workgroup. The exact representation can change, but the
diagnostic message must name what it can actually know.

The first metadata-recording operations:

* release `atomic_store`;
* acquire `atomic_load`;
* release-capable RMWs such as `atomic_fetch_add`, without yet claiming full
  Stream-K final-arriver semantics.

Exit criteria:

* metadata saturation produces deterministic `metadata_full` diagnostics;
* release/acquire metadata is generation-separated across launches;
* diagnostics report atomic source sites when available.

Status: complete. `hip_moi::host_context` now derives an internal
atomic-object table capacity from `storage_bytes`; the default 16 MiB storage
budget provides thousands of records, while small test contexts get a small
bounded table. Release-style atomic stores and release-capable RMW operations
record the atomic address, releasing subgroup, releasing epoch, release site
id, and generation. The table is address-scoped: the scalar value stored,
loaded, or returned by the atomic operation is not part of the current metadata
key. A stale slot is reclaimed through a temporary claim marker, and the probe
sequence starts from a hash of the atomic-object address and producer subgroup.
If no slot can be found or claimed, hip-moi emits a deterministic
`metadata_full` diagnostic carrying the atomic address, byte size, and source
site.

This stage intentionally does not suppress LDS diagnostics. The metadata is
recorded but not yet queried by the LDS conflict predicate. The Stage 3
benchmark shows the first cost baseline after the address-scoped metadata
change: `atomic-metadata-release-store` is 3.44 µs pass-through and 21.1 µs
through `context` on the local RDNA4 machine. Resource counts should be
refreshed before drawing VGPR or SGPR conclusions from this row.

## Stage 4: Happens-Before For LDS Payload Handoffs

Goal: make the minimal hip-stream-k atomic handoff meaningful to the LDS
detector.

Extend the existing LDS conflict check so that an apparent LDS conflict is not
reported when release/acquire atomic synchronization orders the two accesses.
The first conflict predicate should be:

* same generation;
* overlapping LDS byte range;
* different participants;
* at least one write;
* neither access is ordered before the other by the tracked release/acquire
  happens-before state.

The first implementation can be intentionally bounded: support small subgroup
counts, a bounded atomic-object table, and release-store/acquire-load handoffs
first. It does not need to model arbitrary global-memory payloads.

One straightforward correctness-first representation is:

* each subgroup has a logical payload clock, advanced when its LDS epoch
  advances;
* an LDS shadow entry records the writer or reader subgroup and the subgroup's
  clock at the time of the access;
* a release atomic stores the releasing subgroup's current payload clock in the
  atomic-object metadata;
* an acquire atomic that observes the release merges that released clock into
  the acquiring subgroup's acquired state;
* the LDS conflict check asks whether the current subgroup has acquired a clock
  that covers the prior subgroup's recorded access.

The concrete representation can change, but the model must remain explainable
in those terms.

Instrumented tests:

* correct release/acquire flag handoff adapted from RocJITsu hip-stream-k: no
  diagnostic for the ordered LDS handoff;
* missing release/acquire ordering: deterministic diagnostic;
* plain non-atomic flag: deterministic diagnostic or explicit unsupported
  diagnostic, depending on the chosen API;
* stale or wrong flag index: deterministic diagnostic.

Exit criteria:

* the correct RocJITsu-derived flag handoff passes;
* at least two incorrect handoff variants diagnose;
* diagnostics explain both the payload access sites and the atomic sites that
  did or did not order them.

Status: complete for the first minimal release/acquire handoff. The
implementation adds a per-context matrix of acquired epoch tokens indexed by
consumer subgroup and producer subgroup. A release-style atomic records the
producing subgroup and epoch in the atomic-object table. An acquire operation
imports release records for the same atomic address and updates the consumer's
tokens for those producers. The exact-shadow conflict predicate suppresses a
conflict only when the current subgroup has acquired a token covering the prior
subgroup's recorded epoch. This status intentionally accepts address-scoped
over-approximation: extra imported releases can cause false negatives, but
joined release metadata should not cause false positives.

`021_atomic_happens_before_test.hip` covers the key semantic cases: a
release/acquire global flag orders an instrumented LDS payload and reports no
diagnostic; a relaxed flag store with the same LDS payload still reports a
deterministic conflict; and releasing one flag does not order a consumer that
acquires a different relaxed-published flag. The first benchmark row is
`atomic-hb-lds-handoff`: 3.33 µs pass-through and 8.93 µs through `context` on
the local RDNA4 machine after the address-scoped metadata change. Resource
counts should be refreshed before using this row as VGPR or SGPR evidence.

## Stage 5: Release/Acquire Fast Path

Goal: make the first correct atomics implementation cheap enough to be a real
candidate for Stream-K-like kernels.

This fast path still lives under `hip_moi::context`. It is not
`hip_moi::sampled_watchpoint_context`: it remains diagnostic-capable and must
not silently drop conflicts. The target protocol is the RocJITsu hip-stream-k
shape adapted to LDS payloads:

* one or more subgroups write LDS payload;
* a producer subgroup release-stores a flag or counter;
* a consumer subgroup acquire-loads that atomic object;
* the consumer reads LDS payload.

The performance principle is to pay at synchronization points, not at every
LDS access. Atomic operations should update compact per-subgroup ordering
state. LDS instrumentation should keep the existing immediate conflict check
and add only a cheap ordered-before query against that compact state.

The first fast-path representation should aim for:

* a small fixed-capacity or byte-budgeted atomic-object table;
* per-subgroup acquired clocks or tokens, indexed by producer subgroup;
* LDS shadow entries that continue to fit in the existing compact exact-shadow
  shape if possible;
* no scans over all atomic objects during LDS load/store instrumentation;
* deterministic fallback to `metadata_full` or a correctness-first path when
  the fast-path capacity is exceeded.

Exit criteria:

* the Stage 4 correct and incorrect release/acquire tests still pass;
* the hot LDS access path does not perform global atomic metadata lookups;
* a microbenchmark can compare the correctness-first path and the fast path on
  the same RocJITsu-derived flag handoff.

Status: complete for the current release/acquire handoff scope. The first
fast-path change targets the atomic-object table
probe itself. Auto-derived atomic-object capacities now round up to a power of
two, and the device probe-start mapping uses `hash & (capacity - 1)` instead of
runtime modulo. A deliberately bad intermediate experiment rounded capacity
down to a power of two; it made the 4096-workgroup metadata rows fill the table
to 100% and regressed `atomic-metadata-release-store_context` to 1.52 ms and
`atomic-flag-handoff_context` to 4.42 ms.

The second fast-path change targets acquire-side misses. Atomic metadata records
are never deleted within one generation, so a stale slot terminates an
open-addressing probe chain unless another thread is actively claiming that
slot. This mainly helps spin loops before the releasing subgroup has published
metadata. Current local RDNA4 numbers after the address-scoped join change are
21.1 µs for `atomic-metadata-release-store_context`, 45.5 µs for
`atomic-flag-handoff_context`, and 8.93 µs for
`atomic-hb-lds-handoff_context`. The flag-handoff row is the clearest sign that
the acquire side needs a cheaper address-scoped lookup path.

## Stage 6: RMW Atomics As Both Access And Synchronization

Goal: represent read-modify-write atomics correctly enough for counters and
bitmasks.

Start with RocJITsu sources:

* llama.cpp `count-equal.cu` for a tiny global `atomicAdd` reduction;
* hip-stream-k flag protocol if a RMW variant is introduced in that corpus.

Then use `matmul_rdna4.hip` only for Stream-K variants that RocJITsu does not
cover:

* RDNA4 WMMA Split-K `atomicAdd` arrival counter;
* RDNA4 WMMA Stream-K `atomicAdd` final-arriver counter;
* RDNA4 WMMA Stream-K-tree `atomicOr` sibling bitmask.

Implementation requirements:

* an RMW is an atomic read and an atomic write on the atomic object;
* release/acquire semantics on an RMW must update and/or observe the atomic
  object's synchronization metadata;
* diagnostics should distinguish payload races from conflicts on the atomic
  object itself.

Exit criteria:

* a tiny `atomicAdd` reduction reference has an instrumented counterpart;
* an adapted arrival-counter handoff can order LDS payload stores before the
  reducer subgroup reads them;
* an `atomicOr` bitmask test proves that old-value-dependent control flow is
  represented.

Status: complete. The arrival-counter rungs landed in
`023_atomic_rmw_happens_before_test.hip` and
`023_atomic_rmw_happens_before_benchmark.hip`: they cover a producer subgroup
that stores LDS payload, publishes it with a release `fetch_add`, and a
consumer subgroup that observes the counter with an acquire `fetch_add` before
reading the payload. The same test and benchmark also cover a two-RMW
`acq_rel` `fetch_add` chain.

The current implementation supports these RMW handoffs through the same
address-scoped join model as release stores. A release-capable RMW records the
producer subgroup's epoch for the atomic address; an acquire-capable RMW imports
producer records for that address. The old value returned by the RMW is used by
the user kernel's control flow, but not by hip-moi's current synchronization
metadata key. Address+value remains a future precision refinement if false
negative rates on realistic protocols justify the extra VGPR pressure.

Current local RDNA4 rows after the address-scoped join change are 8.57 µs for
`atomic-rmw-arrival-counter_context` and 8.97 µs for
`atomic-rmw-acq-rel-chain_context`.

The bitmask RMW rung landed in
`024_atomic_or_bitmask_happens_before_test.hip` and
`024_atomic_or_bitmask_happens_before_benchmark.hip`. A first subgroup writes
an LDS payload and publishes one bit with release `atomicOr`; a second subgroup
uses the old mask returned by an `acq_rel atomicOr` to decide whether to read
that payload. The relaxed variant still diagnoses. The local RDNA4 benchmark
row is 8.64 µs for `atomic-or-bitmask-handoff_context`.

## Stage 7: RMW Fast Paths

Goal: extend the Stage 5 fast-path idea only where the RMW protocol warrants
it.

Do not optimize all atomics generically up front. The first completed fast
path is address-scoped and internal to the existing API: release-capable RMWs
in workgroups with more than two subgroups populate a direct-mapped
producer-mask cache. Release stores, fence-published relaxed stores, and
two-subgroup RMWs stay on the generic atomic-object table.

Exit criteria:

* each RMW fast path is covered by existing correctness tests and matching
  benchmarks;
* unsupported or capacity-exceeded RMW shapes fall back deterministically;
* the docs say exactly which RMW protocols are optimized.

Status: complete for the current source-level plan. The direct cache stores an
atomic address tag, launch generation, and producer subgroup bit mask. The
generic `atomic_object_record` table remains authoritative for epochs and
source sites. A release-capable RMW sets its producer bit before publishing the
matching generic release record. An acquire uses a matching cache slot to
probe only the producer bits present in that slot, and falls back to the
generic per-producer scan on a miss, stale slot, claimed slot, or address
collision.

The committed fast path is deliberately narrower than the first experiment. A
broader version cached all releases and carried an explicit cache pointer in
`storage_ref`; that raised SGPR pressure in rows that did not benefit. The
committed cache storage is laid out immediately after the generic
atomic-object table and is derived from that pointer on device, so
`storage_ref` does not grow.

Current local RDNA4 rows are:

| Benchmark key | Pass-through | `context` | Context resources |
| --- | ---: | ---: | --- |
| `atomic-rmw-arrival-counter` | 3.45 µs | 8.23 µs | 8 B LDS, 63 SGPR, 23 VGPR, no spills |
| `atomic-rmw-acq-rel-chain` | 3.25 µs | 8.93 µs | 8 B LDS, 63 SGPR, 23 VGPR, no spills |
| `atomic-or-bitmask-handoff` | 3.21 µs | 8.75 µs | 8 B LDS, 63 SGPR, 23 VGPR, no spills |
| `rdna4-wmma-streamk-arrival-counter` | 3.48 µs | 26.6 µs | 4096 B LDS, 79 SGPR, 51 VGPR, no spills |
| `rdna4-wmma-streamk-tree-atomic-or` | 3.77 µs | 43.6 µs | 8192 B LDS, 82 SGPR, 52 VGPR, no spills |

The main positive result is the four-subgroup `atomicOr` tree row: it improves
from the previous 49.2 µs `context` result to 43.6 µs. The limit is equally
important: the cache does not attack exact-shadow LDS instrumentation in the
final fold, and it raises SGPR pressure in the tree row from the previous 78
SGPRs to 82 SGPRs. There are still no spills or private segment use.

## Stage 8: Fences Paired With Atomics

Goal: cover the source patterns that use relaxed atomics plus explicit fences.

Do not model fences alone. Model the pair:

* producer LDS payload writes;
* producer release fence;
* relaxed atomic publication;
* relaxed or acquire atomic observation;
* consumer acquire fence;
* consumer LDS payload reads.

This stage should be driven by corpus evidence. The generic Stream-K comments
in RocJITsu's vendored `hip-matmul/matmul.hip` and the earlier
`matmul_rdna4.hip` examples are the first candidates. RocJITsu llama.cpp
`allreduce.cu` remains a useful counterexample because it uses fences with
non-atomic signaling and should not be mistaken for an atomic protocol.

Exit criteria:

* one relaxed-atomic-plus-fence handoff passes;
* one missing-fence variant diagnoses;
* docs clearly state how the HIP/Clang builtins map to the LLVM IR memory
  model being approximated.

Status: complete for the first standard fence-plus-atomic shape.
`025_atomic_fence_happens_before_test.hip` adds a release fence sequenced before
a relaxed flag store and an acquire fence sequenced after a relaxed flag load.
The fenced variant suppresses the ordered LDS handoff diagnostic; the same
relaxed flag handoff without fences still diagnoses.
`025_atomic_fence_happens_before_benchmark.hip` measures 3.12 µs pass-through
and 6.82 µs through `context` on the local RDNA4 machine.

The implementation records only the simple source-level pattern described
above. A release fence arms the next relaxed atomic publication made through
that `context`; an acquire fence consumes the last relaxed atomic observation
made through that `context`. Some corpus helpers use a relaxed RMW followed by
AMDGPU fences. That pattern is not claimed by this first Stage 8 rung because
its status under the source-level HIP/LLVM memory model needs separate
analysis; it may still be a legitimate target for the later hardware/DBI model.

## Stage 9: Stream-K Integration Tests

Goal: stop testing only distilled handoffs and cover matmul-shaped control
flow.

Preferred order:

1. RocJITsu hip-stream-k simplified math with the original flag protocol;
2. RocJITsu hip-stream-k two-tile ownership shape;
3. `matmul_rdna4.hip` Split-K atomics if a RDNA4 WMMA arrival-counter case is
   needed;
4. `matmul_rdna4.hip` Stream-K-tree only after `atomicOr` support exists.

Keep these as correctness tests first. Add benchmarks only when the semantic
behavior is stable enough that performance numbers are interpretable.

Because hip-moi's source-level diagnostic payload is LDS, these tests should be
clear about what they are validating. RocJITsu-derived Stream-K tests validate
the atomic synchronization protocol and its effect on LDS diagnostics. They do
not claim that hip-moi diagnoses races in Stream-K global partial buffers.

Exit criteria:

* at least one realistic Stream-K-shaped test passes without diagnostics;
* at least one realistic Stream-K-shaped test diagnoses a deliberately broken
  handoff;
* the test README says which source corpus each test came from.

Status: complete for the pre-DBI source-level corpus. The first integration rung landed in
`026_streamk_flag_protocol_test.hip` and
`026_streamk_flag_protocol_benchmark.hip`. It distills RocJITsu hip-stream-k's
owner/helper release/acquire flag protocol to one owner subgroup looping over
two helper flags and folding two LDS helper partials. The ordered variant is
quiet; the broken variant makes the second helper publish with a relaxed store
and diagnoses the corresponding LDS payload handoff. The local RDNA4 benchmark
row is 3.37 µs pass-through and 13.0 µs through `context` after the
address-scoped metadata change.

The second integration rung landed in
`027_streamk_two_tile_flag_protocol_test.hip` and
`027_streamk_two_tile_flag_protocol_benchmark.hip`. It preserves more of the
RocJITsu two-tile ownership shape: four subgroups form two independent
owner/helper tile fixups, with one release/acquire flag per tile. The ordered
variant is quiet; the broken variant makes the second tile's helper publish
with a relaxed store and diagnoses that tile's LDS payload handoff. The local
RDNA4 benchmark row is 3.18 µs pass-through and 13.2 µs through `context`.
The refreshed resource row is 16 B LDS, 93 SGPRs, 60 VGPRs, no spills, and no
private segment.

The third integration rung landed in
`028_rdna4_wmma_streamk_arrival_counter_test.hip` and
`028_rdna4_wmma_streamk_arrival_counter_benchmark.hip`. It is inspired by the
arrival-counter path in `hip-matmul/matmul_rdna4.hip`: two subgroups compute
two RDNA4 WMMA K-slice partials, publish their LDS partials with an
`acq_rel fetch_add` counter, and the final subgroup folds both LDS partials.
The ordered variant is quiet; the relaxed-counter variant diagnoses the final
fold. The local RDNA4 benchmark row is 3.48 µs pass-through and 26.6 µs through
`context`. The refreshed resource row is 4096 B LDS, 79 SGPRs, 51 VGPRs, no
spills, and no private segment. An earlier single-lane fold draft was rejected
because it distorted
the benchmark by making one lane perform every instrumented partial load; the
committed row uses lane-parallel reduction.

The fourth integration rung landed in
`029_rdna4_wmma_streamk_tree_atomic_or_test.hip` and
`029_rdna4_wmma_streamk_tree_atomic_or_benchmark.hip`. It is inspired by the
Stream-K-tree `atomicOr` bitmask idea in `hip-matmul/matmul_rdna4.hip`: four
subgroups compute four RDNA4 WMMA K-slice partials; the first three subgroups
publish bits with release `atomicOr`; the final subgroup waits for those bits,
uses an `acq_rel atomicOr` old-mask observation, and folds all four LDS
partials. The ordered variant is quiet; the relaxed-producer variant diagnoses
the final fold. The local RDNA4 benchmark row is 3.77 µs pass-through and
43.6 µs through `context`. The refreshed resource row is 8192 B LDS, 82 SGPRs,
52 VGPRs, no spills, and no private segment.

Stage 9 has served its purpose for Stage 7. Do not widen this source-level
corpus again until a new source pattern is needed.

## Stage 10: DBI-Oriented Atomic Instruction Seeds

Goal: preserve the bridge to rocjitsu and future dynamic binary
instrumentation.

Keep this track separate from source-level HIP semantics. The first seeds are:

* RocJITsu HipKittens `buffer_atomic_pk_add_bf16`;
* any future RocJITsu or Tensile artifact whose disassembly contains
  `buffer_atomic`, `flat_atomic`, `global_atomic`, `ds_*atomic`, or `s_atomic`.

For each seed, record:

* source path;
* generated instruction;
* address space;
* whether the instruction has source-level memory-ordering semantics available
  in the original HIP code;
* what information rocjitsu would see at DBI time.

Exit criteria:

* source-level tests and DBI instruction seeds are not conflated;
* any DBI note clearly states when the model is hardware-level rather than
  HIP/LLVM-level.

Status: complete as an inventory step. `docs/dbi_atomic_seeds.md` records the
seed table, what rocjitsu would see, and the distinction between workload
atomics and hip-moi detector metadata atomics. The best first future DBI
minimal kernel candidate is HipKittens `buffer_atomic_pk_add_bf16`, because
the instruction is explicit inline assembly rather than an inferred compiler
lowering.

## Stage 11: Exchange And Compare-Exchange Source Shapes

Goal: cover source-level atomic operations that production synchronization
protocols use for locks and token exchange.

Status: complete for the first diagnostic scope. `hip_moi::context` now wraps
`atomic_exchange` and `atomic_compare_exchange_strong`. Exchange is modeled as
a read-modify-write operation: release-capable orders publish the current
subgroup epoch for the atomic address, and acquire-capable orders import
producer epochs for that address.

Compare-exchange is split by outcome. A successful compare-exchange follows
the success memory order and can both publish and acquire. A failed
compare-exchange does not modify the atomic object, so it does not publish a
release record and does not consume a pending release fence. If the failure
memory order is acquire-capable, the failed compare-exchange is modeled as an
acquire load and imports producer epochs for that atomic address.

`030_atomic_exchange_compare_exchange_test.hip` covers:

* release/acquire exchange ordering an LDS payload handoff;
* a lock-like successful acquire compare-exchange after release unlock;
* a failed acquire compare-exchange that observes the published value and
  orders the LDS payload;
* relaxed variants of all three cases, which still diagnose.

`030_atomic_exchange_compare_exchange_benchmark.hip` records local RDNA4 rows:

| Key | Pass-through | `context` | Context resources |
| --- | ---: | ---: | --- |
| `atomic-exchange-handoff` | 3.37 µs | 8.56 µs | 4 B LDS, 62 SGPRs, 23 VGPRs, no spills |
| `atomic-cas-lock-handoff` | 3.06 µs | 7.47 µs | 4 B LDS, 63 SGPRs, 23 VGPRs, no spills |
| `atomic-failed-cas-acquire` | 3.11 µs | 7.10 µs | 4 B LDS, 62 SGPRs, 23 VGPRs, no spills |

## Stage 12: Fences Paired With Relaxed RMWs

Goal: expand raw-fence coverage beyond relaxed store/load flags to the RMW
shape that appears in counter-based protocols.

Status: complete for relaxed `fetch_add`. `031_atomic_fence_rmw_happens_before_test.hip`
uses a release fence sequenced before a relaxed `fetch_add` publication and an
acquire fence sequenced after a relaxed `fetch_add` observation. The paired
fences suppress the ordered LDS handoff diagnostic. The same relaxed RMWs
without fences still diagnose.

The model remains intentionally paired-atomic-only:

* a raw release fence arms the next relaxed atomic operation through the same
  device `context`;
* only an atomic operation that modifies the object can consume that pending
  release fence and publish a release record;
* a raw acquire fence consumes the most recent atomic address observed through
  that context and imports release records for that address;
* a naked fence with no paired atomic operation is not modeled as inter-subgroup
  synchronization.

`031_atomic_fence_rmw_happens_before_benchmark.hip` records the local RDNA4
row:

| Key | Pass-through | `context` | Context resources |
| --- | ---: | ---: | --- |
| `atomic-fence-rmw-handoff` | 3.16 µs | 8.62 µs | 8 B LDS, 57 SGPRs, 23 VGPRs, no spills |

## Stage 13: Fences Paired With Extended Relaxed Atomics

Goal: close the main fence-plus-relaxed-atomic coverage gap for the atomics
that already have release/acquire support: exchange and compare-exchange.

Status: complete for the source-level forms covered by
`030_atomic_exchange_compare_exchange_test.hip`. The new
`033_atomic_fence_extended_test.hip` covers:

* release fence before relaxed exchange paired with relaxed exchange before
  acquire fence;
* release fence before relaxed store paired with successful relaxed
  compare-exchange before acquire fence;
* release fence before relaxed store paired with failed relaxed
  compare-exchange before acquire fence;
* `seq_cst` store/load as the strongest ordinary load/store spelling.

The paired-fence cases suppress the ordered LDS handoff diagnostic. The same
relaxed operations without fences still diagnose. A failed compare-exchange is
only an acquire-side observation; it does not publish a pending release fence
because it did not modify the atomic object.

`033_atomic_fence_extended_benchmark.hip` records the local RDNA4 rows:

| Key | Pass-through | `context` | Context resources |
| --- | ---: | ---: | --- |
| `atomic-fence-exchange` | 3.05 µs | 8.60 µs | 4 B LDS, 61 SGPRs, 21 VGPRs, no spills |
| `atomic-fence-successful-cas` | 2.99 µs | 6.67 µs | 4 B LDS, 62 SGPRs, 21 VGPRs, no spills |
| `atomic-fence-failed-cas` | 3.06 µs | 6.87 µs | 4 B LDS, 62 SGPRs, 21 VGPRs, no spills |
| `atomic-seq-cst-handoff` | 3.09 µs | 8.79 µs | 4 B LDS, 57 SGPRs, 21 VGPRs, no spills |

## Stage 14: General-Context Atomics Optimization

Goal: optimize the diagnostic-capable `hip_moi::context` first, without
creating a separate atomics fast context and without sampling synchronization
metadata.

Status: complete for the first in-place cleanup. The acquire path previously
looked in the direct-mapped RMW producer-mask cache for all workgroup sizes,
then fell back to the generic atomic-object table. That cache is populated only
for release-capable RMWs in workgroups with more than two subgroups. Stage 14
therefore makes one- and two-subgroup acquire imports skip the impossible cache
hit and go directly to the generic table.

This is a generated-code cleanup more than a dramatic latency change. It keeps
VGPR pressure unchanged and introduces no spills. SGPR pressure drops in the
two-subgroup atomics rows, which is useful but less critical than VGPR pressure
for the production-kernel concern that motivates hip-moi.

Selected local RDNA4 rows after the change:

| Key | Pass-through | `context` | Context resources |
| --- | ---: | ---: | --- |
| `atomic-hb-lds-handoff` | 3.32 µs | 8.88 µs | 4 B LDS, 55 SGPRs, 23 VGPRs, no spills |
| `atomic-rmw-arrival-counter` | 3.21 µs | 8.49 µs | 8 B LDS, 57 SGPRs, 23 VGPRs, no spills |
| `atomic-rmw-acq-rel-chain` | 3.22 µs | 8.97 µs | 8 B LDS, 56 SGPRs, 23 VGPRs, no spills |
| `atomic-or-bitmask-handoff` | 3.11 µs | 8.38 µs | 8 B LDS, unchanged VGPR pressure, no spills |
| `atomic-exchange-handoff` | 3.08 µs | 8.61 µs | 4 B LDS, 56 SGPRs, 23 VGPRs, no spills |
| `atomic-cas-lock-handoff` | 2.98 µs | 6.99 µs | 4 B LDS, 57 SGPRs, 23 VGPRs, no spills |
| `atomic-failed-cas-acquire` | 3.05 µs | 6.81 µs | 4 B LDS, 56 SGPRs, 23 VGPRs, no spills |
| `streamk-flag-fixup` | 3.20 µs | 12.5 µs | 12 B LDS, 82 SGPRs, 25 VGPRs, no spills |
| `streamk-two-tile-flag-fixup` | 3.13 µs | 12.5 µs | 16 B LDS, 93 SGPRs, 60 VGPRs, no spills |
| `rdna4-wmma-streamk-arrival-counter` | 3.37 µs | 25.8 µs | 4096 B LDS, 73 SGPRs, 51 VGPRs, no spills |
| `rdna4-wmma-streamk-tree-atomic-or` | 3.62 µs | 42.7 µs | 8192 B LDS, 82 SGPRs, 52 VGPRs, no spills |

## Stage 15: Sampled-Reporting Atomics Correctness

Goal: make the diagnostic-capable sampled-watchpoint backend respect the same
atomic synchronization model as exact shadow.

Status: complete for the first semantic slice. Sampled watchpoint entries
already record subgroup owner and epoch. Stage 15 reuses those fields and asks
the same acquired-token question as exact shadow before emitting a sampled
watchpoint conflict:

```text
acquired_epoch_token[current_owner][prior_owner] >= prior_epoch + 1
```

If the token exists, the sampled conflict candidate is considered ordered by a
supported release/acquire atomic edge and no diagnostic is emitted. If the
release is missing, relaxed, or on a different atomic address, the sampled
reporting path still diagnoses the conflict.

This is not a new publish-only fast path. It applies to
`hip_moi::context` with `backend_kind::sampled_watchpoint` and reports enabled.
`hip_moi::sampled_watchpoint_context` remains publish-only by construction and
does not report races.

Local RDNA4 check:

| Row | Result |
| --- | ---: |
| `HipMoiSampledWatchpointBackend.AtomicReleaseAcquireOrdersSampledDiagnostics` | passed |
| `HipMoiSampledWatchpointBackend.RelaxedAtomicDoesNotOrderSampledDiagnostics` | passed |
| `atomic-hb-lds-handoff_pass-through` | 3.11 µs |
| `atomic-hb-lds-handoff_context` | 8.99 µs |
| `atomic-hb-lds-handoff_context-sampled-reporting` | 76.3 µs |

The sampled-reporting benchmark uses `probe_count=0`, so it scans the entire
watchpoint table and is intentionally much more expensive than exact shadow for
this tiny handoff. Treat it as semantic coverage and as a reminder that
diagnostic sampled reporting is not the Loom-parity publish-only fast path.

## Immediate Next Sessions

1. The current atomics plan is complete through Stage 15.
2. Do not pursue address+value keying unless a later false-negative study
   makes precision, rather than overhead, the blocking problem.
3. If sampled-reporting atomics become performance-relevant, add a low-probe
   benchmark row before optimizing; the current `probe_count=0` row is
   correctness coverage, not a performance target.
4. Additional source-level atomics should be corpus-driven. Likely candidates
   are `fetch_and`, `fetch_xor`, min/max, 64-bit production variants,
   memory-scope-specific tests, and Clang builtin forms that appear in a real
   workload.
5. Any future atomics work should start by deciding whether it belongs to the
   HIP/LLVM source-level detector or to the hardware-level DBI track.
