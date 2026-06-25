<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Atomics Support Plan

This document is the staged implementation plan for adding user-kernel atomics
to hip-moi. The corpus inventory lives in
[`atomics_corpus.md`](atomics_corpus.md). This document says what to build, in
which order, and what each stage must prove before the next one starts.

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
| 7. RMW fast paths | Deferred | Stage 6 rows are measurable and spill-free; do not add benchmark-specific shortcuts until a realistic Stream-K integration row shows which RMW protocol needs a fast path. |
| 8. Fences paired with atomics | Complete for first standard shape | `025_atomic_fence_happens_before_test.hip` and matching benchmark cover release-fence-before-relaxed-store paired with relaxed-load-before-acquire-fence. Relaxed-without-fences still diagnoses. Relaxed RMW followed by fences, as seen in some matmul helpers, remains a separate source-model analysis item. |
| 9. Stream-K integration tests | Not started | Starts after the relevant atomic protocols are supported. |
| 10. DBI-oriented atomic instruction seeds | Not started | Kept separate from source-level HIP diagnostics. |

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

Add a metadata table keyed by atomic object address and released value. For
each release record, record at least:

* address key;
* released value or release sequence identifier;
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
record address, value, releasing subgroup, releasing epoch, release site id,
and generation. A stale slot is reclaimed through a temporary claim marker, and
the probe sequence starts from a hash of the atomic-object address and released
value. If no slot can be found or claimed, hip-moi emits a deterministic
`metadata_full` diagnostic carrying the atomic address, byte size, and source
site.

This stage intentionally does not suppress LDS diagnostics. The metadata is
recorded but not yet queried by the LDS conflict predicate. The Stage 3
benchmark shows the first cost baseline: `atomic-metadata-release-store` is
3.43 µs pass-through and 15.5 µs through `context` on the local RDNA4 machine,
with 23 VGPRs, 37 SGPRs, no private segment, and no spills in the `context`
kernel.

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
producing subgroup and epoch in the atomic-object table. An acquire load that
observes the released value updates the consumer's token for that producer. The
exact-shadow conflict predicate suppresses a conflict only when the current
subgroup has acquired a token covering the prior subgroup's recorded epoch.

`021_atomic_happens_before_test.hip` covers the key semantic cases: a
release/acquire global flag orders an instrumented LDS payload and reports no
diagnostic; a relaxed flag store with the same LDS payload still reports a
deterministic conflict; and releasing one flag does not order a consumer that
acquires a different relaxed-published flag. Stale value reuse remains a good
Stage 5/6 guardrail. The first benchmark row is `atomic-hb-lds-handoff`: 3.34
µs pass-through and 13.6 µs through `context` on the local RDNA4 machine, with
21 VGPRs, 55 SGPRs, 4 B LDS, no private segment, and no spills in the `context`
kernel.

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
metadata. Current local RDNA4 numbers after the Stage 6 value-sensitive metadata
change are 15.5 µs for `atomic-metadata-release-store_context`, 33.5 µs for
`atomic-flag-handoff_context`, and 8.77 µs for
`atomic-hb-lds-handoff_context`.

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

Supporting the `acq_rel` chain required changing atomic metadata from an
address-only key to an `(address, released value)` key. This lets records for
consecutive counter values coexist long enough for the later RMW to acquire the
earlier RMW's release metadata. Current local RDNA4 rows are 7.16 µs for
`atomic-rmw-arrival-counter_context` with 8 B LDS, 23 VGPRs, 54 SGPRs, no
private segment, and no spills, and 8.59 µs for
`atomic-rmw-acq-rel-chain_context` with 8 B LDS, 25 VGPRs, 54 SGPRs, no private
segment, and no spills.

The bitmask RMW rung landed in
`024_atomic_or_bitmask_happens_before_test.hip` and
`024_atomic_or_bitmask_happens_before_benchmark.hip`. A first subgroup writes
an LDS payload and publishes one bit with release `atomicOr`; a second subgroup
uses the old mask returned by an `acq_rel atomicOr` to decide whether to read
that payload. The relaxed variant still diagnoses. The local RDNA4 benchmark
row is 8.52 µs for `atomic-or-bitmask-handoff_context`, with 8 B LDS, 24 VGPRs,
54 SGPRs, no private segment, and no spills.

## Stage 7: RMW Fast Paths

Goal: extend the Stage 5 fast-path idea only where the RMW protocol warrants
it.

Do not optimize all atomics generically up front. Add protocol-specific fast
paths after correctness tests demonstrate the needed semantics:

* `atomicAdd` arrival counters: cache the release state associated with the
  counter value or last-arriver observation;
* `atomicOr` bitmasks: cache the release state associated with published bits
  and the old value returned by the RMW.

Exit criteria:

* each RMW fast path has a correctness-first test and a fast-path test;
* unsupported or capacity-exceeded RMW shapes fall back deterministically;
* the docs say exactly which RMW protocols are optimized.

Status: deferred. The completed Stage 6 microbenchmarks are all spill-free and
cluster around 7 to 9 µs through `context`. That is slow relative to
pass-through but not yet enough evidence to justify a source API that bakes in
a benchmark-specific arrival-counter or bitmask shortcut. The next fast-path
attempt should be driven by a realistic Stream-K integration row: if a concrete
counter or bitmask protocol dominates that row, add a protocol-specific API and
benchmark it there.

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
and 10.2 µs through `context` on the local RDNA4 machine. Resource usage is 4 B
LDS, 3 VGPRs, 10 SGPRs, no spills for pass-through, and 4 B LDS, 23 VGPRs, 56
SGPRs, no spills for `context`.

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

## Immediate Next Sessions

1. Add `tests/reference/atomic_reference_kernels.hip` with the tiny
   RocJITsu-derived `atomicAdd` and release/acquire flag handoff cases.
2. Add compile-only negative reference shapes for plain and relaxed flag
   handoffs.
3. Add the `hip_moi::context` atomic API skeleton as pass-through wrappers.
4. Add pass-through tests that use global atomics to order LDS payload code,
   while still checking ordinary kernel results.
5. Design and document the first atomic object metadata layout before enabling
   diagnostics.
6. Implement release/acquire ordering for LDS conflict suppression, then add
   the release/acquire fast path before widening to RMW protocols.
