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

Fence-only modeling remains out of scope. A fence can matter once it is paired
with an atomic or another operation that can create a synchronizes-with edge,
but a naked fence is not a useful first implementation target.

The first realistic Stream-K seed in RocJITsu uses global memory:

* helpers write global partial accumulators;
* helpers publish a global flag with release semantics;
* owners acquire-load that flag;
* owners read the global partial accumulators.

That means atomics support should not be treated as a small extension to the
current LDS-only detector. The staged plan below introduces global payload
instrumentation deliberately, with a narrow first use case: global handoff
buffers participating in atomic synchronization protocols.

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
| `release_acquire_flag_handoff` | RocJITsu hip-stream-k `device_locks.hpp` | Minimal producer writes payload, release-stores flag, consumer acquire-loads flag, consumer reads payload. |
| `two_helper_flag_handoff` | RocJITsu `simple_streamk` / `two_tile_streamk` | Same protocol with more than one helper so ownership and flag indexing are concrete. |
| `plain_flag_handoff_compile_only` | Derived from hip-stream-k | Negative shape: non-atomic publication. Compile only, because launching it relies on undefined behavior or may hang. |
| `relaxed_flag_handoff_compile_only` | Derived from hip-stream-k | Negative shape: atomic flag without release/acquire ordering. Compile only until hip-moi diagnostics exist. |

Keep these kernels simple. They should preserve the synchronization structure,
not the full GEMM math. The first Stream-K-like references should use small
integer payloads and a tiny number of workgroups.

Exit criteria:

* reference tests compile and safe cases run under CTest;
* compile-only negative shapes are present for later instrumented diagnostics;
* the reference README links to the atomics plan.

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

The first API should also add global payload access methods, because the first
real Stream-K protocol publishes global partial buffers:

```c++
ctx.global_store(ptr, value, HIP_MOI_SITE_ID());
value = ctx.global_load(ptr, HIP_MOI_SITE_ID());
```

These methods should be documented as instrumented payload accesses, not as a
general promise that every global memory access in a HIP program is interposed.

Exit criteria:

* pass-through atomic wrappers compile on device;
* tests verify that the wrappers preserve the uninstrumented kernel results;
* no synchronization diagnostics are claimed yet.

## Stage 3: Atomic Object Metadata

Goal: record enough state for release/acquire handoff without yet modeling all
RMW corner cases.

Add a metadata table keyed by atomic object address. For each atomic object,
record at least:

* address key;
* last released value or release sequence identifier;
* releasing participant;
* releasing participant clock;
* site id for diagnostics;
* generation.

Here, a participant should be the unit that hip-moi currently diagnoses:
subgroup within a workgroup for LDS, and a workgroup/subgroup pair for global
handoff payloads. The exact representation can change, but the diagnostic
message must name what it can actually know.

The first supported operations:

* release `atomic_store`;
* acquire `atomic_load`;
* acquire-release `atomic_fetch_add` as a recorded RMW, without yet depending
  on Stream-K final-arriver logic.

Exit criteria:

* metadata saturation produces deterministic `metadata_full` diagnostics;
* release/acquire metadata is generation-separated across launches;
* diagnostics report atomic source sites when available.

## Stage 4: Happens-Before For Global Payload Handoffs

Goal: make the minimal hip-stream-k handoff meaningful to the detector.

Add global payload shadow records for instrumented `global_load` and
`global_store`. The first conflict predicate should be:

* same generation;
* overlapping global byte range;
* different participants;
* at least one write;
* neither access is ordered before the other by the tracked release/acquire
  happens-before state.

The first implementation can be intentionally bounded: support small test
participant counts and global handoff buffers first. It does not need to be a
complete global-memory race detector for arbitrary production kernels.

Instrumented tests:

* correct release/acquire flag handoff from RocJITsu hip-stream-k shape: no
  diagnostic;
* missing release/acquire ordering: deterministic diagnostic;
* plain non-atomic flag: deterministic diagnostic or explicit unsupported
  diagnostic, depending on the chosen API;
* stale or wrong flag index: deterministic diagnostic.

Exit criteria:

* the correct RocJITsu-derived flag handoff passes;
* at least two incorrect handoff variants diagnose;
* diagnostics explain both the payload access sites and the atomic sites that
  did or did not order them.

## Stage 5: RMW Atomics As Both Access And Synchronization

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
* an arrival-counter handoff can order payload stores before the final reducer;
* an `atomicOr` bitmask test proves that old-value-dependent control flow is
  represented.

## Stage 6: Fences Paired With Atomics

Goal: cover the source patterns that use relaxed atomics plus explicit fences.

Do not model fences alone. Model the pair:

* producer payload writes;
* producer release fence;
* relaxed atomic publication;
* relaxed or acquire atomic observation;
* consumer acquire fence;
* consumer payload reads.

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

## Stage 7: Stream-K Integration Tests

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

Exit criteria:

* at least one realistic Stream-K-shaped test passes without diagnostics;
* at least one realistic Stream-K-shaped test diagnoses a deliberately broken
  handoff;
* the test README says which source corpus each test came from.

## Stage 8: DBI-Oriented Atomic Instruction Seeds

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
4. Add the first global payload access API and tests that prove pass-through
   correctness.
5. Design and document the first atomic object metadata layout before enabling
   diagnostics.
