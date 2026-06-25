<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Atomics Corpus

This document tracks HIP kernels that should guide hip-moi's next semantic
expansion: user-kernel atomics and the memory ordering they create.

Here, an atomic is a target workload operation such as `atomicAdd`,
`atomicOr`, `atomicCAS`, a Clang/HIP atomic builtin, or a lowered AMDGPU atomic
instruction. Atomics used inside hip-moi's own shadow metadata implementation
are not part of this corpus.

## Current hip-moi Reference State

`tests/reference` currently has no target workload atomics. A search for HIP
atomics, Clang atomics, and Stream-K names in that directory found no matches.

That absence is useful information: the first atomics sessions should add new
reference kernels before adding instrumented tests. Those reference kernels
should be ordinary HIP kernels with host-side checks, following the same shape
as the existing reference corpus.

## Seed Sources

### RDNA4 Split-K Atomic Counter

Source:
`/home/benoit/workspace/hip-matmul/matmul_rdna4.hip`,
`Rdna4WmmaF16SplitKAtomics`.

Relevant structure:

* one workgroup computes one split of a matrix tile;
* each workgroup stores its partial accumulator to a global auxiliary buffer;
* thread 0 performs `atomicAdd(counter, 1)`;
* release/acquire fences at agent scope bracket the atomic handoff;
* the last arriving workgroup reduces all partial accumulators and writes the
  final tile.

Useful lines in the source copy:

* kernel class starts at `matmul_rdna4.hip:1086`;
* partial accumulator is stored before the arrival counter at
  `matmul_rdna4.hip:1153`;
* the atomic handoff is called at `matmul_rdna4.hip:1163`;
* the helper uses `atomicAdd` with release/acquire fences at
  `matmul_rdna4.hip:1176`.

This is the simplest matmul-shaped seed because the coordination primitive is a
single monotonic arrival counter.

### RDNA4 Stream-K Atomic Counter

Source:
`/home/benoit/workspace/hip-matmul/matmul_rdna4.hip`,
`Rdna4WmmaF16StreamK`.

Relevant structure:

* work is distributed over the K-iteration space rather than over whole tiles;
* multiple workgroups may contribute partial accumulators for one output tile;
* each contributor stores its local accumulator to a global auxiliary buffer;
* thread 0 performs `atomicAdd` on a tile counter;
* the final arriving workgroup performs the cross-workgroup reduction.

Useful lines in the source copy:

* kernel class starts at `matmul_rdna4.hip:1213`;
* cooperating workgroup range is computed at `matmul_rdna4.hip:1303`;
* partial accumulator store and atomic handoff are at `matmul_rdna4.hip:1320`;
* the helper uses `atomicAdd` with release/acquire fences at
  `matmul_rdna4.hip:1360`.

This is the first realistic Stream-K seed. It should become a reference test
after smaller atomic-counter handoff tests are in place.

### RDNA4 Stream-K Tree Atomic Bitmask

Source:
`/home/benoit/workspace/hip-matmul/matmul_rdna4.hip`,
`Rdna4WmmaF16StreamKTree`.

Relevant structure:

* workgroups reduce partial accumulators through a tree;
* each tree step stores an accumulator node to a global auxiliary buffer;
* thread 0 performs `atomicOr(counter, bit)`;
* the old counter value tells a workgroup whether its sibling has already
  published the sibling accumulator;
* a workgroup that observes its sibling bit continues the tree reduction.

Useful lines in the source copy:

* kernel class starts at `matmul_rdna4.hip:1401`;
* tree-local accumulator and sibling addresses are formed at
  `matmul_rdna4.hip:1518`;
* the accumulator is stored before the atomic bit publication at
  `matmul_rdna4.hip:1536`;
* the helper uses `atomicOr` with release/acquire fences at
  `matmul_rdna4.hip:1580`.

This seed is more important than a plain counter once hip-moi starts modeling
read-modify-write operations as both synchronization operations and memory
accesses.

### Generic Stream-K And Hybrid Stream-K Tree

Source:
`/home/benoit/workspace/hip-matmul/matmul.hip`.

Relevant structure:

* `MmtKernel_StreamK_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2`
  starts at `matmul.hip:2071`;
* its atomic counter handoff is at `matmul.hip:2282`;
* its helper explains the relaxed atomic plus explicit agent-scope fences at
  `matmul.hip:2390`;
* `MmtKernel_StreamKTree_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2`
  starts at `matmul.hip:2451`;
* the hybrid Stream-K tree starts at `matmul.hip:2832`.

These kernels are larger and less directly aligned with the current RDNA4 test
suite, but they preserve useful design comments and a broader Stream-K shape.
They should inform the reference extraction, especially if the smaller RDNA4
versions accidentally erase an important synchronization detail.

## Extraction Plan

The atomics corpus should grow in stages.

1. Add tiny reference kernels.

   Start with full-kernel HIP examples that isolate one semantic question at a
   time: global `atomicAdd` arrival counter, shared-memory `atomicAdd`, global
   `atomicOr` bit publication, and an `atomicCAS` lock or flag. Each should
   have a host-side oracle and no hip-moi instrumentation.

2. Add fence-plus-atomic publication kernels.

   These should make the intended synchronizes-with edge explicit: producer
   writes payload, producer performs release operation, consumer observes that
   operation with acquire semantics, consumer reads payload. Include incorrect
   variants that omit the ordering operation or use a non-atomic flag.

3. Extract a small RDNA4 Split-K reference.

   Use `Rdna4WmmaF16SplitKAtomics` as the source shape, but keep the first
   extraction smaller than the production benchmark corpus. The goal is to
   preserve the atomic handoff and final-reducer behavior while keeping
   compile time and debugging cost low.

4. Extract a small RDNA4 Stream-K reference.

   Preserve the work-centric K-iteration distribution and the last-arriver
   `atomicAdd` reduction.

5. Extract a small RDNA4 Stream-K-tree reference.

   Preserve the `atomicOr` bitmask protocol and sibling-publication logic.
   This is the first reference that should force the design to reason about
   read-modify-write atomics as both reads and writes.

6. Add instrumented counterparts only after the semantic API is designed.

   Instrumented tests should not blindly mirror the whole reference corpus at
   first. Each new instrumented case should correspond to one implemented
   semantic feature: atomic access representation, release/acquire ordering,
   failed versus successful CAS, global versus LDS atomic scope, or Stream-K
   handoff.

## Open Design Questions For The Corpus

The corpus should help answer these questions before the implementation grows
too much API surface:

* Which atomic operations does hip-moi represent as memory accesses, and with
  what byte ranges?
* Which operations create ordering edges, and at which scope?
* How should failed compare-and-swap be represented?
* Does the first implementation model only LDS atomics, or also global atomics
  when they synchronize global partial accumulators as in Stream-K?
* How does a source-level HIP API express the distinction between relaxed,
  acquire, release, acquire-release, and sequentially consistent operations?
* How much of this survives the later DBI direction, where atomics are decoded
  from AMDGPU instructions rather than from a source-level API?
