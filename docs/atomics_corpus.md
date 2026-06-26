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

## Current hip-moi Corpus State

`tests/reference/atomic_reference_kernels.hip` now contains the first
RocJITsu-derived safe atomics references and compile-only broken handoff
shapes. `tests/instrumented` contains source-level HIP atomics tests through
release/acquire flags, RMW handoffs, fence-plus-atomic handoffs, and
Stream-K-shaped RDNA4 WMMA rows. `docs/dbi_atomic_seeds.md` separately records
instruction-level DBI seeds.

## Seed Sources

The priority rule is RocJITsu-first. Prefer
`/home/benoit/workspace/rocjitsu-test-corpus` for reference and instrumented
tests. Use `/home/benoit/workspace/hip-matmul/matmul_rdna4.hip` only for
Stream-K examples that go beyond what RocJITsu currently provides, especially
RDNA4 WMMA arrival counters and Stream-K-tree bitmasks.
For hip-moi's source-level detector, extracted diagnostic tests should keep the
race payload in LDS; global atomics are synchronization seeds, not an expansion
to ordinary global load/store race detection.

### RocJITsu Test Corpus: hip-stream-k Release/Acquire Flags

Source:
`/home/benoit/workspace/rocjitsu-test-corpus/corpus/fuzz-targets`.

The strongest new seed is the extracted `hip-stream-k` corpus:

* `cases/hip-stream-k/simple_streamk/case.json`
* `cases/hip-stream-k/two_tile_streamk/case.json`
* `third_party/hip-stream-k/include/streamk/device/device_locks.hpp`
* `third_party/hip-stream-k/include/streamk/kernel/kernel_simple_streamk.hpp`
* `third_party/hip-stream-k/include/streamk/kernel/kernel_two_tile_streamk.hpp`

Both runnable cases are CDNA3-targeted GEMM cases with shape
`m=256, n=256, k=256, grid=4` and correctness validation enabled.

The synchronization protocol is small and directly relevant:

* flags are initialized to 1, meaning locked;
* helper CTAs store partial accumulators to a global partials buffer;
* helpers execute a full-workgroup barrier so all waves have issued their
  partial stores;
* one lane calls `unlock`, which uses `__hip_atomic_store` with
  `__ATOMIC_RELEASE` and `__HIP_MEMORY_SCOPE_AGENT`;
* owner CTAs call `wait_for_unlock`, which spins on `__hip_atomic_load` with
  `__ATOMIC_ACQUIRE` and `__HIP_MEMORY_SCOPE_AGENT`;
* after the acquire load observes the released flag, the owner reads the
  helper's partial accumulator and folds it into the final tile.

Useful lines in the source copy:

* `device_locks.hpp:36` defines the release-store `unlock`;
* `device_locks.hpp:48` defines the acquire-load `wait_for_unlock`;
* `kernel_simple_streamk.hpp:271` stores partials, synchronizes the block, and
  calls `unlock`;
* `kernel_simple_streamk.hpp:290` waits for helper CTAs and reads their
  partials;
* `kernel_two_tile_streamk.hpp:267` has the same helper publication path;
* `simple_streamk.hpp:105` and `two_tile_streamk.hpp:51` document the host-side
  flag initialization contract.

This is now the preferred first real Stream-K seed. Compared to the
`hip-matmul` Stream-K classes below, it isolates an explicit release/acquire
flag protocol rather than hiding synchronization inside an arrival-counter
helper.

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

### RocJITsu Test Corpus: Additional Atomics Signals

The same `rocjitsu-test-corpus` checkout has several lower-priority atomics
signals.

`third_party/hip-matmul/matmul.hip` repeats the generic Stream-K seeds already
listed above: `atomicAdd` for a counter handoff and `atomicOr` for tree-style
bit publication.

Stage 16 adds source-level `atomic_fetch_and` and `atomic_fetch_xor` coverage as
bitmask protocol siblings of `atomicOr`, not because the current corpus has a
stronger AND/XOR signal than OR. The goal is to exercise the same old-value
dependent RMW synchronization shape with bit clearing and bit toggling. The
same search did not find a strong source-level LDS handoff requiring min/max;
min/max remains better classified as instruction-level or DBI coverage until a
real source synchronization protocol needs it.

`third_party/llama.cpp/ggml/src/ggml-cuda/count-equal.cu` contains a simple
device-side `atomicAdd` from one thread per block into an integer output
counter. This is a useful tiny global-atomic reduction seed, but it is not a
memory-ordering handoff: the atomic combines counts rather than publishing
payload data to a consumer.

`third_party/llama.cpp/ggml/src/ggml-cuda/allreduce.cu` is a useful non-atomic
counterexample. Its comments explicitly choose volatile stores plus
`__threadfence_system()` for a cross-GPU signal because `atomicAdd_system()` is
not portable on the target systems discussed there. This should not be treated
as an atomics seed, but it is relevant once the corpus expands from atomics to
fences paired with non-atomic signaling.

`third_party/hipkittens/include/common/macros.cuh` defines
`buffer_atomic_pk_add_bf16` with inline AMDGPU assembly. This is not a
source-level HIP memory-model seed for the first implementation pass, but it is
important for the later DBI direction because it gives us a concrete
instruction-level atomic form to decode.

The `cases/llama.cpp/noncont_batched_matmul_hazard` override contains
host-side `std::atomic` bookkeeping. That is not a device-kernel atomics seed
for hip-moi.

The packaged Tensile corpus is useful for RocJITsu coverage but is not
currently a strong source-level atomics seed:

* generated `TensileLibrary.yaml` files contain 2794 `streamKAtomic` entries,
  all set to 0 in this checkout;
* a scan for `buffer_atomic`, `flat_atomic`, `global_atomic`, `ds_*atomic`, and
  `s_atomic` in the packaged artifacts found no textual generated instruction
  hits;
* some sparse and gradient configs use nonzero `GlobalSplitU`, but that does
  not by itself establish an emitted atomic in these generated artifacts.

The IREE subcorpus contains scatter-shaped MLIR cases, but the checked-in MLIR
uses `iree_linalg_ext.scatter` with `unique_indices(true)`. That is useful
general DBI corpus material, but not an atomics seed unless a later lowering
inspection proves that the generated GPU code emits atomic instructions.

The broad search also found many host-side or non-HIP atomics in vendored
CPU/SYCL/Metal/WebGPU/Vulkan sources. Those are not hip-moi source-level HIP
atomics seeds.

## Corpus Extraction Priorities

The implementation roadmap is in [`atomics_plan.md`](atomics_plan.md). This
section only records the corpus order that should feed that plan.

1. Add RocJITsu-derived tiny reference kernels.

   Start with full-kernel HIP examples that isolate one semantic question at a
   time. The first two should come from RocJITsu: llama.cpp `count-equal.cu`
   for a tiny global `atomicAdd`, and hip-stream-k `device_locks.hpp` for a
   release/acquire flag handoff. Each safe case should have a host-side oracle
   and no hip-moi instrumentation.

2. Add RocJITsu-derived release/acquire publication kernels.

   These should make the intended synchronizes-with edge explicit: producer
   writes payload, producer performs release operation, consumer observes that
   operation with acquire semantics, consumer reads payload. Include incorrect
   variants that omit the ordering operation or use a non-atomic flag.

3. Extract a small hip-stream-k-style release/acquire flag reference from
   RocJITsu.

   Use the RocJITsu `hip-stream-k` extract as the source shape. The first
   version should preserve only the core protocol: helper writes payload,
   helper publishes a release flag, owner acquire-loads the flag, owner reads
   payload. This is the best bridge from tiny atomics tests to real Stream-K
   synchronization.

4. Extract a RocJITsu two-helper or two-tile Stream-K reference.

   Preserve the original flag indexing and owner/helper relationship from
   `simple_streamk` or `two_tile_streamk`, while keeping the math small.

5. Add `matmul_rdna4.hip` Split-K only if RDNA4 WMMA arrival counters become
   the next missing Stream-K shape.

   Use `Rdna4WmmaF16SplitKAtomics` as the source shape, but keep the first
   extraction smaller than the production benchmark corpus. The goal is to
   preserve the atomic handoff and final-reducer behavior while keeping
   compile time and debugging cost low.

6. Add `matmul_rdna4.hip` Stream-K only if RocJITsu's flag-based Stream-K
   corpus is no longer enough.

   Preserve the work-centric K-iteration distribution and the last-arriver
   `atomicAdd` reduction.

7. Add `matmul_rdna4.hip` Stream-K-tree after `atomicOr` support exists.

   Preserve the `atomicOr` bitmask protocol and sibling-publication logic.
   This is the first reference that should force the design to reason about
   read-modify-write atomics as both reads and writes.

8. Add instruction-level DBI-oriented atomic seeds from RocJITsu.

   Keep these separate from source-level semantic tests. The first candidates
   are HipKittens `buffer_atomic_pk_add_bf16` and any future Tensile artifacts
   whose disassembly actually contains `buffer_atomic`, `flat_atomic`,
   `global_atomic`, `ds_*atomic`, or `s_atomic`.

9. Add instrumented counterparts only after the semantic API is designed.

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
