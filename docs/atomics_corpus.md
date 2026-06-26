<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Atomics Source Provenance

This document records the source kernels that shaped hip-moi's current
source-level atomics support. It is provenance, not a roadmap. The stable
atomics model is [`atomics.md`](atomics.md), current benchmark interpretation
is in [`../benchmarks/README.md`](../benchmarks/README.md), and DBI-oriented
instruction seeds are in [`dbi_atomic_seeds.md`](dbi_atomic_seeds.md).

For hip-moi's source-level detector, atomics are synchronization operations.
The diagnostic payload remains LDS access. Global atomics are included because
real kernels use global flags, counters, and bitmasks to order work; this does
not make hip-moi a general global-memory race detector.

## Current Coverage Map

| Source idea | Origin | hip-moi coverage |
| --- | --- | --- |
| Release/acquire flag handoff | RocJITsu `hip-stream-k` device locks | Reference kernels, `021_atomic_happens_before_test.hip`, `026_streamk_flag_protocol_test.hip`, and matching benchmarks. |
| Two-tile owner/helper flag protocol | RocJITsu `hip-stream-k` two-tile case | `027_streamk_two_tile_flag_protocol_test.hip` and matching benchmark. |
| Arrival counter RMW | `hip-matmul/matmul_rdna4.hip` Split-K and Stream-K kernels | `023_atomic_rmw_happens_before_test.hip`, `028_rdna4_wmma_streamk_arrival_counter_test.hip`, and matching benchmarks. |
| Stream-K-tree bitmask RMW | `hip-matmul/matmul_rdna4.hip` Stream-K-tree kernel | `024_atomic_or_bitmask_happens_before_test.hip`, `029_rdna4_wmma_streamk_tree_atomic_or_test.hip`, and matching benchmarks. |
| Exchange and compare-exchange | Lock-like source patterns and HIP atomics API coverage | `030_atomic_exchange_compare_exchange_test.hip` and matching benchmark. |
| Fences paired with relaxed atomics | Stream-K helper comments and HIP/Clang fence patterns | `025_atomic_fence_happens_before_test.hip`, `031_atomic_fence_rmw_happens_before_test.hip`, `033_atomic_fence_extended_test.hip`, and matching benchmarks. |
| Bitwise AND/XOR RMWs | Siblings of old-value-dependent bitmask protocols | `034_atomic_bitwise_happens_before_test.hip` and matching benchmark. |
| Instruction-level atomics | RocJITsu, HipKittens, hip-matmul, hip-fpsan, and sanitizer-strategy audit notes | Tracked separately in `dbi_atomic_seeds.md`; these are DBI seeds, not necessarily source-level synchronization protocols. |

## RocJITsu Hip-Stream-K Flags

The strongest source-level seed came from
`/home/benoit/workspace/rocjitsu-test-corpus/corpus/fuzz-targets`.

Relevant files:

* `cases/hip-stream-k/simple_streamk/case.json`;
* `cases/hip-stream-k/two_tile_streamk/case.json`;
* `third_party/hip-stream-k/include/streamk/device/device_locks.hpp`;
* `third_party/hip-stream-k/include/streamk/kernel/kernel_simple_streamk.hpp`;
* `third_party/hip-stream-k/include/streamk/kernel/kernel_two_tile_streamk.hpp`.

The protocol is compact and directly relevant:

* helper workgroups write partial accumulators;
* a full-workgroup barrier ensures the helper's waves have issued their
  partial stores;
* one lane publishes a release flag through `__hip_atomic_store`;
* the owner waits with an acquire `__hip_atomic_load`;
* after the acquire observes the release, the owner reads the helper partial.

hip-moi adapts this protocol so the payload is LDS, which keeps the detector's
payload scope unchanged while preserving the synchronization structure.

The first distilled source-level rows are:

* `tests/reference/atomic_reference_kernels.hip`;
* `tests/instrumented/021_atomic_happens_before_test.hip`;
* `benchmarks/021_atomic_happens_before_benchmark.hip`.

The Stream-K-shaped integration rows are:

* `tests/instrumented/026_streamk_flag_protocol_test.hip`;
* `benchmarks/026_streamk_flag_protocol_benchmark.hip`;
* `tests/instrumented/027_streamk_two_tile_flag_protocol_test.hip`;
* `benchmarks/027_streamk_two_tile_flag_protocol_benchmark.hip`.

## RDNA4 Stream-K Counters And Bitmasks

`/home/benoit/workspace/hip-matmul/matmul_rdna4.hip` supplied the RDNA4 WMMA
counter and bitmask shapes that RocJITsu did not cover directly.

The arrival-counter source shapes are:

* `Rdna4WmmaF16SplitKAtomics`;
* `Rdna4WmmaF16StreamK`.

Both use a global counter to identify the last arriving contributor. hip-moi's
source-level extraction keeps the synchronization idea but localizes the
diagnostic payload to LDS partials:

* `tests/instrumented/028_rdna4_wmma_streamk_arrival_counter_test.hip`;
* `benchmarks/028_rdna4_wmma_streamk_arrival_counter_benchmark.hip`.

The bitmask-tree source shape is:

* `Rdna4WmmaF16StreamKTree`.

That kernel uses `atomicOr` and the returned old bitmask to decide whether a
sibling has already published its partial. hip-moi's extraction again keeps
the control-flow and synchronization idea while keeping the diagnostic payload
in LDS:

* `tests/instrumented/029_rdna4_wmma_streamk_tree_atomic_or_test.hip`;
* `benchmarks/029_rdna4_wmma_streamk_tree_atomic_or_benchmark.hip`.

## Smaller Source-Level Seeds

`third_party/llama.cpp/ggml/src/ggml-cuda/count-equal.cu` contains a tiny
device-side `atomicAdd` reduction. It is useful as a global RMW sanity seed,
but it is not a payload handoff: the atomic combines counts rather than
publishing data for another subgroup to consume.

`third_party/llama.cpp/ggml/src/ggml-cuda/allreduce.cu` is a useful
counterexample. Its comments choose volatile stores plus
`__threadfence_system()` for a cross-GPU signal because `atomicAdd_system()` is
not portable for the systems discussed there. It is not part of the current
source-level atomics package, but it remains relevant if future work expands
from atomics to non-atomic signaling plus fences.

`third_party/hipkittens/include/common/macros.cuh` defines
`buffer_atomic_pk_add_bf16` with inline AMDGPU assembly. That is not a
source-level HIP memory-model seed, but it is important for the DBI direction
because the instruction mnemonic and operands are explicit in source.

## Deliberate Non-Seeds

The RocJITsu checkout also contains sources that look atomic-related but did
not drive source-level hip-moi coverage:

* host-side `std::atomic` bookkeeping in vendored tests;
* generated Tensile YAML entries where `streamKAtomic` is disabled;
* IREE scatter-shaped MLIR cases using `unique_indices(true)`;
* CPU, SYCL, Metal, WebGPU, or Vulkan atomics in vendored sources.

These may be useful for future DBI or broad sanitizer corpus work, but they do
not currently justify new HIP source-level atomics API surface.
