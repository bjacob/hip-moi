<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Scope And Fast Paths

hip-moi currently has two different device-side shapes.

`hip_moi::context` is the general diagnostic context. It is the right place for
correctness-first behavior: exact shadow checks, sampled reporting mode,
metadata saturation diagnostics, host reporting, and future synchronization
models.

`hip_moi::sampled_watchpoint_context` is a narrower sampled publish-only fast
path. It exists for benchmark-sensitive kernels that use full-workgroup
barriers and want a low-overhead Loom-style metadata publication path. It omits
the cold diagnostic state carried by `hip_moi::context`.

The benchmark row named `context + sampled_watchpoint` is different from
`sampled_watchpoint_context`. The former keeps the full `hip_moi::context`
object live and selects its sampled backend. The latter constructs the
dedicated fast-view class, which carries only the watchpoint table, generation,
subgroup size, and a local epoch counter. That split is the current way to keep
apples-to-apples measurements of the general path and the publish-only fast
path.

This split is intentional. The current fastest RDNA4 matmul path uses local-only
epoch tracking inside `sampled_watchpoint_context`; represented watchpoints
carry that local epoch, and no reporting path consumes a global epoch word. That
is a valid fast-path assumption for the current publish-only benchmark. It is
not a general answer to synchronization.

## Current Benchmark Shape

The vendored `benchmarks/prod_16x8_benchmark.hip` row compares:

* noop matmul,
* Jakub-style sampled Loom publish-only instrumentation,
* hip-moi general `context` with the `sampled_watchpoint` backend,
* hip-moi `sampled_watchpoint_context` publish-only instrumentation.

At the current `4096^3` production shape, the narrow hip-moi
`sampled_watchpoint_context` row is below sampled Loom on this benchmark, while
the general `context + sampled_watchpoint` row remains much slower. That is
expected: the fast row is intentionally publish-only and omits general
diagnostic state. The remaining near-term project risk is now workload breadth,
starting with an attention block, rather than more matmul-only heroics.

The first attention benchmark is not yet the production-pressure endpoint. It
uses RDNA4 WMMA and instruments all LDS traffic, but its fixed LDS footprint is
only 4352 B on a device with 64 KiB of workgroup LDS. Source mining points to a
next attention row shaped by production MHA parameters: head dimension 128,
query heads 32 or 64, KV heads 4 or 8, long sequences, and causal/no-mask
variants. The microkernel should be compile-probed for LDS usage, VGPRs, and
spills before instrumentation, because closeness to the LDS/VGPR limits is now
the point of the next benchmark.

## Next Scopes

The next non-negotiable scope increase is workload breadth. The first likely
candidate is an attention block, because it should stress phase structure, LDS
reuse, and multiple tiled operations more realistically than one isolated
matmul. The right first step is a benchmark/reference workload, not a new hidden
special case in the library.

The next negotiable semantic increase is synchronization finer than global
`__syncthreads()`, likely involving atomics. That work belongs first in
`hip_moi::context`, because atomics require a synchronization model beyond a
single local epoch advanced at full-workgroup barriers.

Fence-only reasoning remains insufficient. Fences become meaningful for
inter-thread synchronization when paired with operations, typically atomics,
that can create synchronizes-with edges. Any atomics work should model those
edges explicitly rather than trying to stretch the sampled publish-only fast
path beyond its assumptions.
