<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Scope And Fast Paths

This document states the current scope boundaries. For metadata fields,
watchpoint layout, epoch handling, and diagnostic predicates, see
[`instrumentation_model.md`](instrumentation_model.md).

## Diagnostic Scope

hip-moi is currently subgroup-scoped. The diagnostic condition is a same-epoch
LDS conflict between different subgroups in the same workgroup, with at least
one write.

The current implementation intentionally does not diagnose conflicts wholly
inside one subgroup. That choice is not a statement about the HIP/LLVM memory
model; it is a scoping decision made because the current delivery target is
comparison with Loom-style subgroup-scoped instrumentation.

The implemented synchronization model is full-workgroup barriers:

```c++
ctx.syncthreads();
```

Atomics, release/acquire ordering, fences paired with atomics, and
subgroup-local synchronization are future semantic work.

## Context Split

`hip_moi::context` is the general diagnostic context. It owns the semantic
surface for:

* exact-shadow checking;
* sampled-watchpoint reporting mode;
* `metadata_full` diagnostics;
* host reporting through `HIP_MOI_CHECK` and `host_context` destructors;
* future synchronization models beyond full-workgroup barriers.

`hip_moi::sampled_watchpoint_context` is a narrow publish-only fast view. It is
for benchmark-sensitive kernels that want to measure low-overhead sampled
watchpoint publication. It does not report races.

The benchmark row named `context + sampled_watchpoint` is therefore not the
same thing as `sampled_watchpoint_context`:

| Row | Device object | Diagnostic-capable? | Why it exists |
| --- | --- | --- | --- |
| `context + sampled_watchpoint` | `hip_moi::context` | Yes, when reporting is enabled | Measures the general API path with the sampled backend selected. |
| `sampled_watchpoint_context` | `hip_moi::sampled_watchpoint_context` | No | Measures the narrow publish-only fast path. |

The split is intentional. The fast view removes diagnostic state, runtime
backend selection, global subgroup epoch storage, and saturation reports from
the hot kernel. Those omissions are exactly why it can have lower VGPR pressure
and fewer spills. They are also why it is not a substitute for the diagnostic
context.

## Benchmark Scope

The benchmark suite currently covers:

* tiny matmul wave-scaling rows for fast overhead triage;
* a production-shaped FP16 RDNA4 matmul extraction from Jakub's benchmark;
* dense attention rows that stress scalar LDS score and softmax-weight scratch;
* D128/V128 attention pressure rows with larger K/V LDS staging;
* no-score/register-handoff attention rows that are closer to mature
  flash-attention structure.

The benchmark catalog, resource-pressure table, mode definitions, and current
RDNA4 timings live in [`../benchmarks/README.md`](../benchmarks/README.md).

The main current performance lesson is not "always use the fast view." The
lesson is narrower:

* publish-only sampled metadata can be made very cheap when the hot path keeps
  little state live;
* diagnostic-capable paths carry real overhead because they preserve reporting,
  saturation handling, and future semantic room;
* attention-like workloads expose different bottlenecks than isolated matmul,
  especially when scalar score/weight LDS scratch is instrumented.

## Next Scope Increase

The next semantic expansion is atomics. It belongs first in
`hip_moi::context`, because atomics require a real synchronization model beyond
the local epoch used by `sampled_watchpoint_context`.

The atomics design should specify:

* which HIP/Clang operations are represented;
* which LLVM/HIP memory-ordering rules are being modeled;
* what metadata records the synchronization state;
* how conflicts are diagnosed;
* which shortcuts, if any, are allowed in publish-only benchmark paths.

Fence-only modeling remains out of scope. A fence becomes useful for
inter-thread synchronization only together with an operation, typically an
atomic, that can create a synchronizes-with edge.
