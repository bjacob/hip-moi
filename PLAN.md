# PLAN for hip-moi: HIP memory-ordering instrumentation library

## Delivery Objective

hip-moi is now in a delivery phase. The primary recipient is Jakub. The project
should produce precise, inspectable information that helps Jakub:

* improve Loom;
* participate in the LLVM GPU ThreadSanitizer RFC discussion;
* understand what a HIP source-level instrumentation API can and cannot do;
* compare diagnostic capability against publish-only performance paths;
* reason about overhead using latency, VGPR pressure, spills, private segment
  size, LDS usage, and generated-code shape.

The repository should no longer read like a worklog. Durable facts belong in
`README.md`, `docs/`, `benchmarks/README.md`, and focused source comments.
`PLAN.md` records only current scope, status, and next work.

## Project Purpose

hip-moi has two purposes.

1. Provide a source-level HIP instrumentation prototype for LDS
   memory-ordering diagnostics under the HIP/LLVM memory model.
2. Provide a concrete stepping stone toward future assembly-level
   instrumentation, where the relevant model is the hardware execution model
   and Loom-style subgroup-scoped instrumentation is the main comparison point.

The second purpose drives current prioritization. hip-moi is not trying to be a
general per-thread HIP race detector. The active product surface is
subgroup-scoped and focused on cross-subgroup LDS interactions inside a
workgroup.

## Current Public Surface

The diagnostic-capable device context is:

```c++
hip_moi::context
```

The host-side owner is:

```c++
hip_moi::host_context
```

The publish-only fast view is:

```c++
hip_moi::sampled_watchpoint_context
```

All active LDS access helpers require an explicit LDS byte offset:

```c++
ctx.lds_store_at(ptr, value, /*lds_byte_offset=*/offset, HIP_MOI_SITE_ID());
value = ctx.lds_load_at(ptr, /*lds_byte_offset=*/offset, HIP_MOI_SITE_ID());
```

The pointer performs the actual LDS access. The explicit offset is the address
representation recorded by the metadata backends.

## Detector Contract

The current diagnostic condition is:

* two instrumented LDS accesses overlap in represented LDS byte range;
* the accesses come from different subgroups in the same workgroup;
* the accesses are in the same epoch;
* at least one access is a write.

`ctx.syncthreads()` performs a real full-workgroup barrier and advances the
epoch. The current implemented synchronization model is not a model for
atomics, fences, release/acquire ordering, or subgroup-local barriers.

Fence-only modeling is intentionally out of scope. Fence semantics become
useful for inter-thread synchronization when paired with operations, typically
atomics, that can create synchronizes-with edges.

## Implementation Status

The active implementation has two metadata backends in `hip_moi::context`.

`backend_kind::exact_shadow`:

* stores one packed 64-bit entry per 4-byte LDS cell;
* records access kind, subgroup owner, epoch, generation, and truncated site id;
* atomically exchanges the shadow cell at each instrumented access;
* emits deterministic diagnostics for represented same-epoch cross-subgroup
  conflicts.

`backend_kind::sampled_watchpoint`:

* stores packed 64-bit watchpoint records in a global table;
* represents contiguous 4-byte LDS cell ranges;
* selects subgroup/site instances through a deterministic sampled policy;
* lets one selected lane publish metadata;
* optionally checks displaced and probed watchpoint entries in reporting mode.

`hip_moi::sampled_watchpoint_context` is narrower:

* sampled watchpoint publication only;
* compile-time publish-only policy;
* local per-thread epoch counter advanced at full-workgroup barriers;
* no diagnostic buffer, no `metadata_full` reports, no conflict checking.

That fast view exists to measure a Loom-like publication path with low live
state. It is not a sanitizer mode because it cannot report races.

The exact metadata fields, bit layouts, algorithms, and predicates are
documented in `docs/instrumentation_model.md`.

## Benchmark Status

The benchmark suite is documented in `benchmarks/README.md`. Current benchmark
rows use these names:

* `pass-through`: same kernel shape without instrumentation;
* `Jakub-Sampled-Loom`: Jakub's sampled publish-only HIP comparison path;
* `context + sampled_watchpoint`: general diagnostic-capable hip-moi context
  with sampled backend selected;
* `sampled_watchpoint_context`: narrow hip-moi publish-only fast path;
* `exact shadow`: precise exact-shadow checking, present only in the tiny
  matmul wave-scaling benchmark.

Current stable conclusions:

* On the production-shaped matmul extraction,
  `sampled_watchpoint_context` is faster than the local Jakub-Sampled-Loom row.
* The general `context + sampled_watchpoint` row remains much slower because it
  keeps diagnostic state and runtime flexibility live in the hot kernel.
* Dense attention benchmarks are scalar-LDS stress tests and remain expensive.
* No-score/register-handoff attention benchmarks are closer to production
  attention structure and much closer to the pass-through baseline.
* LDS pressure alone is not sufficient to characterize overhead; VGPR pressure,
  spills, private segment size, and code size are first-class metrics.

## Test Status

The active test corpus is documented in `tests/README.md` and
`tests/instrumented/README.md`.

The instrumented suite now focuses on:

* host reporting behavior;
* site ids;
* simulated hard-synchronization diagnostics;
* shadow and watchpoint ABI predicates;
* explicit LDS-offset APIs;
* exact-shadow diagnostics;
* sampled-watchpoint diagnostics and publish-only fast execution;
* RDNA4 matmul and attention-shaped correctness tests.

The reference suite remains useful as concrete uninstrumented HIP code for
safe-kernel validation and benchmark-shape provenance. Racy reference kernels
are compile-only unless an instrumented test consumes the shape safely.

## Documentation Status

Current entry points:

* `README.md`: project orientation and reading map;
* `docs/README.md`: documentation index;
* `docs/instrumentation_model.md`: canonical implementation model;
* `docs/context.md`: context allocation and usage examples;
* `docs/tutorial/README.md`: tested tutorial examples;
* `benchmarks/README.md`: benchmark catalog, modes, resource pressure, and
  current RDNA4 results.

Documentation should keep using the same terms:

* workgroup, not block;
* subgroup for the instrumentation owner;
* lane for a thread's index inside a subgroup;
* pass-through for uninstrumented benchmark rows;
* Jakub-Sampled-Loom for Jakub's local HIP comparison path;
* watchpoint for hip-moi's software sampled range record, not a hardware
  mechanism.

The Markdown sweep is complete as of this session: the old chronological
`PLAN.md` has been replaced by this delivery plan, stale attention benchmark
promises have been rewritten as implemented benchmark mappings, and the test
READMEs now describe the current detector scope.

## Next Work

1. Write a Loom/RFC comparison document.

   This should map hip-moi concepts to real Loom, Jakub-Sampled-Loom, and the
   RFC vocabulary: metadata layout, ownership, epoch advancement, access-time
   checking, sampling, false-negative sources, and diagnostic capabilities.

2. Write a benchmark interpretation document.

   The benchmark README records numbers. A separate delivery document should
   explain what those numbers imply for Loom and the RFC: which overheads come
   from metadata traffic, which come from register pressure, which paths spill,
   and which workload shapes are most informative.

3. Tighten source comments around benchmark-local Jakub-Sampled-Loom code.

   The benchmark sources contain a local comparison implementation. Comments
   should make clear that this is Jakub's HIP prototype shape, not upstream
   Loom itself.

4. Plan the next semantic expansion: atomics.

   Atomics belong first in `hip_moi::context`, not in
   `sampled_watchpoint_context`. The design should state exactly which LLVM/HIP
   operations create synchronization, how epochs or happens-before state are
   represented, and how diagnostics remain explainable.

## Non-Goals

These are explicitly out of scope for the current delivery phase:

* reintroducing per-thread same-subgroup diagnostics;
* claiming that `sampled_watchpoint_context` is a diagnostic sanitizer mode;
* modeling fences without the atomic or other operation that can create a
  synchronizes-with edge;
* adding public API for speculative regular-access coalescing without a proven
  low-overhead implementation;
* treating benchmark-local Jakub-Sampled-Loom code as authoritative upstream
  Loom.

## Session Workflow

At the end of each substantive session:

1. update `PLAN.md`;
2. ensure tests pass;
3. run `git clang-format`;
4. commit the session.
