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
* at least one access is a write;
* `hip_moi::context` cannot prove that supported atomic release/acquire
  synchronization orders the two accesses.

`ctx.syncthreads()` performs a real full-workgroup barrier and advances the
epoch. The lower-level workgroup spelling
`ctx.release_fence(hip_moi::atomic_memory_scope::workgroup); ctx.barrier();
ctx.acquire_fence(hip_moi::atomic_memory_scope::workgroup);` is also supported;
`ctx.barrier()` is the epoch boundary, and the fence wrappers emit native
fences so source instrumentation can mirror HIP or Clang builtin spellings.
`hip_moi::context` also models release/acquire ordering through instrumented
atomic operations by recording address-scoped release metadata and pairwise
acquired-epoch tokens between producer and consumer subgroups.

Fence-only modeling is intentionally out of scope. Workgroup fences need
`ctx.barrier()` to order LDS diagnostics; atomic fences need a paired atomic
operation that can create synchronizes-with edges.

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
* The ping-pong benchmarks are scheduling-sensitive RDNA4 rows. The private-LDS
  row uses the same optimized `setprio`/`sched_barrier`/WMMA kernel shape that
  the ATT probe validates. The wide cooperative row adds real cross-subgroup
  LDS sharing. Both now time pass-through, `context + sampled_watchpoint`, and
  `sampled_watchpoint_context`.
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
* RDNA4 matmul and attention-shaped correctness tests;
* RDNA4 ping-pong-shaped correctness tests with private LDS staging, minimal
  cooperative LDS staging, wider pairwise cooperative LDS staging, `setprio`,
  `sched_barrier`, WMMA, and exact-shadow diagnostics for intentionally
  unsynchronized cooperative shapes.
* an optimized RDNA4 ping-pong ATT probe and timing benchmark that validates
  dynamic `s_setprio`/LDS/WMMA ordering in ROCprof's decoded per-wave trace
  output, including complementary LDS-priority signatures across representative
  SIMD selections, and then reports pass-through versus sampled hip-moi
  latency.

The reference suite remains useful as concrete uninstrumented HIP code for
safe-kernel validation and benchmark-shape provenance. Racy reference kernels
are compile-only unless an instrumented test consumes the shape safely.

## Documentation Status

Current entry points:

* `README.md`: project orientation and reading map;
* `docs/README.md`: documentation index;
* `docs/instrumentation_model.md`: canonical implementation model;
* `docs/context.md`: context allocation and usage examples;
* `docs/tutorial/README.md`: tested tutorial sequence with local definitions,
  per-example intent, and links to deeper model/context docs;
* `docs/pingpong.md`: source survey and scope decision for ping-pong
  scheduling, `setprio`, `sched_barrier`, RDNA4 suitability, generated-code
  inspection, and optimized ATT validation;
* `docs/atomics.md`: stable delivery-facing atomics model, including supported
  API, address-scoped release metadata, acquired epoch tokens, paired fences,
  precision trade-offs, and current RDNA4 performance interpretation;
* `docs/atomics_plan.md`: staged atomics roadmap and current implementation
  status;
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
READMEs now describe the current detector scope. The atomics delivery pass has
also split the stable atomics model into `docs/atomics.md`, leaving
`docs/atomics_plan.md` as history and future-work tracking.

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

4. Keep ping-pong ATT validation as the guardrail for ping-pong benchmark work.

   The optimized probe now validates pass-through and sampled hip-moi dynamic
   instruction streams through ROCprof UI JSON. SIMD 0/1 traces validate the
   `1010` LDS-priority role and SIMD 2/3 traces validate the complementary
   `0101` role. On gfx10+ the local ATT workflow selects one SIMD ID per run,
   so this confirms complementary role schedules but not same-cycle activity of
   both roles in one trace. The current private-LDS timing row uses this
   validated kernel shape; future ping-pong variants should pass the same
   generated-code and ATT checks before latency numbers are treated as
   meaningful.

5. Use the completed atomics package for delivery discussion.

   The source-level atomics package is complete through Stage 16. The stable
   description is now `docs/atomics.md`: it defines address-scoped release
   records, pairwise acquired epoch tokens, supported atomic operations,
   paired-fence semantics, address-only false-negative risk, sampled-reporting
   caveats, and current RDNA4 cost. `docs/atomics_plan.md` remains the
   implementation history and future-work tracker.

   The short version for discussion is: `hip_moi::context` supports
   release/acquire load/store, fetch-add/or/and/xor, exchange, successful and
   failed compare-exchange, `seq_cst` sanity coverage, and atomic fences paired
   with relaxed atomics. All refreshed atomics `context` rows are spill-free.
   Small two-subgroup rows are roughly 6.7 to 9.0 µs through `context`, while
   Stream-K-shaped integration rows range from 12.5 µs to 42.7 µs. VGPR
   pressure is controlled; remaining cost is mostly metadata protocol work and
   global metadata traffic.

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
