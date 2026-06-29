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
  exact metadata-record/import timing, precision trade-offs, and current RDNA4
  performance interpretation;
* `docs/loom_rfc_comparison.md`: mapping between hip-moi, the compiler-rt RFC,
  HRX/Loom as summarized in Jakub's materials, and the benchmark-local
  Jakub-Sampled-Loom path;
* `docs/benchmark_interpretation.md`: interpretation of benchmark modes,
  apples-to-apples comparisons, resource metrics, and current evidence;
* `docs/dbi_transition.md`: transition brief from source-level hip-moi lessons
  to rocjitsu DBI requirements, first experiments, and open model questions;
* `docs/atomics_corpus.md`: source provenance for current atomics tests and
  benchmarks;
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
also made `docs/atomics.md` the stable atomics model. The Loom/RFC and
benchmark interpretation passes are complete in `docs/loom_rfc_comparison.md`
and `docs/benchmark_interpretation.md`. The atomics delivery polish pass added
a front-loaded delivery summary, a precise metadata timeline, and a fast-path
decision table for Jakub-facing discussion. The DBI transition pass added
`docs/dbi_transition.md`, which states what rocjitsu should preserve, what it
can change, and why the first DBI experiment should be LDS address
reconstruction. The delivery cleanup pass removed served-purpose checkpoint
docs: the staged atomics implementation plan and the attention source-mining
worklog. Durable atomics provenance now lives in `docs/atomics_corpus.md`; the
attention conclusions live in `benchmarks/README.md` and
`docs/benchmark_interpretation.md`.


## CDNA4/gfx950 Port Status

The gfx950 port now has an initial CDNA4-specific test and benchmark set under
`tests/instrumented` and `benchmarks`, guarded by the gfx950 CMake paths:

* MFMA attention block with LDS softmax scratch: `010_cdna4_mfma_attention_block`;
* MFMA D128 attention block with LDS softmax scratch: `011_cdna4_d128_attention_block`;
* MFMA register handoff: `013_cdna4_mfma_register_handoff`;
* MFMA no-score LDS attention: `014_cdna4_mfma_no_score_lds_attention`;
* MFMA D128 no-score LDS attention: `015_cdna4_d128_no_score_lds_attention`;
* MFMA ping-pong LDS rows: `016`, `017`, and `018`;
* MFMA Stream-K rows: `028` and `029`;
* MFMA attention LDS alias handoff: `036`.

The current CDNA4 MFMA helpers have been checked against the CDNA4 ISA manual's
general MFMA input/output layout and the local LLVM gfx950 builtin definitions.
The committed rows intentionally use the legacy wave64 `16x16x16f16` MFMA shape
where that matches the existing RDNA4 tile structure or the `hip-matmul` source
material. The gfx950-only `<8 x half>` builtins, especially
`__builtin_amdgcn_mfma_f32_16x16x32_f16`, are available in the local LLVM tree
and should be used for future ports that need the native CDNA4 K=32 f16 tile.

Remaining RDNA4-specific tests that still need CDNA4 counterparts are the
larger attention row: `012`. Their CDNA4 ports
should be based on the verified MFMA layout helpers, the rocjitsu gfx950 MFMA
layout corpus, and the CDNA4 ISA manual rather than RDNA4 WMMA lane formulas.

## Next Work

1. Import newly mined production hazard patterns into the corpus.

   The newest corpus review found three useful pattern families. The import must
   be sanitized: do not name source repositories, incident ids, bundle paths,
   or verbatim external reports in OSS files. Only generalized kernel
   mechanisms, shapes, tests, and benchmark rows should land here.

   Two source-level import targets are done. The production attention-forward
   LDS alias handoff is now represented by:

   * `tests/instrumented/035_attention_lds_alias_handoff_test.hip`;
   * `tests/instrumented/036_rdna4_wmma_attention_lds_alias_handoff_test.hip`;
   * `benchmarks/036_rdna4_wmma_attention_lds_alias_handoff_benchmark.hip`.

   The scalar test covers the missing-barrier diagnostic, the clean conditional
   barrier, and a non-aliased no-barrier case. The RDNA4 test keeps the same
   mechanism around a WMMA tile. The benchmark times the synchronized variant
   with pass-through, `context + sampled_watchpoint`, and
   `sampled_watchpoint_context` rows, and `benchmarks/README.md` records the
   current latency/resource data.

   A sanitized split-K/PostGSU-style two-level flag reduction is now
   represented by:

   * `tests/instrumented/037_streamk_two_level_reduction_test.hip`;
   * `benchmarks/037_streamk_two_level_reduction_benchmark.hip`.

   This row has four producer subgroups, two pair-reducer subgroups, and one
   final reducer subgroup. It exercises chained release/acquire flag handoffs
   with LDS payload diagnostics. Current RDNA4 timing is 3.71 µs
   pass-through and 27.9 µs through `context`, with 768 B LDS, 69 SGPRs, 44
   VGPRs, and no spills in the `context` row.

   The second target is an assembly-level wait-count/pack-ordering family. It
   is not a HIP source-level race test because the relevant failure is a
   hardware instruction-scheduling/data-dependency hazard, not a HIP/LLVM
   memory-model violation. Keep it on the DBI track: add a future rocjitsu seed
   that decodes LDS reads, nearby `s_waitcnt lgkmcnt(...)`, and pack
   instructions, then checks whether the wait-count is strong enough for the
   consumed VGPR data.

   The third target is a global fp32 atomic accumulator family. It is important
   for atomics, but it is not a current hip-moi source-level LDS payload
   testcase: the payload is a global accumulator and the suspected issue is at
   the hardware atomic/L2 level. Keep the source-level detector focused on
   global atomics as synchronization for LDS accesses. Record this family as a
   DBI atomic instruction seed and revisit only if hip-moi's scope expands to
   global-memory diagnostics.

2. Keep ping-pong ATT validation as the guardrail for ping-pong benchmark work.

   The optimized probe now validates pass-through and sampled hip-moi dynamic
   instruction streams through ROCprof UI JSON. SIMD 0/1 traces validate the
   `1010` LDS-priority role and SIMD 2/3 traces validate the complementary
   `0101` role. On gfx10+ the local ATT workflow selects one SIMD ID per run,
   so this confirms complementary role schedules but not same-cycle activity of
   both roles in one trace. The current private-LDS timing row uses this
   validated kernel shape; future ping-pong variants should pass the same
   generated-code and ATT checks before latency numbers are treated as
   meaningful.

3. Use the completed atomics package for delivery discussion.

   The source-level atomics package is complete enough for the current delivery
   phase. The stable description is now `docs/atomics.md`: it defines
   address-scoped release records, pairwise acquired epoch tokens, supported
   atomic operations, paired-fence semantics, address-only false-negative risk,
   sampled-reporting caveats, and current RDNA4 cost. Source provenance is in
   `docs/atomics_corpus.md`.

   The short version for discussion is: `hip_moi::context` supports
   release/acquire load/store, fetch-add/or/and/xor, exchange, successful and
   failed compare-exchange, `seq_cst` sanity coverage, and atomic fences paired
   with relaxed atomics. All refreshed atomics `context` rows are spill-free.
   Small two-subgroup rows are roughly 7 to 9 µs through `context`, while
   Stream-K-shaped integration rows range from 12.4 µs to 45.2 µs, with the
   six-subgroup two-level flag reduction at 27.9 µs. VGPR pressure is
   controlled; remaining cost is mostly metadata protocol work and global
   metadata traffic. A deep acquire-path audit rejected two tempting local
   shortcuts: conditional acquired-token publication broke the four-subgroup
   `atomicOr` tree, and a special two-subgroup direct lookup regressed the
   shared-context flag microbenchmark. The next meaningful atomics speedup
   should be protocol-aware or DBI-informed, not another small generic
   table-loop trim.

4. Use `docs/dbi_transition.md` as the bridge to rocjitsu work.

   The next concrete DBI step is an LDS address reconstruction experiment:
   decode LDS load/store instructions in a minimal uninstrumented kernel,
   compute active-lane LDS byte ranges, and compare those ranges against the
   explicit offsets used by the matching hip-moi source-level test. This proves
   that DBI can remove the `lds_byte_offset` API compromise while preserving
   the diagnostic payload.

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
