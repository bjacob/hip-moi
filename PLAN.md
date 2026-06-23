# PLAN for hip-moi: HIP memory-ordering instrumentation library

## Purpose

hip-moi has two purposes.

1. Instrument HIP programs at the source/API level to diagnose LDS
   memory-ordering bugs under the HIP/LLVM memory model.
2. Act as a prototype and thinking tool for a later assembly-level sanitizer,
   where the relevant model is the hardware execution model and where Jakub's
   Loom-style subgroup-level approach is the primary comparison point.

The second purpose now drives near-term prioritization. hip-moi is no longer
trying to be a general thread-level HIP race detector. The product surface is
subgroup-scoped, and success should be judged against Loom-like instrumentation
on Jakub's RDNA4 matmul benchmark.

## Memory-Model Ground Rules

HIP is modeled after C++, and HIP programs lower through LLVM IR. We refer to
the LLVM IR memory model because hip-moi may instrument code that uses Clang
builtins for operations such as fences, which are not fully specified as HIP
language APIs but do have well-defined LLVM IR lowerings.

For HIP-source instrumentation, correctness is judged at the HIP/LLVM level.
GPU execution folklore such as lockstep behavior is not part of that language
memory model.

For the future assembly-level effort, the model changes. There, hardware
details such as lockstep execution, instruction issue, and load/store queue
behavior are legitimate. That is the reason hip-moi now focuses on
subgroup-to-subgroup LDS interactions: it is the part most relevant to the
intended assembly-level path.

## Current State

The public device API is:

```c++
hip_moi::context
```

The public host owner is:

```c++
hip_moi::host_context
```

The old `thread_level_context` and `subgroup_level_context` split has been
removed. `hip_moi::context` is subgroup-scoped. It reports same-epoch LDS
conflicts between different subgroups in the same workgroup. It intentionally
does not report conflicts whose participants are wholly inside one subgroup.

There is also a narrower performance view:

```c++
hip_moi::sampled_watchpoint_context
```

This is not the general semantic center of hip-moi. It is a sampled
publish-only fast path for benchmark-sensitive kernels with full-workgroup
barriers and no in-kernel diagnostic reporting. It exists so we can study
low-overhead Loom-like instrumentation without forcing the full diagnostic
`context` object and its cold state into the hottest production kernels.

The legacy record/log/scan design has been removed from the active API and test
suite. The implementation now has two explicit-offset subgroup-level paths:

* `backend_kind::exact_shadow` performs immediate exact shadow-table checks,
* `backend_kind::sampled_watchpoint` performs selected-lane watchpoint checks.

The active device access API requires a caller-supplied LDS byte offset:

```c++
ctx.lds_store_at(ptr, value, /*lds_byte_offset=*/offset, site);
ctx.lds_load_at(ptr, /*lds_byte_offset=*/offset, site);
```

Benchmark-sensitive code can also select the backend at compile time:

```c++
ctx.lds_store_at<hip_moi::backend_kind::sampled_watchpoint>(
    ptr, value, /*lds_byte_offset=*/offset, site);
ctx.lds_load_at<hip_moi::backend_kind::sampled_watchpoint>(
    ptr, /*lds_byte_offset=*/offset, site);
```

That avoids keeping both exact-shadow and sampled-watchpoint paths live in the
same optimized kernel. The non-templated overloads remain useful as the regular
API when backend selection comes from `host_context_options`.

This is closer to Jakub's HIP prototype and to what compiler or assembly-level
instrumentation would naturally know. Pointer-only `lds_load` / `lds_store`
helpers were deleted because preserving them kept the old backend alive and
made benchmark work measure the wrong thing.

The top-level CMake build enables `CMAKE_EXPORT_COMPILE_COMMANDS` by default so
clangd and other tooling can use the generated compilation database. The
generated root `compile_commands.json` is intentionally ignored by git. The
repo also carries a local `.clangd` that tells system clangd where the TheRock
ROCm tree lives and forces project `.hpp` headers to parse as HIP; otherwise
clangd can infer `-x c++-header` for headers even when the including `.hip`
files have the correct `-x hip` compile command.

## Current Detector Contract

A diagnostic of interest is:

* two instrumented LDS accesses overlap in byte range,
* the accesses are from different subgroups in the same workgroup,
* the accesses are in the same synchronization epoch,
* at least one access is a write.

Instrumented accesses check and update shadow state online. `ctx.syncthreads()`
performs a real workgroup barrier and advances the epoch; ordinary kernels do
not need a final flush. `ctx.finish()` remains only as a compatibility alias for
explicit epoch advancement in tests and experiments.

`ctx.simulate_syncthreads(participates, site)` exists only for tests and hard
synchronization diagnostics. It lets the suite exercise barrier-divergence
reporting without launching a kernel that would hang on a real divergent
`__syncthreads()`.

## What Loom Changes

Real Loom and Jakub's Loom-flavored HIP implementation point to a different
center of gravity.

Real Loom:

* inserts `sanitizer.race.access` around workgroup-memory loads, stores, and
  atomics,
* inserts `sanitizer.race.sync` for workgroup barriers,
* computes a workgroup-local LDS byte offset for each access,
* maps that offset to a global shadow entry,
* packs access kind, owner, epoch, generation, and site id into one 64-bit
  shadow entry,
* performs the conflict check immediately at the instrumented access,
* increments a per-workgroup epoch header at barriers,
* branches to a cold diagnostic path only when the compact predicate fires.

Jakub's exact Loom HIP path mirrors this shape with direct helper calls. Jakub's
sampled Loom path is the more performance-relevant data point: it stores exact
dword-cell ranges in a small global watchpoint table, but only selected lanes
publish and check metadata. Sampling and table overwrites can introduce false
negatives, but the represented dword-range conflicts are exact.

The lesson for hip-moi is: stop trying to make per-lane record collection cheap
enough. The hot path needs online shadow/watchpoint checks, compact metadata,
minimal global traffic, and low VGPR pressure.

## Scope Guardrails

The fastest current sampled path should remain a specialized mode, not the new
conceptual center of hip-moi.

The general `hip_moi::context` / `hip_moi::host_context` path remains the home
for correctness-first diagnostics, storage saturation handling, reporting, and
future synchronization models. It should stay capable of representing
synchronization state beyond full-workgroup barriers.

The `hip_moi::sampled_watchpoint_context` path is allowed to be aggressive:
publish-only, sampled, full-workgroup-barrier-oriented, and benchmark-driven.
Optimizations such as local-only epoch tracking are acceptable there because
represented watchpoints carry the local epoch and no reporting path consumes a
global epoch word. Those assumptions must not leak back into the general
context as if they solved finer-grained synchronization.

`docs/context.md` now documents the practical split with examples: general
`context + sampled_watchpoint` usage for diagnostics, direct
`sampled_watchpoint_context` usage for publish-only fast rows, one-watchpoint
benchmark policy setup, and the per-workgroup storage-ref array pattern needed
for multi-workgroup kernels.

`docs/tutorial/` now includes the same split as an executable tutorial. In
particular, `006_sampled_watchpoint_context.hip` shows the high-performance
publish-only recipe used by the benchmarks: sampled backend host allocation,
one watchpoint entry, static publish-only policy, direct
`sampled_watchpoint_context` construction, explicit LDS byte offsets, and no
`HIP_MOI_CHECK` on the fast path. The companion `.hip` files are intentionally
commented as readable standalone programs so readers can move from the tutorial
README to compilable examples without already knowing hip-moi's terminology.

This split is important for the next scope increases:

* non-negotiable: broaden end-to-end workloads beyond one isolated matmul, with
  an attention block as the likely first candidate;
* negotiable but important: support workloads with synchronization finer than
  global `__syncthreads()`, likely involving atomics and release/acquire-style
  reasoning.

## Target Architecture

The target device backend is a Loom-like subgroup-level detector.

The main hot-path metadata should be:

* a per-workgroup epoch word,
* a compact shadow or watchpoint table in global memory,
* a cold diagnostic buffer,
* small launch/generation metadata.

The main hot-path operations should be:

* compute the LDS byte offset,
* normalize to 4-byte LDS cells,
* load the current epoch,
* pack a compact current access record,
* compare against existing shadow/watchpoint metadata,
* update the shadow/watchpoint metadata,
* emit a diagnostic only on conflict.

For the publish-only sampled fast view, the epoch can be a per-thread local
counter advanced at instrumented full-workgroup barriers. The general diagnostic
context should retain a real synchronization-state representation that can
later grow to atomics and non-barrier happens-before edges.

For subgroup-level hip-moi, the encoded owner should be the subgroup id rather
than the exact thread id. Same-subgroup conflicts remain out of scope.

The old access-record and coalescing-record data structures are no longer part
of the active implementation. If a future debug/reference backend is useful, it
should be reintroduced deliberately instead of accidentally shaping the hot
path.

## Exact Shadow Path

The first implementation target is an exact Loom-style shadow path, adapted to
subgroup-level semantics.

Each shadow entry represents one LDS granule, initially a 4-byte dword cell. A
64-bit entry packs:

* access kind,
* subgroup id,
* epoch,
* generation,
* site id.

On each instrumented access, the context computes:

```text
entry = shadow_base + workgroup_offset + header_size
      + ((lds_byte_offset >> granule_shift) * sizeof(uint64_t))
```

The conflict predicate is:

```text
prior entry is nonempty
same generation
same epoch
different subgroup
conflicting access kind
```

This path is useful because it is simple, close to real Loom, and can validate
the storage layout and diagnostic plumbing. It may still be too slow for the
main performance target because every participating lane can perform global
shadow traffic.

## Sampled Watchpoint Path

The main performance target is a sampled-Loom-style watchpoint path.

Each watchpoint entry represents an exact LDS dword-cell range:

```text
{start_cell, cell_count, access_kind, subgroup_id, epoch, generation}
```

Only selected lanes publish and check watchpoints. The selection is
deterministic for a launch and mixed from generation, workgroup id, subgroup id,
and site id. The sampled backend now exposes the same basic tuning axes as
Jakub's sampled Loom HIP prototype:

* watchpoint count,
* sample skip,
* probe count,
* delay iterations,
* publish-only versus reporting mode.

The sampled selector uses 32-bit mixing and power-of-two masks rather than
64-bit modulo arithmetic on the hot path. Sampled capacities should be chosen
as powers of two for good distribution and for fair comparison with the Loom
benchmark row.

The sampled access API now has two policy layers. Runtime policy fields in
`host_context_options` preserve launch-time flexibility. A compile-time
`sampled_watchpoint_policy<SampleSkip, ProbeCount, DelayIters, ReportConflicts,
StaticWatchpointCapacity>` can instead be supplied as a second template argument to
`lds_load_at<backend_kind::sampled_watchpoint, Policy>` or
`lds_store_at<backend_kind::sampled_watchpoint, Policy>`. That lets hot
publish-only rows erase runtime policy loads and reporting branches. The final
static-capacity argument is optional and defaults to zero, which means "use the
runtime watchpoint capacity." A value of one lets the sampled hot-path view fold
slot selection to zero for one-watchpoint benchmark rows.

Selection is site-aware, but conflict rendezvous is range-aware: the watchpoint
slot is keyed by the watched LDS range, epoch, and generation, not by source
site or subgroup. Reporting mode publishes first, checks the displaced
watchpoint entry, and then optionally probes additional slots. That keeps the
one-probe path meaningful for common same-range conflicts while still allowing
larger probe counts or full-table scans for tests and diagnostic-heavy runs.

This path deliberately permits false negatives from sampling, limited probes, or
table overwrites. It must not create false positives for the represented
dword-cell ranges.

## Source Site IDs and Regular Access Patterns

`site_id` is now a compact static-site token carried into shadow/watchpoint
metadata and diagnostics:

```c++
ctx.lds_store_at(&lds[index], value, /*lds_byte_offset=*/offset, HIP_MOI_SITE_ID());
```

A nonzero site id does not currently enable a separate coalescing data
structure. The lane-record coalescing implementation was removed because it
still wrote per-lane metadata before proving anything, which defeated the
optimization goal.

The underlying idea remains useful as a possible future direction: recognize
regular lane-to-address patterns online and publish/check one subgroup-level
range. That work should be Loom-inspired and should avoid collecting per-lane
records in global memory. Candidate patterns include contiguous per-lane
accesses, fixed-stride accesses, descending fixed-stride accesses, and the vector
LDS accesses used by the RDNA4 WMMA benchmark.

Dynamic instance identity is out of scope for now. If a static site has dynamic
behavior that an online proof cannot validate, it should fall back
conservatively to the selected shadow/watchpoint backend.

## LDS Offset API

Real Loom gets LDS byte offsets from compiler lowering. A HIP library usually
only sees pointers. Jakub's HIP helpers pass byte offsets explicitly, which is
the right shape for benchmark-driven work.

The active API is explicit-offset only:

```c++
ctx.lds_store_at(ptr, value, /*lds_byte_offset=*/offset, site);
ctx.lds_load_at(ptr, /*lds_byte_offset=*/offset, site);
```

Later, we can investigate whether HIP/Clang exposes a reliable way to derive
LDS offsets from pointers, but performance work should not wait on that
question.

## Context Storage

`host_context_options::storage_bytes` remains the primary user-facing storage
knob and defaults to 16 MiB.

The target storage layout should be based on shadow/watchpoint needs rather
than access-record buffers:

* one or more dispatch/generation slots,
* per-workgroup epoch headers,
* per-workgroup exact shadow entries or sampled watchpoint entries,
* diagnostics and diagnostic counters.

The current user-facing knobs are deliberately small:

* `storage_bytes`,
* selected `backend_kind`,
* exact-shadow entry capacity override,
* sampled-watchpoint capacity override,
* diagnostic capacity,
* subgroup capacity.

Future backend tuning may add higher-level knobs:

* LDS granule shift,
* local memory byte span to shadow,
* workgroup capacity,
* watchpoint count,
* probe count,
* sample skip,
* diagnostic capacity.

Storage saturation should produce diagnostics or conservative fallback, not
silent corruption. Sampling-related misses are acceptable only when the user has
selected the sampled path and the documentation calls out that false negatives
are possible.

## Host API and Reporting

The standard host shape remains:

```c++
hip_moi::host_context_options options;
options.storage_bytes = 16 * 1024 * 1024;

hip_moi::host_context moi(options);
kernel<<<grid, block>>>(moi.launch_ref());
HIP_MOI_CHECK(moi);
```

Scope-based handling through the `host_context` destructor remains a first-class
usage pattern. By default, unconsumed diagnostics are printed and abort the
process. Advanced users may disable destructor reporting and destructor aborting
separately.

The cold diagnostic path should report enough information to connect a conflict
back to a static site and LDS location:

* access kinds,
* current and prior subgroup ids,
* epoch,
* generation,
* current and prior site ids,
* LDS byte offset or dword cell range,
* storage saturation or sampling warnings when relevant.

## Benchmark Integration

The top-level `benchmarks/` directory now vendors the focused RDNA4 benchmark
families extracted from Jakub's
`sanitizer-strategy/rdna4_matmul/rdna4_matmul.hip`. Jakub's repository remains
the upstream reference for the broader experimental harness, but day-to-day
hip-moi performance work should run from this repository.

The small `w2_2x4_benchmark.hip` family compares:

* noop,
* sampled Loom,
* hip-moi exact shadow through explicit LDS-offset APIs,
* hip-moi general `context` with the `sampled_watchpoint` backend,
* hip-moi narrower `sampled_watchpoint_context` fast view when the default
  publish-only sampled knobs make that comparison valid.

The production `prod_16x8_benchmark.hip` extraction is now the main
optimization target. It keeps Jakub's real fp16 production 16x8 row shape:
8 waves, 16x8 WMMA tiles, `KGroup=2`, the pipelined LDS staging pattern, and
runtime `BENCH_M/N/K` sizes. It compares only the rows that matter for the
current Loom-parity loop:

* noop,
* sampled Loom publish-only,
* hip-moi publish-only through the general `context` with the
  `sampled_watchpoint` backend,
* hip-moi publish-only through `sampled_watchpoint_context`.

Both extracted benchmarks now construct the hip-moi device context once near
kernel entry and pass it through the instrumented access helpers. That better
matches the intended source-instrumentation shape than rebuilding a context
inside every instrumented access wrapper. The default sampled publish-only
rows additionally construct the narrow `sampled_watchpoint_context` view so the
benchmark can measure the cost of the fast path separately from the full
diagnostic-capable context. The compact 2/4/8-wave extraction and the
production extraction both use this same row split; the compact benchmark no
longer labels a static-policy general `context` row as
`sampled_watchpoint_context`.

The sampled rows now share one fair knob set:

```text
SAMPLED_WATCHPOINTS, SAMPLED_SKIP, SAMPLED_PROBES, SAMPLED_DELAY,
SAMPLED_REPORTS
```

By default this is `watchpoints=1`, `skip=32`, `probes=1`, `delay=32`, and
`reports=off`, matching Jakub's sampled-Loom publish-only configuration. The
benchmark prints these effective knobs and names sampled rows as
`publish_only` or `reporting`. The headline per-iteration latency prints in
microseconds for values below 1 ms and in milliseconds otherwise; elapsed
measurement windows remain in milliseconds.

The current tiny go-to shapes are:

```bash
./benchmarks/build_w2_2x4_benchmark.sh
./benchmarks/build_w2_2x4_benchmark.sh w4_4x16
./benchmarks/build_w2_2x4_benchmark.sh w8_16x8
```

Use the 2-wave shape for fast intra-session iteration and the 2/4/8 set when a
change specifically targets tiny-shape overhead or wave-count scaling.

The current production go-to benchmark is:

```bash
./benchmarks/build_prod_16x8_benchmark.sh
```

Run the production benchmark before a session-ending commit when
performance-sensitive code changed. The current focused baseline at
`BENCH_M=BENCH_N=BENCH_K=4096` is roughly:

```text
noop                                      1.16 ms
sampled Loom                              8.65 ms
hip-moi context + sampled_watchpoint     25.9 ms
hip-moi sampled_watchpoint_context        3.38 ms
```

The current corrected compact baselines are:

```text
shape      noop       sampled Loom  exact shadow  context+sampled  sampled_context
w2 2x4     2.80 µs   4.69 µs       8.91 µs       4.85 µs          3.49 µs
w4 4x16    3.15 µs   5.91 µs       14.0 µs       7.45 µs          4.31 µs
w8 16x8    3.20 µs   5.79 µs       13.0 µs       7.74 µs          4.62 µs
```

The old append-only `BENCHMARK_LOG.md` has been removed. It was useful while we
were discovering the shape of the benchmark, but the durable workflow is now:
keep the vendored benchmark sources current, summarize important benchmark
state in this plan, and include fresh benchmark numbers in commit messages or
review notes when performance-sensitive code changes.

The current hip-moi rows exercise the explicit-offset exact-shadow path, the
general `context` with the `sampled_watchpoint` backend, and the narrow
`sampled_watchpoint_context` sampled fast path. Under the fair publish-only
production default, the narrow sampled view is now substantially faster than
the sampled-Loom row on the vendored 16x8 benchmark, while the general
`context` row remains much slower. Dense sampling, reporting-on rows, and
future workload families are still expected to expose overhead that the current
matmul does not.

Historical codegen audits after backend specialization on the extracted 2-wave
benchmark showed the main pressure points:

* before backend specialization, hip-moi exact and sampled both compiled to the
  same 79-VGPR, 37.6-KB-code kernel because runtime backend dispatch kept both
  implementations live;
* compile-time backend selection reduced exact shadow to 52 VGPR and 19.5 KB of
  code;
* compile-time backend selection plus cheaper sampled selection reduced sampled
  watchpoints to 57 VGPR and 15.2 KB of code;
* the generated metadata still reports `uses_flat_scratch=1`, although
  `ScratchSize` is 0 for the hot kernels.

After adding fair sampled knobs, a codegen audit before the latest sampled
reporting-slot fix showed sampled hip-moi at 71 VGPR and 18.5 KB of code on the
same extracted 2-wave benchmark. Refresh audits from the rebuilt benchmark
executable before relying on exact VGPR/code-size numbers; sidecar code object
files can be stale after header-only changes.

## Test Corpus

The active instrumented suite is intentionally smaller after dropping
thread-level mode. Its filenames use contiguous three-digit prefixes so the
directory order reflects the intended progression through the active API and
backend tests.

Kept:

* one single-subgroup diagnostic-free smoke test,
* host API and destructor behavior tests,
* source-site id diagnostic tests,
* simulated hard-synchronization diagnostics,
* compact shadow/watchpoint ABI tests,
* explicit LDS-offset API tests,
* exact-shadow backend tests,
* sampled-watchpoint backend tests.

Removed:

* the old single-subgroup thread-level ladder,
* thread-level mirrors of multi-subgroup tests,
* single-subgroup RDNA4 tests whose only role was thread-level coverage,
* legacy record/log multi-subgroup tests,
* legacy coalescing proof and coalesced-conflict tests.

New tests should follow the implementation path:

1. exact shadow-entry packing/unpacking and conflict predicates,
2. explicit LDS-offset access APIs,
3. exact shadow-path positive and negative tests,
4. sampled watchpoint positive and negative tests,
5. sampling false-negative documentation tests where deterministic behavior can
   be forced by knobs,
6. RDNA4 WMMA benchmark-shaped tests using the same helper style as Jakub,
7. online regular-pattern optimization tests if that direction becomes active.

Reference tests remain useful as uninstrumented kernels that compile and run,
especially as a source of later multi-subgroup or RDNA4 shapes.

## Implementation Roadmap

### Session 1: Shadow ABI Skeleton

Status: implemented. `hip_moi/shadow.hpp` defines the compact exact shadow
entry and sampled watchpoint entry helpers, with focused tests covering
pack/unpack behavior and conflict predicates.

Add Loom-style constants and compact record helpers:

* access-kind enum values compatible with the existing diagnostics,
* 64-bit exact shadow-entry pack/unpack helpers,
* sampled watchpoint pack/unpack helpers,
* generation and epoch helpers,
* host/device tests for bit layout where practical.

No benchmark win is expected from this session.

### Session 2: Storage Layout for Loom Backend

Status: implemented. `host_context` now allocates shadow epoch storage, exact
shadow entries, sampled watchpoints, and a generation value inside the existing
byte-budget layout. The old access-record/coalescing storage has been removed.

Teach `host_context` to allocate a shadow/watchpoint backend layout from
`storage_bytes`. The storage layout represents:

* per-workgroup epoch headers,
* exact shadow entries or sampled watchpoints,
* diagnostics,
* counters needed by `HIP_MOI_CHECK`.

### Session 3: Explicit LDS-Offset APIs

Status: implemented. `context` now provides `lds_load_at` and `lds_store_at`
helpers that accept an explicit LDS byte offset. The extracted
sanitizer-strategy benchmark now routes its hip-moi row through these overloads,
so the go-to benchmark measures offset-aware hip-moi instrumentation rather than
the older implicit-pointer path. Pointer-only `lds_load` and `lds_store` were
deleted during cleanup.

Update focused tests, tutorials, and benchmark glue to pass explicit LDS byte
offsets at the hip-moi helper sites.

### Session 4: Exact Shadow Backend

Status: implemented for explicit-offset accesses. `lds_load_at` and
`lds_store_at` now use the exact shadow table when exact-shadow storage is
present, emit immediate diagnostics for same-epoch cross-subgroup conflicts, and
report metadata saturation for out-of-range LDS offsets. Calls without explicit
LDS offsets no longer exist in the public API.

Implement the exact subgroup-level shadow path:

* compute shadow entry address from LDS byte offset,
* load epoch,
* pack current entry,
* update/check prior entry,
* emit deterministic diagnostics.

Use this to validate storage, epochs, diagnostics, and benchmark integration.
Expect correctness progress more than performance parity.

### Session 5: Sampled Watchpoint Backend

Status: first cut implemented. `host_context_options::backend` can select
`backend_kind::sampled_watchpoint`, and the device context records selected-lane
sampled watchpoints for explicit-offset accesses. Focused tests cover
synchronized non-diagnostics and racy selected-access diagnostics. The extracted
benchmark now prints both `hip_moi_exact_shadow` and
`hip_moi_sampled_watchpoint` rows.

Current benchmark result: sampled-watchpoint correctness plumbing exists, but it
does not yet improve latency relative to exact shadow. This says the next
optimization should attack per-lane wrapper work, selected-lane calculation,
context creation, and register pressure, not merely metadata write count.

Implement the sampled-Loom-style backend:

* dword-cell range normalization,
* selected-lane publication,
* bounded probe scans,
* watchpoint overwrite behavior,
* report counting and cold diagnostics.

This is the first session expected to materially improve benchmark numbers.

### Session 6: Legacy Cleanup

Status: implemented. The record/log backend, lane-record coalescing data
structures, pointer-only LDS APIs, and their stale tests/docs have been removed.
The remaining implementation is the explicit-offset shadow/watchpoint path.

This cleanup is meant to make the next performance sessions honest: if a
benchmark moves, it is because the Loom-style path moved, not because a legacy
fallback accidentally absorbed the work.

### Session 7: Benchmark-Driven Tightening

Status: implemented for the matmul-only phase. The first tightening pass added
compile-time backend selection for the explicit-offset access helpers and
replaced sampled watchpoint lane/slot selection with cheaper 32-bit mixing and
masking. On the 2-wave benchmark, sampled hip-moi moved from roughly 0.010 ms
after legacy cleanup to roughly 0.007 ms, while sampled Loom remains roughly
0.005 ms. The same pass made the 4-wave and 8-wave sampled rows roughly
0.011 ms.

The benchmark harness was then changed to create the hip-moi context once per
kernel and pass it through the helper calls. This reduced sampled hip-moi again
to roughly 0.006 ms on 2-wave and 0.009 ms on 4/8-wave. Code size dropped, but
VGPR use rose to 63 for the sampled row, so this is not a free lunch; it mostly
confirms that repeated context lookup was real overhead and that context live
range is now a central tuning problem.

The fairness pass then aligned sampled knobs between sampled Loom and hip-moi,
fixed per-launch hip-moi generation in the benchmark, and split sampled row
names into publish-only versus reporting. With fair default knobs
(`watchpoints=1`, `skip=32`, `probes=1`, `delay=32`, `reports=off`), sampled
hip-moi is now approximately tied with sampled Loom on 2-wave and trails it by
about 0.002 ms on the 4/8-wave rows. Dense sampling (`skip=1`) and reporting
mode show hip-moi still has avoidable policy/checking overhead. The same pass
also fixed sampled reporting rendezvous so same-range conflicts meet in the
same watchpoint slot instead of depending on source-site or subgroup hash
collisions.

The next pass added a compile-time sampled policy API and changed the extracted
benchmark to use it only for the default publish-only hip-moi row. With precise
latency printing, sampled hip-moi moved from `0.00483` to `0.00425` ms on
2-wave, from `0.00760` to `0.00523` ms on 4-wave, and from `0.00790` to
`0.00573` ms on 8-wave. The benchmark deliberately falls back to the runtime
policy path for non-default sampled knobs, so dense/reporting sweeps remain
honest.

The main benchmark was then updated with the same static publish-only hip-moi
sampled row for Jakub's production fp16 16x8 matmul. The focused
`benchmarks/prod_16x8_benchmark.hip` extracts that row family and is now the
main optimization gate. With fair publish-only knobs
(`watchpoints=1`, `skip=32`, `probes=1`, `delay=32`, `reports=off`) and
`BENCH_M=BENCH_N=BENCH_K=4096`, the initial focused baseline was roughly
`1.17 ms` noop, `8.63 ms` sampled Loom, and `17.6 ms` hip-moi sampled
watchpoints. That put hip-moi at about 2x sampled Loom on the production shape,
even though the tiny 2/4/8-wave benchmark had reached parity. That result is
what redirected the next performance work to the production extraction.

Use the 2/4/8-wave benchmark set to tune tiny-shape overhead:

* watchpoint count,
* probe count,
* sample skip,
* VGPR pressure,
* inlining decisions,
* cold diagnostic isolation.

Inspect generated code when benchmark movement is surprising. The main danger is
spilling caused by instrumentation VGPR usage; global-memory traffic from spills
can dominate the actual sanitizer work.

The first generated-code audit on the focused production extraction confirmed
that the new goal is not to invent a different sampling algorithm first. The
goal is to make hip-moi's sampled hot path compile into something much closer
to Jakub's sampled Loom path:

```text
row                  private bytes  VGPR spills  SGPRs  code size
noop                            0            0     23   0x01a28
sampled Loom                  320          244     57   0x0f528
hip-moi static sampled       1024          751     50   0x1d340
hip-moi runtime sampled      1456         1203    107   0x44328
```

The disassembly tells the same story. The sampled Loom row is about 11.8k
assembly lines, while the hip-moi static sampled row is about 21.8k. The hip-moi
static row has roughly 354 scratch loads and 221 scratch stores versus 75 and
72 for sampled Loom. It also has 2624 `s_nop` occurrences because the
compile-time delay policy unrolled the 32-iteration delay at every hot call
site. That is the first low-risk fix.

Production sampled roadmap:

1. Stop unrolling the sampled delay. Keep the default static publish-only policy
   values, but lower the delay to a compact counted loop or a small noinline
   helper so the compiler does not replicate 32 `s_nop`s at every access site.
   This should reduce code size and may reduce spills by shortening hot blocks.
   Status: implemented with `#pragma unroll 1` on the sampled delay loops. On
   the production extraction this reduced static sampled code size from
   `0x1d340` to `0x1c06c` and dropped counted `s_nop` occurrences from 2624 to
   82. It did not reduce private memory or VGPR spills; the static sampled row
   still uses 1024 private bytes and spills roughly 769 VGPRs. That confirms the
   next major problem is live state, not merely delay-loop code volume.
2. Introduce a sampled hot-path device view separate from the full public
   `context`. The public `host_context` / `context` API can remain stable, but
   benchmark-sensitive kernels should be able to carry a tiny sampled view with
   only the watchpoint pointer, generation, epoch/header pointer, and static
   policy facts needed by `lds_load_at` / `lds_store_at`. It should not keep
   diagnostic buffers, exact-shadow pointers, generic options, or barrier-test
   state live in the production sampled access path.
   Status: implemented for publish-only sampled policies as
   `sampled_watchpoint_context`. The focused production benchmark now routes the
   static publish-only hip-moi fast row through this view. That dropped the
   `sampled_watchpoint_context` production row from roughly `16.9 ms` after the
   delay-loop cleanup to `7.35 ms`, beating sampled Loom's roughly `8.63 ms` on
   the same run. Codegen improved from 1024 private bytes and 769 VGPR spills to
   116 private bytes and 40 VGPR spills. This confirms that the main production
   gap was the full context's live state, not the sampled watchpoint algorithm.
3. Add a full-workgroup-barrier sampled epoch path. In the production benchmark,
   every `ctx.syncthreads()` is a full workgroup barrier, and sampled Loom uses
   one per-workgroup epoch header. hip-moi still carries the more general
   subgroup-epoch machinery from earlier designs. For this sampled fast path,
   use the Loom-shaped sequence: barrier, one workgroup epoch increment by one
   thread, barrier. Keep subgroup-specific epochs out of the hot sampled path
   unless a later benchmark requires them.
   Status: implemented in the sampled publish-only view with a local per-thread
   epoch and no subgroup-state fields in the hot context. Existing host
   allocation is adapted through `make_sampled_watchpoint_context()`, but the
   hot context no longer carries subgroup-state capacity or loops over subgroup
   epochs. The focused production `sampled_watchpoint_context` row dropped again
   from roughly
   `7.35 ms` to `5.27 ms`, while sampled Loom was roughly `8.59 ms`.
   A later tightening pass made this path more Loom-shaped by replacing the
   ordinary epoch store/increment plus `__threadfence()` with atomic epoch
   updates between workgroup barriers. That improved latency from roughly
   `4.57 ms` after slot specialization to `4.42 ms`, even though private memory
   rose from 68 to 88 bytes and VGPR spills from 16 to 25. The win appears to
   come from removing device-wide fences from the barrier path, not from smaller
   generated code.
   The sampled view then stopped reloading the epoch from global memory at each
   sampled access. Each thread now keeps a local epoch copy and increments it
   after the instrumented barrier. The first version kept a thread-0 global
   epoch update for Loom-shape compatibility and moved the focused production
   row to roughly `3.81 ms`. The current publish-only fast path drops the global
   epoch update entirely, because represented watchpoints carry the local epoch
   value and reporting is out of scope for this view. That moved the row to the
   current roughly `3.41`-`3.44 ms` range, with 68 private bytes and 16 VGPR
   spills.
4. Specialize hot-path constants that are fixed in the production row:
   32 threads per subgroup, power-of-two watchpoint capacity, one probe,
   publish-only reporting, and known access sizes. Avoid generic range loops,
   division/modulo, and policy loads in the static sampled path.
   Status: partially implemented for one-watchpoint publish-only rows. The
   static sampled policy can now declare `StaticWatchpointCapacity=1`, and the
   sampled hot-path view folds watchpoint slot selection to zero in that case.
   The focused production `sampled_watchpoint_context` row moved from roughly
   `5.27 ms` to `4.57 ms`. Codegen improved from 100 private bytes and 32 VGPR
   spills to 68 private bytes and 16 VGPR spills, and code size dropped from
   `0x11284` to `0x0aff4`. Remaining low-risk specialization work is mostly
   access-size templating and any still-visible generic range loops.
   A naive access-size template experiment should not be repeated in the same
   form. A corrected recheck after the local-only epoch work measured roughly
   `3.96 ms` versus the current `3.46 ms`: it shrank code size to `0x8834`, but
   increased private memory from 68 to 152 bytes, VGPR spills from 16 to 49, and
   scratch loads/stores from 16/16 to 31/32. Future range specialization needs
   to be validated against generated code immediately.
5. Isolate cold diagnostics and storage validation. Publish-only mode should not
   inline reporting, metadata-full diagnostics, or slow validation branches into
   every instrumented access. If those paths are still required for safety,
   move them behind cold noinline helpers or an explicitly checked setup phase.
6. Only after the generated code resembles sampled Loom should we revisit more
   ambitious regular-pattern or coalescing-style ideas. Status: the
   publish-only sampled row is now well below sampled Loom on the vendored
   production benchmark, so the next decision point is workload breadth rather
   than more matmul-only heroics.

For each performance-sensitive step, record both latency and generated-code
metrics. The important codegen gates are private segment size, VGPR spill count,
SGPR pressure, approximate code size, and whether the hot kernel uses flat
scratch. The win condition for the matmul-only phase was sampled hip-moi
publish-only below sampled Loom publish-only on `prod_16x8_benchmark.hip` with
the fair default knobs; that has been achieved.

### Session 8: Broaden End-To-End Workloads

Status: first correctness test in progress.

The next non-negotiable scope increase is an end-to-end workload beyond one
isolated matmul. The likely first candidate is an attention block. This should
stress whether hip-moi's current fast path handles realistic phase structure,
LDS reuse, and multiple cooperating tiled kernels, rather than merely one
production matmul shape.

Before adding a benchmark row, keep one smaller instrumented correctness test in
the active suite. `tests/instrumented/009_attention_block_test.hip` is that
first rung: one workgroup runs a scalar Q/K/V attention block, stages K/V tiles
through LDS, performs online softmax accumulation, compares against a host
reference, and exercises both the diagnostic-capable exact context and the
publish-only sampled fast context. It is intentionally scalar and audit-friendly;
the benchmark version should be WMMA-heavy and more production-shaped.

The next rung is `tests/instrumented/010_rdna4_wmma_attention_block_test.hip`.
It is RDNA4-gated, grows the shape to 32 queries by 32 keys, uses two subgroups
in one workgroup, stages packed K/V fragments through LDS, computes QK and PV
with RDNA4 WMMA, and compares both exact-context and sampled-fast-context
instrumented kernels against a host reference. This remains a correctness test,
but its fragment layout and phase structure should be close enough to guide the
future benchmark extraction.

The first benchmark version should be a benchmark/reference workload before it
is a new detector feature. Keep the row structure familiar:

* noop baseline,
* sampled Loom-style comparison when feasible,
* hip-moi sampled publish-only fast path,
* optionally the general `context` path for correctness-oriented spot checks.

Avoid introducing attention-specific assumptions into `host_context` or storage
layout. If the attention benchmark requires new instrumentation shapes, prefer
explicit helper/API additions over hidden special cases.

### Session 9: Finer-Grained Synchronization

Status: design-only.

Workloads that synchronize through atomics rather than global `__syncthreads()`
are a separate semantic expansion. This belongs first in the general
`hip_moi::context` model, not in the sampled publish-only fast view.

The likely direction is to pair atomic instrumentation with synchronization
state that can express release/acquire-style edges. Fence-only reasoning remains
insufficient: fences matter when paired with operations, usually atomics, that
can create inter-thread synchronization. The local-only epoch optimization in
`sampled_watchpoint_context` should be treated as out of scope for this work
unless an equivalent proof is designed for the new synchronization model.

### Possible Later Work: Online Regular-Pattern Summaries

If benchmark analysis shows enough repeated regular LDS shapes to justify it,
revisit online subgroup regular-pattern detection. This should use subgroup
intrinsics to prove contiguous or fixed-stride access ranges without first
writing per-lane records to global memory.

## Near-Term Non-Goals

* Reintroducing thread-level same-subgroup diagnostics.
* Reintroducing the record/log/scan backend as a performance path.
* Diagnosing naked fences without atomics. Fence-only reasoning does not create
  synchronizes-with edges; any future fence work must be paired with atomics.
* Preserving the original race. The value proposition is deterministic
  diagnostics, while minimizing the chance that instrumentation both hides a
  race and fails to report it.
* Fully automatic instrumentation of arbitrary HIP code. The short-term path is
  explicit helper calls, explicit site ids, and explicit LDS offsets where the
  benchmark needs them.
