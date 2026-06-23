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

## Current Detector Contract

A diagnostic of interest is:

* two instrumented LDS accesses overlap in byte range,
* the accesses are from different subgroups in the same workgroup,
* the accesses are in the same synchronization epoch,
* at least one access is a write.

`ctx.syncthreads()` performs a real workgroup barrier, closes the current epoch,
checks conflicts for that epoch, and advances subgroup epochs. `ctx.finish()`
closes the final epoch for kernels that end after instrumented accesses without
another barrier.

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

Only selected lanes publish and check watchpoints. The selection should be
deterministic for a launch and mixed from generation, workgroup id, subgroup id,
site id, and possibly the access range. Knobs such as watchpoint count, probe
count, and sample skip should mirror Jakub's benchmark enough that we can make
meaningful comparisons.

The current sampled selector is intentionally cheap: it uses 32-bit mixing and
power-of-two masks rather than 64-bit modulo arithmetic. This is part of the
hot-path contract; sampled capacities should be chosen as powers of two for
good distribution.

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
kernel<<<grid, block>>>(moi.device_ref());
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

Jakub's `sanitizer-strategy/rdna4_matmul` benchmark is now the guiding
performance loop.

The local `hip-moi-benchmark` branch contains an extracted benchmark family
derived from `rdna4_matmul/rdna4_matmul.hip`. It compares:

* noop,
* sampled Loom,
* hip-moi exact shadow through explicit LDS-offset APIs,
* hip-moi sampled watchpoints through explicit LDS-offset APIs.

The benchmark now constructs the hip-moi device context once near kernel entry
and passes it through the instrumented access helpers. That better matches the
intended source-instrumentation shape than rebuilding a context inside every
instrumented access wrapper.

The current go-to shapes are:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh
./rdna4_matmul/build_w2_2x4_benchmark.sh w4_4x16
./rdna4_matmul/build_w2_2x4_benchmark.sh w8_16x8
```

Use the 2-wave shape for fast intra-session iteration. Run all three before a
session-ending commit when performance-sensitive code has changed.

Append the raw output of those three commands to `BENCHMARK_LOG.md` at each
commit. For doc-only commits, either append a fresh run or explicitly note that
the commit is performance-equivalent to the previous logged entry.

The current hip-moi rows exercise the explicit-offset exact-shadow and
sampled-watchpoint paths. The sampled path is not yet competitive with Jakub's
sampled Loom row; the next benchmark goal is to shrink hot-path helper work,
VGPR pressure, and selected-lane overhead.

Latest codegen audit on the extracted 2-wave benchmark:

* before backend specialization, hip-moi exact and sampled both compiled to the
  same 79-VGPR, 37.6-KB-code kernel because runtime backend dispatch kept both
  implementations live;
* compile-time backend selection reduced exact shadow to 52 VGPR and 19.5 KB of
  code;
* compile-time backend selection plus cheaper sampled selection reduced sampled
  watchpoints to 57 VGPR and 15.2 KB of code;
* the generated metadata still reports `uses_flat_scratch=1`, although
  `ScratchSize` is 0 for the hot kernels.

## Test Corpus

The active instrumented suite is intentionally smaller after dropping
thread-level mode.

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

Status: in progress. The first tightening pass added compile-time backend
selection for the explicit-offset access helpers and replaced sampled
watchpoint lane/slot selection with cheaper 32-bit mixing and masking. On the
2-wave benchmark, sampled hip-moi moved from roughly 0.010 ms after legacy
cleanup to roughly 0.007 ms, while sampled Loom remains roughly 0.005 ms. The
same pass made the 4-wave and 8-wave sampled rows roughly 0.011 ms.

The benchmark harness was then changed to create the hip-moi context once per
kernel and pass it through the helper calls. This reduced sampled hip-moi again
to roughly 0.006 ms on 2-wave and 0.009 ms on 4/8-wave. Code size dropped, but
VGPR use rose to 63 for the sampled row, so this is not a free lunch; it mostly
confirms that repeated context lookup was real overhead and that context live
range is now a central tuning problem.

Use the 2/4/8-wave benchmark set to tune:

* watchpoint count,
* probe count,
* sample skip,
* VGPR pressure,
* inlining decisions,
* cold diagnostic isolation.

Inspect generated code when benchmark movement is surprising. The main danger is
spilling caused by instrumentation VGPR usage; global-memory traffic from spills
can dominate the actual sanitizer work.

Next likely target: remove repeated benchmark helper overhead around
`make_hip_moi_context()` and context construction, or add a slimmer sampled
hot-path wrapper, while preserving the ordinary user-facing `host_context` API.
After the benchmark harness change, the likely library-side version of this is
a smaller sampled hot-path context/view that carries only the fields needed by
the sampled backend, so caching the context does not keep exact-shadow and
diagnostic-heavy state live across the whole kernel.

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
