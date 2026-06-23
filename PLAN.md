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

The current implementation still uses the earlier record/log/scan design:

* ordinary accesses append `access_record` metadata,
* site-id coalescing can record lane-carrying proof metadata,
* epoch close scans records and emits deterministic diagnostics,
* storage is partitioned into typed record buffers.

That design was useful to establish the API and test semantics, but it is not
the long-term performance path. It performs too many global metadata writes and
creates too much scan work. It should become a reference/debug backend while the
main path moves to a Loom-inspired online detector.

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

The current access-record and coalescing-record data structures may remain as a
debug/reference backend while the new path is brought up. They should not be
used by the performance benchmark once the Loom-like backend exists.

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

This path deliberately permits false negatives from sampling, limited probes, or
table overwrites. It must not create false positives for the represented
dword-cell ranges.

## Source Site IDs and Coalescing

The default `site_id` remains zero, meaning no site-specific optimization:

```c++
ctx.lds_store(&lds[index], value);
```

A nonzero site id opts a static access site into possible optimization:

```c++
ctx.lds_store(&lds[index], value, HIP_MOI_SITE_ID());
```

The earlier coalescing implementation proved that regular lane-to-address
patterns can be recognized, but it did so by first writing one lane-carrying
record per participating lane. That defeats the optimization goal.

The new coalescing target is online:

* at an opted-in site, use subgroup intrinsics to inspect lane-local offsets,
* prove common regular patterns directly,
* publish one subgroup-level range when the proof succeeds,
* fall back to sampled per-lane watchpoints or exact shadow checks when the
  proof fails.

The first regular patterns to support are:

* contiguous per-lane accesses,
* fixed-stride per-lane accesses,
* descending fixed-stride accesses,
* vector LDS accesses used by the RDNA4 WMMA benchmark.

Dynamic instance identity is out of scope for now. If a static site has dynamic
behavior that the online proof cannot validate, it should fall back
conservatively.

## LDS Offset API

Real Loom gets LDS byte offsets from compiler lowering. A HIP library usually
only sees pointers. Jakub's HIP helpers pass byte offsets explicitly, which is
the right short-term shape for benchmark-driven work.

Keep the current high-level API:

```c++
ctx.lds_store(ptr, value, site);
ctx.lds_load(ptr, site);
```

Add explicit-offset overloads for the Loom-like fast path:

```c++
ctx.lds_store_at(ptr, value, /*lds_byte_offset=*/offset, site);
ctx.lds_load_at(ptr, /*lds_byte_offset=*/offset, site);
```

The explicit-offset overloads are the benchmark path. Later, we can investigate
whether `lds_load(ptr, site)` and `lds_store(ptr, value, site)` can derive LDS
offsets reliably enough on HIP/Clang, but performance work should not wait on
that question.

## Context Storage

`host_context_options::storage_bytes` remains the primary user-facing storage
knob and defaults to 16 MiB.

The target storage layout should be based on shadow/watchpoint needs rather
than access-record buffers:

* one or more dispatch/generation slots,
* per-workgroup epoch headers,
* per-workgroup exact shadow entries or sampled watchpoint entries,
* diagnostics and diagnostic counters,
* optional debug/reference buffers.

Typed capacities for access records and coalescing records should be demoted to
debug/reference backend options. The main backend should expose higher-level
knobs:

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
* hip-moi with site-id coalescing enabled.

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

The current hip-moi row is orders of magnitude slower than sampled Loom because
it still uses the record/log/scan path. The near-term benchmark goal is to make
the hip-moi row exercise the new Loom-like backend and exit the multi-millisecond
record-scan regime on the 2-wave shape.

## Test Corpus

The active instrumented suite is intentionally smaller after dropping
thread-level mode.

Kept:

* one single-subgroup diagnostic-free smoke test,
* host API and destructor behavior tests,
* core multi-subgroup conflict tests,
* RDNA4 multi-subgroup WMMA tests in data-tiled and row-major layouts,
* source-site id tests,
* coalescing proof and coalesced-conflict tests,
* simulated hard-synchronization diagnostics.

Removed:

* the old single-subgroup thread-level ladder,
* thread-level mirrors of multi-subgroup tests,
* single-subgroup RDNA4 tests whose only role was thread-level coverage.

New tests should follow the implementation path:

1. exact shadow-entry packing/unpacking and conflict predicates,
2. explicit LDS-offset access APIs,
3. exact shadow-path positive and negative tests,
4. sampled watchpoint positive and negative tests,
5. sampling false-negative documentation tests where deterministic behavior can
   be forced by knobs,
6. site-id coalescing tests for online regular-pattern proofs,
7. RDNA4 WMMA benchmark-shaped tests using the same helper style as Jakub.

Reference tests remain useful as uninstrumented kernels that compile and run,
especially as a source of later multi-subgroup or RDNA4 shapes.

## Implementation Roadmap

### Session 1: Shadow ABI Skeleton

Add Loom-style constants and compact record helpers:

* access-kind enum values compatible with the existing diagnostics,
* 64-bit exact shadow-entry pack/unpack helpers,
* sampled watchpoint pack/unpack helpers,
* generation and epoch helpers,
* host/device tests for bit layout where practical.

No benchmark win is expected from this session.

### Session 2: Storage Layout for Loom Backend

Teach `host_context` to allocate a shadow/watchpoint backend layout from
`storage_bytes`.

Keep existing record buffers available for the debug/reference backend, but make
the new storage layout able to represent:

* per-workgroup epoch headers,
* exact shadow entries or sampled watchpoints,
* diagnostics,
* counters needed by `HIP_MOI_CHECK`.

### Session 3: Explicit LDS-Offset APIs

Add `lds_load_at` and `lds_store_at` overloads. Initially they may route through
the current backend for compatibility. Then wire them to the exact shadow path.

Update one or two focused tests and the extracted benchmark glue to pass
explicit LDS byte offsets at the hip-moi helper sites.

### Session 4: Exact Shadow Backend

Implement the exact subgroup-level shadow path:

* compute shadow entry address from LDS byte offset,
* load epoch,
* pack current entry,
* update/check prior entry,
* emit deterministic diagnostics.

Use this to validate storage, epochs, diagnostics, and benchmark integration.
Expect correctness progress more than performance parity.

### Session 5: Sampled Watchpoint Backend

Implement the sampled-Loom-style backend:

* dword-cell range normalization,
* selected-lane publication,
* bounded probe scans,
* watchpoint overwrite behavior,
* report counting and cold diagnostics.

This is the first session expected to materially improve benchmark numbers.

### Session 6: Online Site-ID Coalescing

Replace lane-record coalescing on the benchmark path with online subgroup
coalescing:

* prove contiguous and fixed-stride patterns with subgroup intrinsics,
* publish one subgroup-level range when proof succeeds,
* fall back when proof fails,
* keep the old record-based coalescing path only as debug/reference support.

This is the main path toward making hip-moi competitive with Jakub's sampled
Loom row.

### Session 7: Benchmark-Driven Tightening

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

## Near-Term Non-Goals

* Reintroducing thread-level same-subgroup diagnostics.
* Making the record/log/scan backend competitive.
* Diagnosing naked fences without atomics. Fence-only reasoning does not create
  synchronizes-with edges; any future fence work must be paired with atomics.
* Preserving the original race. The value proposition is deterministic
  diagnostics, while minimizing the chance that instrumentation both hides a
  race and fails to report it.
* Fully automatic instrumentation of arbitrary HIP code. The short-term path is
  explicit helper calls, explicit site ids, and explicit LDS offsets where the
  benchmark needs them.
