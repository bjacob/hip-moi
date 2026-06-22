# PLAN for hip-moi: HIP memory-ordering instrumentation library

## Purpose

hip-moi has two purposes.

1. Instrument HIP programs at the source/API level to diagnose LDS memory-ordering
   bugs under the HIP/LLVM memory model.
2. Act as a prototype and thinking tool for a later assembly-level sanitizer,
   where the relevant model is the hardware execution model and where Jakub's
   Loom-style subgroup-level approach is the primary comparison point.

The second purpose now drives near-term prioritization. hip-moi is no longer
trying to be a general thread-level HIP race detector. The product surface is
subgroup-scoped and should be evaluated against Loom-like instrumentation.

## Memory-Model Ground Rules

HIP is modeled after C++, and HIP programs lower through LLVM IR. We refer to
the LLVM IR memory model because hip-moi may instrument code that uses Clang
builtins for operations such as fences, which are not fully specified as HIP
language APIs but do have well-defined LLVM IR lowerings.

For HIP-source instrumentation, correctness is judged at the HIP/LLVM level.
GPU execution folklore such as lockstep behavior is not part of that language
memory model.

For the future assembly-level effort, the model changes. There, hardware details
such as lockstep execution, instruction issue, and load/store queue behavior are
legitimate. That is the reason hip-moi now focuses on subgroup-to-subgroup LDS
interactions: it is the part most relevant to the intended assembly-level path.

## Current API

The public device API is:

```c++
hip_moi::context
```

The public host owner is:

```c++
hip_moi::host_context
```

The old `thread_level_context` and `subgroup_level_context` split has been
removed. `hip_moi::context` is the former subgroup-scoped detector, renamed to
be the only context. It reports same-epoch LDS conflicts between different
subgroups in the same workgroup. It intentionally does not report conflicts
whose participants are wholly inside one subgroup.

The standard kernel shape is:

```c++
__global__ void kernel(hip_moi::context::storage_ref storage) {
  hip_moi::context::config cfg{
      /*thread_count=*/static_cast<int>(blockDim.x),
      /*threads_per_subgroup=*/32,
      /*subgroup_count=*/2,
  };
  hip_moi::context ctx(storage, cfg);

  ctx.init_workgroup();
  // ctx.lds_load / ctx.lds_store
  // ctx.syncthreads() for real barriers and epoch boundaries
  ctx.finish(); // close the final epoch when needed
}
```

The standard host shape is:

```c++
hip_moi::host_context_options options;
options.storage_bytes = 16 * 1024 * 1024;
options.subgroup_capacity = 64;

hip_moi::host_context moi(options);
kernel<<<grid, block>>>(moi.device_ref());
HIP_MOI_CHECK(moi);
```

Scope-based handling through the `host_context` destructor is also a first-class
usage pattern. By default, unconsumed diagnostics are printed and abort the
process. Advanced users may disable destructor reporting and destructor aborting
separately.

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

## Coalescing

Coalescing is subgroup-scoped only.

The default `site_id` is zero, meaning exact logging only:

```c++
ctx.lds_store(&lds[index], value);
```

A nonzero site id opts a static access site into possible coalescing:

```c++
ctx.lds_store(&lds[index], value, HIP_MOI_SITE_ID());
```

Current coalescing stores one lane-carrying `coalescing_access_record` per
participating lane, then tries to summarize regular lane-to-address patterns at
epoch close. The summary representation can describe contiguous, fixed-stride,
and descending fixed-stride access patterns. If proof fails or storage is
missing/full, the access path falls back to exact records where possible.

The remaining performance target is to make coalescing materialize as a real
optimization in Jakub's benchmark: fewer global metadata writes, less scanning,
and lower VGPR pressure. The current design is correct and useful for
experiments, but not yet competitive with Loom.

## Context Storage

`host_context_options::storage_bytes` is the primary storage knob and defaults
to 16 MiB. The host context partitions that byte budget into typed buffers for
access records, diagnostics, subgroup state, counters, and optional coalescing
metadata.

Typed capacities remain advanced overrides:

* negative: derive from `storage_bytes`,
* positive: force the capacity,
* zero: disable optional coalescing buffers.

The access-record and diagnostic capacities cannot be zero.

Storage saturation should produce `metadata_full` diagnostics or conservative
fallback, not silent corruption. If both coalescing fallback and exact records
saturate, hip-moi may miss specific conflicts; the user response is to increase
`storage_bytes` or narrow instrumentation scope.

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

Reference tests remain useful as uninstrumented kernels that compile and run,
especially as a source of later multi-subgroup or RDNA4 shapes.

## Jakub Benchmark Integration

Jakub's `sanitizer-strategy/rdna4_matmul` benchmark now has a local branch
`hip-moi-benchmark` with a `fp16_wmma_tiled_prod_16x8_hip_moi_subgroup` row.
That row uses the current exact hip-moi context path with site-id coalescing
disabled, one hip-moi context per workgroup, and a configurable
`HIP_MOI_STORAGE_BYTES` budget.

Tiny-shape smoke results show the current exact path is orders of magnitude
slower than Loom. That is expected and is the point of having the row: it gives
us a direct measurement target while we reshape hip-moi toward a Loom-like fast
path.

## Immediate Priorities

1. Keep the narrowed API building cleanly and all retained tests passing.
2. Add benchmark rows for:
   * exact context with no coalescing,
   * context with site-id coalescing enabled at selected regular sites.
3. Study Jakub's Loom implementation and HIP prototype for the fast-path shape:
   compact metadata, minimal global memory traffic, and low VGPR pressure.
4. Redesign the hot path so regular subgroup-wide LDS accesses avoid per-lane
   global metadata writes where possible.
5. Add targeted instrumented tests only when they exercise behavior needed by
   the benchmark or the Loom-inspired design.

## Near-Term Non-Goals

* Reintroducing thread-level same-subgroup diagnostics.
* Diagnosing naked fences without atomics. Fence-only reasoning does not create
  synchronizes-with edges; any future fence work must be paired with atomics.
* Preserving the original race. The value proposition is deterministic
  diagnostics, while minimizing the chance that instrumentation both hides a
  race and fails to report it.
