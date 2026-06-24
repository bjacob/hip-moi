<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# hip-moi Tutorial

This document is the tutorial. The numbered `.hip` files in this directory are
compiled companions: they keep the examples honest under CTest, and they give
readers complete programs to run or modify after reading.

The tutorial has two goals:

1. Show the source-level API shape that an instrumented HIP kernel uses.
2. Make the diagnostic path and the publish-only fast path visibly different.

For the exact metadata layouts and conflict predicates, use
[`../instrumentation_model.md`](../instrumentation_model.md). For allocation
details and larger context examples, use [`../context.md`](../context.md).

## Terms Used Here

This tutorial uses a small vocabulary consistently:

* **LDS** is HIP `__shared__` memory. hip-moi currently instruments LDS
  accesses only when the source code calls a hip-moi access helper.
* **Workgroup** means a HIP block. This matches the terminology used in the
  implementation docs.
* **Subgroup** is the owner unit recorded by hip-moi metadata. The current
  RDNA4 examples use 32 threads per subgroup.
* **Lane** is a thread's index inside its subgroup.
* **Epoch** is the interval between instrumented full-workgroup barriers. A
  cross-subgroup conflict is reportable only when both accesses are in the same
  epoch.
* **LDS byte offset** is the byte offset of the accessed object inside the
  kernel's shared-memory layout. hip-moi records this offset rather than only
  the C++ pointer value.
* **Site id** is a source-site identifier, usually produced by
  `HIP_MOI_SITE_ID()`, that helps diagnostics and sampling distinguish static
  instrumentation sites.
* **Watchpoint** means hip-moi's software sampled range record. It is not a
  hardware watchpoint.

Those definitions are expanded in
[`../instrumentation_model.md#scope`](../instrumentation_model.md#scope),
[`../instrumentation_model.md#site-ids`](../instrumentation_model.md#site-ids),
and
[`../instrumentation_model.md#sampled-watchpoint-entry`](../instrumentation_model.md#sampled-watchpoint-entry).

## Running The Tutorial

Run every tutorial program through CTest:

```sh
ctest --test-dir /home/benoit/workspace/hip-moi-build -R HipMoiTutorial --output-on-failure
```

Some tutorial programs intentionally fail when run directly. Their CTest
wrappers expect the abort and check the diagnostic text.

The examples are:

| File | Purpose | Expected direct behavior |
| --- | --- | --- |
| [`001_passing_syncthreads.hip`](001_passing_syncthreads.hip) | First safe instrumented LDS handoff. | Succeeds. |
| [`002_failing_cross_subgroup_race.hip`](002_failing_cross_subgroup_race.hip) | First explicit diagnostic. | Prints a diagnostic and aborts through `HIP_MOI_CHECK`. |
| [`003_destructor_fallback.hip`](003_destructor_fallback.hip) | Scope-based host diagnostic handling. | Prints a diagnostic and aborts in the destructor. |
| [`004_rdna4_wmma_data_tiled_matmul.hip`](004_rdna4_wmma_data_tiled_matmul.hip) | Real RDNA4 WMMA kernel shape with instrumented LDS staging. | Succeeds on RDNA4 builds. |
| [`005_context_cross_subgroup_race.hip`](005_context_cross_subgroup_race.hip) | Minimal diagnostic path used as contrast for the fast view. | Prints a diagnostic and aborts through `HIP_MOI_CHECK`. |
| [`006_sampled_watchpoint_context.hip`](006_sampled_watchpoint_context.hip) | Publish-only sampled fast view. | Succeeds; it does not report diagnostics. |

## The Two Device-Side Shapes

hip-moi currently exposes two device-side shapes:

```c++
hip_moi::context
hip_moi::sampled_watchpoint_context
```

`hip_moi::context` is the general diagnostic context. Use it when the host
should be able to consume diagnostics with `HIP_MOI_CHECK` or through
`hip_moi::host_context`'s destructor. The context can run exact-shadow
instrumentation or sampled-watchpoint instrumentation. Both are described in
[`../instrumentation_model.md#public-contexts`](../instrumentation_model.md#public-contexts).

`hip_moi::sampled_watchpoint_context` is narrower. It is a publish-only fast
view used by the high-performance benchmark rows. It publishes sampled
watchpoint metadata but does not report races. The trade-off is described in
[`../instrumentation_model.md#publish-only-fast-path`](../instrumentation_model.md#publish-only-fast-path)
and in [`../context.md#choosing-a-context`](../context.md#choosing-a-context).

## Example 001: A Synchronized LDS Handoff

Intent: this is the first safe example. It teaches the mechanical conversion
from a plain HIP kernel using LDS and `__syncthreads()` to a diagnostic-capable
hip-moi kernel.

The plain HIP pattern is:

```c++
if(threadIdx.x == 0)
{
    value = 41;
}

__syncthreads();

if(threadIdx.x == 1)
{
    out[0] = value + 1;
}
```

The instrumented version uses three pieces:

1. A host-owned metadata object:

   ```c++
   hip_moi::host_context moi(options);
   ```

2. A device-side diagnostic context:

   ```c++
   hip_moi::context ctx(storage, cfg);
   ctx.init_workgroup();
   ```

3. Wrapped LDS accesses and an instrumented barrier:

   ```c++
   ctx.lds_store_at(&value, 41, /*lds_byte_offset=*/0);
   ctx.syncthreads();
   int observed = ctx.lds_load_at(&value, /*lds_byte_offset=*/0);
   ```

`hip_moi::host_context` owns detector metadata in device global memory. The
kernel receives that metadata as a `hip_moi::context::storage_ref` value. The
larger allocation story is in [`../context.md#byte-budget`](../context.md#byte-budget).

The `ctx.init_workgroup()` call initializes per-workgroup detector state. The
`ctx.syncthreads()` call performs a real `__syncthreads()` and advances the
hip-moi epoch. The epoch model is summarized in
[`../instrumentation_model.md#synchronization-model`](../instrumentation_model.md#synchronization-model).

The `lds_byte_offset` argument is part of the represented access. In this
example `value` is the only shared object, so its offset is zero. In larger
kernels the offset must distinguish different shared arrays and different
elements inside those arrays.

After the kernel launch, the host explicitly consumes diagnostics:

```c++
HIP_MOI_CHECK(moi);
```

This safe example should produce no diagnostics. The complete program is
[`001_passing_syncthreads.hip`](001_passing_syncthreads.hip).

## Example 002: Diagnosing A Cross-Subgroup Race

Intent: this is the smallest diagnostic example. It shows the condition hip-moi
currently reports: same-epoch LDS access from two different subgroups, with at
least one write.

The kernel launches 64 threads and configures two 32-thread subgroups:

```c++
hip_moi::context::config cfg{
    /*thread_count=*/static_cast<int>(blockDim.x),
    /*threads_per_subgroup=*/32,
    /*subgroup_count=*/2,
};
```

Thread 0 is in subgroup 0. Thread 32 is in subgroup 1. Both touch the same LDS
scalar at byte offset zero, and there is no `ctx.syncthreads()` between them:

```c++
if(threadIdx.x == 0)
{
    ctx.lds_store_at(&value, 123, /*lds_byte_offset=*/0);
}
if(threadIdx.x == 32)
{
    out[0] = ctx.lds_load_at(&value, /*lds_byte_offset=*/0);
}
```

Because no instrumented barrier advanced the epoch, both accesses are in the
same epoch. `HIP_MOI_CHECK(moi)` synchronizes the device, copies diagnostics to
the host, prints the diagnostic, and aborts:

```text
hip-moi diagnostic 0: kind=access_conflict ... first_subgroup=...
hip-moi: HIP_MOI_CHECK failed at ...
```

The diagnostic predicate is defined in
[`../instrumentation_model.md#exact-shadow-access-algorithm`](../instrumentation_model.md#exact-shadow-access-algorithm).
The complete program is
[`002_failing_cross_subgroup_race.hip`](002_failing_cross_subgroup_race.hip).

## Example 003: Scope-Based Checking

Intent: this example shows that explicit `HIP_MOI_CHECK(moi)` is not the only
legitimate host-side reporting style.

`HIP_MOI_CHECK(moi)` is useful when the program wants to decide exactly where
diagnostics are consumed. `hip_moi::host_context` also checks for unconsumed
diagnostics in its destructor by default:

```c++
{
    hip_moi::host_context moi(options);
    kernel<<<grid, block>>>(..., moi.launch_ref());
}
```

If the kernel emitted diagnostics and the program did not consume them, the
destructor reports them and aborts. This behavior is deliberate, not just a
fallback for mistakes. Programs that want scope-based handling can use it as
their primary reporting pattern.

Advanced users can disable destructor reporting and destructor aborting
separately. Diagnostic consumption is summarized in
[`../instrumentation_model.md#diagnostic-record`](../instrumentation_model.md#diagnostic-record);
context selection is discussed in
[`../context.md#choosing-a-context`](../context.md#choosing-a-context).

The complete program is
[`003_destructor_fallback.hip`](003_destructor_fallback.hip).

## Example 004: A Real RDNA4 WMMA Matmul Shape

Intent: this example moves beyond scalar toy kernels. It shows that the same
instrumentation pattern applies to a real RDNA4 WMMA kernel that stages vector
fragments through LDS.

The program contains both a plain kernel and an instrumented kernel. Both
compute a 16x16x16 FP16-to-FP32 WMMA matmul. The A, B, and C matrices use a
data-tiled layout: each lane owns one contiguous vector fragment. This layout
matches the WMMA operand shape and avoids row-major unpacking in the kernel.

Only the LDS publication and consumption are wrapped:

```c++
ctx.lds_store_at(&a_shared[lane],
                 load_global_fragment(a_global, lane),
                 /*lds_byte_offset=*/kASharedByteOffset
                     + lane * static_cast<int>(sizeof(f16x8_t)));

ctx.syncthreads();

f16x8_t a = ctx.lds_load_at(&a_shared[lane],
                            /*lds_byte_offset=*/kASharedByteOffset
                                + lane * static_cast<int>(sizeof(f16x8_t)));
```

The WMMA intrinsic and global stores remain ordinary kernel code. This is
important: hip-moi instruments selected LDS operations; it does not become a
replacement execution framework for the whole kernel.

This example uses one 32-thread subgroup, so it is a correctness and API-shape
example rather than a race diagnostic example. It is gated by the CMake RDNA4
architecture check. The complete program is
[`004_rdna4_wmma_data_tiled_matmul.hip`](004_rdna4_wmma_data_tiled_matmul.hip).

The larger benchmark suite uses the same discipline: every LDS access in an
instrumented benchmark path is routed through the selected instrumentation
path. See [`../../benchmarks/README.md`](../../benchmarks/README.md).

## Example 005: The Diagnostic Context In Its Simplest Failure Mode

Intent: this example deliberately repeats the simple cross-subgroup race shape
so it can be compared directly with Example 006.

The important point is not that this kernel is different from Example 002. The
important point is which device-side object it uses:

```c++
hip_moi::context ctx(storage, cfg);
```

This is the diagnostic-capable path. The host can call:

```c++
HIP_MOI_CHECK(moi);
```

and get a user-facing diagnostic. The complete program is
[`005_context_cross_subgroup_race.hip`](005_context_cross_subgroup_race.hip).

## Example 006: Publish-Only Sampled Watchpoints

Intent: this example shows the high-performance path used by benchmark rows
named `sampled_watchpoint_context`. It is not a diagnostic example.

The fast path starts with a compile-time sampled policy:

```c++
using publish_only_policy = hip_moi::sampled_watchpoint_policy<
    /*SampleSkip=*/32,
    /*ProbeCount=*/1,
    /*DelayIters=*/32,
    /*ReportConflicts=*/false,
    /*StaticWatchpointCapacity=*/1>;
```

The fields mean:

* `SampleSkip=32`: deterministically sample only some subgroup/site instances;
* `ProbeCount=1`: one probe would be used by reporting policies, but this
  example is publish-only;
* `DelayIters=32`: match the benchmark comparison delay knob;
* `ReportConflicts=false`: publish metadata but do not diagnose conflicts;
* `StaticWatchpointCapacity=1`: tell the compiler the watchpoint table has one
  entry, so slot selection can fold to zero.

Those knobs are defined in
[`../context.md#backend-storage`](../context.md#backend-storage). The packed
watchpoint record is defined in
[`../instrumentation_model.md#sampled-watchpoint-entry`](../instrumentation_model.md#sampled-watchpoint-entry).

The host still allocates metadata through `hip_moi::host_context`, but it uses
sampled-watchpoint publish-only options:

```c++
hip_moi::host_context_options options =
    hip_moi::make_one_watchpoint_publish_only_options();
```

Inside the kernel, the generic storage ref is converted into the narrower fast
view:

```c++
hip_moi::sampled_watchpoint_context::config cfg{
    /*threads_per_subgroup=*/32,
};
hip_moi::sampled_watchpoint_context ctx =
    hip_moi::make_sampled_watchpoint_context(storage, cfg);
```

Then LDS accesses use the fast API:

```c++
ctx.lds_store_at<publish_only_policy>(
    &values[index],
    index * 3,
    /*lds_byte_offset=*/index * sizeof(int),
    HIP_MOI_SITE_ID());

ctx.syncthreads();

int loaded = ctx.lds_load_at<publish_only_policy>(
    &values[neighbor],
    /*lds_byte_offset=*/neighbor * sizeof(int),
    HIP_MOI_SITE_ID());
```

There is intentionally no `HIP_MOI_CHECK(moi)` in this example.
`sampled_watchpoint_context` does not report conflicts. A successful run means
the kernel computed the right values while publishing sampled watchpoint
metadata; it does not mean the program was proven race-free.

The complete program is
[`006_sampled_watchpoint_context.hip`](006_sampled_watchpoint_context.hip). The
benchmark rows that measure this path are documented in
[`../../benchmarks/README.md#benchmark-modes`](../../benchmarks/README.md#benchmark-modes).

## What To Read Next

After this tutorial:

* Read [`../instrumentation_model.md`](../instrumentation_model.md) to see the
  exact fields recorded by exact shadow entries and sampled watchpoints.
* Read [`../context.md`](../context.md) for storage sizing, context options,
  one-watchpoint publish-only setup, and multi-workgroup storage patterns.
* Read [`../../benchmarks/README.md`](../../benchmarks/README.md) to connect
  the tutorial paths to measured RDNA4 overhead.
* Read [`../../tests/instrumented/README.md`](../../tests/instrumented/README.md)
  to see the correctness test ladder beyond these small examples.
