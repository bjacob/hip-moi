<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# hip-moi Tutorial

This README is the tutorial. The numbered `.hip` files beside it are compiled
companions: they keep the examples honest, provide CTest coverage, and give
readers a place to experiment after reading.

hip-moi now exposes one device API:

```c++
hip_moi::context
```

The context is subgroup-scoped. It reports same-epoch LDS conflicts between
different subgroups in a workgroup, and intentionally does not try to diagnose
conflicts wholly inside one subgroup. This matches the project’s current
benchmarking target: compare hip-moi against Loom-style subgroup-level
instrumentation and then optimize that path.

Run the tutorial programs through CTest:

```sh
ctest --test-dir /home/benoit/workspace/hip-moi-build -R HipMoiTutorial --output-on-failure
```

The failing examples intentionally abort when run directly. Their CTest wrappers
expect that failure and check the diagnostic text.

## A Synchronized LDS Handoff

Start with an ordinary HIP kernel that communicates through LDS and uses
`__syncthreads()` to order the producer before the consumer. To instrument it,
pass a `hip_moi::context::storage_ref`, create a `hip_moi::context`, replace LDS
accesses with `ctx.lds_store` / `ctx.lds_load`, and replace the barrier with
`ctx.syncthreads()`.

The host side owns storage with `hip_moi::host_context`:

```c++
hip_moi::host_context_options options;
options.storage_bytes = 64 * 1024;
options.subgroup_capacity = 1;

hip_moi::host_context moi(options);

hipLaunchKernelGGL(producer_consumer_kernel,
                   dim3(1),
                   dim3(kThreadCount),
                   0,
                   0,
                   device_output,
                   moi.device_ref());

HIP_MOI_CHECK(moi);
```

The compiled companion is `001_passing_syncthreads.hip`.

## Diagnosing a Cross-Subgroup Race

The first interesting failure needs at least two subgroups in one workgroup. A
plain kernel can have subgroup 0 write an LDS value and subgroup 1 read it
without a barrier. The instrumented kernel makes the subgroup partition explicit:

```c++
hip_moi::context::config cfg{
    /*thread_count=*/static_cast<int>(blockDim.x),
    /*threads_per_subgroup=*/32,
    /*subgroup_count=*/2,
};
hip_moi::context ctx(storage, cfg);
```

Because there is no `ctx.syncthreads()`, both accesses are in the same epoch.
`HIP_MOI_CHECK(moi)` reports the conflict to `stderr` and aborts:

```text
hip-moi diagnostic 0: kind=access_conflict ... first_subgroup=...
hip-moi: HIP_MOI_CHECK failed at ...
```

The compiled companion is `002_failing_cross_subgroup_race.hip`.

## Scope-Based Checking

Calling `HIP_MOI_CHECK(moi)` gives precise control over where diagnostics are
consumed. It is also valid to rely on `hip_moi::host_context`’s destructor: if
diagnostics are unconsumed, the destructor reports them and aborts by default.
Advanced users can disable destructor reporting and aborting separately.

The compiled companion is `003_destructor_fallback.hip`.

## Real Matmul Shape

`004_rdna4_wmma_data_tiled_matmul.hip` is a real RDNA4 WMMA example. It first
shows a plain data-tiled 16x16x16 f16 WMMA kernel, then instruments the LDS
publication and consumption with `hip_moi::context`. It is gated by the CMake
RDNA4 architecture check.

## Coalescing

`006_context_coalescing.hip` shows how to opt in to coalescing at a static LDS
access site:

```c++
constexpr hip_moi::site_id site = HIP_MOI_SITE_ID();
ctx.lds_store(&shared[lane], value, site);
```

A nonzero `site_id` says the access site is eligible for automatic coalescing.
The detector still falls back to exact records when the observed lane-to-address
pattern is not regular enough to summarize safely.
