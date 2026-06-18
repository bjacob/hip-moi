<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# hip-moi Tutorial

This directory contains small, self-contained HIP programs that demonstrate the
user-facing hip-moi workflow. They are built by CMake and registered as CTests
so the examples keep compiling and running as the library changes.

The examples are numbered in the same progression style as the instrumented
tests:

* `001_passing_syncthreads.hip`: a correctly synchronized producer/consumer.
* `002_failing_same_epoch_race.hip`: a race diagnosed by `HIP_MOI_CHECK`.
* `003_destructor_fallback.hip`: the same race, but with the check forgotten so
  `hip_moi::host_context` reports from its destructor.

Run them through CTest:

```sh
ctest --test-dir /home/benoit/workspace/hip-moi-build -R HipMoiTutorial --output-on-failure
```

The failing examples intentionally abort when run directly. Their CTest wrappers
expect that failure and check the diagnostic text.

## Basic Pattern

Create a host context, pass its device-side storage view to the kernel, then
check after launch:

```c++
hip_moi::host_context moi(small_context_options());

hipLaunchKernelGGL(producer_consumer_kernel,
                   dim3(1),
                   dim3(kThreadCount),
                   0,
                   0,
                   device_output,
                   moi.device_ref());
check_hip(hipGetLastError(), "hipGetLastError");

HIP_MOI_CHECK(moi);
```

Inside the kernel, bind the device context to that storage and use hip-moi APIs
for the LDS accesses you want checked:

```c++
hip_moi::config cfg{
    /*thread_count=*/static_cast<int>(blockDim.x),
    /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
    /*subgroup_count=*/1,
};
hip_moi::context ctx(storage, cfg);

ctx.init_workgroup();

if(threadIdx.x == 0)
{
    ctx.lds_store(&value, 41);
}

ctx.syncthreads();

if(threadIdx.x == 1)
{
    out[0] = ctx.lds_load(&value) + 1;
}
```

`ctx.syncthreads()` is both the real workgroup barrier and, in the MVP model,
the epoch boundary that tells hip-moi the accesses are ordered.

## Diagnosed Failure

If the barrier is missing, the two instrumented accesses are in the same epoch:

```c++
if(threadIdx.x == 0)
{
    ctx.lds_store(&value, 123);
}
if(threadIdx.x == 1)
{
    out[0] = ctx.lds_load(&value);
}
```

`HIP_MOI_CHECK(moi)` reports a diagnostic to `stderr` and aborts:

```text
hip-moi diagnostic 0: kind=access_conflict ...
hip-moi: HIP_MOI_CHECK failed at ...
```

If the user forgets `HIP_MOI_CHECK(moi)`, the default
`hip_moi::host_context` destructor still reports unconsumed diagnostics and
aborts:

```text
hip-moi: unconsumed diagnostics in host_context destructor; aborting
```
