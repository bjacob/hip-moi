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
* `003_destructor_fallback.hip`: the same race diagnosed by the scope-based
  `hip_moi::host_context` destructor path.
* `004_rdna4_wmma_data_tiled_matmul.hip`: a real RDNA4/gfx12 WMMA matmul
  example using data-tiled packed A/B fragments and host-reference output
  checking. This example is built and registered only for gfx12-family HIP
  offload architectures.

Run them through CTest:

```sh
ctest --test-dir /home/benoit/workspace/hip-moi-build -R HipMoiTutorial --output-on-failure
```

The failing examples intentionally abort when run directly. Their CTest wrappers
expect that failure and check the diagnostic text.

## Explicit Check Pattern

Create a host context, pass its device-side storage view to the kernel, then
check at the point where you want diagnostics reported:

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

This style gives precise control over where diagnostics are consumed. It is a
good fit when one host context is reused across launches, when a test wants to
assert details immediately, or when the program wants the source location of the
explicit check in the failure message.

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

The destructor path is also a first-class usage pattern. Let a
`hip_moi::host_context` live for the scope you want checked; when it is
destroyed, any unconsumed diagnostics are reported to `stderr` and abort by
default:

```text
hip-moi: unconsumed diagnostics in host_context destructor; aborting
```

The destructor style is less precise than `HIP_MOI_CHECK(moi)`, but it is a
legitimate concise mode for users who want checked scopes with minimal host-side
ceremony.

## RDNA4 Data-Tiled WMMA Matmul

`004_rdna4_wmma_data_tiled_matmul.hip` is a larger example that looks more like
a real kernel. It uses all 32 threads in one workgroup, stages per-lane
`f16x8_t` fragments through LDS, calls the RDNA4 WMMA builtin, and checks the
full 16x16 output tile against a host-side reference matmul.

The A/B inputs are in data-tiled layout: each lane owns one contiguous fragment
for each matrix. For `_Float16` WMMA fragments, that means lane `i` starts at
byte offset `i * 16`:

```c++
int offset = lane * kFragmentElements + idx;
a[offset]  = static_cast<_Float16>(a_value(m, k));
b[offset]  = static_cast<_Float16>(b_value(n, k));
```

The kernel instruments the LDS fragment stores and loads, then uses the real
RDNA4 intrinsic:

```c++
ctx.lds_store(&a_shared[lane], load_global_fragment(a_global, lane));
ctx.lds_store(&b_shared[lane], load_global_fragment(b_global, lane));

ctx.syncthreads();

f16x8_t a = ctx.lds_load(&a_shared[lane]);
f16x8_t b = ctx.lds_load(&b_shared[lane]);
acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
```

The example is CMake-gated because the builtin is RDNA4/gfx12-specific. On a
gfx12 build it appears as:

```sh
ctest --test-dir /home/benoit/workspace/hip-moi-build \
  -R HipMoiTutorial.004Rdna4WmmaDataTiledMatmul --output-on-failure
```
