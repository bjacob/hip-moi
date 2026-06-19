<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# hip-moi Tutorial

This README is the tutorial. The numbered `.hip` files beside it are compiled
companions: they keep the examples honest, provide CTest coverage, and give
readers a place to experiment after reading.

Run the tutorial programs through CTest:

```sh
ctest --test-dir /home/benoit/workspace/hip-moi-build -R HipMoiTutorial --output-on-failure
```

The failing examples intentionally abort when run directly. Their CTest wrappers
expect that failure and check the diagnostic text.

## A Synchronized LDS Handoff

Start with an ordinary HIP kernel that communicates through LDS and uses
`__syncthreads()` to order the producer before the consumer:

```c++
__global__ void plain_producer_consumer_kernel(int* out)
{
    __shared__ int value;

    if(threadIdx.x == 0)
    {
        value = 41;
    }

    __syncthreads();

    if(threadIdx.x == 1)
    {
        out[0] = value + 1;
    }
}
```

To instrument the same kernel, pass a hip-moi storage view, create a
`hip_moi::context`, replace the LDS accesses with `ctx.lds_store` and
`ctx.lds_load`, and replace the barrier with `ctx.syncthreads()`:

```c++
__global__ void producer_consumer_kernel(int* out, hip_moi::context_storage_ref storage)
{
    __shared__ int value;

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
}
```

`ctx.syncthreads()` still performs the real workgroup barrier. In the MVP model
it also advances hip-moi's shadow epoch, so the store and load are known to be
ordered and no diagnostic is produced.

On the host, create a `hip_moi::host_context`, pass `moi.device_ref()` to the
kernel, and call `HIP_MOI_CHECK(moi)` at the point where diagnostics should be
reported:

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

The compiled companion is `001_passing_syncthreads.hip`.

## Diagnosing a Missing Barrier

Now remove the barrier from the plain HIP shape. This is the kind of bug hip-moi
is meant to make deterministic:

```c++
__global__ void plain_same_epoch_race_kernel(int* out)
{
    __shared__ int value;

    if(threadIdx.x == 0)
    {
        value = 123;
    }
    if(threadIdx.x == 1)
    {
        out[0] = value;
    }
}
```

The instrumented version is structurally the same, but the accesses go through
the context:

```c++
__global__ void same_epoch_race_kernel(int* out, hip_moi::context_storage_ref storage)
{
    __shared__ int value;

    hip_moi::config cfg{
        /*thread_count=*/static_cast<int>(blockDim.x),
        /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
        /*subgroup_count=*/1,
    };
    hip_moi::context ctx(storage, cfg);

    ctx.init_workgroup();

    if(threadIdx.x == 0)
    {
        ctx.lds_store(&value, 123);
    }
    if(threadIdx.x == 1)
    {
        out[0] = ctx.lds_load(&value);
    }
}
```

Because there is no `ctx.syncthreads()`, both accesses are in the same epoch.
`HIP_MOI_CHECK(moi)` reports the conflict to `stderr` and aborts:

```text
hip-moi diagnostic 0: kind=access_conflict ...
hip-moi: HIP_MOI_CHECK failed at ...
```

The compiled companion is `002_failing_same_epoch_race.hip`.

## Scope-Based Checking

The destructor path is also a first-class usage pattern. The kernel is the same
missing-barrier kernel as above; only the host-side consumption changes. Instead
of calling `HIP_MOI_CHECK(moi)` explicitly, let the context own a checked scope:

```c++
{
    hip_moi::host_context moi(small_context_options());

    hipLaunchKernelGGL(same_epoch_race_kernel,
                       dim3(1),
                       dim3(kThreadCount),
                       0,
                       0,
                       device_output,
                       moi.device_ref());
    check_hip(hipGetLastError(), "hipGetLastError");
}
```

When the `host_context` is destroyed, any unconsumed diagnostics are reported to
`stderr` and abort by default:

```text
hip-moi: unconsumed diagnostics in host_context destructor; aborting
```

This style is less precise than `HIP_MOI_CHECK(moi)`, but it is a legitimate
mode for users who want checked scopes with minimal host-side ceremony.

The compiled companion is `003_destructor_fallback.hip`.

## A Real Matmul Kernel

The larger example is a real single-workgroup 16x16x16 matmul using the RDNA4
WMMA builtin. RDNA4 and the data-tiled layout are details of this concrete
example; the tutorial point is how a real LDS-staging kernel changes when
instrumented.

The plain kernel stages each thread's A and B fragments through LDS, runs the
WMMA instruction, and writes each thread's C accumulator fragment contiguously:

```c++
__global__ void plain_data_tiled_wmma_kernel(const _Float16* a_global,
                                             const _Float16* b_global,
                                             float*          c_global)
{
    __shared__ f16x8_t a_shared[kThreadCount];
    __shared__ f16x8_t b_shared[kThreadCount];

    int lane = static_cast<int>(threadIdx.x) & 31;

    a_shared[lane] = load_global_fragment(a_global, lane);
    b_shared[lane] = load_global_fragment(b_global, lane);

    __syncthreads();

    f16x8_t a   = a_shared[lane];
    f16x8_t b   = b_shared[lane];
    f32x8_t acc = {};
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);

    __syncthreads();

    for(int elem = 0; elem < 8; ++elem)
    {
        c_global[data_tiled_fragment_offset(lane, elem)] = acc[elem];
    }
}
```

The matrix layout is data-tiled: each lane owns one contiguous fragment of A, B,
and C. For `_Float16` A/B WMMA fragments, lane `i` starts at byte offset
`i * 16`; for `float` C accumulator fragments, lane `i` starts at byte offset
`i * 32`. In element units, the helper is the same:

```c++
__host__ __device__ int data_tiled_fragment_offset(int lane, int elem)
{
    return lane * kFragmentElements + elem;
}

int offset = data_tiled_fragment_offset(lane, idx);
a[offset]  = static_cast<_Float16>(a_value(m, k));
b[offset]  = static_cast<_Float16>(b_value(n, k));
```

The instrumented version changes only the LDS-related operations and the
barriers. Global A/B loads, WMMA, packed C stores, and the host-side reference
matmul stay ordinary code:

```c++
__global__ void instrumented_data_tiled_wmma_kernel(
    const _Float16* a_global,
    const _Float16* b_global,
    float* c_global,
    hip_moi::context_storage_ref storage)
{
    __shared__ f16x8_t a_shared[kThreadCount];
    __shared__ f16x8_t b_shared[kThreadCount];

    hip_moi::context ctx = make_context(storage);
    ctx.init_workgroup();

    int lane = static_cast<int>(threadIdx.x) & 31;

    ctx.lds_store(&a_shared[lane], load_global_fragment(a_global, lane));
    ctx.lds_store(&b_shared[lane], load_global_fragment(b_global, lane));

    ctx.syncthreads();

    f16x8_t a   = ctx.lds_load(&a_shared[lane]);
    f16x8_t b   = ctx.lds_load(&b_shared[lane]);
    f32x8_t acc = {};
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);

    ctx.syncthreads();

    for(int elem = 0; elem < 8; ++elem)
    {
        c_global[data_tiled_fragment_offset(lane, elem)] = acc[elem];
    }
}
```

The companion program runs both the plain and instrumented kernels and checks
both packed outputs against the same host-side reference matmul. It is
CMake-gated because the builtin is RDNA4/gfx12-specific:

```sh
ctest --test-dir /home/benoit/workspace/hip-moi-build \
  -R HipMoiTutorial.004Rdna4WmmaDataTiledMatmul --output-on-failure
```

The compiled companion is `004_rdna4_wmma_data_tiled_matmul.hip`.
