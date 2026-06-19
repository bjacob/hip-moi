<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# MVP reference kernels

`mvp_reference_kernels.hip` is a self-checking GTest HIP binary containing the
first MVP corpus in uninstrumented form.

Each reference case is represented as a small C++ type with:

* metadata such as `name()`, `block_dim()`, and `grid_dim()`,
* a `__global__ static void run(int *out)` kernel,
* an `expected(int *out)` host-side oracle for launched safe cases.

The host harness uses concrete virtual `ReferenceCase` subclasses and passes
them to a non-templated runner; virtual dispatch stays entirely on the host side.
The launched safe cases are exposed to GTest as a parameterized suite, so CTest
can report one named entry per reference kernel.

This is intentionally similar in spirit to the kernel objects in
`/home/benoit/workspace/hip-matmul/matmul_rdna4.hip`, but much smaller and
specific to hip-moi's reference corpus.

The safe reference kernels are launched and numerically checked. The
diagnostic-positive references are full `__global__` kernels too, but are not
launched by this executable because they intentionally contain racy or otherwise
hard synchronization patterns. Once hip-moi exists, corresponding instrumented
tests should assert that those shapes produce diagnostics.

This split gives us three useful properties before the library exists:

* every reference case is concrete code, not a snippet,
* safe reference cases compile, run, and validate basic outputs,
* racy reference cases still compile and remain available as templates for the
  real diagnostic tests.

## Current MVP layers

The launched no-diagnostic corpus currently includes:

* tiny scalar cases preserving the original first examples,
* cooperative array cases where all or most threads participate,
* looping cases that reuse LDS across many synchronization epochs,
* 2D tiled and pseudorandomized LDS access patterns,
* simple matmul-inspired kernels with cooperative LDS loads and host-side
  reference matmul checks,
* more involved matmul-inspired kernels with K loops, double buffering, skewed
  LDS layouts, multiple workgroups, and host-side reference matmul checks.

The compile-only diagnostic-positive corpus currently includes:

* same-epoch write/read and write/write conflicts,
* partially overlapping object/subobject writes,
* all-thread same-index array conflicts,
* divergent-barrier hard cases that should eventually be handled through
  simulation mode rather than launched directly.

Future instrumented tests should reuse the same case shapes and assert:

* launched safe cases produce no diagnostics,
* compile-only diagnostic-positive cases produce deterministic diagnostics,
* hard divergent-barrier cases are only tested in a mode that cannot hang.
