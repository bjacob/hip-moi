<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Reference Kernels

This directory contains uninstrumented HIP kernels used as concrete source
shapes for tests and benchmarks.

The reference kernels serve two purposes:

* safe kernels are launched and checked against host-side expected outputs;
* diagnostic-positive kernels are kept as compile-only source shapes when
  launching them uninstrumented would rely on undefined behavior or could hang.

The active files are:

* `mvp_reference_kernels.hip`: the original broad reference corpus, including
  scalar LDS patterns, cooperative array patterns, looped epochs, 2D tiling,
  pseudorandom LDS layouts, and matmul-inspired kernels.
* `rdna4_jakub_matmul_reference.hip`: a `gfx12`-gated RDNA4 WMMA reference
  derived from Jakub's `sanitizer-strategy/rdna4_matmul` corpus.

## Harness Shape

Each launched reference case is represented as a small C++ type with:

* host metadata such as `name()`, `block_dim()`, and `grid_dim()`;
* a `__global__ static void run(int* out)` kernel;
* an `expected(int* out)` host oracle.

The host runner uses virtual dispatch only on the host side. Device code stays
ordinary HIP. GTest exposes the safe cases as a parameterized suite, so the
CTest output names the individual reference kernel that failed.

## Safe Reference Coverage

The launched no-diagnostic corpus includes:

* single-thread and same-thread scalar LDS cases;
* cooperative all-thread array cases;
* looping cases that reuse LDS across multiple full-workgroup barriers;
* 2D tiled and pseudorandomized LDS access patterns;
* simple matmul-inspired kernels with host reference checks;
* chunked, double-buffered, skewed-layout, and multi-workgroup matmul-inspired
  kernels;
* Jakub-derived RDNA4 packed FP16 WMMA matmul schedules: no-pipeline,
  pipelined, and double-buffered.

The reference corpus does not yet contain target workload atomics. The seed
kernels and extraction plan for that next expansion are tracked in
[`../../docs/atomics_corpus.md`](../../docs/atomics_corpus.md).

## Compile-Only Diagnostic Shapes

The compile-only corpus includes:

* same-epoch write/read and write/write conflicts;
* partially overlapping object and subobject writes;
* all-thread same-index array conflicts;
* divergent-barrier hard cases;
* Jakub-derived RDNA4 WMMA missing-barrier variants for load/compute and
  compute/load reuse mistakes.

These kernels are deliberately not launched by the reference self-test. The
corresponding instrumented tests should exercise the shape through hip-moi and
assert either deterministic diagnostics or a non-hanging simulated diagnostic
mode.

## Relationship To Instrumented Tests

Reference tests are not the specification for the detector. They are source
shapes and sanity checks. The detector contract is documented in
[`../../docs/instrumentation_model.md`](../../docs/instrumentation_model.md).

When an instrumented test adopts a reference shape, every LDS access in the
instrumented kernel should go through hip-moi unless the test is explicitly
comparing against a plain-HIP kernel.
