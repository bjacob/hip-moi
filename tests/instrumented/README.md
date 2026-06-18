<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Instrumented tests

This directory contains tests that use the hip-moi API. It should grow in
lockstep with implemented library capabilities, not by bulk-instrumenting the
whole reference corpus ahead of support.

Instrumented tests should mirror the reference kernels in
`../reference/mvp_reference_kernels.hip`:

* diagnostic-free references should produce zero diagnostics,
* diagnostic-positive references should produce deterministic diagnostics,
* racy references should not rely on numerical kernel outputs as their oracle.

`safe_mvp_test.hip` is intentionally tiny: it verifies that the API can be
included from a HIP kernel, storage can be passed in, and a same-thread
instrumented LDS store/load can run with two logged accesses and zero
diagnostics.

`race_mvp_test.hip` starts the diagnostic-positive suite with a same-epoch
write/read conflict on one LDS address. It asserts diagnostic metadata rather
than numerical kernel output.
