<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Future instrumented tests

This directory is reserved for tests that use the hip-moi API.

Initial instrumented tests should mirror the reference kernels in
`../reference/mvp_reference_kernels.hip`:

* diagnostic-free references should produce zero diagnostics,
* diagnostic-positive references should produce deterministic diagnostics,
* racy references should not rely on numerical kernel outputs as their oracle.
