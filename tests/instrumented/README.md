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

`001_safe_mvp_test.hip` is intentionally tiny: it verifies that the API can be
included from a HIP kernel, storage can be passed in, and a same-thread
instrumented LDS store/load can run with two logged accesses and zero
diagnostics.

`002_race_mvp_test.hip` starts the diagnostic-positive suite with a same-epoch
write/read conflict on one LDS address. It asserts diagnostic metadata rather
than numerical kernel output.

`003_host_context_test.hip` exercises the user-facing host layer:
`hip_moi::host_context`, `HIP_MOI_CHECK`, explicit diagnostic consumption, and
scope-based destructor handling of unconsumed diagnostics. It uses GTest stderr
capture and death tests to check both reporting and abort behavior.

`004_basic_conflict_predicate_test.hip` broadens the raw detector-contract
coverage for same-epoch byte ranges: read/read, write/write, non-overlap,
adjacent ranges, and overlapping subobjects.

`005_epoch_boundary_test.hip` exercises uniform `ctx.syncthreads()` as the MVP
epoch boundary: separated same-address accesses should not report, while a new
same-epoch conflict after a barrier should still report in the new epoch.

`006_all_thread_array_test.hip` moves beyond hand-picked threads: all threads in
the workgroup participate in simple LDS array write/read patterns, including one
intentionally unsynchronized neighbor-read diagnostic case.
