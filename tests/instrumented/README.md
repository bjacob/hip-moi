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

Files whose workgroups contain only one subgroup include `single_subgroup` in
their names. When such a file has deterministic thread-level race diagnostics
that are entirely intra-subgroup, it should have subgroup-level companion checks
asserting zero diagnostics. Diagnostic-free single-subgroup tests generally do
not need redundant subgroup-level mirrors.

`001_single_subgroup_safe_mvp_test.hip` is intentionally tiny: it verifies that the API can be
included from a HIP kernel, storage can be passed in, and a same-thread
instrumented LDS store/load can run with two logged accesses and zero
diagnostics.

`002_single_subgroup_race_mvp_test.hip` starts the diagnostic-positive suite with a same-epoch
write/read conflict on one LDS address. It asserts diagnostic metadata rather
than numerical kernel output, and checks that the same single-subgroup conflict
is intentionally not a subgroup-level diagnostic.

`003_host_context_test.hip` exercises the user-facing host layer:
`hip_moi::host_context`, `HIP_MOI_CHECK`, explicit diagnostic consumption, and
scope-based destructor handling of unconsumed diagnostics. It uses GTest stderr
capture and death tests to check both reporting and abort behavior.

`004_single_subgroup_basic_conflict_predicate_test.hip` broadens the raw detector-contract
coverage for same-epoch byte ranges: read/read, write/write, non-overlap,
adjacent ranges, and overlapping subobjects.

`005_single_subgroup_epoch_boundary_test.hip` exercises uniform `ctx.syncthreads()` as the MVP
epoch boundary: separated same-address accesses should not report, while a new
same-epoch conflict after a barrier should still report in the new epoch.

`006_single_subgroup_all_thread_array_test.hip` moves beyond hand-picked threads: all threads in
the workgroup participate in simple LDS array write/read patterns, including one
intentionally unsynchronized neighbor-read diagnostic case.

`007_single_subgroup_metadata_capacity_test.hip` covers access-log overflow, diagnostic counters
that exceed the stored diagnostic buffer, and the host-facing stderr message for
truncated diagnostic buffers.

`008_single_subgroup_loop_epoch_test.hip` stresses repeated epoch advancement in loops, including
safe producer/consumer loops, all-thread per-slot loops, repeated missing-barrier
diagnostics, and diagnostic epoch numbering across iterations.

`009_single_subgroup_tiled_lds_test.hip` covers 2D tiled LDS patterns: row-major, transpose,
skewed stride, blocked layout, diagonal gather, striped load/store, and an
unsynchronized transpose diagnostic case.

`010_single_subgroup_matmul_like_test.hip` covers small cooperative LDS matmul idioms: simple
2x2 and 4x4 tiles, a chunked K loop, and a scalar missing-barrier diagnostic.
The output-producing kernels consume explicit small integer input matrices and
compare against a host-side reference matmul.

`011_single_subgroup_epoch_log_lifetime_test.hip` verifies that access-log storage is reused at
epoch boundaries, so long synchronized loops can run with capacity sized for one
epoch rather than the whole kernel.

`012_single_subgroup_matmul_pipeline_test.hip` covers double-buffered and pipeline-like matmul
LDS patterns, including safe ping-pong buffering and diagnostic-positive buffer
reuse/partial-overwrite cases. Safe output cases use explicit small integer
inputs and a host-side reference matmul oracle.

`013_single_subgroup_rdna4_wmma_row_major_test.hip` is gated to RDNA4/gfx12 targets. It uses a
real `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` intrinsic with all 32
threads, conventional row-major LDS tiles, single-buffer and double-buffer safe
cases, and a diagnostic-positive row overwrite. Safe cases use non-uniform
small integer-valued `_Float16` inputs that compare exactly against a host-side
reference matmul.

`014_single_subgroup_rdna4_wmma_data_tiled_test.hip` is also gated to RDNA4/gfx12 targets. It
uses the same intrinsic but with data-tiled packed fragments. Each thread's A/B
fragment is a contiguous 16-byte object at byte offset `lane * 16`, and each
thread's C accumulator fragment is a contiguous 32-byte object at byte offset
`lane * 32`, stored with one `f32x8_t` vector store. The test includes a
diagnostic-positive neighbor-fragment overwrite. The packed A/B/C fragments are
generated from logical tiles and checked against the same exact host-side
reference matmul.

Across the `single_subgroup` files, representative thread-level
diagnostic-positive cases have subgroup-level companions that assert zero
diagnostics. This is deliberate: in a one-subgroup workgroup, subgroup-level
mode has no cross-subgroup pair to report.

`015_thread_level_subgroup_test.hip` starts multi-subgroup coverage for
`thread-level` mode. It uses a 64-thread workgroup split into two 32-thread
subgroups, checks `thread_level_context` subgroup identity helpers, verifies
subgroup ids in access records, and asserts both same-subgroup and
cross-subgroup same-epoch diagnostics.

`016_subgroup_level_bootstrap_test.hip` starts `subgroup-level` mode coverage.
It uses `subgroup_level_context` with the existing per-thread access records as
a bootstrap, but changes the conflict predicate to subgroup identity:
cross-subgroup same-epoch conflicts report, while same-subgroup conflicts
intentionally do not report by this mode's contract.

`017_subgroup_level_multisubgroup_test.hip` broadens `subgroup-level` coverage
to a 128-thread, four-subgroup workgroup. It covers array, loop, tiled, and
matmul-shaped LDS sharing patterns with both safe and missing-barrier
cross-subgroup cases.

`018_rdna4_multisubgroup_wmma_data_tiled_test.hip` is the first real
multi-subgroup RDNA4 WMMA test under both modes. It uses a 64-thread workgroup
split into two 32-thread subgroups, data-tiled vector fragments, double-buffered
two-tile LDS staging, exact host-reference output checks, and a missing-barrier
cross-subgroup diagnostic case checked in both `thread-level` and
`subgroup-level` modes.

`019_rdna4_multisubgroup_wmma_row_major_test.hip` is the conventional
row-major companion to `018`. It uses two subgroups, per-subgroup row-major
A/B/C tiles, double-buffered LDS staging, exact host-reference output checks,
and a missing-barrier cross-subgroup diagnostic case checked in both modes.

`020_site_id_test.hip` covers exact source-site id plumbing. It checks that
`HIP_MOI_SITE_ID()` produces nonzero compile-time ids, default accesses still
record `site_id == 0`, explicit site ids are stored in access records, and both
thread-level and subgroup-level diagnostics carry site ids without changing the
detector's exact behavior.

`021_coalescing_opportunity_test.hip` covers the first conservative use of
nonzero site ids for regular-access summaries. Thread-level contiguous and
fixed-stride sites produce one coalescing summary at `ctx.syncthreads()`, while
default-site accesses, repeated dynamic instances, and irregular address
patterns remain exact-only.

`022_subgroup_level_coalescing_test.hip` covers subgroup-level coalescing proof
logs and summaries. It verifies that default site ids do not write proof
records, opted-in sites write one proof record per lane, contiguous and
fixed-stride lane patterns summarize, repeated lanes are rejected, and
independent subgroups produce separate summaries.

`023_subgroup_level_coalesced_conflict_test.hip` starts using subgroup-level
coalesced summaries in conflict detection. It checks summary-vs-summary
diagnostics, summary-vs-exact diagnostics, read/read silence, disjoint
summaries, and fixed-stride gaps that overlap in span but not in represented
per-lane byte ranges.
