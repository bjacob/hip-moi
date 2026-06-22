<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Instrumented Tests

The instrumented suite is now intentionally narrow. hip-moi exposes one
device-side API, `hip_moi::context`, and that context diagnoses same-epoch LDS
conflicts between different subgroups in a workgroup.

The suite keeps one single-subgroup smoke test so the basic API stays exercised,
then focuses on the cross-subgroup behavior that matters for comparison with
Loom-style instrumentation.

Current files:

* `001_single_subgroup_safe_mvp_test.hip`: one diagnostic-free API smoke test.
* `003_host_context_test.hip`: `hip_moi::host_context`, `HIP_MOI_CHECK`,
  destructor reporting/aborting, byte-budget allocation, and diagnostic text.
* `017_context_multisubgroup_test.hip`: core cross-subgroup conflict and epoch
  behavior.
* `018_rdna4_multisubgroup_wmma_data_tiled_test.hip`: RDNA4 WMMA data-tiled
  multi-subgroup cases.
* `019_rdna4_multisubgroup_wmma_row_major_test.hip`: RDNA4 WMMA row-major
  multi-subgroup cases.
* `020_site_id_test.hip`: source-site ids in access records and diagnostics.
* `022_context_coalescing_test.hip`: coalescing access logs and summaries.
* `023_context_coalesced_conflict_test.hip`: conflict detection using coalesced
  summaries.
* `024_hard_synchronization_negative_test.hip`: non-hanging simulated barrier
  divergence diagnostics.

The removed single-subgroup ladder was useful while hip-moi still had a
thread-level detector. It is deliberately gone from the active corpus so the
tests track the project’s current Loom-comparison focus.
