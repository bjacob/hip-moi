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

Instrumented tests should route every LDS access in the instrumented kernel
through hip-moi. Reference kernels and plain tutorial kernels may remain
uninstrumented, but once a test case is exercising the instrumentation API, raw
`__shared__` reads or writes are not part of the intended test shape.

Current files:

* `001_single_subgroup_safe_mvp_test.hip`: one diagnostic-free API smoke test.
* `002_host_context_test.hip`: `hip_moi::host_context`, `HIP_MOI_CHECK`,
  destructor reporting/aborting, byte-budget allocation, and diagnostic text.
* `003_site_id_test.hip`: source-site ids in diagnostics.
* `004_hard_synchronization_negative_test.hip`: non-hanging simulated barrier
  divergence diagnostics.
* `005_shadow_abi_test.hip`: host/device shadow-packing ABI checks.
* `006_lds_offset_api_test.hip`: explicit LDS byte-offset instrumentation.
* `007_exact_shadow_backend_test.hip`: deterministic exact-shadow conflict
  diagnostics.
* `008_sampled_watchpoint_backend_test.hip`: sampled watchpoint diagnostics.
* `009_attention_block_test.hip`: scalar attention-shaped Q/K/V tiled LDS
  correctness test for the exact context and sampled fast context.
* `010_rdna4_wmma_attention_block_test.hip`: RDNA4-only WMMA-heavy attention
  correctness test, used as the stepping stone toward an attention benchmark.
* `011_rdna4_d128_attention_block_test.hip`: RDNA4-only D128/V128 attention
  correctness test shaped by the AITER/llama.cpp source-mined production
  signal, used as the rung before the next attention benchmark.
* `012_rdna4_d128_attention_pressure_test.hip`: RDNA4-only D128/V128 attention
  pressure correctness tests for full K/V double-buffering at about 19.25 KiB
  of LDS and a wider 32-key pressure tile at about 38.25 KiB of LDS.

The removed single-subgroup ladder was useful while hip-moi still had a
thread-level detector. It is deliberately gone from the active corpus so the
tests track the project’s current Loom-comparison focus.
