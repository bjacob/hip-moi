<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Instrumented Tests

The instrumented suite checks hip-moi's active diagnostic and publish-only
paths. The diagnostic context is `hip_moi::context`; it diagnoses same-epoch
LDS conflicts between different subgroups in a workgroup. The publish-only fast
view is `hip_moi::sampled_watchpoint_context`; tests use it to ensure the fast
path runs correctly, not to assert race diagnostics.

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
  correctness test for exact-shadow and sampled fast execution.

Architecture-specific attention and ping-pong families:

* RDNA4/gfx120* tests:
  `010_rdna4_wmma_attention_block_test.hip`,
  `011_rdna4_d128_attention_block_test.hip`,
  `012_rdna4_d128_attention_pressure_test.hip`,
  `013_rdna4_wmma_register_handoff_test.hip`,
  `014_rdna4_wmma_no_score_lds_attention_test.hip`,
  `015_rdna4_d128_no_score_lds_attention_test.hip`,
  `016_rdna4_pingpong_private_lds_test.hip`,
  `017_rdna4_pingpong_cooperative_lds_test.hip`, and
  `018_rdna4_pingpong_wide_cooperative_lds_test.hip`.
* CDNA4/gfx950 tests:
  `010_cdna4_mfma_attention_block_test.hip`,
  `011_cdna4_d128_attention_block_test.hip`,
  `012_cdna4_d128_attention_pressure_test.hip`,
  `013_cdna4_mfma_register_handoff_test.hip`,
  `014_cdna4_mfma_no_score_lds_attention_test.hip`,
  `015_cdna4_d128_no_score_lds_attention_test.hip`,
  `016_cdna4_pingpong_private_lds_test.hip`,
  `017_cdna4_pingpong_cooperative_lds_test.hip`, and
  `018_cdna4_pingpong_wide_cooperative_lds_test.hip`.
* gfx1250 tests:
  `010_gfx1250_wmma_attention_block_test.hip`,
  `011_gfx1250_d128_attention_block_test.hip`,
  `012_gfx1250_d128_attention_pressure_test.hip`,
  `013_gfx1250_wmma_register_handoff_test.hip`,
  `014_gfx1250_wmma_no_score_lds_attention_test.hip`,
  `015_gfx1250_d128_no_score_lds_attention_test.hip`,
  `016_gfx1250_pingpong_private_lds_test.hip`,
  `017_gfx1250_pingpong_cooperative_lds_test.hip`, and
  `018_gfx1250_pingpong_wide_cooperative_lds_test.hip`.

These families cover dense attention, D128 pressure, register handoff,
no-score/weight-LDS attention, and ping-pong LDS sharing shapes using the
architecture-specific matrix instructions available on each target.

* `019_atomic_api_test.hip`: pass-through atomic API skeleton for global
  release/acquire flags around LDS payload code.
* `020_atomic_metadata_test.hip`: bounded atomic-object metadata capacity,
  launch-generation reuse, and deterministic metadata-full diagnostics.
* `021_atomic_happens_before_test.hip`: release/acquire atomic synchronization
  suppresses an ordered LDS handoff while relaxed publication still diagnoses.
* `023_atomic_rmw_happens_before_test.hip`: release/acquire `fetch_add`
  arrival counters and two-RMW `acq_rel` chains order LDS payload handoffs.
* `024_atomic_or_bitmask_happens_before_test.hip`: old-value-dependent
  `atomicOr` bitmask control flow orders LDS payload when the returned mask
  observes the releasing operation.
* `025_atomic_fence_happens_before_test.hip`: release-fence-before-relaxed
  publication paired with relaxed-observation-before-acquire-fence.
* `026_streamk_flag_protocol_test.hip`: RocJITsu hip-stream-k-shaped
  owner/helper flag fixup distilled to LDS partial payloads.
* `027_streamk_two_tile_flag_protocol_test.hip`: two independent
  Stream-K-style owner/helper tile fixups, one release/acquire flag per tile.

Architecture-specific WMMA/MFMA Stream-K families:

* Arrival-counter tests:
  `028_rdna4_wmma_streamk_arrival_counter_test.hip`,
  `028_cdna4_mfma_streamk_arrival_counter_test.hip`, and
  `028_gfx1250_wmma_streamk_arrival_counter_test.hip`.
* Tree `atomicOr` tests:
  `029_rdna4_wmma_streamk_tree_atomic_or_test.hip`,
  `029_cdna4_mfma_streamk_tree_atomic_or_test.hip`, and
  `029_gfx1250_wmma_streamk_tree_atomic_or_test.hip`.

These tests publish architecture-specific matrix-instruction K-slice LDS
partials and then use atomics to order the final fold.

* `030_atomic_exchange_compare_exchange_test.hip`: exchange and
  compare-exchange synchronization shapes, including successful lock-like CAS
  and failed acquire CAS.
* `031_atomic_fence_rmw_happens_before_test.hip`: release/acquire fences paired
  with relaxed RMW atomics.
* `032_workgroup_barrier_fence_test.hip`: lower-level
  release-fence/barrier/acquire-fence spelling for full-workgroup epoch
  boundaries.
* `033_atomic_fence_extended_test.hip`: release/acquire fences paired with
  relaxed exchange and compare-exchange, plus a `seq_cst` load/store sanity
  case.
* `034_atomic_bitwise_happens_before_test.hip`: release/acquire `fetch_and`
  bit clearing and `fetch_xor` bit toggling as old-value-dependent bitmask
  synchronization shapes.
* `035_attention_lds_alias_handoff_test.hip`: scalar attention-pipeline LDS
  alias handoff where a missing full-workgroup barrier reports, a conditional
  barrier is clean, and non-aliased LDS slots do not need the handoff barrier.
* Architecture-specific attention LDS-alias handoff tests:
  `036_rdna4_wmma_attention_lds_alias_handoff_test.hip`,
  `036_cdna4_mfma_attention_lds_alias_handoff_test.hip`, and
  `036_gfx1250_wmma_attention_lds_alias_handoff_test.hip`. These are paired
  with matching benchmarks where available.
* `037_streamk_two_level_reduction_test.hip`: Stream-K-style two-level flag
  reduction where four producer subgroups feed two pair reducers, then one
  final reducer consumes the pair partials.

The removed single-subgroup ladder was useful while hip-moi still had a
per-thread detector. It is deliberately gone from the active corpus so the tests
track the project's current Loom-comparison focus.
