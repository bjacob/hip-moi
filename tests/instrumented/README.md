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
* `010_rdna4_wmma_attention_block_test.hip`: RDNA4-only WMMA-heavy attention
  correctness test paired with the D16 dense attention benchmark.
* `011_rdna4_d128_attention_block_test.hip`: RDNA4-only D128/V128 attention
  correctness test shaped by the AITER/llama.cpp source-mined production
  signal and paired with the D128 dense attention benchmark.
* `012_rdna4_d128_attention_pressure_test.hip`: RDNA4-only D128/V128 attention
  pressure correctness tests for full K/V double-buffering at about 19.25 KiB
  of LDS and a wider 32-key pressure tile at about 38.25 KiB of LDS.
* `013_rdna4_wmma_register_handoff_test.hip`: RDNA4-only QK-to-PV register
  handoff test that reshapes WMMA accumulator fragments into the next WMMA
  operand without dense score/weight LDS scratch.
* `014_rdna4_wmma_no_score_lds_attention_test.hip`: RDNA4-only two-key-tile
  linear attention-shaped test that uses the register handoff and instruments
  K/V LDS staging without materializing dense score/weight scratch.
* `015_rdna4_d128_no_score_lds_attention_test.hip`: RDNA4-only D128/V128
  version of the no-score/weight-LDS attention test, paired with the D128
  no-score attention benchmark.
* `016_rdna4_pingpong_private_lds_test.hip`: RDNA4-only ping-pong-shaped WMMA
  test with `setprio` and `sched_barrier` where each subgroup stages and
  consumes only its own LDS slots.
* `017_rdna4_pingpong_cooperative_lds_test.hip`: RDNA4-only ping-pong-shaped
  WMMA test where subgroup 0 stages B fragments in LDS for every subgroup to
  consume; the synchronized case is clean and the unsynchronized case reports
  diagnostics.
* `018_rdna4_pingpong_wide_cooperative_lds_test.hip`: RDNA4-only four-subgroup
  ping-pong-shaped WMMA test where each pair has an even subgroup stage shared
  B fragments for both subgroups in the pair.
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
* `028_rdna4_wmma_streamk_arrival_counter_test.hip`: RDNA4-only WMMA
  Stream-K-shaped arrival-counter test where two subgroups publish K-slice LDS
  partials and the final subgroup folds them.
* `029_rdna4_wmma_streamk_tree_atomic_or_test.hip`: RDNA4-only WMMA
  Stream-K-tree-shaped `atomicOr` bitmask test where three subgroups publish
  K-slice LDS partials and the final subgroup folds them.
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
* `036_rdna4_wmma_attention_lds_alias_handoff_test.hip`: RDNA4-only WMMA-shaped
  version of the LDS alias handoff, paired with the matching benchmark.

The removed single-subgroup ladder was useful while hip-moi still had a
per-thread detector. It is deliberately gone from the active corpus so the tests
track the project's current Loom-comparison focus.
