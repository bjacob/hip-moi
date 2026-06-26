# hip-moi Benchmarks

This directory contains the RDNA4 performance benchmarks used to guide
hip-moi's sampled publish-only implementation. The matmul benchmarks are
focused extractions from Jakub's
`sanitizer-strategy/rdna4_matmul/rdna4_matmul.hip`; the attention benchmarks are
hip-moi-native workloads grown from the instrumented test corpus and from the
llama.cpp/AITER production-shape signals summarized in this README and in
[`docs/benchmark_interpretation.md`](../docs/benchmark_interpretation.md).

For interpretation of what the benchmark modes and current numbers imply, read
[`docs/benchmark_interpretation.md`](../docs/benchmark_interpretation.md).

All benchmark kernels route every LDS access through the selected
instrumentation path. The `pass-through` rows are the only uninstrumented rows.
Masked attention builds, when available, are triage-only and are not the
headline results in this document.

The comparison rows are:

* `pass-through`: the same kernel shape with no instrumentation.
* `Jakub-Sampled-Loom`: Jakub's sampled publish-only comparison path.
* `context + sampled_watchpoint`: the general diagnostic-capable
  `hip_moi::context` using the sampled-watchpoint backend.
* `sampled_watchpoint_context`: the narrow publish-only fast path.
* `exact shadow`: available only in the tiny matmul wave-scaling benchmark.

Benchmarks are disabled in the default CMake build. Enable them with:

```bash
cmake -S . -B ../hip-moi-build -DHIP_MOI_BUILD_BENCHMARKS=ON
cmake --build ../hip-moi-build --target hip_moi_benchmark_prod_16x8
../hip-moi-build/benchmarks/hip_moi_benchmark_prod_16x8
```

The standalone scripts in this directory compile and run individual benchmark
families directly with `hipcc`. They use `ROCM_ROOT` when provided, otherwise
they try the local TheRock SDK path used during development. They default to
`gfx1201`.

Common timing knobs:

```bash
MIN_MS=500 WARMUP_MS=500 ./benchmarks/build_prod_16x8_benchmark.sh
```

Common sampled instrumentation knobs:

```bash
SAMPLED_WATCHPOINTS=1 SAMPLED_SKIP=32 SAMPLED_PROBES=1 SAMPLED_DELAY=32 ./benchmarks/build_prod_16x8_benchmark.sh
SAMPLED_REPORTS=1 ./benchmarks/build_prod_16x8_benchmark.sh
```

The current headline rows use the fair publish-only defaults printed by the
benchmarks: `watchpoints=1`, `skip=32`, `probes=1`, `delay=32`, and
`reports=off`.

## Codegen Inspection

Several scripts in this directory extract HIP fatbins, unbundle RDNA4 device
objects, and report metadata or instruction counts with LLVM tools. They are
sanity checks for whether a benchmark or test object still has the generated
code shape that the measurement assumes.

The current ping-pong inspection scripts are:

```bash
benchmarks/inspect_pingpong_codegen.sh
EXPECTED_LDS_PRIORITY_SIGNATURE=1010 benchmarks/run_pingpong_att_validation.sh private-pass-through
ATT_SIMD_SELECT=0x2 EXPECTED_LDS_PRIORITY_SIGNATURE=0101 benchmarks/run_pingpong_att_validation.sh private-pass-through
```

`inspect_pingpong_codegen.sh` targets the RDNA4 ping-pong GTest executables.
`run_pingpong_att_validation.sh` builds an optimized RDNA4 ATT probe and checks
ROCprof's decoded per-wave instruction stream for the expected
`s_setprio`/LDS/WMMA ordering and, optionally, the expected LDS-priority
signature. See [`docs/pingpong.md`](../docs/pingpong.md) for the `setprio`,
`sched_barrier`, and ATT-trace caveats.

`019_atomic_flag_handoff_benchmark.hip` is the first atomics handoff benchmark:
global release/acquire atomics order a raw LDS payload. The `context` row
records release-side atomic-object metadata, but this benchmark intentionally
uses a raw LDS payload rather than the diagnostic LDS API.

`020_atomic_metadata_benchmark.hip` isolates the release-metadata cost: each
workgroup release-stores one unique global flag, and the `context` row records
that atomic object in the bounded metadata table.

`021_atomic_happens_before_benchmark.hip` is the first diagnostic-semantics
atomics benchmark. A global release/acquire flag orders an instrumented LDS
payload handoff, and the `context` row uses the acquired-epoch matrix to avoid
reporting the ordered LDS access pair. It also has a
`context-sampled-reporting` row that checks the same synchronization rule
through the sampled-watchpoint reporting backend.

`023_atomic_rmw_happens_before_benchmark.hip` is the first RMW atomics
benchmark. It covers both a release/acquire `fetch_add` arrival counter and a
two-RMW `acq_rel` counter chain before the final subgroup reads LDS payload.

`024_atomic_or_bitmask_happens_before_benchmark.hip` is the first bitmask RMW
benchmark. It uses an `atomicOr` old value to decide whether the second subgroup
reads LDS payload written by the first subgroup.

`025_atomic_fence_happens_before_benchmark.hip` is the first fence-plus-atomic
benchmark. It models a release fence before relaxed atomic publication and an
acquire fence after relaxed atomic observation.

`030_atomic_exchange_compare_exchange_benchmark.hip` covers exchange and
compare-exchange source shapes: release/acquire exchange, successful acquire
CAS after a release unlock, and failed acquire CAS as an acquire load.

`031_atomic_fence_rmw_happens_before_benchmark.hip` extends raw-fence coverage
to relaxed RMWs: release fence before relaxed `fetch_add`, and relaxed
`fetch_add` before acquire fence.

`033_atomic_fence_extended_benchmark.hip` closes the main paired-fence gap for
the currently supported source-level atomics: relaxed exchange, successful
relaxed compare-exchange, failed relaxed compare-exchange, and `seq_cst`
load/store.

`034_atomic_bitwise_happens_before_benchmark.hip` extends source-level bitwise
RMW coverage beyond `atomicOr`. It covers `atomic_fetch_and` bit clearing and
`atomic_fetch_xor` bit toggling, with old-value-dependent control flow deciding
when to read an instrumented LDS payload.

`026_streamk_flag_protocol_benchmark.hip` is the first Stream-K-shaped
integration benchmark. It keeps the RocJITsu hip-stream-k owner/helper flag
protocol but distills the payload to LDS partials so hip-moi can diagnose the
handoff.

`027_streamk_two_tile_flag_protocol_benchmark.hip` is the second Stream-K
integration row. It keeps the flag protocol distilled to LDS partials, but
splits the work into two independent tile fixups, each with its own owner
subgroup and helper subgroup.

`028_rdna4_wmma_streamk_arrival_counter_benchmark.hip` is the first
WMMA-flavored Stream-K integration row. It keeps the diagnostic payload in LDS,
but adds RDNA4 WMMA arithmetic and an arrival-counter-style `fetch_add` before
the final subgroup folds LDS partials.

`036_rdna4_wmma_attention_lds_alias_handoff_benchmark.hip` is the synchronized
benchmark partner for the attention LDS-alias handoff tests. It models the
case where one logical attention tile is consumed from a reusable LDS slot
before the next logical tile is written into the same slot. The benchmark times
the correct barrier-protected variant; the paired tests cover the missing
barrier diagnostic.

## Benchmark Catalog

`Fast VGPRs` refers to the `sampled_watchpoint_context` row. Spill and private
segment notes are included when that fast row is not spill-free.

| Key | Benchmark source | Paired test/reference | Origin | Shape | LDS use | Fast VGPRs |
| --- | --- | --- | --- | --- | --- | --- |
| `matmul-wave-w2` | `w2_2x4_benchmark.hip` | `tests/reference/rdna4_jakub_matmul_reference.hip` | Jakub RDNA4 matmul extraction | 2 waves, 2x4 WMMA tiles, M=32 N=64 K=16 | 3072 B | 48, no spills |
| `matmul-wave-w4` | `w2_2x4_benchmark.hip` | `tests/reference/rdna4_jakub_matmul_reference.hip` | Jakub RDNA4 matmul extraction | 4 waves, 4x16 WMMA tiles, M=64 N=256 K=16 | 10240 B | 122, no spills |
| `matmul-wave-w8` | `w2_2x4_benchmark.hip` | `tests/reference/rdna4_jakub_matmul_reference.hip` | Jakub RDNA4 matmul extraction | 8 waves, 16x8 WMMA tiles, M=256 N=128 K=16 | 12288 B | 98, no spills |
| `matmul-prod-16x8` | `prod_16x8_benchmark.hip` | `tests/reference/rdna4_jakub_matmul_reference.hip` | Jakub production fp16 WMMA row extraction | 8 waves, 16x8 WMMA tiles, KGroup=2, M=N=K=4096 | 24576 B | 256, 16 spills, 68 B private |
| `attention-d16-dense` | `010_rdna4_wmma_attention_block_benchmark.hip` | `tests/instrumented/010_rdna4_wmma_attention_block_test.hip` | hip-moi D16 WMMA attention stress row | seq=12288, head_dim=16, value_dim=16, dense score and weight LDS | 4352 B | 146, no spills |
| `attention-d16-no-score` | `014_rdna4_wmma_no_score_lds_attention_benchmark.hip` | `tests/instrumented/014_rdna4_wmma_no_score_lds_attention_test.hip` | llama.cpp-style register-handoff direction, D16 shape | seq=12288, head_dim=16, value_dim=16, K/V LDS only | 1024 B | 65, no spills |
| `attention-d128-dense` | `011_rdna4_d128_attention_block_benchmark.hip` | `tests/instrumented/011_rdna4_d128_attention_block_test.hip` | AITER/llama.cpp D128 shape signal | seq=8192, q_heads=64, kv_heads=8, gqa=8, head_dim=value_dim=128, dense score and weight LDS | 4352 B | 250, no spills |
| `attention-d128-pressure-full-kv16` | `012_rdna4_d128_attention_pressure_benchmark.hip` | `tests/instrumented/012_rdna4_d128_attention_pressure_test.hip` | llama.cpp-scale LDS pressure candidate | seq=8192, D128/V128, 16-key tile, full K/V double-buffering | 19712 B | 256, 22 spills, 92 B private |
| `attention-d128-pressure-wide-k32` | `012_rdna4_d128_attention_pressure_benchmark.hip` | `tests/instrumented/012_rdna4_d128_attention_pressure_test.hip` | explicit high-LDS pressure variant | seq=8192, D128/V128, 32-key tile, wider double-buffering | 39168 B | 256, 120 spills, 352 B private |
| `attention-d128-no-score` | `015_rdna4_d128_no_score_lds_attention_benchmark.hip` | `tests/instrumented/015_rdna4_d128_no_score_lds_attention_test.hip` | production-faithful register-handoff direction, D128 shape | seq=12288, q_heads=64, kv_heads=8, gqa=8, head_dim=value_dim=128, K/V LDS only | 1024 B | 122, no spills |
| `attention-lds-alias-handoff` | `036_rdna4_wmma_attention_lds_alias_handoff_benchmark.hip` | `tests/instrumented/036_rdna4_wmma_attention_lds_alias_handoff_test.hip` | sanitized production-hazard pattern: reusable attention LDS slot needs a handoff barrier | 256 workgroups, 2 subgroups/workgroup, one 16x16x16 WMMA tile consumes a tail-K fragment before the next logical tile reuses the same LDS slot | 512 B | 27, no spills |
| `pingpong-private-lds` | `016_rdna4_pingpong_att_probe.hip` | `tests/instrumented/016_rdna4_pingpong_private_lds_test.hip`, `run_pingpong_att_validation.sh` | hip-moi RDNA4 ping-pong scheduling probe | 2 waves, 4 K tiles, private A/B LDS double-buffering, alternating `setprio`, WMMA live work | 4096 B | 44, no spills |
| `pingpong-wide-cooperative-lds` | `018_rdna4_pingpong_wide_cooperative_lds_benchmark.hip` | `tests/instrumented/018_rdna4_pingpong_wide_cooperative_lds_test.hip` | hip-moi RDNA4 ping-pong sharing probe | 4 waves, 2 cooperating wave pairs, 4 K tiles, even wave stages shared B fragments, alternating `setprio`, WMMA live work | 6144 B | 48, no spills |
| `atomic-flag-handoff` | `019_atomic_flag_handoff_benchmark.hip` | `tests/instrumented/019_atomic_api_test.hip`, `tests/reference/atomic_reference_kernels.hip` | RocJITsu hip-stream-k release/acquire flag protocol, adapted to LDS payload | 4096 workgroups, 2 subgroups/workgroup, global release/acquire flag orders raw LDS payload | 4 B | n/a (`context` only) |
| `atomic-metadata-release-store` | `020_atomic_metadata_benchmark.hip` | `tests/instrumented/020_atomic_metadata_test.hip` | hip-moi release-metadata microbenchmark | 4096 workgroups, 2 subgroups/workgroup, one unique global release store per workgroup | 0 B | n/a (`context` only) |
| `atomic-hb-lds-handoff` | `021_atomic_happens_before_benchmark.hip` | `tests/instrumented/021_atomic_happens_before_test.hip` | RocJITsu hip-stream-k release/acquire flag protocol, adapted to instrumented LDS payload | 256 workgroups, 2 subgroups/workgroup, release/acquire flag orders instrumented LDS payload | 4 B | n/a (`context` only) |
| `atomic-rmw-arrival-counter` | `023_atomic_rmw_happens_before_benchmark.hip` | `tests/instrumented/023_atomic_rmw_happens_before_test.hip` | Stream-K-style arrival-counter core, distilled into a two-subgroup LDS handoff | 256 workgroups, 2 subgroups/workgroup, release `fetch_add` publishes payload and acquire `fetch_add` consumes it | 8 B | n/a (`context` only) |
| `atomic-rmw-acq-rel-chain` | `023_atomic_rmw_happens_before_benchmark.hip` | `tests/instrumented/023_atomic_rmw_happens_before_test.hip` | Stream-K-style arrival-counter chain stress case | 256 workgroups, 2 subgroups/workgroup, producer and consumer both use `acq_rel fetch_add` | 8 B | n/a (`context` only) |
| `atomic-or-bitmask-handoff` | `024_atomic_or_bitmask_happens_before_benchmark.hip` | `tests/instrumented/024_atomic_or_bitmask_happens_before_test.hip` | Stream-K-tree-style sibling bitmask core | 256 workgroups, 2 subgroups/workgroup, release `atomicOr` publishes one bit and `acq_rel atomicOr` consumes the old mask | 8 B | n/a (`context` only) |
| `atomic-fence-handoff` | `025_atomic_fence_happens_before_benchmark.hip` | `tests/instrumented/025_atomic_fence_happens_before_test.hip` | Standard fence-plus-atomic handoff | 256 workgroups, 2 subgroups/workgroup, release fence before relaxed flag store and acquire fence after relaxed flag load | 4 B | n/a (`context` only) |
| `atomic-exchange-handoff` | `030_atomic_exchange_compare_exchange_benchmark.hip` | `tests/instrumented/030_atomic_exchange_compare_exchange_test.hip` | Exchange-style token handoff | 256 workgroups, 2 subgroups/workgroup, release exchange publishes payload and acquire exchange consumes it | 4 B | n/a (`context` only) |
| `atomic-cas-lock-handoff` | `030_atomic_exchange_compare_exchange_benchmark.hip` | `tests/instrumented/030_atomic_exchange_compare_exchange_test.hip` | Lock-like compare-exchange handoff | 256 workgroups, 2 subgroups/workgroup, release unlock store followed by successful acquire CAS lock acquisition | 4 B | n/a (`context` only) |
| `atomic-failed-cas-acquire` | `030_atomic_exchange_compare_exchange_benchmark.hip` | `tests/instrumented/030_atomic_exchange_compare_exchange_test.hip` | Failed compare-exchange acquire-load shape | 256 workgroups, 2 subgroups/workgroup, release flag store followed by failed acquire CAS that observes the published value | 4 B | n/a (`context` only) |
| `atomic-fence-rmw-handoff` | `031_atomic_fence_rmw_happens_before_benchmark.hip` | `tests/instrumented/031_atomic_fence_rmw_happens_before_test.hip` | Raw-fence-plus-RMW handoff | 256 workgroups, 2 subgroups/workgroup, release fence before relaxed `fetch_add` and relaxed `fetch_add` before acquire fence | 8 B | n/a (`context` only) |
| `atomic-fence-exchange` | `033_atomic_fence_extended_benchmark.hip` | `tests/instrumented/033_atomic_fence_extended_test.hip` | Raw-fence-plus-exchange handoff | 256 workgroups, 2 subgroups/workgroup, release fence before relaxed exchange and relaxed exchange before acquire fence | 4 B | n/a (`context` only) |
| `atomic-fence-successful-cas` | `033_atomic_fence_extended_benchmark.hip` | `tests/instrumented/033_atomic_fence_extended_test.hip` | Raw-fence-plus-successful-CAS handoff | 256 workgroups, 2 subgroups/workgroup, release fence before relaxed unlock store and successful relaxed CAS before acquire fence | 4 B | n/a (`context` only) |
| `atomic-fence-failed-cas` | `033_atomic_fence_extended_benchmark.hip` | `tests/instrumented/033_atomic_fence_extended_test.hip` | Raw-fence-plus-failed-CAS handoff | 256 workgroups, 2 subgroups/workgroup, release fence before relaxed flag store and failed relaxed CAS before acquire fence | 4 B | n/a (`context` only) |
| `atomic-seq-cst-handoff` | `033_atomic_fence_extended_benchmark.hip` | `tests/instrumented/033_atomic_fence_extended_test.hip` | Strong load/store sanity row | 256 workgroups, 2 subgroups/workgroup, `seq_cst` store/load orders an instrumented LDS payload | 4 B | n/a (`context` only) |
| `atomic-and-bitmask-handoff` | `034_atomic_bitwise_happens_before_benchmark.hip` | `tests/instrumented/034_atomic_bitwise_happens_before_test.hip` | Bitmask RMW clearing shape, modeled after Stream-K-tree-style old-value protocols | 256 workgroups, 2 subgroups/workgroup, release `fetch_and` clears a producer bit and acquire `fetch_and` observes the old mask before reading payload | 4 B | n/a (`context` only) |
| `atomic-xor-bitmask-handoff` | `034_atomic_bitwise_happens_before_benchmark.hip` | `tests/instrumented/034_atomic_bitwise_happens_before_test.hip` | Bitmask RMW toggling shape, modeled after Stream-K-tree-style old-value protocols | 256 workgroups, 2 subgroups/workgroup, release `fetch_xor` toggles a producer bit and acquire `fetch_xor` observes the old mask before reading payload | 4 B | n/a (`context` only) |
| `streamk-flag-fixup` | `026_streamk_flag_protocol_benchmark.hip` | `tests/instrumented/026_streamk_flag_protocol_test.hip`, `tests/reference/atomic_reference_kernels.hip` | RocJITsu hip-stream-k owner/helper flag protocol, distilled to LDS partial payloads | 256 workgroups, 3 subgroups/workgroup, one owner loops over two helper release/acquire flags and folds helper partials | 12 B | n/a (`context` only) |
| `streamk-two-tile-flag-fixup` | `027_streamk_two_tile_flag_protocol_benchmark.hip` | `tests/instrumented/027_streamk_two_tile_flag_protocol_test.hip`, `tests/reference/atomic_reference_kernels.hip` | RocJITsu hip-stream-k two-tile ownership shape, distilled to LDS partial payloads | 256 workgroups, 4 subgroups/workgroup, two independent owner/helper tile fixups with one release/acquire flag per tile | 16 B | n/a (`context` only) |
| `rdna4-wmma-streamk-arrival-counter` | `028_rdna4_wmma_streamk_arrival_counter_benchmark.hip` | `tests/instrumented/028_rdna4_wmma_streamk_arrival_counter_test.hip` | `hip-matmul/matmul_rdna4.hip` Stream-K arrival-counter idea, localized to LDS payload diagnostics | 256 workgroups, 2 subgroups/workgroup, two K-slice RDNA4 WMMA partials, `acq_rel fetch_add` arrival counter, final subgroup folds LDS partials | 4096 B | n/a (`context` only) |
| `rdna4-wmma-streamk-tree-atomic-or` | `029_rdna4_wmma_streamk_tree_atomic_or_benchmark.hip` | `tests/instrumented/029_rdna4_wmma_streamk_tree_atomic_or_test.hip` | `hip-matmul/matmul_rdna4.hip` Stream-K-tree `atomicOr` idea, localized to LDS payload diagnostics | 256 workgroups, 4 subgroups/workgroup, four K-slice RDNA4 WMMA partials, first three subgroups publish bits with release `atomicOr`, final subgroup folds LDS partials after `acq_rel atomicOr` | 8192 B | n/a (`context` only) |

## Shapes and Resource Pressure

This table describes the uninstrumented `pass-through` kernel shape. LDS
percentages assume the local RDNA4 test device's 64 KiB workgroup LDS limit.
VGPR counts come from the bundled RDNA4 code-object metadata for the
`pass-through` row.

| Key | Shape | LDS pressure | Pass-through VGPR pressure |
| --- | --- | ---: | ---: |
| `matmul-wave-w2` | 2 waves, 2x4 WMMA tiles, M=32 N=64 K=16 | 3072 B, 4.7% | 42 |
| `matmul-wave-w4` | 4 waves, 4x16 WMMA tiles, M=64 N=256 K=16 | 10240 B, 15.6% | 114 |
| `matmul-wave-w8` | 8 waves, 16x8 WMMA tiles, M=256 N=128 K=16 | 12288 B, 18.8% | 90 |
| `matmul-prod-16x8` | 8 waves, 16x8 WMMA tiles, KGroup=2, M=N=K=4096 | 24576 B, 37.5% | 225 |
| `attention-d16-dense` | seq=12288, head_dim=16, value_dim=16, dense score and weight LDS | 4352 B, 6.6% | 82 |
| `attention-d16-no-score` | seq=12288, head_dim=16, value_dim=16, K/V LDS only | 1024 B, 1.6% | 50 |
| `attention-d128-dense` | seq=8192, q_heads=64, kv_heads=8, gqa=8, head_dim=value_dim=128, dense score and weight LDS | 4352 B, 6.6% | 218 |
| `attention-d128-pressure-full-kv16` | seq=8192, D128/V128, 16-key tile, full K/V double-buffering | 19712 B, 30.1% | 232 |
| `attention-d128-pressure-wide-k32` | seq=8192, D128/V128, 32-key tile, wider double-buffering | 39168 B, 59.8% | 227 |
| `attention-d128-no-score` | seq=12288, q_heads=64, kv_heads=8, gqa=8, head_dim=value_dim=128, K/V LDS only | 1024 B, 1.6% | 178 |
| `attention-lds-alias-handoff` | 256 workgroups, 2 subgroups/workgroup, one 16x16x16 WMMA tile consumes a tail-K fragment before the next logical tile reuses the same LDS slot | 512 B, 0.8% | 19 |
| `pingpong-private-lds` | 2 waves, 4 K tiles, private A/B LDS double-buffering, alternating `setprio`, WMMA live work | 4096 B, 6.3% | 23 |
| `pingpong-wide-cooperative-lds` | 4 waves, 2 cooperating wave pairs, 4 K tiles, even wave stages shared B fragments, alternating `setprio`, WMMA live work | 6144 B, 9.4% | 24 |
| `atomic-flag-handoff` | 4096 workgroups, 2 subgroups/workgroup, global release/acquire flag orders raw LDS payload | 4 B, <0.1% | 3 |
| `atomic-metadata-release-store` | 4096 workgroups, 2 subgroups/workgroup, one unique global release store per workgroup | 0 B, 0.0% | 2 |
| `atomic-hb-lds-handoff` | 256 workgroups, 2 subgroups/workgroup, release/acquire flag orders instrumented LDS payload | 4 B, <0.1% | 3 |
| `atomic-rmw-arrival-counter` | 256 workgroups, 2 subgroups/workgroup, release/acquire `fetch_add` arrival counter orders instrumented LDS payload | 8 B, <0.1% | 4 |
| `atomic-rmw-acq-rel-chain` | 256 workgroups, 2 subgroups/workgroup, two-RMW `acq_rel fetch_add` chain orders instrumented LDS payload | 8 B, <0.1% | 4 |
| `atomic-or-bitmask-handoff` | 256 workgroups, 2 subgroups/workgroup, old-value-dependent `atomicOr` bitmask orders instrumented LDS payload | 8 B, <0.1% | 4 |
| `atomic-fence-handoff` | 256 workgroups, 2 subgroups/workgroup, release/acquire fences pair relaxed atomic flag operations around instrumented LDS payload | 4 B, <0.1% | 3 |
| `atomic-exchange-handoff` | 256 workgroups, 2 subgroups/workgroup, release/acquire exchange orders instrumented LDS payload | 4 B, <0.1% | 4 |
| `atomic-cas-lock-handoff` | 256 workgroups, 2 subgroups/workgroup, release unlock store and successful acquire CAS lock acquisition order instrumented LDS payload | 4 B, <0.1% | 3 |
| `atomic-failed-cas-acquire` | 256 workgroups, 2 subgroups/workgroup, release store and failed acquire CAS order instrumented LDS payload | 4 B, <0.1% | 4 |
| `atomic-fence-rmw-handoff` | 256 workgroups, 2 subgroups/workgroup, release/acquire fences pair relaxed RMW operations around instrumented LDS payload | 8 B, <0.1% | 4 |
| `atomic-fence-exchange` | 256 workgroups, 2 subgroups/workgroup, release/acquire fences pair relaxed exchange operations around instrumented LDS payload | 4 B, <0.1% | 3 |
| `atomic-fence-successful-cas` | 256 workgroups, 2 subgroups/workgroup, release/acquire fences pair a relaxed unlock store with successful relaxed CAS around instrumented LDS payload | 4 B, <0.1% | 3 |
| `atomic-fence-failed-cas` | 256 workgroups, 2 subgroups/workgroup, release/acquire fences pair a relaxed flag store with failed relaxed CAS around instrumented LDS payload | 4 B, <0.1% | 3 |
| `atomic-seq-cst-handoff` | 256 workgroups, 2 subgroups/workgroup, `seq_cst` store/load orders instrumented LDS payload | 4 B, <0.1% | 3 |
| `atomic-and-bitmask-handoff` | 256 workgroups, 2 subgroups/workgroup, old-value-dependent `fetch_and` bit clearing orders instrumented LDS payload | 4 B, <0.1% | 4 |
| `atomic-xor-bitmask-handoff` | 256 workgroups, 2 subgroups/workgroup, old-value-dependent `fetch_xor` bit toggling orders instrumented LDS payload | 4 B, <0.1% | 4 |
| `streamk-flag-fixup` | 256 workgroups, 3 subgroups/workgroup, one owner loops over two helper flags and folds two LDS helper partials | 12 B, <0.1% | 4 |
| `streamk-two-tile-flag-fixup` | 256 workgroups, 4 subgroups/workgroup, two independent owner/helper tile fixups with one release/acquire flag per tile | 16 B, <0.1% | 3 |
| `rdna4-wmma-streamk-arrival-counter` | 256 workgroups, 2 subgroups/workgroup, two K-slice RDNA4 WMMA partials, arrival counter, final subgroup folds LDS partials | 4096 B, 6.3% | 21 |
| `rdna4-wmma-streamk-tree-atomic-or` | 256 workgroups, 4 subgroups/workgroup, four K-slice RDNA4 WMMA partials, `atomicOr` bitmask tree, final subgroup folds LDS partials | 8192 B, 12.5% | 36 |

## Benchmark Modes

The result table uses compact column names so the rows stay readable. This
table expands those benchmark modes.

| Results column | Benchmark mode | What it measures |
| --- | --- | --- |
| `pass-through` | Uninstrumented kernel | Baseline kernel latency with the same workload shape and no hip-moi or Jakub-Sampled-Loom instrumentation. |
| `Jakub-Sampled-Loom` | Jakub-Sampled-Loom | Publish-only sampled instrumentation modeled after Jakub's Loom-flavored HIP prototype. |
| `exact shadow` | hip-moi exact shadow | Precise shadow-memory checking through explicit LDS-offset APIs. Present only in the tiny matmul wave-scaling benchmark. |
| `context + sampled_watchpoint` | General hip-moi context with sampled-watchpoint backend | Diagnostic-capable hip-moi API path using sampled watchpoints. This keeps more state live than the publish-only fast path. |
| `sampled_watchpoint_context` | hip-moi sampled publish-only fast path | Narrow fast-view context optimized for Loom-parity publish-only sampling. This is the main performance target. |
| `context` | General hip-moi context | Used by atomics benchmarks when the feature is not part of sampled watchpoint instrumentation. |
| `context-sampled-reporting` | General hip-moi context with sampled-watchpoint reporting | Atomics correctness coverage for sampled diagnostics. This is not the publish-only fast path; current rows may scan many watchpoint entries. |

## Current RDNA4 Results

Matmul and attention rows were measured on 2026-06-24 on device 0, AMD Radeon
RX 9070, `gfx1201`, 28 CUs. The ping-pong rows were measured on the same machine
on 2026-06-25. Atomics and the attention LDS-alias handoff row were last
refreshed on 2026-06-26. Latencies below 1 ms are printed in microseconds
(`µs`); larger latencies are printed in milliseconds. Most rows use
`MIN_MS=100` and `WARMUP_MS=100`.
`attention-d16-dense` uses `MIN_MS=500` and `WARMUP_MS=500` because full dense
score/weight instrumentation makes it much slower than the matmul rows.

| Key | pass-through | Jakub-Sampled-Loom | exact shadow | `context + sampled_watchpoint` | `sampled_watchpoint_context` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `matmul-wave-w2` | 2.80 µs | 4.69 µs | 8.91 µs | 4.85 µs | 3.49 µs |
| `matmul-wave-w4` | 3.15 µs | 5.91 µs | 14.0 µs | 7.45 µs | 4.31 µs |
| `matmul-wave-w8` | 3.20 µs | 5.79 µs | 13.0 µs | 7.74 µs | 4.62 µs |
| `matmul-prod-16x8` | 1.16 ms | 8.70 ms | n/a | 26.2 ms | 3.35 ms |
| `attention-d16-dense` | 6.56 ms | 118 ms | n/a | 148 ms | 64.0 ms |
| `attention-d16-no-score` | 1.06 ms | 1.89 ms | n/a | 2.41 ms | 1.49 ms |
| `attention-d128-dense` | 4.25 ms | 102 ms | n/a | 105 ms | 38.8 ms |
| `attention-d128-pressure-full-kv16` | 5.87 ms | 156 ms | n/a | 132 ms | 48.4 ms |
| `attention-d128-pressure-wide-k32` | 8.87 ms | 183 ms | n/a | 162 ms | 75.5 ms |
| `attention-d128-no-score` | 3.44 ms | 10.9 ms | n/a | 22.0 ms | 7.29 ms |
| `attention-lds-alias-handoff` | 3.16 µs | n/a | n/a | 6.74 µs | 4.64 µs |
| `pingpong-private-lds` | 3.94 µs | n/a | n/a | 9.10 µs | 6.64 µs |
| `pingpong-wide-cooperative-lds` | 5.41 µs | n/a | n/a | 18.9 µs | 9.67 µs |

The atomics rows use a separate result table because most atomics coverage is
implemented only for `hip_moi::context`. The
`context-sampled-reporting` column is present only where sampled diagnostics
are intentionally part of the semantic coverage.

| Key | pass-through | `context` | `context-sampled-reporting` |
| --- | ---: | ---: | ---: |
| `atomic-flag-handoff` | 8.07 µs | 43.2 µs | n/a |
| `atomic-metadata-release-store` | 3.43 µs | 18.3 µs | n/a |
| `atomic-hb-lds-handoff` | 3.11 µs | 8.96 µs | 76.3 µs |
| `atomic-rmw-arrival-counter` | 3.18 µs | 8.13 µs | n/a |
| `atomic-rmw-acq-rel-chain` | 3.21 µs | 8.92 µs | n/a |
| `atomic-or-bitmask-handoff` | 3.10 µs | 7.96 µs | n/a |
| `atomic-fence-handoff` | 3.01 µs | 6.96 µs | n/a |
| `atomic-exchange-handoff` | 3.07 µs | 8.19 µs | n/a |
| `atomic-cas-lock-handoff` | 2.98 µs | 7.01 µs | n/a |
| `atomic-failed-cas-acquire` | 3.06 µs | 6.86 µs | n/a |
| `atomic-fence-rmw-handoff` | 3.09 µs | 8.45 µs | n/a |
| `atomic-fence-exchange` | 3.08 µs | 8.61 µs | n/a |
| `atomic-fence-successful-cas` | 3.00 µs | 6.69 µs | n/a |
| `atomic-fence-failed-cas` | 3.06 µs | 6.87 µs | n/a |
| `atomic-seq-cst-handoff` | 3.09 µs | 8.84 µs | n/a |
| `atomic-and-bitmask-handoff` | 2.89 µs | 8.94 µs | n/a |
| `atomic-xor-bitmask-handoff` | 3.02 µs | 8.91 µs | n/a |
| `streamk-flag-fixup` | 3.22 µs | 12.8 µs | n/a |
| `streamk-two-tile-flag-fixup` | 3.07 µs | 12.4 µs | n/a |
| `rdna4-wmma-streamk-arrival-counter` | 3.40 µs | 25.5 µs | n/a |
| `rdna4-wmma-streamk-tree-atomic-or` | 3.66 µs | 45.2 µs | n/a |

## Reading The Suite

The matmul wave rows are the small-shape guardrail: they expose overhead at
tiny dynamic sizes and make wave-count scaling obvious. `matmul-prod-16x8` is
the main matmul performance gate because it preserves Jakub's production shape
and already runs close to the VGPR ceiling.

The dense attention rows intentionally materialize score and softmax-weight
tiles in LDS. They are valuable scalar-LDS stress tests, but source mining now
suggests that mature attention implementations try to avoid this dense LDS
handoff. The no-score rows therefore represent the current production-faithful
attention direction: K/V fragments are staged through LDS, while the QK-to-PV
handoff stays in registers. The pressure rows remain useful because they push
LDS and VGPR usage into the regime Jakub expects to be interesting for
instrumentation overhead.

The attention LDS-alias handoff row is a correctness-driven production-hazard
extraction rather than a high-pressure throughput row. Its paired tests include
the missing-barrier diagnostic; the benchmark times only the synchronized
variant because that is the meaningful performance shape. The current fast row
uses 27 VGPRs, no spills, and 512 B LDS.

The general `context + sampled_watchpoint` row tracks the diagnostic-capable API
path. The publish-only performance comparison row is
`sampled_watchpoint_context`.

The ping-pong rows are scheduling-sensitive guardrails. They are not intended
to model high LDS or VGPR pressure. The private row uses the same private-LDS
`setprio`/`sched_barrier`/WMMA kernel shape that the ATT validation script
checks for complementary per-SIMD priority signatures. The wide cooperative row
keeps the same scheduling idiom but adds real cross-subgroup LDS sharing: in
each pair, the even subgroup stages B fragments and both subgroups consume
them.

The atomics rows now use address-scoped release metadata. A release records the
atomic address, producer subgroup, epoch, source site, and launch generation.
An acquire imports producer records for that address; the scalar value stored,
loaded, or returned by a RMW is not part of the current metadata key.
Address+value remains a possible future precision refinement, but it is not the
default implementation.

The first general-context atomics optimization is intentionally small: one- and
two-subgroup acquire imports skip the direct-mapped RMW producer-mask cache
probe, because that cache is populated only for workgroups with more than two
subgroups. This keeps synchronization metadata exhaustive. The main benefit is
cleaner generated code and lower SGPR pressure in the two-subgroup atomics
rows; VGPR pressure stays unchanged and all refreshed atomics rows remain
spill-free.

A later atomics performance audit tried two smaller local shortcuts and
rejected both. First, avoiding the acquired-token `atomicMax` when a volatile
read already showed a new-enough token caused false diagnostics in the
four-subgroup `atomicOr` tree. The acquired-token write is therefore a required
publication step, not just a redundant metadata update. Second, a special
two-subgroup direct-producer lookup badly regressed the
`atomic-flag-handoff` shared-context microbenchmark. The current generic
two-subgroup path remains the stable implementation.

The atomic flag handoff row records release-side atomic-object metadata for a
raw LDS payload. The metadata release-store row isolates the release-side
table-recording cost. The happens-before LDS handoff row is the first row where
atomic metadata affects LDS diagnostics: a release/acquire global flag orders
an instrumented LDS payload handoff, while the deliberately relaxed variant
still diagnoses. These rows show that the current address-scoped publication
and acquire paths are correct enough for the staged model, but not yet cheap;
`atomic-flag-handoff_context` is 43.2 µs after address-scoped acquire
imports.

The `atomic-hb-lds-handoff_context-sampled-reporting` row checks that sampled
diagnostics consult the same acquired-epoch tokens as exact shadow. It uses
`probe_count=0`, which scans the whole watchpoint table, so the 76.3 µs result
is semantic coverage rather than a candidate fast path.

The atomic RMW rows cover `fetch_add` arrival counters, a two-RMW `acq_rel`
chain, and old-value-dependent bitmask control flow through `atomicOr`,
`fetch_and`, and `fetch_xor`. The old value returned by the RMW still drives
user control flow, but hip-moi no longer uses that value as synchronization
metadata. The current two-subgroup RMW latencies remain in the 8 to 9 µs range
through `context`.

The atomic fence handoff row supports the standard shape where a release fence
is sequenced before a relaxed atomic store, and an acquire fence is sequenced
after a relaxed atomic load that observed that store. The first implementation
keeps fence state in the per-thread device context object: release fences arm
the next relaxed atomic publication, and acquire fences consume the last
relaxed atomic observation made through that context.

The exchange and compare-exchange rows broaden source-level atomics coverage
without changing the address-scoped metadata model. Exchange is treated as a
RMW: release-capable exchange publishes the producer epoch, and
acquire-capable exchange imports producer epochs for the same atomic address.
Compare-exchange is outcome-sensitive. A successful compare-exchange follows
the success order and can publish/acquire; a failed compare-exchange does not
modify the atomic object, so it cannot publish, but an acquire-capable failure
order is modeled as an acquire load. The failed-CAS row exists specifically to
keep that subtle case exercised.

The fence/RMW row extends raw-fence support from relaxed store/load flags to
relaxed RMW counters. A release fence arms the next relaxed RMW publication,
and an acquire fence consumes the most recent relaxed RMW observation. Naked
fences are still not modeled as inter-subgroup synchronization.

The extended fence rows apply that same paired-fence rule to the remaining
source-level RMW forms currently implemented by hip-moi. Relaxed exchange and
successful relaxed compare-exchange can both publish a pending release fence
because they modify the atomic object. Failed relaxed compare-exchange cannot
publish, but it can be the acquire-side observation consumed by a following
acquire fence. The `atomic-seq-cst-handoff` row is a sanity check for the
strongest ordinary load/store spelling, not a new synchronization mechanism.

The atomics fast path is intentionally narrow: release-capable RMWs in
workgroups with more than two subgroups populate a direct-mapped
address-scoped producer-mask cache. Acquires use that cache to skip generic
per-producer probes when the cache slot matches, and fall back to the generic
atomic-object table on misses or collisions. Release stores, fence-published
relaxed stores, and two-subgroup RMWs stay on the generic path because the
cache did not pay for itself on those shapes.

The Stream-K flag fixup rows are integration rows rather than one-edge
microbenchmarks. They preserve RocJITsu hip-stream-k owner/helper flag
protocols but distill the payload to LDS partials so hip-moi can diagnose the
handoff. The one-owner/two-helper row is now 12.8 µs through `context`; the
two-tile ownership row is now 12.4 µs through `context`.

The RDNA4 WMMA Stream-K arrival-counter row is the first atomics integration
row with WMMA arithmetic. It is not a direct global-partial Stream-K GEMM: the
diagnostic payload is intentionally kept in LDS so hip-moi can test whether an
arrival-counter synchronization edge orders the final fold of the partials.
The current row is 25.5 µs through `context`. An earlier single-lane reduction
draft was rejected because it produced an unrepresentative 209 µs `context`
latency by making one lane perform every instrumented partial load. The
committed row uses lane-parallel reduction and is the better signal for the
next fast-path decision.

The RDNA4 WMMA Stream-K-tree `atomicOr` row is the first WMMA-shaped
bitmask-tree case. Four subgroups compute WMMA
partials; the first three subgroups publish bits with release `atomicOr`; the
final subgroup waits for those bits, performs an `acq_rel atomicOr`, and folds
all four LDS partials. The direct RMW cache lowers this row from the
previous 49.2 µs `context` result to 45.2 µs, at the cost of higher SGPR
pressure and no change in VGPRs or spills.

The current atomics `context` resource refresh found no spills:

| Key | Context LDS | Context SGPRs | Context VGPRs | Spills/private |
| --- | ---: | ---: | ---: | --- |
| `streamk-flag-fixup` | 12 B | 84 | 26 | none, 0 B |
| `streamk-two-tile-flag-fixup` | 16 B | 94 | 61 | none, 0 B |
| `rdna4-wmma-streamk-arrival-counter` | 4096 B | 75 | 51 | none, 0 B |
| `rdna4-wmma-streamk-tree-atomic-or` | 8192 B | 84 | 52 | none, 0 B |
| `atomic-hb-lds-handoff` | 4 B | 56 | 23 | none, 0 B |
| `atomic-rmw-arrival-counter` | 8 B | 58 | 23 | none, 0 B |
| `atomic-rmw-acq-rel-chain` | 8 B | 57 | 23 | none, 0 B |
| `atomic-exchange-handoff` | 4 B | 57 | 23 | none, 0 B |
| `atomic-cas-lock-handoff` | 4 B | 57 | 23 | none, 0 B |
| `atomic-failed-cas-acquire` | 4 B | 56 | 23 | none, 0 B |
| `atomic-fence-handoff` | 4 B | 57 | 23 | none, 0 B |
| `atomic-fence-rmw-handoff` | 8 B | 57 | 23 | none, 0 B |
| `atomic-fence-exchange` | 4 B | 61 | 21 | none, 0 B |
| `atomic-fence-successful-cas` | 4 B | 62 | 21 | none, 0 B |
| `atomic-fence-failed-cas` | 4 B | 62 | 21 | none, 0 B |
| `atomic-seq-cst-handoff` | 4 B | 60 | 21 | none, 0 B |
| `atomic-and-bitmask-handoff` | 4 B | 59 | 23 | none, 0 B |
| `atomic-xor-bitmask-handoff` | 4 B | 59 | 23 | none, 0 B |
