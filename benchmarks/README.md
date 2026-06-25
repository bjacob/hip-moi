# hip-moi Benchmarks

This directory contains the RDNA4 performance benchmarks used to guide
hip-moi's sampled publish-only implementation. The matmul benchmarks are
focused extractions from Jakub's
`sanitizer-strategy/rdna4_matmul/rdna4_matmul.hip`; the attention benchmarks are
hip-moi-native workloads grown from the instrumented test corpus and from the
llama.cpp/AITER shape signals recorded in
[`attention_source_mining.md`](attention_source_mining.md).

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
global release/acquire atomics order a raw LDS payload. Starting with atomics
Stage 3, the `context` row records release-side atomic-object metadata, but
hip-moi does not yet use that metadata to suppress LDS diagnostics.

`020_atomic_metadata_benchmark.hip` isolates the Stage 3 metadata cost: each
workgroup release-stores one unique global flag, and the `context` row records
that atomic object in the bounded metadata table.

`021_atomic_happens_before_benchmark.hip` is the first diagnostic-semantics
atomics benchmark. A global release/acquire flag orders an instrumented LDS
payload handoff, and the `context` row uses the acquired-epoch matrix to avoid
reporting the ordered LDS access pair.

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
| `pingpong-private-lds` | `016_rdna4_pingpong_att_probe.hip` | `tests/instrumented/016_rdna4_pingpong_private_lds_test.hip`, `run_pingpong_att_validation.sh` | hip-moi RDNA4 ping-pong scheduling probe | 2 waves, 4 K tiles, private A/B LDS double-buffering, alternating `setprio`, WMMA live work | 4096 B | 44, no spills |
| `pingpong-wide-cooperative-lds` | `018_rdna4_pingpong_wide_cooperative_lds_benchmark.hip` | `tests/instrumented/018_rdna4_pingpong_wide_cooperative_lds_test.hip` | hip-moi RDNA4 ping-pong sharing probe | 4 waves, 2 cooperating wave pairs, 4 K tiles, even wave stages shared B fragments, alternating `setprio`, WMMA live work | 6144 B | 48, no spills |
| `atomic-flag-handoff` | `019_atomic_flag_handoff_benchmark.hip` | `tests/instrumented/019_atomic_api_test.hip`, `tests/reference/atomic_reference_kernels.hip` | RocJITsu hip-stream-k release/acquire flag protocol, adapted to LDS payload | 4096 workgroups, 2 subgroups/workgroup, global release/acquire flag orders raw LDS payload | 4 B | n/a; `context` row is 24 VGPRs, no spills |
| `atomic-metadata-release-store` | `020_atomic_metadata_benchmark.hip` | `tests/instrumented/020_atomic_metadata_test.hip` | hip-moi Stage 3 metadata microbenchmark | 4096 workgroups, 2 subgroups/workgroup, one unique global release store per workgroup | 0 B | n/a; `context` row is 23 VGPRs, no spills |
| `atomic-hb-lds-handoff` | `021_atomic_happens_before_benchmark.hip` | `tests/instrumented/021_atomic_happens_before_test.hip` | RocJITsu hip-stream-k release/acquire flag protocol, adapted to instrumented LDS payload | 256 workgroups, 2 subgroups/workgroup, release/acquire flag orders instrumented LDS payload | 4 B | n/a; `context` row is 21 VGPRs, no spills |

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
| `pingpong-private-lds` | 2 waves, 4 K tiles, private A/B LDS double-buffering, alternating `setprio`, WMMA live work | 4096 B, 6.3% | 23 |
| `pingpong-wide-cooperative-lds` | 4 waves, 2 cooperating wave pairs, 4 K tiles, even wave stages shared B fragments, alternating `setprio`, WMMA live work | 6144 B, 9.4% | 24 |
| `atomic-flag-handoff` | 4096 workgroups, 2 subgroups/workgroup, global release/acquire flag orders raw LDS payload | 4 B, <0.1% | 3 |
| `atomic-metadata-release-store` | 4096 workgroups, 2 subgroups/workgroup, one unique global release store per workgroup | 0 B, 0.0% | 2 |
| `atomic-hb-lds-handoff` | 256 workgroups, 2 subgroups/workgroup, release/acquire flag orders instrumented LDS payload | 4 B, <0.1% | 3 |

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
| `context` | General hip-moi context | Used by atomics benchmarks when the feature is not part of sampled watchpoint instrumentation. Stage 2 atomics use this only as a pass-through wrapper around HIP/Clang atomics. |

## Current RDNA4 Results

Matmul and attention rows were measured on 2026-06-24 on device 0, AMD Radeon
RX 9070, `gfx1201`, 28 CUs. The ping-pong rows were measured on the same machine
on 2026-06-25. Latencies below 1 ms are printed in microseconds (`µs`); larger
latencies are printed in milliseconds. Most rows use `MIN_MS=100` and
`WARMUP_MS=100`. `attention-d16-dense` uses `MIN_MS=500` and `WARMUP_MS=500`
because full dense score/weight instrumentation makes it much slower than the
matmul rows.

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
| `pingpong-private-lds` | 3.94 µs | n/a | n/a | 9.10 µs | 6.64 µs |
| `pingpong-wide-cooperative-lds` | 5.41 µs | n/a | n/a | 18.9 µs | 9.67 µs |

The atomics rows use a separate result table because the first atomics API is
implemented only for `hip_moi::context`, not for sampled watchpoint modes.

| Key | pass-through | `context` |
| --- | ---: | ---: |
| `atomic-flag-handoff` | 7.32 µs | 30.6 µs |
| `atomic-metadata-release-store` | 3.43 µs | 13.4 µs |
| `atomic-hb-lds-handoff` | 3.34 µs | 13.6 µs |

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

The atomic flag handoff row is no longer a pure API-wrapper guardrail: the
`context` kernel records release-side atomic-object metadata. The pass-through
kernel reports 4 B LDS, 3 VGPRs, 10 SGPRs, and no spills. The `context` kernel
reports 4 B LDS, 24 VGPRs, 42 SGPRs, no scratch/private segment, and no spills.
It is still not a diagnostic benchmark because Stage 3 does not query this
metadata from the LDS conflict check.

The atomic metadata release-store row isolates the table-recording cost. The
pass-through kernel reports 0 B LDS, 2 VGPRs, 5 SGPRs, and no spills. The
`context` kernel reports 0 B LDS, 23 VGPRs, 34 SGPRs, no scratch/private
segment, and no spills. The disassembly shows the expected bounded metadata
probe/claim path, including global loads and a 64-bit compare-and-swap for
claiming stale slots. The first implementation uses address hashing before
linear probing; a previous all-probes-from-zero prototype was rejected because
it made table fill effectively quadratic. The current Stage 5 implementation
derives power-of-two atomic-object capacities and maps the hash with a mask,
avoiding runtime division in the probe-start calculation.

The atomic happens-before LDS handoff row is the first row where atomic
metadata affects LDS diagnostics. The pass-through kernel reports 4 B LDS, 3
VGPRs, 10 SGPRs, and no spills. The `context` kernel reports 4 B LDS, 21 VGPRs,
55 SGPRs, no scratch/private segment, and no spills. This benchmark uses a
fresh generation kernel argument for each measured launch so the exact-shadow
entries and atomic-object records remain launch-separated without timing
host-side metadata copies.
