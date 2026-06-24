# hip-moi Benchmarks

These RDNA4 benchmarks are the performance-facing companions to the correctness
tests. The matmul benchmarks are focused extractions from Jakub's
`sanitizer-strategy/rdna4_matmul/rdna4_matmul.hip`; the attention benchmarks
are hip-moi-native workloads grown from the instrumented attention tests.

All current benchmarks focus on subgroup-level, full-workgroup-barrier LDS
instrumentation. They do not exercise atomics or finer-grained synchronization.
Instrumented benchmark rows are expected to route every LDS access in the
benchmarked kernel through the selected instrumentation path; the `noop` rows
are the only uninstrumented rows.

The common comparison rows are:

* `noop`: same kernel shape with no instrumentation,
* `sampled Loom`: Jakub-style sampled-Loom publish-only instrumentation,
* `context + sampled_watchpoint`: full diagnostic-capable `hip_moi::context`
  using the `sampled_watchpoint` backend,
* `sampled_watchpoint_context`: narrow publish-only fast-view context.

The RDNA4 WMMA matmul wave-scaling benchmark also includes
`hip-moi exact shadow` as a correctness/performance reference row.

## Build And Run

The scripts use `ROCM_ROOT` when provided, otherwise they try the local TheRock
SDK path used during development. They default to `gfx1201`.

### RDNA4 WMMA Matmul Wave-Scaling Benchmarks

Use these for quick intra-session experiments and wave-count scaling checks.
All three shapes are built from `benchmarks/w2_2x4_benchmark.hip`.

```bash
./benchmarks/build_w2_2x4_benchmark.sh
./benchmarks/build_w2_2x4_benchmark.sh w4_4x16
./benchmarks/build_w2_2x4_benchmark.sh w8_16x8
```

### RDNA4 WMMA Matmul Production 16x8 Benchmark

Use this as the main matmul-focused performance gate.

```bash
./benchmarks/build_prod_16x8_benchmark.sh
BENCH_M=4096 BENCH_N=4096 BENCH_K=4096 ./benchmarks/build_prod_16x8_benchmark.sh
```

### RDNA4 WMMA Attention Block Benchmark

Use this as the first larger end-to-end workload beyond isolated matmul.

```bash
./benchmarks/build_attention_block_benchmark.sh
BENCH_SEQ=8192 ./benchmarks/build_attention_block_benchmark.sh
```

The attention script also accepts a compile-time LDS site mask for overhead
attribution:

```bash
HIP_MOI_ATTENTION_SITE_MASK=2 ./benchmarks/build_attention_block_benchmark.sh
```

The default mask is `0xf`, meaning all LDS accesses are instrumented. The bits
are `0x1` for K/V fragment staging, `0x2` for score scratch, `0x4` for softmax
weight scratch, and `0x8` for row-scale/row-sum scratch. Masked builds are
triage-only; the benchmark numbers below use the default all-sites mask.

The companion codegen probe compiles masked variants, unbundles the AMDGPU code
object, and prints register/spill metadata plus coarse instruction-class counts:

```bash
./benchmarks/inspect_attention_block_codegen.sh
./benchmarks/inspect_attention_block_codegen.sh 0x2 0x4 0x6
```

### RDNA4 WMMA No-Score/Weight-LDS Attention Benchmark

Use this as the register-handoff attention row. It keeps the RDNA4 WMMA QK and
PV phases, but only stages K/V fragments through LDS. The QK-to-PV handoff stays
in registers, so there is no dense score or softmax-weight LDS scratch.

```bash
./benchmarks/build_attention_no_score_lds_benchmark.sh
BENCH_SEQ=16384 ./benchmarks/build_attention_no_score_lds_benchmark.sh
```

### RDNA4 WMMA D128 No-Score/Weight-LDS Attention Benchmark

Use this as the D128/V128 register-handoff attention row. It scales the
no-score/weight-LDS idea to eight QK head fragments and eight PV value
fragments.

```bash
./benchmarks/build_attention_d128_no_score_lds_benchmark.sh
BENCH_SEQ=16384 ./benchmarks/build_attention_d128_no_score_lds_benchmark.sh
```

### RDNA4 WMMA D128 Attention Benchmark

Use this as the source-mined D128/V128 attention row with AITER-style outer
shape labels.

```bash
./benchmarks/build_attention_d128_benchmark.sh
BENCH_SEQ=16384 ./benchmarks/build_attention_d128_benchmark.sh
```

It accepts the same `HIP_MOI_ATTENTION_SITE_MASK` knob as the smaller attention
block benchmark, and the headline numbers below use the default all-sites mask.
It also has a companion codegen probe:

```bash
./benchmarks/inspect_attention_d128_codegen.sh
./benchmarks/inspect_attention_d128_codegen.sh 0x1 0x2 0x4 0x8
```

### RDNA4 WMMA D128 Attention LDS-Pressure Benchmark

Use this as the source-mined LDS-pressure attention row. It runs two D128/V128
candidate layouts in one executable: full K/V double-buffering for a 16-key tile
and a wider 32-key double-buffer pressure tile.

```bash
./benchmarks/build_attention_d128_pressure_benchmark.sh
BENCH_SEQ=16384 ./benchmarks/build_attention_d128_pressure_benchmark.sh
```

It accepts the same `HIP_MOI_ATTENTION_SITE_MASK` knob as the other attention
benchmarks, and the headline numbers below use the default all-sites mask.

### Shared Knobs

Timing knobs:

```bash
MIN_MS=500 WARMUP_MS=500 ./benchmarks/build_prod_16x8_benchmark.sh
```

Sampled instrumentation knobs:

```bash
SAMPLED_WATCHPOINTS=1 SAMPLED_SKIP=32 SAMPLED_PROBES=1 SAMPLED_DELAY=32 ./benchmarks/build_prod_16x8_benchmark.sh
SAMPLED_REPORTS=1 ./benchmarks/build_prod_16x8_benchmark.sh
```

## CMake

Benchmarks are disabled in the default build. Enable them explicitly:

```bash
cmake -S . -B ../hip-moi-build -DHIP_MOI_BUILD_BENCHMARKS=ON
```

The CMake targets are:

| Benchmark | Target | Source |
| --- | --- | --- |
| RDNA4 WMMA matmul wave-scaling, w2 2x4 | `hip_moi_benchmark_w2_2x4` | `w2_2x4_benchmark.hip` |
| RDNA4 WMMA matmul wave-scaling, w4 4x16 | `hip_moi_benchmark_w4_4x16` | `w2_2x4_benchmark.hip` |
| RDNA4 WMMA matmul wave-scaling, w8 16x8 | `hip_moi_benchmark_w8_16x8` | `w2_2x4_benchmark.hip` |
| RDNA4 WMMA matmul production 16x8 | `hip_moi_benchmark_prod_16x8` | `prod_16x8_benchmark.hip` |
| RDNA4 WMMA attention block | `hip_moi_benchmark_attention_block` | `attention_block_benchmark.hip` |
| RDNA4 WMMA no-score/weight-LDS attention | `hip_moi_benchmark_attention_no_score_lds` | `attention_no_score_lds_benchmark.hip` |
| RDNA4 WMMA D128 no-score/weight-LDS attention | `hip_moi_benchmark_attention_d128_no_score_lds` | `attention_d128_no_score_lds_benchmark.hip` |
| RDNA4 WMMA D128 attention | `hip_moi_benchmark_attention_d128` | `attention_d128_benchmark.hip` |
| RDNA4 WMMA D128 attention LDS-pressure | `hip_moi_benchmark_attention_d128_pressure` | `attention_d128_pressure_benchmark.hip` |

Example:

```bash
cmake --build ../hip-moi-build --target hip_moi_benchmark_attention_block
../hip-moi-build/benchmarks/hip_moi_benchmark_attention_block
```

The benchmark targets are RDNA4-only and are skipped unless
`CMAKE_HIP_ARCHITECTURES` names a `gfx12*` target.

## Current RDNA4 Measurements

Measured on 2026-06-24 on device 0, AMD Radeon RX 9070, `gfx1201`, 28 CUs.
All rows used the default fair sampled knobs printed by the benchmarks:
`watchpoints=1`, `skip=32`, `probes=1`, `delay=32`, `reports=off`, with
`min_ms=100` and `warmup_ms=100`, except where a benchmark subsection says
otherwise.

The headline per-iteration latency is printed in microseconds for values below
1 ms and milliseconds otherwise. Elapsed measurement windows such as `total=`
and `warmup=` remain in milliseconds.

### RDNA4 WMMA Matmul Wave-Scaling Benchmarks

These are quick iteration benchmarks for tiny-shape overhead and wave-count
scaling.

| Benchmark | Shape | noop | sampled Loom | hip-moi exact shadow | hip-moi `context + sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `hip_moi_benchmark_w2_2x4` | w2 2x4, M=32 N=64 K=16 | 2.80 µs | 4.69 µs | 8.91 µs | 4.85 µs | 3.49 µs |
| `hip_moi_benchmark_w4_4x16` | w4 4x16, M=64 N=256 K=16 | 3.15 µs | 5.91 µs | 14.0 µs | 7.45 µs | 4.31 µs |
| `hip_moi_benchmark_w8_16x8` | w8 16x8, M=256 N=128 K=16 | 3.20 µs | 5.79 µs | 13.0 µs | 7.74 µs | 4.62 µs |

### RDNA4 WMMA Matmul Production 16x8 Benchmark

`hip_moi_benchmark_prod_16x8` is the main matmul-focused performance signal.
It keeps Jakub's fp16 production 16x8 row shape: 8 waves, 16x8 WMMA tiles,
`KGroup=2`, pipelined LDS staging, and runtime `BENCH_M/N/K` sizes.

| Shape | noop | sampled Loom | hip-moi `context + sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | ---: | ---: | ---: | ---: |
| w8 16x8, M=4096 N=4096 K=4096 | 1.16 ms | 8.70 ms | 26.2 ms | 3.35 ms |

### RDNA4 WMMA Attention Block Benchmark

`hip_moi_benchmark_attention_block` is the first larger end-to-end workload
beyond isolated matmul. It defaults to `seq=12288`, intentionally larger than
the production matmul benchmark. It uses RDNA4 WMMA for both QK and PV, two
subgroups per workgroup, one workgroup per 32-query block, K/V fragment staging
through LDS, and instrumented LDS scratch for scores, softmax weights, row
scales, and row sums.

The reported TFLOP/s is an effective QK+PV matmul-rate proxy; softmax and
scalar phase work are intentionally not modeled as FLOPs.
These rows were refreshed with `min_ms=500` and `warmup_ms=500` because full
LDS instrumentation makes the instrumented rows substantially slower than the
matmul benchmarks.

| Shape | noop | sampled Loom | hip-moi `context + sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | ---: | ---: | ---: | ---: |
| seq=12288, head_dim=16, value_dim=16 | 6.56 ms | 118 ms | 148 ms | 64.0 ms |

On the same default all-sites build, the device metadata from the bundled
AMDGPU object is:

| Row | SGPRs | VGPRs | VGPR spills | Private segment |
| --- | ---: | ---: | ---: | ---: |
| noop | 18 | 82 | 0 | 0 B |
| sampled Loom | 95 | 205 | 0 | 0 B |
| hip-moi `context + sampled_watchpoint` | 97 | 256 | 386 | 748 B |
| hip-moi `sampled_watchpoint_context` | 45 | 146 | 0 | 0 B |

This is the clearest current reason to keep the narrow fast-view context
separate from the general diagnostic-capable context: the general context spills
in this workload, while the fast view is spill-free and uses fewer registers
than the sampled-Loom comparison row.

For attribution, a `seq=4096`, `min_ms=300`, `warmup_ms=300` pass on
2026-06-24 with compile-time site masks showed:

| Mask | Sites instrumented | noop | sampled Loom | hip-moi `context + sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | --- | ---: | ---: | ---: | ---: |
| `0xf` | all | 1.19 ms | 38.7 ms | 36.4 ms | 24.6 ms |
| `0x1` | K/V only | 1.18 ms | 1.38 ms | 1.64 ms | 1.29 ms |
| `0x2` | scores only | 1.19 ms | 30.2 ms | 28.8 ms | 16.4 ms |
| `0x4` | weights only | 1.19 ms | 19.7 ms | 17.4 ms | 13.1 ms |
| `0x6` | scores and weights | 1.20 ms | 36.6 ms | 37.8 ms | 23.1 ms |
| `0x8` | row scratch only | 1.19 ms | 3.18 ms | 3.24 ms | 2.11 ms |
| `0x9` | K/V and row scratch | 1.19 ms | 3.33 ms | 3.33 ms | 2.17 ms |
| `0xe` | scores, weights, rows | 1.19 ms | 37.3 ms | 41.7 ms | 23.5 ms |

The masked builds change the generated kernel shape and are not headline
apples-to-apples numbers. They are useful for localizing the cost: K/V fragment
staging is close to free here, while score and weight scratch account for nearly
all of the attention instrumentation overhead.

The dynamic access pressure explains this split. Per key tile and workgroup,
the K/V class performs 192 vector `f16x8` LDS accesses. The score class performs
1536 scalar float LDS accesses: 512 stores after QK, 512 loads for the max pass,
and 512 more loads for the weight pass. The weight class performs 1024 scalar
half LDS accesses: 512 stores and 512 loads. Row scratch is smaller: 576 scalar
float accesses per key tile plus one final 512-load epilogue per workgroup.

The codegen probe for masks `0x2`, `0x4`, `0x6`, and `0x9` showed that
`sampled_watchpoint_context` remains spill-free and far smaller than both
comparison rows:

| Mask | Row | Instructions | Atomics | VGPRs | VGPR spills | Private segment |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `0x2` | sampled Loom | 1696 | 14 | 140 | 0 | 0 B |
| `0x2` | `context + sampled_watchpoint` | 5797 | 41 | 182 | 0 | 0 B |
| `0x2` | `sampled_watchpoint_context` | 1056 | 12 | 110 | 0 | 0 B |
| `0x4` | sampled Loom | 1519 | 13 | 125 | 0 | 0 B |
| `0x4` | `context + sampled_watchpoint` | 5162 | 37 | 174 | 0 | 0 B |
| `0x4` | `sampled_watchpoint_context` | 887 | 10 | 100 | 0 | 0 B |
| `0x6` | sampled Loom | 2731 | 23 | 164 | 0 | 0 B |
| `0x6` | `context + sampled_watchpoint` | 10279 | 77 | 252 | 0 | 0 B |
| `0x6` | `sampled_watchpoint_context` | 1415 | 20 | 124 | 0 | 0 B |
| `0x9` | sampled Loom | 3391 | 26 | 172 | 0 | 0 B |
| `0x9` | `context + sampled_watchpoint` | 12592 | 89 | 235 | 0 | 0 B |
| `0x9` | `sampled_watchpoint_context` | 1709 | 22 | 122 | 0 | 0 B |

#### Attention Resource Pressure

The current attention benchmark should not be mistaken for the final
production-pressure attention workload. Its source-level shared storage is:

| LDS object | Bytes |
| --- | ---: |
| K fragment staging | 512 B |
| V fragment staging | 512 B |
| score scratch | 2048 B |
| softmax weight scratch | 1024 B |
| row-scale / row-sum scratch | 256 B |
| total | 4352 B |

The bundled RDNA4 object reports the same `group_segment_fixed_size: 4352`. On
the local `gfx1201` test device, workgroup LDS is 64 KiB, so this benchmark uses
only about 6.6% of available LDS. It is a useful instrumentation stressor, but
it is not close to the resource-saturation regime that motivated the next
attention investigation.

The all-sites codegen table above also shows the register picture: the
publish-only `sampled_watchpoint_context` row uses 146 VGPRs and no spills,
while sampled Loom uses 205 VGPRs and no spills. The general
`context + sampled_watchpoint` row spills heavily, but that path is intentionally
not the publish-only fast path.

#### Production Attention Source Mining

Two outside codebases provide the strongest current guidance.

* llama.cpp's `ggml-cuda` flash-attention sources expose RDNA-relevant WMMA
  paths around head dimensions 64, 80, 96, 112, and 128, plus broader MMA
  template machinery for shapes such as 192/128, 256/256, 320/256, 512/512, and
  576/512. Its actual RDNA4 dispatch is narrower than the template list: for
  AITER-like D128 long-sequence GQA shapes it chooses a WMMA FATTN path when the
  rocWMMA flash-attention build path is enabled, or an RDNA MMA-F16 path with
  about 18-19 KiB of LDS otherwise. The larger entries are useful pressure
  signals, not automatically dispatch-selected RDNA4 benchmark rows.
* AITER's MHA benchmark/test coverage points to production inference shapes:
  bf16/fp16, head dimension 128, query heads 32 or 64, KV heads 4 or 8, sequence
  lengths from 1024 through 16384, batch sizes 1/4/8, and causal/no-mask
  variants. Its newest handwritten attention ASM is not a direct `gfx1201` RDNA4
  import, but the shape parameters are the right production benchmark
  vocabulary.

The next attention benchmark should be built by compile-probing candidate
microkernels for LDS usage, VGPRs, and spills before instrumentation. A good next
target is an RDNA4-compatible, hip-moi-native attention row with production-style
outer parameters (`head_dim=128`, many heads, long sequence). If that
representative row does not approach the LDS/VGPR pressure Jakub wants, add a
clearly labeled production-derived pressure variant instead of pretending the
literal D128 production row already saturates the machine.

See [attention_source_mining.md](attention_source_mining.md) for the detailed
shape and dispatch extraction.

The attention optimization foray leaves a narrow, useful conclusion. The current
benchmark is production-like in that it uses RDNA4 WMMA for QK and PV, multiple
subgroups, K/V LDS staging, and online softmax state. It is also deliberately a
scalar-scratch stress benchmark: it materializes the score tile and the softmax
weights in LDS so that the QK-to-softmax-to-PV handoff stays simple and
auditable.

The latest llama.cpp source read makes that distinction sharper. Its RDNA
MMA-F16 flash-attention path keeps QK results in accumulator fragments, tracks
row max/sum state in registers, and reshapes those QK fragments directly into
the VKQ/PV MMA operand path. That is evidence that a mature implementation can
avoid dense LDS `scores` and `weights` scratch entirely. hip-moi should keep the
current dense-score rows as scalar-LDS stress tests, but the next
production-faithful attention benchmark should be built only after an isolated
RDNA4 WMMA register-transpose correctness test proves that handoff.

That distinction matters for prioritization. If a future attention benchmark
mostly instruments K/V staging plus row state, the `0x9` proxy suggests the
current `sampled_watchpoint_context` path is close enough to move on. If a
target kernel really does materialize dense scalar score/weight scratch in LDS,
then this benchmark is the right stress signal and the next optimization cannot
be about WMMA fragment staging. It has to reduce repeated scalar
score/weight-site cost.

The obvious source-level attempt was to hoist or cache the per-site sampled
selection decision for dense scalar score/weight loops, so losing sampled sites
could bypass repeated seed/lane/range setup. A first implementation session
tried two versions and deliberately did not keep either in the hot benchmark
path:

* a branch-out version that duplicated raw and instrumented loop bodies. It
  created private memory and scratch traffic in `sampled_watchpoint_context` and
  regressed the quick all-sites row;
* a lighter prepared-site object that force-inlined the selection result and
  avoided scratch. It restored codegen shape but did not produce a stable
  full-size benchmark win.

A future attempt at this idea needs a more compiler-friendly representation
than a source-level prepared-site object, and should be accepted only if the
full-size all-sites attention row improves without introducing private segment
usage.

### RDNA4 WMMA No-Score/Weight-LDS Attention Benchmark

`hip_moi_benchmark_attention_no_score_lds` grows
`014_rdna4_wmma_no_score_lds_attention_test.hip` into a benchmark-sized
workload. It uses the same D16/V16 outer shape as the smaller attention block
benchmark and defaults to `seq=12288`, but it removes the dense score and
softmax-weight LDS scratch. K/V fragments are the only LDS payload, and every
K/V LDS load/store is routed through the selected instrumentation row.

This benchmark is the first direct performance signal for the production-style
direction discovered in the source-mining pass: keep QK results in registers,
reshape them into the PV operand path, and avoid materializing score/weight
tiles in LDS.

Measured on 2026-06-24 with `min_ms=100`, `warmup_ms=100`, all K/V LDS
instrumented, and the default sampled knobs:

| Shape | noop | sampled Loom | hip-moi `context + sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | ---: | ---: | ---: | ---: |
| seq=12288, head_dim=16, value_dim=16 | 1.06 ms | 1.89 ms | 2.41 ms | 1.49 ms |

This is a very different signal from the dense-score attention benchmark. The
fast hip-moi publish-only row is close to noop and faster than sampled Loom,
which supports the current hypothesis that dense scalar score/weight LDS
scratch, not K/V fragment staging, was the dominant attention instrumentation
cost.

Source-level LDS storage is only the two fragment-staging arrays:

| LDS object | Bytes |
| --- | ---: |
| K fragment staging | 512 B |
| V fragment staging | 512 B |
| total source LDS | 1024 B |

### RDNA4 WMMA D128 No-Score/Weight-LDS Attention Benchmark

`hip_moi_benchmark_attention_d128_no_score_lds` grows
`015_rdna4_d128_no_score_lds_attention_test.hip` into the D128/V128 version of
the register-handoff benchmark. It keeps the same outer labels as the D128
attention rows (`q_heads=64`, `kv_heads=8`, `gqa=8`) and defaults to
`seq=12288`, but it only stages K/V fragments through LDS. QK accumulation spans
eight D128 head fragments; the reshaped QK tile is then reused across eight
V128 value fragments.

Measured on 2026-06-24 with `min_ms=100`, `warmup_ms=100`, all K/V LDS
instrumented, and the default sampled knobs:

| Shape | noop | sampled Loom | hip-moi `context + sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | ---: | ---: | ---: | ---: |
| seq=12288, q_heads=64, kv_heads=8, gqa=8, head_dim=128, value_dim=128 | 3.44 ms | 10.9 ms | 22.0 ms | 7.29 ms |

This is the higher-pressure confirmation of the D16 no-score result: when the
dense scalar score/weight LDS scratch is removed, the fast hip-moi publish-only
row remains faster than sampled Loom even at D128/V128. The general
`context + sampled_watchpoint` row still carries substantially more state and is
not the hot-path comparison target.

Source-level LDS storage is still only the two fragment-staging arrays:

| LDS object | Bytes |
| --- | ---: |
| K fragment staging | 512 B |
| V fragment staging | 512 B |
| total source LDS | 1024 B |

### RDNA4 WMMA D128 Attention Benchmark

`hip_moi_benchmark_attention_d128` grows the
`011_rdna4_d128_attention_block_test.hip` correctness rung into a benchmark. It
uses the production-representative D128/V128 per-token shape mined from
AITER/llama.cpp signals, labels the intended GQA setting as `q_heads=64`,
`kv_heads=8`, `gqa=8`, and defaults to `seq=8192`.

The kernel still stages one K or V fragment through LDS at a time, so it is a
D128 representative workload rather than an LDS-saturation pressure variant.
The increased head/value dimensions do raise the WMMA and accumulator pressure:
QK loops over eight head fragments and PV loops over eight value fragments.
Every LDS access is routed through the selected benchmark row.

Measured on 2026-06-24 with `min_ms=100` and `warmup_ms=100`:

| Shape | noop | sampled Loom | hip-moi `context + sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | ---: | ---: | ---: | ---: |
| seq=8192, q_heads=64, kv_heads=8, gqa=8, head_dim=128, value_dim=128 | 4.25 ms | 102 ms | 105 ms | 38.8 ms |

The D128 version changes the register story much more than the LDS story. Source
shared storage is still 4352 B because only one K or V fragment is staged at a
time, but the noop row already uses 218 VGPRs. The all-sites codegen probe
reported:

| LDS object | Bytes |
| --- | ---: |
| K fragment staging | 512 B |
| V fragment staging | 512 B |
| score scratch | 2048 B |
| softmax weight scratch | 1024 B |
| row-old-scale scratch | 128 B |
| row-sum scratch | 128 B |
| total source LDS | 4352 B |
| bundled RDNA4 `group_segment_fixed_size` | 4352 B |

On the local `gfx1201` test device, workgroup LDS is 64 KiB, so this D128
benchmark uses about 6.6% of available LDS. Its pressure is primarily register
and scalar-scratch instrumentation pressure, not LDS-capacity pressure.

| Row | Instructions | Atomics | SGPRs | VGPRs | VGPR spills | Private segment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| noop | 1042 | 0 | 17 | 218 | 0 | 0 B |
| sampled Loom | 21942 | 215 global | 95 | 256 | 105 | 352 B |
| hip-moi `context + sampled_watchpoint` | 61047 | 469 flat | 100 | 256 | 337 | 992 B |
| hip-moi `sampled_watchpoint_context` | 6205 | 198 flat | 39 | 250 | 0 | 0 B |

The fast hip-moi row is still substantially smaller than sampled Loom in static
instruction count and remains faster in the headline timing. The current
publish-only fast row is also spill-free again: scalar-sized LDS accesses use a
compile-time range path, while vector accesses keep the older runtime-shaped
path to avoid regressing vector-heavy matmul kernels. Later pressure-oriented
attention work should still treat VGPR pressure as a first-class concern rather
than only counting dynamic instrumented LDS operations.

The site-mask timing pass shows that dense scalar scratch remains the main
dynamic cost center, while D128 makes K/V fragment staging and row-state scratch
less negligible:

| Mask | Sites instrumented | noop | sampled Loom | hip-moi `context + sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | --- | ---: | ---: | ---: | ---: |
| `0xf` | all | 4.25 ms | 102 ms | 105 ms | 38.8 ms |
| `0x1` | K/V only | 4.22 ms | 8.87 ms | 14.1 ms | 6.04 ms |
| `0x2` | scores only | 4.22 ms | 63.7 ms | 63.9 ms | 40.4 ms |
| `0x4` | weights only | 4.17 ms | 61.1 ms | 64.7 ms | 25.9 ms |
| `0x6` | scores and weights | 4.25 ms | 96.3 ms | 93.8 ms | 51.9 ms |
| `0x8` | row scratch only | 4.19 ms | 24.7 ms | 32.2 ms | 19.9 ms |
| `0x9` | K/V and row scratch | 4.23 ms | 26.7 ms | 34.4 ms | 20.1 ms |
| `0xe` | scores, weights, rows | 4.26 ms | 101 ms | 102 ms | 51.8 ms |

As with the smaller attention benchmark, masked builds change the generated
kernel shape and are not additive decompositions. They are useful for locating
the expensive site classes, not for predicting the exact all-sites latency.

Per key tile and workgroup, the K/V class performs 1536 vector `f16x8` LDS
accesses: eight D128 head fragments for QK and eight D128 value fragments for
PV. Scores and weights are unchanged from the D16 benchmark at 1536 scalar
float score accesses and 1024 scalar half weight accesses per key tile. Row
scratch becomes more important: scaling the eight value-fragment accumulators
per key tile creates 4096 scalar old-scale loads, and the final output epilogue
adds 4096 scalar row-sum loads per workgroup.

Two takeaways fall out of this pass. First, specializing scalar-sized
publish-only accesses cut the all-sites fast row from the previous `59.3 ms` to
`38.8 ms` and removed its private segment use. The deliberately rejected broader
variant also specialized vector accesses; it preserved the D128 attention win
but regressed the production matmul fast row, so vector LDS accesses stay on the
runtime-shaped path for now. Second, the score/weight/row masks remain the
expensive site classes, but the masked rows are no longer a reliable additive
model of the all-sites fast row. A future production-pressure variant should
therefore probe whether a mature attention kernel avoids this dense LDS
score/weight materialization; if it does not, the next optimization target is
still repeated scalar-site instrumentation cost under high VGPR pressure.

### RDNA4 WMMA D128 Attention LDS-Pressure Benchmark

`hip_moi_benchmark_attention_d128_pressure` grows
`012_rdna4_d128_attention_pressure_test.hip` into benchmark rows. It keeps the
D128/V128, `q_heads=64`, `kv_heads=8`, `gqa=8`, `seq=8192` shape labels, but
uses larger source shared-memory layouts than the representative D128 benchmark.

| Candidate | Key tile | Source LDS | Approx. 64 KiB LDS use |
| --- | ---: | ---: | ---: |
| `full_kv16` | 16 | 19712 B | 30.1% |
| `wide_k32` | 32 | 39168 B | 59.8% |

Measured on 2026-06-24 with `min_ms=100`, `warmup_ms=100`, all-sites
instrumentation, and the default sampled knobs:

| Candidate | noop | sampled Loom | hip-moi `context + sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | ---: | ---: | ---: | ---: |
| `full_kv16`, seq=8192 | 5.87 ms | 156 ms | 132 ms | 48.4 ms |
| `wide_k32`, seq=8192 | 8.87 ms | 183 ms | 162 ms | 75.5 ms |

The `full_kv16` row is the better immediate benchmark candidate: it matches the
source-mined llama.cpp D128 LDS scale more closely and is faster in every row.
The `wide_k32` row remains useful as an explicit high-LDS pressure stressor,
but it should be read as a pressure variant rather than a literal production
dispatch clone. In both candidates, the fast hip-moi row remains substantially
faster than sampled Loom; the general `context + sampled_watchpoint` row also
beats sampled Loom here, but it remains the larger correctness-oriented path and
is not the publish-only hot path.

A codegen/performance cleanup pass on 2026-06-24 tested two tempting
`sampled_watchpoint_context` micro-optimizations and rejected both. A static
subgroup-size policy hint removed no meaningful generated-code structure on the
D128 probe. A narrower naturally-aligned one-granule scalar path changed the
all-sites D128 codegen by only about one VGPR and left the pressure benchmark
within noise: roughly `48 ms` / `75 ms` for the two fast rows. The production
matmul guardrail also stayed at about `3.4 ms`. These were not kept in the API
or implementation. The next credible optimization target is larger-grain:
reduce repeated per-site sampled-selection setup in dense score/weight/row
scratch without introducing private memory or spills.

A follow-up session tested that larger-grain idea directly. The first variant
prepared selected-site state for all dense score/weight/row accesses and
regressed the fast pressure rows to about `103 ms` / `215 ms`. A narrower
lane-0 softmax-only variant still regressed them to about `64 ms` / `204 ms`.
Both prototypes were removed. The practical lesson is that source-level
prepared-site state is too expensive for this benchmark shape, even when its
semantic intent is attractive. The next performance direction should be either
a different compiler-friendly representation of static-site decisions, or a
kernel/workload study that determines whether production attention avoids dense
LDS score/weight materialization altogether.

The source read now favors the second path. Treat `full_kv16` and `wide_k32` as
dense-score pressure rows, not as the final production-faithful attention model.
The first concrete coding steps are now in the test suite:
`013_rdna4_wmma_register_handoff_test.hip` moves from the QK accumulator layout
to the PV B-fragment layout without storing a dense score/weight tile in LDS,
and `014_rdna4_wmma_no_score_lds_attention_test.hip` grows that into a
two-key-tile attention-shaped correctness test. The D16
`hip_moi_benchmark_attention_no_score_lds` benchmark grows from that test.
`015_rdna4_d128_no_score_lds_attention_test.hip` and
`hip_moi_benchmark_attention_d128_no_score_lds` then repeat the same idea at
D128/V128. These register-handoff rows are now the production-faithful attention
direction; the dense score/weight rows remain scalar-LDS stress tests rather
than the presumed production target.
