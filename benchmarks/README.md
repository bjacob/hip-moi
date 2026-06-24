# hip-moi Benchmarks

These RDNA4 benchmarks are the performance-facing companions to the correctness
tests. The matmul benchmarks are focused extractions from Jakub's
`sanitizer-strategy/rdna4_matmul/rdna4_matmul.hip`; the attention benchmark is
a hip-moi-native workload grown from the instrumented attention tests.

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

Example:

```bash
cmake --build ../hip-moi-build --target hip_moi_benchmark_attention_block
../hip-moi-build/benchmarks/hip_moi_benchmark_attention_block
```

The benchmark targets are RDNA4-only and are skipped unless
`CMAKE_HIP_ARCHITECTURES` names a `gfx12*` target.

## Current RDNA4 Measurements

Measured on 2026-06-23 on device 0, AMD Radeon RX 9070, `gfx1201`, 28 CUs.
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
| w8 16x8, M=4096 N=4096 K=4096 | 1.16 ms | 8.65 ms | 25.9 ms | 3.38 ms |

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

For attribution, a quick `seq=4096`, `min_ms=200`, `warmup_ms=200` pass with
compile-time site masks showed:

| Mask | Sites instrumented | noop | sampled Loom | hip-moi `context + sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | --- | ---: | ---: | ---: | ---: |
| `0xf` | all | 2.82 ms | 40.1 ms | 31.7 ms | 18.7 ms |
| `0x1` | K/V only | 2.70 ms | 2.76 ms | 1.68 ms | 1.28 ms |
| `0x2` | scores only | 1.19 ms | 29.8 ms | 28.3 ms | 17.9 ms |
| `0x4` | weights only | 1.19 ms | 20.5 ms | 18.6 ms | 13.1 ms |
| `0x8` | row scratch only | 1.19 ms | 3.09 ms | 3.12 ms | 2.14 ms |
| `0xe` | scores, weights, rows | 1.19 ms | 35.3 ms | 34.2 ms | 18.5 ms |

The masked builds change the generated kernel shape and are not headline
apples-to-apples numbers. They are useful for localizing the cost: K/V fragment
staging is close to free here, while score and weight scratch account for nearly
all of the attention instrumentation overhead.
