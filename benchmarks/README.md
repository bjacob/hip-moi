# hip-moi Benchmarks

The matmul benchmarks are focused extractions from Jakub's
`sanitizer-strategy/rdna4_matmul/rdna4_matmul.hip`. The attention benchmark is
a hip-moi-native workload grown from the instrumented correctness tests. They
are vendored here so hip-moi performance work can be reproduced without
reaching back into the external strategy repository.

They compare the main rows needed for the current Loom-parity work:

* noop LDS matmul,
* Jakub's sampled-Loom publish-only instrumentation,
* hip-moi general `context` with the `sampled_watchpoint` backend,
* hip-moi narrower `sampled_watchpoint_context` fast view.

The compact 2/4/8-wave benchmark family also includes the hip-moi exact-shadow
row as a correctness/performance reference. In both benchmark families,
`context + sampled_watchpoint` means the full diagnostic-capable
`hip_moi::context` running its sampled backend, while
`sampled_watchpoint_context` means the dedicated publish-only fast-view class.

The benchmarks intentionally focus on subgroup-level, full-workgroup-barrier
LDS instrumentation. They are not the correctness test suite and they do not
exercise atomics or finer-grained synchronization.

## Quick Scripts

The scripts use `ROCM_ROOT` when provided, otherwise they try the local TheRock
SDK path used during development. They default to `gfx1201`.

```bash
./benchmarks/build_w2_2x4_benchmark.sh
./benchmarks/build_w2_2x4_benchmark.sh w4_4x16
./benchmarks/build_w2_2x4_benchmark.sh w8_16x8
./benchmarks/build_prod_16x8_benchmark.sh
./benchmarks/build_attention_block_benchmark.sh
```

Use the two-wave shape for quick intra-session experiments. Use the production
16x8 benchmark as the current main gate for matmul-focused
performance-sensitive commits. Use the attention block benchmark for the first
larger end-to-end workload beyond isolated matmul.

Useful knobs:

```bash
BENCH_M=4096 BENCH_N=4096 BENCH_K=4096 ./benchmarks/build_prod_16x8_benchmark.sh
MIN_MS=500 WARMUP_MS=500 ./benchmarks/build_prod_16x8_benchmark.sh
SAMPLED_WATCHPOINTS=1 SAMPLED_SKIP=32 SAMPLED_PROBES=1 SAMPLED_DELAY=32 ./benchmarks/build_prod_16x8_benchmark.sh
SAMPLED_REPORTS=1 ./benchmarks/build_prod_16x8_benchmark.sh
BENCH_SEQ=8192 ./benchmarks/build_attention_block_benchmark.sh
```

## CMake

Benchmarks are disabled in the default build. Enable them explicitly:

```bash
cmake -S . -B ../hip-moi-build -DHIP_MOI_BUILD_BENCHMARKS=ON
cmake --build ../hip-moi-build --target hip_moi_benchmark_prod_16x8
../hip-moi-build/benchmarks/hip_moi_benchmark_prod_16x8
cmake --build ../hip-moi-build --target hip_moi_benchmark_attention_block
../hip-moi-build/benchmarks/hip_moi_benchmark_attention_block
```

The CMake targets are:

* `hip_moi_benchmark_w2_2x4`,
* `hip_moi_benchmark_w4_4x16`,
* `hip_moi_benchmark_w8_16x8`,
* `hip_moi_benchmark_prod_16x8`,
* `hip_moi_benchmark_attention_block`.

They are RDNA4-only and are skipped unless `CMAKE_HIP_ARCHITECTURES` names a
`gfx12*` target.

## Current RDNA4 Numbers

Measured on 2026-06-23 on device 0, AMD Radeon RX 9070, `gfx1201`, 28 CUs.
All rows used the default fair sampled knobs printed by the benchmarks:
`watchpoints=1`, `skip=32`, `probes=1`, `delay=32`, `reports=off`, with
`min_ms=100` and `warmup_ms=100`. The headline per-iteration latency is printed
in microseconds for values below 1 ms and milliseconds otherwise; elapsed
measurement windows such as `total=` and `warmup=` remain in milliseconds.

The compact rows are useful for quick iteration and wave-count scaling:

| Shape | noop | sampled Loom | hip-moi exact shadow | hip-moi `context` + `sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | ---: | ---: | ---: | ---: | ---: |
| w2 2x4, M=32 N=64 K=16 | 2.80 µs | 4.69 µs | 8.91 µs | 4.85 µs | 3.49 µs |
| w4 4x16, M=64 N=256 K=16 | 3.15 µs | 5.91 µs | 14.0 µs | 7.45 µs | 4.31 µs |
| w8 16x8, M=256 N=128 K=16 | 3.20 µs | 5.79 µs | 13.0 µs | 7.74 µs | 4.62 µs |

The production-shaped row is the current main performance signal:

| Shape | noop | sampled Loom | hip-moi `context` + `sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | ---: | ---: | ---: | ---: |
| w8 16x8, M=4096 N=4096 K=4096 | 1.16 ms | 8.65 ms | 25.9 ms | 3.38 ms |

The attention row is the first larger end-to-end workload beyond isolated
matmul. It defaults to `seq=12288`, which is intentionally larger than the
production matmul row and lands near 2x the `sampled_watchpoint_context`
latency measured for `prod_16x8`. It uses RDNA4 WMMA for both QK and PV, two
subgroups per workgroup, one workgroup per 32-query block, and K/V fragment
staging through LDS. Its reported TFLOP/s is an effective QK+PV matmul-rate
proxy; softmax and scalar phase work are intentionally not modeled as FLOPs.

| Shape | noop | sampled Loom | hip-moi `context` + `sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | ---: | ---: | ---: | ---: |
| seq=12288, head_dim=16, value_dim=16 | 6.89 ms | 7.28 ms | 7.73 ms | 6.94 ms |
