# hip-moi Benchmarks

These RDNA4 benchmarks are focused extractions from Jakub's
`sanitizer-strategy/rdna4_matmul/rdna4_matmul.hip`. They are vendored here so
hip-moi performance work can be reproduced without reaching back into the
external strategy repository.

They compare the main rows needed for the current Loom-parity work:

* noop LDS matmul,
* Jakub's sampled-Loom publish-only instrumentation,
* hip-moi general `context` with the `sampled_watchpoint` backend,
* hip-moi sampled-watchpoint instrumentation through the narrower
  `sampled_watchpoint_context` fast view.

The compact 2/4/8-wave benchmark family also includes the hip-moi exact-shadow
row as a correctness/performance reference.

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
```

Use the two-wave shape for quick intra-session experiments. Use the production
16x8 benchmark as the current main gate for performance-sensitive commits.

Useful knobs:

```bash
BENCH_M=4096 BENCH_N=4096 BENCH_K=4096 ./benchmarks/build_prod_16x8_benchmark.sh
MIN_MS=500 WARMUP_MS=500 ./benchmarks/build_prod_16x8_benchmark.sh
SAMPLED_WATCHPOINTS=1 SAMPLED_SKIP=32 SAMPLED_PROBES=1 SAMPLED_DELAY=32 ./benchmarks/build_prod_16x8_benchmark.sh
SAMPLED_REPORTS=1 ./benchmarks/build_prod_16x8_benchmark.sh
```

## CMake

Benchmarks are disabled in the default build. Enable them explicitly:

```bash
cmake -S . -B ../hip-moi-build -DHIP_MOI_BUILD_BENCHMARKS=ON
cmake --build ../hip-moi-build --target hip_moi_benchmark_prod_16x8
../hip-moi-build/benchmarks/hip_moi_benchmark_prod_16x8
```

The CMake targets are:

* `hip_moi_benchmark_w2_2x4`,
* `hip_moi_benchmark_w4_4x16`,
* `hip_moi_benchmark_w8_16x8`,
* `hip_moi_benchmark_prod_16x8`.

They are RDNA4-only and are skipped unless `CMAKE_HIP_ARCHITECTURES` names a
`gfx12*` target.

## Current RDNA4 Numbers

Measured on 2026-06-23 on device 0, AMD Radeon RX 9070, `gfx1201`, 28 CUs.
All rows used the default fair sampled knobs printed by the benchmarks:
`watchpoints=1`, `skip=32`, `probes=1`, `delay=32`, `reports=off`, with
`min_ms=100` and `warmup_ms=100`.

The compact rows are useful for quick iteration and wave-count scaling:

| Shape | noop | sampled Loom | hip-moi exact shadow | hip-moi `context` + `sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | ---: | ---: | ---: | ---: | ---: |
| w2 2x4, M=32 N=64 K=16 | 0.00285 ms | 0.00477 ms | 0.00902 ms | 0.00481 ms | 0.00434 ms |
| w4 4x16, M=64 N=256 K=16 | 0.00312 ms | 0.00589 ms | 0.0138 ms | 0.00746 ms | 0.00572 ms |
| w8 16x8, M=256 N=128 K=16 | 0.00323 ms | 0.00578 ms | 0.0129 ms | 0.00781 ms | 0.00633 ms |

The production-shaped row is the current main performance signal:

| Shape | noop | sampled Loom | hip-moi `context` + `sampled_watchpoint` | hip-moi `sampled_watchpoint_context` |
| --- | ---: | ---: | ---: | ---: |
| w8 16x8, M=4096 N=4096 K=4096 | 1.16 ms | 8.64 ms | 26.1 ms | 3.43 ms |
