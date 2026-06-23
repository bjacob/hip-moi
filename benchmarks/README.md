# hip-moi Benchmarks

These RDNA4 benchmarks are focused extractions from Jakub's
`sanitizer-strategy/rdna4_matmul/rdna4_matmul.hip`. They are vendored here so
hip-moi performance work can be reproduced without reaching back into the
external strategy repository.

They compare three rows:

* noop LDS matmul,
* Jakub's sampled-Loom publish-only instrumentation,
* hip-moi sampled-watchpoint publish-only instrumentation.

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
