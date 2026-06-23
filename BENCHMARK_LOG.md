# hip-moi benchmark log

Append the raw 2/4/8-wave RDNA4 benchmark output at each hip-moi commit.

The benchmark lives in Jakub's local `sanitizer-strategy` repository. Run:

```bash
cd /home/benoit/workspace/sanitizer-strategy
./rdna4_matmul/build_w2_2x4_benchmark.sh
./rdna4_matmul/build_w2_2x4_benchmark.sh w4_4x16
./rdna4_matmul/build_w2_2x4_benchmark.sh w8_16x8
```

Use the 2-wave shape for fast intra-session iteration. Use the full 2/4/8 set
before a session-ending commit when benchmark-sensitive code changed.

## 2026-06-23 baseline before Loom-like backend

hip-moi commit measured: `418450b` (`Center plan on Loom-like backend`)
sanitizer-strategy benchmark commit: `8e3d1e3`

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=32 N=64 K=16 waves=2 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w2_2x4_noop                                   0.003 ms      0.02 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.221 ms  iters=34740  warmup= 100.178 ms  warmup_iters=34476
fp16_wmma_tiled_w2_2x4_sampled_loom_tsan                      0.005 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.094 ms  iters=21051  warmup= 100.004 ms  warmup_iters=21051
fp16_wmma_tiled_w2_2x4_hip_moi_coalescing                     6.023 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 102.397 ms  iters=17  warmup= 103.023 ms  warmup_iters=17
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w4_4x16
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=64 N=256 K=16 waves=4 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w4_4x16_noop                                  0.003 ms      0.17 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.280 ms  iters=32077  warmup= 100.276 ms  warmup_iters=31819
fp16_wmma_tiled_w4_4x16_sampled_loom_tsan                     0.006 ms      0.09 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.117 ms  iters=17014  warmup= 100.113 ms  warmup_iters=16962
fp16_wmma_tiled_w4_4x16_hip_moi_coalescing                  125.275 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 125.275 ms  iters=1  warmup= 125.170 ms  warmup_iters=1
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w8_16x8
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=256 N=128 K=16 waves=8 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w8_16x8_noop                                  0.003 ms      0.33 TFLOP/s    0.2% of 191 TFLOP/s  total= 100.353 ms  iters=31117  warmup= 100.274 ms  warmup_iters=30818
fp16_wmma_tiled_w8_16x8_sampled_loom_tsan                     0.006 ms      0.18 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.121 ms  iters=17203  warmup= 100.068 ms  warmup_iters=17182
fp16_wmma_tiled_w8_16x8_hip_moi_coalescing                  633.183 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 633.183 ms  iters=1  warmup= 635.521 ms  warmup_iters=1
```

## 2026-06-23 after shadow ABI skeleton

hip-moi commit measured: this commit (`Add Loom-style shadow ABI helpers`)
sanitizer-strategy benchmark commit: `8e3d1e3`

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=32 N=64 K=16 waves=2 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w2_2x4_noop                                   0.003 ms      0.02 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.039 ms  iters=34685  warmup= 100.013 ms  warmup_iters=34441
fp16_wmma_tiled_w2_2x4_sampled_loom_tsan                      0.005 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.051 ms  iters=21128  warmup= 100.123 ms  warmup_iters=21027
fp16_wmma_tiled_w2_2x4_hip_moi_coalescing                     5.945 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 101.062 ms  iters=17  warmup= 101.777 ms  warmup_iters=17
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w4_4x16
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=64 N=256 K=16 waves=4 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w4_4x16_noop                                  0.003 ms      0.17 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.048 ms  iters=31799  warmup= 100.074 ms  warmup_iters=31794
fp16_wmma_tiled_w4_4x16_sampled_loom_tsan                     0.006 ms      0.09 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.147 ms  iters=16984  warmup= 100.045 ms  warmup_iters=16921
fp16_wmma_tiled_w4_4x16_hip_moi_coalescing                  122.007 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 122.007 ms  iters=1  warmup= 121.560 ms  warmup_iters=1
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w8_16x8
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=256 N=128 K=16 waves=8 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w8_16x8_noop                                  0.003 ms      0.33 TFLOP/s    0.2% of 191 TFLOP/s  total= 100.401 ms  iters=31776  warmup= 100.114 ms  warmup_iters=30816
fp16_wmma_tiled_w8_16x8_sampled_loom_tsan                     0.006 ms      0.18 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.537 ms  iters=17400  warmup= 100.205 ms  warmup_iters=17376
fp16_wmma_tiled_w8_16x8_hip_moi_coalescing                  623.629 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 623.629 ms  iters=1  warmup= 635.914 ms  warmup_iters=1
```

## 2026-06-23 after Loom backend storage layout

hip-moi commit measured: this commit (`Add Loom backend storage layout`)
sanitizer-strategy benchmark commit: `8e3d1e3`

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=32 N=64 K=16 waves=2 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w2_2x4_noop                                   0.003 ms      0.02 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.007 ms  iters=34712  warmup= 100.026 ms  warmup_iters=34452
fp16_wmma_tiled_w2_2x4_sampled_loom_tsan                      0.005 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.313 ms  iters=20989  warmup= 100.064 ms  warmup_iters=20875
fp16_wmma_tiled_w2_2x4_hip_moi_coalescing                     6.118 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 104.006 ms  iters=17  warmup= 102.010 ms  warmup_iters=17
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w4_4x16
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=64 N=256 K=16 waves=4 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w4_4x16_noop                                  0.003 ms      0.17 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.068 ms  iters=31787  warmup= 100.089 ms  warmup_iters=31815
fp16_wmma_tiled_w4_4x16_sampled_loom_tsan                     0.006 ms      0.09 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.070 ms  iters=16924  warmup= 100.001 ms  warmup_iters=16892
fp16_wmma_tiled_w4_4x16_hip_moi_coalescing                  125.161 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 125.161 ms  iters=1  warmup= 123.810 ms  warmup_iters=1
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w8_16x8
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=256 N=128 K=16 waves=8 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w8_16x8_noop                                  0.003 ms      0.34 TFLOP/s    0.2% of 191 TFLOP/s  total= 100.006 ms  iters=31981  warmup= 100.025 ms  warmup_iters=30763
fp16_wmma_tiled_w8_16x8_sampled_loom_tsan                     0.006 ms      0.18 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.069 ms  iters=17449  warmup= 100.066 ms  warmup_iters=17432
fp16_wmma_tiled_w8_16x8_hip_moi_coalescing                  634.801 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 634.801 ms  iters=1  warmup= 636.625 ms  warmup_iters=1
```

## 2026-06-23 after explicit LDS-offset APIs

hip-moi commit measured: this commit (`Add explicit LDS offset access APIs`)
sanitizer-strategy benchmark commit: `8e3d1e3`

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=32 N=64 K=16 waves=2 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w2_2x4_noop                                   0.003 ms      0.02 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.417 ms  iters=35737  warmup= 100.090 ms  warmup_iters=35158
fp16_wmma_tiled_w2_2x4_sampled_loom_tsan                      0.005 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.610 ms  iters=21321  warmup= 100.035 ms  warmup_iters=21212
fp16_wmma_tiled_w2_2x4_hip_moi_coalescing                     5.883 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 105.891 ms  iters=18  warmup= 105.790 ms  warmup_iters=18
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w4_4x16
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=64 N=256 K=16 waves=4 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w4_4x16_noop                                  0.003 ms      0.17 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.080 ms  iters=32047  warmup= 100.025 ms  warmup_iters=31787
fp16_wmma_tiled_w4_4x16_sampled_loom_tsan                     0.006 ms      0.09 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.116 ms  iters=17101  warmup= 100.108 ms  warmup_iters=16943
fp16_wmma_tiled_w4_4x16_hip_moi_coalescing                  122.517 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 122.517 ms  iters=1  warmup= 122.790 ms  warmup_iters=1
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w8_16x8
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=256 N=128 K=16 waves=8 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w8_16x8_noop                                  0.003 ms      0.33 TFLOP/s    0.2% of 191 TFLOP/s  total= 100.030 ms  iters=31950  warmup= 100.035 ms  warmup_iters=30732
fp16_wmma_tiled_w8_16x8_sampled_loom_tsan                     0.006 ms      0.18 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.076 ms  iters=17459  warmup= 100.006 ms  warmup_iters=17459
fp16_wmma_tiled_w8_16x8_hip_moi_coalescing                  585.826 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 585.826 ms  iters=1  warmup= 587.829 ms  warmup_iters=1
```

## 2026-06-23 after exact shadow backend for explicit offsets

hip-moi commit measured: this commit (`Add exact shadow backend for offset accesses`)
sanitizer-strategy benchmark commit: `8e3d1e3`

Note: the extracted benchmark still calls `lds_load`/`lds_store`, not the new
explicit-offset `lds_load_at`/`lds_store_at` APIs, so these numbers continue to
track the legacy coalescing path rather than the exact-shadow hot path.

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=32 N=64 K=16 waves=2 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w2_2x4_noop                                   0.003 ms      0.02 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.014 ms  iters=34403  warmup= 100.040 ms  warmup_iters=34416
fp16_wmma_tiled_w2_2x4_sampled_loom_tsan                      0.005 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.063 ms  iters=21002  warmup= 100.022 ms  warmup_iters=20918
fp16_wmma_tiled_w2_2x4_hip_moi_coalescing                     5.999 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 101.983 ms  iters=17  warmup= 101.912 ms  warmup_iters=17
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w4_4x16
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=64 N=256 K=16 waves=4 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w4_4x16_noop                                  0.003 ms      0.17 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.105 ms  iters=32029  warmup= 100.562 ms  warmup_iters=31428
fp16_wmma_tiled_w4_4x16_sampled_loom_tsan                     0.006 ms      0.09 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.128 ms  iters=16923  warmup= 100.223 ms  warmup_iters=16960
fp16_wmma_tiled_w4_4x16_hip_moi_coalescing                  123.793 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 123.793 ms  iters=1  warmup= 122.519 ms  warmup_iters=1
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w8_16x8
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=256 N=128 K=16 waves=8 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w8_16x8_noop                                  0.003 ms      0.33 TFLOP/s    0.2% of 191 TFLOP/s  total= 100.024 ms  iters=31912  warmup= 100.061 ms  warmup_iters=30808
fp16_wmma_tiled_w8_16x8_sampled_loom_tsan                     0.006 ms      0.18 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.112 ms  iters=17321  warmup= 100.289 ms  warmup_iters=17371
fp16_wmma_tiled_w8_16x8_hip_moi_coalescing                  609.967 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 609.967 ms  iters=1  warmup= 610.128 ms  warmup_iters=1
```

## 2026-06-23 after benchmark hookup to explicit LDS offsets

hip-moi commit measured: `21c7ee3` (`Add exact shadow backend for offset accesses`)
sanitizer-strategy benchmark commit: `87dcbaf`

The extracted benchmark's hip-moi row now calls `lds_load_at` and
`lds_store_at`, so it measures the exact-shadow offset path instead of the old
record/log/coalescing path.

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=32 N=64 K=16 waves=2 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w2_2x4_noop                                   0.003 ms      0.02 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.016 ms  iters=34415  warmup= 100.081 ms  warmup_iters=34442
fp16_wmma_tiled_w2_2x4_sampled_loom_tsan                      0.005 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.013 ms  iters=21056  warmup= 100.009 ms  warmup_iters=20899
fp16_wmma_tiled_w2_2x4_hip_moi_exact_shadow                   0.076 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.011 ms  iters=1318  warmup= 100.513 ms  warmup_iters=1307
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w4_4x16
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=64 N=256 K=16 waves=4 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w4_4x16_noop                                  0.003 ms      0.17 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.098 ms  iters=33099  warmup= 100.107 ms  warmup_iters=31822
fp16_wmma_tiled_w4_4x16_sampled_loom_tsan                     0.006 ms      0.09 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.811 ms  iters=17043  warmup= 100.394 ms  warmup_iters=17110
fp16_wmma_tiled_w4_4x16_hip_moi_exact_shadow               benchmark_tsan_reports=800
fp16_wmma_tiled_w4_4x16_hip_moi_exact_shadow                  0.120 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.481 ms  iters=835  warmup= 100.287 ms  warmup_iters=836
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w8_16x8
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=256 N=128 K=16 waves=8 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w8_16x8_noop                                  0.003 ms      0.33 TFLOP/s    0.2% of 191 TFLOP/s  total= 100.249 ms  iters=31961  warmup= 100.139 ms  warmup_iters=31108
fp16_wmma_tiled_w8_16x8_sampled_loom_tsan                     0.006 ms      0.18 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.158 ms  iters=17381  warmup= 100.383 ms  warmup_iters=17368
fp16_wmma_tiled_w8_16x8_hip_moi_exact_shadow                  2.215 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 101.883 ms  iters=46  warmup= 101.906 ms  warmup_iters=46
```

## 2026-06-23 after first sampled-watchpoint backend

hip-moi commit measured: this commit (`Add sampled watchpoint backend`)
sanitizer-strategy benchmark commit: `0a91080`

The extracted benchmark now prints separate hip-moi rows for exact shadow and
sampled watchpoints. The first sampled backend is wired through
`host_context_options::backend`, but it does not yet reduce latency relative to
exact shadow on this benchmark.

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=32 N=64 K=16 waves=2 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w2_2x4_noop                                   0.003 ms      0.02 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.015 ms  iters=34340  warmup= 100.014 ms  warmup_iters=34344
fp16_wmma_tiled_w2_2x4_sampled_loom_tsan                      0.005 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.023 ms  iters=21006  warmup= 100.219 ms  warmup_iters=20800
fp16_wmma_tiled_w2_2x4_hip_moi_exact_shadow                   0.076 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.130 ms  iters=1320  warmup= 100.265 ms  warmup_iters=1312
fp16_wmma_tiled_w2_2x4_hip_moi_sampled_watchpoint             0.076 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.036 ms  iters=1321  warmup= 101.372 ms  warmup_iters=1334
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w4_4x16
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=64 N=256 K=16 waves=4 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w4_4x16_noop                                  0.003 ms      0.17 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.215 ms  iters=32298  warmup= 100.456 ms  warmup_iters=31831
fp16_wmma_tiled_w4_4x16_sampled_loom_tsan                     0.006 ms      0.09 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.348 ms  iters=17008  warmup= 100.049 ms  warmup_iters=16969
fp16_wmma_tiled_w4_4x16_hip_moi_exact_shadow               benchmark_tsan_reports=800
fp16_wmma_tiled_w4_4x16_hip_moi_exact_shadow                  0.078 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.215 ms  iters=1292  warmup= 100.430 ms  warmup_iters=1295
fp16_wmma_tiled_w4_4x16_hip_moi_sampled_watchpoint            0.076 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.284 ms  iters=1326  warmup= 101.105 ms  warmup_iters=1340
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w8_16x8
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=256 N=128 K=16 waves=8 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w8_16x8_noop                                  0.003 ms      0.33 TFLOP/s    0.2% of 191 TFLOP/s  total= 100.153 ms  iters=31599  warmup= 100.098 ms  warmup_iters=31084
fp16_wmma_tiled_w8_16x8_sampled_loom_tsan                     0.006 ms      0.18 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.441 ms  iters=17371  warmup= 100.195 ms  warmup_iters=17301
fp16_wmma_tiled_w8_16x8_hip_moi_exact_shadow                  2.206 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 101.479 ms  iters=46  warmup= 101.832 ms  warmup_iters=46
fp16_wmma_tiled_w8_16x8_hip_moi_sampled_watchpoint            2.206 ms      0.00 TFLOP/s    0.0% of 191 TFLOP/s  total= 101.456 ms  iters=46  warmup= 101.526 ms  warmup_iters=46
```

## 2026-06-23 after legacy record/log cleanup

hip-moi commit measured: this commit
sanitizer-strategy benchmark commit: `76181cc`

The cleanup removed the old record/log backend, lane-record coalescing storage,
and pointer-only LDS helpers. The benchmark now exercises only explicit-offset
exact-shadow and sampled-watchpoint paths.

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=32 N=64 K=16 waves=2 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w2_2x4_noop                                   0.003 ms      0.02 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.089 ms  iters=34642  warmup= 100.083 ms  warmup_iters=34670
fp16_wmma_tiled_w2_2x4_sampled_loom_tsan                      0.005 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.027 ms  iters=20952  warmup= 100.092 ms  warmup_iters=20962
fp16_wmma_tiled_w2_2x4_hip_moi_exact_shadow                   0.011 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.196 ms  iters=9041  warmup= 100.291 ms  warmup_iters=9008
fp16_wmma_tiled_w2_2x4_hip_moi_sampled_watchpoint             0.010 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.127 ms  iters=9708  warmup= 100.144 ms  warmup_iters=9385
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w4_4x16
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=64 N=256 K=16 waves=4 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w4_4x16_noop                                  0.003 ms      0.17 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.465 ms  iters=32463  warmup= 100.001 ms  warmup_iters=32197
fp16_wmma_tiled_w4_4x16_sampled_loom_tsan                     0.006 ms      0.09 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.026 ms  iters=17079  warmup= 100.051 ms  warmup_iters=17013
fp16_wmma_tiled_w4_4x16_hip_moi_exact_shadow                  0.020 ms      0.03 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.075 ms  iters=4924  warmup= 100.621 ms  warmup_iters=4950
fp16_wmma_tiled_w4_4x16_hip_moi_sampled_watchpoint            0.020 ms      0.03 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.137 ms  iters=4977  warmup= 100.004 ms  warmup_iters=4971
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w8_16x8
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=256 N=128 K=16 waves=8 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w8_16x8_noop                                  0.003 ms      0.32 TFLOP/s    0.2% of 191 TFLOP/s  total= 100.165 ms  iters=30888  warmup= 100.068 ms  warmup_iters=30885
fp16_wmma_tiled_w8_16x8_sampled_loom_tsan                     0.006 ms      0.18 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.024 ms  iters=17124  warmup= 100.120 ms  warmup_iters=17084
fp16_wmma_tiled_w8_16x8_hip_moi_exact_shadow                  0.018 ms      0.06 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.500 ms  iters=5682  warmup= 100.062 ms  warmup_iters=5569
fp16_wmma_tiled_w8_16x8_hip_moi_sampled_watchpoint            0.017 ms      0.06 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.317 ms  iters=5817  warmup= 100.014 ms  warmup_iters=5804
```

## 2026-06-23 after backend-specialized sampled hot path

hip-moi commit measured: this commit
sanitizer-strategy benchmark commit: `1e3d398`

This pass added compile-time backend selection overloads for explicit-offset
`lds_load_at` / `lds_store_at` calls and changed sampled lane/slot selection
from 64-bit modulo arithmetic to 32-bit mixing plus power-of-two masking.

Codegen audit on the extracted 2-wave benchmark:

```text
sampled Loom:      69 VGPR, codeLenInByte=7208,  ScratchSize=0, uses_flat_scratch=0
hip-moi exact:     52 VGPR, codeLenInByte=19484, ScratchSize=0, uses_flat_scratch=1
hip-moi sampled:   57 VGPR, codeLenInByte=15200, ScratchSize=0, uses_flat_scratch=1
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=32 N=64 K=16 waves=2 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w2_2x4_noop                                   0.003 ms      0.02 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.363 ms  iters=34814  warmup= 100.192 ms  warmup_iters=34881
fp16_wmma_tiled_w2_2x4_sampled_loom_tsan                      0.005 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 103.202 ms  iters=21203  warmup= 100.106 ms  warmup_iters=20980
fp16_wmma_tiled_w2_2x4_hip_moi_exact_shadow                   0.010 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.030 ms  iters=10089  warmup= 100.400 ms  warmup_iters=10129
fp16_wmma_tiled_w2_2x4_hip_moi_sampled_watchpoint             0.007 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.072 ms  iters=14133  warmup= 100.476 ms  warmup_iters=14126
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w4_4x16
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=64 N=256 K=16 waves=4 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w4_4x16_noop                                  0.003 ms      0.17 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.543 ms  iters=32480  warmup= 100.212 ms  warmup_iters=31930
fp16_wmma_tiled_w4_4x16_sampled_loom_tsan                     0.006 ms      0.09 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.268 ms  iters=16955  warmup= 100.109 ms  warmup_iters=16964
fp16_wmma_tiled_w4_4x16_hip_moi_exact_shadow                  0.016 ms      0.03 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.020 ms  iters=6328  warmup= 100.009 ms  warmup_iters=6328
fp16_wmma_tiled_w4_4x16_hip_moi_sampled_watchpoint            0.011 ms      0.05 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.018 ms  iters=9039  warmup= 100.001 ms  warmup_iters=9039
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w8_16x8
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=256 N=128 K=16 waves=8 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w8_16x8_noop                                  0.003 ms      0.32 TFLOP/s    0.2% of 191 TFLOP/s  total= 100.026 ms  iters=30904  warmup= 100.014 ms  warmup_iters=30908
fp16_wmma_tiled_w8_16x8_sampled_loom_tsan                     0.006 ms      0.18 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.004 ms  iters=17117  warmup= 100.033 ms  warmup_iters=17122
fp16_wmma_tiled_w8_16x8_hip_moi_exact_shadow                  0.015 ms      0.07 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.050 ms  iters=6668  warmup= 100.494 ms  warmup_iters=6644
fp16_wmma_tiled_w8_16x8_hip_moi_sampled_watchpoint            0.011 ms      0.10 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.482 ms  iters=9268  warmup= 100.098 ms  warmup_iters=9246
```

## 2026-06-23 after cached hip-moi benchmark context

hip-moi commit measured: `822635a` (`Specialize sampled backend hot path`)
sanitizer-strategy benchmark commit: `14027b7`

No hip-moi source changed for this entry. The benchmark harness now constructs
the hip-moi context once near kernel entry and passes it through the access
helpers, instead of rebuilding the context inside each instrumented LDS wrapper.
This better matches the intended hand-instrumented source shape.

Codegen audit on the extracted 2-wave benchmark:

```text
sampled Loom:      69 VGPR, codeLenInByte=7208,  ScratchSize=0, uses_flat_scratch=0
hip-moi exact:     61 VGPR, codeLenInByte=15916, ScratchSize=0, uses_flat_scratch=1
hip-moi sampled:   63 VGPR, codeLenInByte=11448, ScratchSize=0, uses_flat_scratch=1
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=32 N=64 K=16 waves=2 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w2_2x4_noop                                   0.003 ms      0.02 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.046 ms  iters=34603  warmup= 100.093 ms  warmup_iters=34635
fp16_wmma_tiled_w2_2x4_sampled_loom_tsan                      0.005 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.135 ms  iters=21194  warmup= 100.062 ms  warmup_iters=20937
fp16_wmma_tiled_w2_2x4_hip_moi_exact_shadow                   0.009 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.067 ms  iters=11212  warmup= 100.294 ms  warmup_iters=11177
fp16_wmma_tiled_w2_2x4_hip_moi_sampled_watchpoint             0.006 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.153 ms  iters=16760  warmup= 100.120 ms  warmup_iters=16686
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w4_4x16
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=64 N=256 K=16 waves=4 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w4_4x16_noop                                  0.003 ms      0.17 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.080 ms  iters=31795  warmup= 100.087 ms  warmup_iters=31786
fp16_wmma_tiled_w4_4x16_sampled_loom_tsan                     0.006 ms      0.09 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.011 ms  iters=16917  warmup= 100.010 ms  warmup_iters=16918
fp16_wmma_tiled_w4_4x16_hip_moi_exact_shadow                  0.014 ms      0.04 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.010 ms  iters=7300  warmup= 100.038 ms  warmup_iters=7299
fp16_wmma_tiled_w4_4x16_hip_moi_sampled_watchpoint            0.009 ms      0.06 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.789 ms  iters=11179  warmup= 100.014 ms  warmup_iters=11180
```

Command:

```bash
./rdna4_matmul/build_w2_2x4_benchmark.sh w8_16x8
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=256 N=128 K=16 waves=8 min_ms=100.0 warmup_ms=100.0
fp16_wmma_tiled_w8_16x8_noop                                  0.003 ms      0.33 TFLOP/s    0.2% of 191 TFLOP/s  total= 100.058 ms  iters=31256  warmup= 100.015 ms  warmup_iters=30904
fp16_wmma_tiled_w8_16x8_sampled_loom_tsan                     0.006 ms      0.18 TFLOP/s    0.1% of 191 TFLOP/s  total= 102.151 ms  iters=17143  warmup= 100.134 ms  warmup_iters=17165
fp16_wmma_tiled_w8_16x8_hip_moi_exact_shadow                  0.013 ms      0.08 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.067 ms  iters=7752  warmup= 100.161 ms  warmup_iters=7674
fp16_wmma_tiled_w8_16x8_hip_moi_sampled_watchpoint            0.009 ms      0.12 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.116 ms  iters=11195  warmup= 100.230 ms  warmup_iters=11150
```

## 2026-06-23 after sampled fairness pass

hip-moi commit measured: this commit
sanitizer-strategy benchmark commit: `0bd939d`

This pass made the sampled benchmark rows more apples-to-apples:

* both sampled rows use `SAMPLED_WATCHPOINTS`, `SAMPLED_SKIP`,
  `SAMPLED_PROBES`, `SAMPLED_DELAY`, and `SAMPLED_REPORTS`;
* the default sampled knobs are `watchpoints=1`, `skip=32`, `probes=1`,
  `delay=32`, `reports=off`;
* row names now say `publish_only` or `reporting`;
* hip-moi generation is overridden from the per-launch runtime generation in
  the benchmark, matching Loom's per-launch generation behavior.
* sampled watchpoint reporting now publishes first and checks the displaced
  watchpoint entry; the watchpoint slot is keyed by watched LDS range, epoch,
  and generation rather than by subgroup or source site.

2-wave calibration matrix, with `probes=1` and `delay=32`:

```text
watchpoints  skip  reports  sampled_loom  hip_moi_sampled
1            1     off      0.007 ms      0.009 ms
1            32    off      0.005 ms      0.005 ms
64           1     off      0.007 ms      0.009 ms
64           32    off      0.005 ms      0.005 ms
1            1     on       0.010 ms      0.012 ms
1            32    on       0.005 ms      0.005 ms
64           1     on       0.009 ms      0.012 ms
64           32    on       0.005 ms      0.005 ms
```

Command:

```bash
HIP_MOI_ROOT=/home/benoit/workspace/hip-moi ./rdna4_matmul/build_w2_2x4_benchmark.sh
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=32 N=64 K=16 waves=2 min_ms=100.0 warmup_ms=100.0
sampled knobs: watchpoints=1 skip=32 probes=1 delay=32 reports=off
fp16_wmma_tiled_w2_2x4_noop                                   0.003 ms      0.02 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.083 ms  iters=35606  warmup= 100.034 ms  warmup_iters=34606
fp16_wmma_tiled_w2_2x4_sampled_loom_publish_only              0.005 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.486 ms  iters=21298  warmup= 100.294 ms  warmup_iters=21360
fp16_wmma_tiled_w2_2x4_hip_moi_exact_shadow                   0.009 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.009 ms  iters=11172  warmup= 100.094 ms  warmup_iters=11166
fp16_wmma_tiled_w2_2x4_hip_moi_sampled_watchpoint_publish_only    0.005 ms      0.01 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.135 ms  iters=20746  warmup= 100.175 ms  warmup_iters=20705
```

Command:

```bash
HIP_MOI_ROOT=/home/benoit/workspace/hip-moi ./rdna4_matmul/build_w2_2x4_benchmark.sh w4_4x16
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=64 N=256 K=16 waves=4 min_ms=100.0 warmup_ms=100.0
sampled knobs: watchpoints=1 skip=32 probes=1 delay=32 reports=off
fp16_wmma_tiled_w4_4x16_noop                                  0.003 ms      0.17 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.020 ms  iters=31762  warmup= 100.010 ms  warmup_iters=31765
fp16_wmma_tiled_w4_4x16_sampled_loom_publish_only             0.006 ms      0.09 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.155 ms  iters=16904  warmup= 100.010 ms  warmup_iters=16815
fp16_wmma_tiled_w4_4x16_hip_moi_exact_shadow                  0.014 ms      0.04 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.243 ms  iters=7211  warmup= 100.594 ms  warmup_iters=7253
fp16_wmma_tiled_w4_4x16_hip_moi_sampled_watchpoint_publish_only    0.008 ms      0.07 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.219 ms  iters=13152  warmup= 100.022 ms  warmup_iters=13129
```

Command:

```bash
HIP_MOI_ROOT=/home/benoit/workspace/hip-moi ./rdna4_matmul/build_w2_2x4_benchmark.sh w8_16x8
```

Output:

```text
device 0: AMD Radeon RX 9070, gcnArch=gfx1201, CUs=28
bench shape: M=256 N=128 K=16 waves=8 min_ms=100.0 warmup_ms=100.0
sampled knobs: watchpoints=1 skip=32 probes=1 delay=32 reports=off
fp16_wmma_tiled_w8_16x8_noop                                  0.003 ms      0.32 TFLOP/s    0.2% of 191 TFLOP/s  total= 100.017 ms  iters=30853  warmup= 100.102 ms  warmup_iters=30884
fp16_wmma_tiled_w8_16x8_sampled_loom_publish_only             0.006 ms      0.18 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.202 ms  iters=17304  warmup= 100.039 ms  warmup_iters=17072
fp16_wmma_tiled_w8_16x8_hip_moi_exact_shadow                  0.013 ms      0.08 TFLOP/s    0.0% of 191 TFLOP/s  total= 100.013 ms  iters=7767  warmup= 100.065 ms  warmup_iters=7772
fp16_wmma_tiled_w8_16x8_hip_moi_sampled_watchpoint_publish_only    0.008 ms      0.13 TFLOP/s    0.1% of 191 TFLOP/s  total= 100.261 ms  iters=12608  warmup= 100.141 ms  warmup_iters=12608
```
