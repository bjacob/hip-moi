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
