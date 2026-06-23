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
