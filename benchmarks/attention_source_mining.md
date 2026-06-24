<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Attention Source Mining

This document records the extraction of two different signals:

* which outer attention shapes look production-representative;
* what a mature implementation chooses for those shapes.

The local source-mining snapshots were:

* llama.cpp: `51eae8c`
* AITER: `de79999`

The goal is not to vendor either implementation. The goal is to explain why the
hip-moi attention benchmarks use their current outer shapes, LDS layouts, and
register-handoff variants.

## AITER: Representative Outer Shapes

AITER is the clearer source for production outer shapes. Its MHA C++ benchmark
README centers the forward benchmark around:

* BF16, BSHD layout, batch mode;
* `hdim_q=128`, `hdim_v=128`;
* query heads 32 or 64;
* KV heads 8 or 4;
* sequence lengths 1024, 2048, 4096, 8192, 10240, plus 16384 for selected
  batch-1 rows;
* causal and no-mask variants;
* batch sizes 1, 4, and 8.

The README's first command-line example is especially direct:

```text
prec=bf16, batch=1, heads=64, dim=128, seq=8192, causal mask
```

Its performance tables then broaden that to the repeated production grid:

```text
batch in {1, 4, 8}
q_heads / kv_heads in {32/8, 64/8, 64/4}
seq_q = seq_kv in {1024, 2048, 4096, 8192, 10240}
head_dim = value_dim = 128
causal in {false, true}
```

The AITER Python tests add breadth rather than a narrower headline target. They
exercise fp16 and bf16, layouts BSHD/BHSD/SBHD/KVPACKED, GQA ratios 1 and 8,
causal and non-causal modes, local attention, dropout, bias, deterministic
mode, odd dimensions, and sequence lengths up to 2048. That is useful for later
correctness coverage, but the C++ README table is the stronger performance
shape signal.

AITER's newer sink/ASM tests are `gfx1250`-specific, so they are not a direct
`gfx1201` RDNA4 import. They still reinforce the same basic production choices:
D64 and D128, GQA, causal mode, long equal Q/K sequences, and packed/varlen
forms.

## llama.cpp: Mature RDNA Dispatch Signal

llama.cpp encodes production choices in `ggml_cuda_get_best_fattn_kernel`. Users
do not manually pick among the instantiated kernels. The dispatch code chooses
based on head dimension, value dimension, query length, KV length, GQA ratio,
mask availability, alignment, data type, target GPU family, and build flags.

For RDNA4, these availability facts are central:

* `amd_wmma_available(cc)` is true for RDNA3/RDNA4.
* `amd_mfma_available(cc)` is false on RDNA and true only for CDNA.
* the older WMMA flash-attention path is enabled on HIP RDNA4 only when
  `GGML_HIP_ROCWMMA_FATTN` is set and rocWMMA major version is greater than 1.

For long-sequence AITER-like prefill shapes, assume:

* `Q->ne[0]` is the head dimension;
* `Q->ne[1]` is the query sequence length, so AITER-like rows are much larger
  than 32;
* `K->ne[1]` is a multiple of `FATTN_KQ_STRIDE`, which is 256;
* strides are 16-byte aligned;
* the causal path has a mask and no ALiBi/max-bias term, so llama's GQA
  optimization applies when `q_heads / kv_heads >= 2`.

Under those assumptions, the headline AITER shapes map as follows.

| Outer shape | GQA ratio | If rocWMMA FATTN is enabled | Otherwise on RDNA4 |
| --- | ---: | --- | --- |
| D128, q_heads=32, kv_heads=8 | 4 | `WMMA_F16<D=128, cols=32>` | `MMA_F16<128,128,ncols1=16,ncols2=4>` |
| D128, q_heads=64, kv_heads=8 | 8 | `WMMA_F16<D=128, cols=32>` | `MMA_F16<128,128,ncols1=8,ncols2=8>` |
| D128, q_heads=64, kv_heads=4 | 16 | `WMMA_F16<D=128, cols=32>` | `MMA_F16<128,128,ncols1=8,ncols2=8>` |
| D64, q_heads=64, kv_heads=8 | 8 | `WMMA_F16<D=64, cols=32>` | `MMA_F16<64,64,ncols1=8,ncols2=8>` |

The RDNA MMA-F16 dynamic LDS estimates for those fallback choices are modest:

| Fallback RDNA MMA-F16 choice | Estimated LDS |
| --- | ---: |
| `MMA_F16<64,64,ncols1=8,ncols2=8>` | about 10.1 KiB |
| `MMA_F16<128,128,ncols1=8,ncols2=8>` | about 18.1 KiB |
| `MMA_F16<128,128,ncols1=16,ncols2=4>` | about 19.2 KiB |

The rocWMMA FATTN path uses static LDS in the WMMA kernel. For default
half-accumulator precision, the rough source-level LDS footprints are:

| WMMA FATTN choice | Estimated static LDS |
| --- | ---: |
| `WMMA_F16<D=64, cols=32>` | about 21 KiB |
| `WMMA_F16<D=128, cols=32>` | about 25 KiB |
| `WMMA_F16<D=256, cols=32>` | about 33 KiB |

If the WMMA path is forced into float accumulation, the KQ scratch doubles and
the D128/D256 footprints rise to roughly 42 KiB and 50 KiB respectively.

The larger llama.cpp template/config list is still useful, but it must not be
read as "these are all equally mature RDNA4 choices." In particular:

* D192/V128, D512/V512, and D576/V512 are represented in validation or template
  machinery, but the RDNA4 dispatch does not obviously select the RDNA MMA-F16
  path for them.
* D256/V256 is a possible WMMA FATTN choice when the rocWMMA path is enabled,
  but it is not the AITER headline shape.
* The RDNA MMA-F16 config table contains entries whose LDS estimates are useful
  pressure signals, but some of those entries are not dispatch-selected on
  `gfx1201`.

## llama.cpp: Score And Weight Scratch Signal

The most important follow-up source-mining result is about where the QK scores
and softmax weights live. The current hip-moi attention benchmarks materialize a
dense score tile and a dense softmax-weight tile in LDS because that makes the
QK-to-softmax-to-PV handoff simple, explicit, and easy to instrument. That is a
good stress test for scalar LDS instrumentation, but it is not obviously what a
mature attention kernel does.

The upstream llama.cpp RDNA MMA-F16 path points in the other direction. In
`ggml/src/ggml-cuda/fattn-mma-f16.cuh`, the QK result is kept in accumulator
fragment arrays such as `KQ_C`; row state is tracked in register arrays such as
`KQ_max`, `KQ_max_scale`, and `KQ_rowsum`; and the QK accumulator fragments are
converted directly into the B operand fragments for the following VKQ/PV MMA
step. The inspected path did not expose dense LDS arrays analogous to hip-moi's
current `scores[query][key]` and `weights[query][key]` scratch.

That made the hip-moi correctness ladder clearer. Before measuring a larger
attention benchmark, the RDNA4 WMMA register-handoff problem had to be isolated:

* QK accumulation currently gives each lane values shaped like
  `qk_acc[elem] -> (query_row=rdna4_wmma_acc_m(lane, elem), key_col=lane & 15)`.
* PV wants a B operand shaped like
  `p_fragment[elem] -> (query_row=lane & 15, key=rdna4_wmma_f16_k(lane, elem))`.
* Bridging those layouts without LDS score/weight scratch requires a
  subgroup-scoped register transpose, likely through lane permutation/shuffle
  operations.

The small surgical test is
`tests/instrumented/013_rdna4_wmma_register_handoff_test.hip`. It proves the
register handoff directly, then feeds the reshaped fragment to a second PV WMMA,
with both exact-context and sampled-fast-context launches. The working lane
exchange uses `ds_bpermute`; a temporary `readlane` probe was not sufficient for
the dynamic cross-half permutation.

The first attention-shaped user of that primitive is
`tests/instrumented/014_rdna4_wmma_no_score_lds_attention_test.hip`. It runs two
key tiles, instruments the remaining K/V LDS staging, keeps the QK-to-PV handoff
in registers, and deliberately omits softmax so the host reference stays small
and exact. The corresponding benchmark is
`benchmarks/014_rdna4_wmma_no_score_lds_attention_benchmark.hip`; the D128/V128
version is
`benchmarks/015_rdna4_d128_no_score_lds_attention_benchmark.hip`.

The D128 dense-score benchmarks remain valuable as scalar LDS instrumentation
stress cases, but they should not be treated as the most production-faithful
attention endpoint.

## Implications For hip-moi

The attention benchmark set separates two questions:

1. Production representativeness: start from AITER-like outer shapes, especially
   BF16/FP16, D128/V128, q_heads=64, kv_heads=8 or 4, long equal Q/K sequences,
   and causal plus no-mask variants.
2. Resource pressure: compile-probe RDNA4 microkernels and record fixed LDS,
   VGPRs, spills, and private segment size before interpreting instrumentation
   overhead.

The source-mined production choices do not automatically saturate 64 KiB of LDS
on `gfx1201`. A mature D128 path looks more like 18-25 KiB of LDS depending on
the llama build path, not 60+ KiB. If Jakub's intended stress case requires
nearly full LDS, then our benchmark should say so explicitly and may need a
production-derived pressure variant rather than a literal llama.cpp-selected
D128 row.

The concrete production-shape target represented in the current suite is:

```text
RDNA4, FP16 or BF16 inputs, D128/V128, q_heads=64, kv_heads=8 or 4,
seq_q=seq_kv at least 8192,
microkernel compile-probed before instrumentation.
```

## Production-Pressure LDS Variant

The latest mining pass sharpened what "production-pressure" should mean. It
should not mean adding unused padding just to reach 64 KiB of LDS. A realistic
pressure variant should spend LDS on data that a flash-attention-like kernel
would plausibly stage or reuse:

* full K/V head-dimension tiles, not one 16-element fragment at a time;
* enough key columns per block to amortize global reads;
* double-buffered K/V staging when the kernel is pipelined;
* score, weight, row max, and row sum scratch only when the algorithm really
  materializes them.

The current D128 benchmark is intentionally small:

| Region | Bytes |
| --- | ---: |
| one K fragment tile | 512 |
| one V fragment tile | 512 |
| score scratch | 2048 |
| softmax weight scratch | 1024 |
| row max/sum scratch | 256 |
| total | 4352 |

For a D128/V128, two-subgroup workgroup with 16 query rows per subgroup, the
first realistic LDS pressure step is to stage complete K and V head-dimension
tiles for a 16-key tile:

```text
K full tile = 8 fragments * 32 lanes * 16 bytes = 4096 bytes
V full tile = 8 fragments * 32 lanes * 16 bytes = 4096 bytes
score scratch = 2 subgroups * 16 queries * 16 keys * 4 bytes = 2048 bytes
weight scratch = 2 subgroups * 16 queries * 16 keys * 2 bytes = 1024 bytes
row scratch = 2 subgroups * 16 queries * 2 values * 4 bytes = 256 bytes
single-buffer total = 11520 bytes
double-buffered K/V total = 19712 bytes
```

The double-buffered total is about 19.25 KiB, which is close to the llama.cpp
RDNA MMA-F16 D128 fallback estimate. That makes it a good next "literal
production-inspired" benchmark target: it is still far from 64 KiB, but it is
using LDS for the same reasons a mature implementation does.

If Jakub specifically wants a near-LDS-pressure benchmark, the cleanest variant
is not padding, but a wider key tile:

```text
K full tile, 32 keys = 8192 bytes
V full tile, 32 keys = 8192 bytes
double-buffered K/V = 32768 bytes
score scratch = 2 subgroups * 16 queries * 32 keys * 4 bytes = 4096 bytes
weight scratch = 2 subgroups * 16 queries * 32 keys * 2 bytes = 2048 bytes
row scratch = 256 bytes
total = 39168 bytes
```

That is about 38.25 KiB before any extra alignment, mask, or float-accumulator
scratch. It is less literal than the 16-key tile, but it is still derived from
real flash-attention tiling pressure: larger K/V staging, a larger score tile,
and pipelined reuse. This should be labelled a D128 production-pressure variant,
not a llama.cpp clone.

The implemented benchmark mapping is:

1. `attention-d128-dense` keeps the 4352 B D128 shape as the scalar-scratch
   instrumentation stress row.
2. `attention-d128-pressure-full-kv16` uses D128 full-K/V double-buffering and
   targets about 19 KiB of LDS.
3. `attention-d128-pressure-wide-k32` uses a wider 32-key tile and targets about
   38 KiB of LDS.
4. `attention-d128-no-score` keeps the QK-to-PV handoff in registers and uses
   LDS only for K/V staging.

Each benchmark uses the familiar rows when applicable:

* pass-through;
* Jakub-Sampled-Loom-style;
* `hip_moi::context + sampled_watchpoint`, if still informative;
* `hip_moi::sampled_watchpoint_context`.
