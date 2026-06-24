<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Attention Source Mining

This note records the current extraction of two different signals:

* which outer attention shapes look production-representative;
* what a mature implementation chooses for those shapes.

The local source-mining snapshots were:

* llama.cpp: `51eae8c`
* AITER: `de79999`

The goal is not to vendor either implementation. The goal is to choose the next
hip-moi attention benchmark shape with less guesswork.

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

## Implications For hip-moi

The next benchmark should separate two questions:

1. Production representativeness: start from AITER-like outer shapes, especially
   BF16/FP16, D128/V128, q_heads=64, kv_heads=8 or 4, long equal Q/K sequences,
   and causal plus no-mask variants.
2. Resource pressure: compile-probe candidate RDNA4 microkernels before
   instrumentation and record fixed/dynamic LDS, VGPRs, spills, and private
   segment size.

The source-mined production choices do not automatically saturate 64 KiB of LDS
on `gfx1201`. A mature D128 path looks more like 18-25 KiB of LDS depending on
the llama build path, not 60+ KiB. If Jakub's intended stress case requires
nearly full LDS, then our benchmark should say so explicitly and may need a
production-derived pressure variant rather than a literal llama.cpp-selected
D128 row.

The first concrete benchmark target should therefore be:

```text
RDNA4, FP16 or BF16 inputs, D128/V128, q_heads=64, kv_heads=8 or 4,
seq_q=seq_kv at least 8192, causal first, no-mask second,
microkernel compile-probed before instrumentation.
```

Only after that probe should hip-moi add the familiar rows:

* noop;
* sampled Loom-style;
* `hip_moi::context + sampled_watchpoint`, if still informative;
* `hip_moi::sampled_watchpoint_context`.
