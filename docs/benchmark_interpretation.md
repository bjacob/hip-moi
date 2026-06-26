<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Benchmark Interpretation

This document explains how to read hip-moi's benchmark suite. The raw benchmark
catalog and RDNA4 tables live in [`benchmarks/README.md`](../benchmarks/README.md).
This document is the interpretation layer: what the rows are meant to prove,
which comparisons are apples-to-apples, and what the current data suggests for
Loom, the RFC, and future DBI work.

## Benchmark Modes

The main benchmark modes are:

| Mode | Interpretation |
| --- | --- |
| `pass-through` | Baseline kernel shape with no hip-moi or Jakub-Sampled-Loom instrumentation. |
| `Jakub-Sampled-Loom` | Benchmark-local HIP implementation of Jakub's sampled Loom-like policy. It is not upstream Loom. |
| `sampled_watchpoint_context` | hip-moi's narrow publish-only fast path. This is the closest hip-moi comparison to Jakub-Sampled-Loom. |
| `context + sampled_watchpoint` | General diagnostic-capable hip-moi context using sampled watchpoints. This carries more state than the publish-only context. |
| `exact shadow` | General hip-moi context using the exact dword-cell shadow table. |
| `context` | General hip-moi context for atomics benchmarks. It models diagnostics and source-level synchronization. |

The benchmark suite intentionally separates publish-only cost from diagnostic
cost. A publish-only row that runs fast is useful, but it is not a sanitizer
unless a separate reporting path exists and its false-negative behavior is
defined.

## Resource Metrics Matter

Latency is not enough. The benchmark tables also track:

* LDS bytes used by the user kernel;
* pass-through VGPR pressure;
* instrumentation-row VGPR pressure when relevant;
* spills and private segment size;
* SGPR pressure for atomics rows.

VGPR pressure is the most important generated-code warning sign. Production
matrix and attention kernels often use nearly the entire VGPR budget before
instrumentation. A small number of additional live values in the instrumentation
path can trigger spills and therefore global-memory traffic. SGPR pressure still
matters, but it is more negotiable on the current workloads.

## Matmul Rows

The small matmul wave rows are latency guardrails. They are useful because all
instrumentation overhead is exposed: the user work is tiny, so a few extra
instructions or live values show up immediately.

The production-shaped `matmul-prod-16x8` row is the main Loom-parity result.
The current table shows:

* `pass-through`: 1.16 ms;
* `Jakub-Sampled-Loom`: 8.70 ms;
* `context + sampled_watchpoint`: 26.2 ms;
* `sampled_watchpoint_context`: 3.35 ms.

The interpretation is narrow but important: when hip-moi uses a publish-only
context that keeps little state live, the cost can be below Jakub's local
sampled Loom-like policy on this extracted matmul. The same result does not say
that diagnostic hip-moi is faster than Loom. It says the publish-only API shape
can be compiled into a competitive hot path.

## Attention Rows

The attention rows split into two families.

Dense attention rows materialize score and softmax-weight scratch in LDS. They
stress scalar LDS instrumentation and are useful for finding overhead cliffs,
but source mining suggests mature attention implementations often avoid this
dense LDS handoff.

No-score attention rows are more production-faithful for the current direction:
K/V fragments are staged through LDS, while the QK-to-PV handoff stays in
registers. These rows are less dramatic than the dense rows but more useful for
predicting realistic instrumentation overhead.

The high-pressure D128 rows deliberately push LDS and VGPR usage closer to the
regime Jakub expects to matter. They should be used to catch spill regressions.
They are not evidence that every production attention kernel uses that exact
LDS layout.

The attention LDS-alias handoff row is a different kind of attention benchmark:
it is a production-hazard extraction. It measures the synchronized case where a
reusable LDS slot is read as one logical tile and then written as the next
logical tile. The paired tests, not the benchmark row, cover the missing-barrier
diagnostic.

## Ping-Pong Rows

The ping-pong rows are scheduling guardrails, not headline sanitizer results.
They answer a different question: can hip-moi instrumentation coexist with the
`setprio`/`sched_barrier`/WMMA scheduling idiom without invalidating the kernel
shape being measured?

The private-LDS row has ATT validation for complementary per-SIMD priority
signatures. The wide cooperative row adds real cross-subgroup LDS sharing. New
ping-pong variants should pass generated-code and ATT checks before their
latencies are treated as meaningful.

## Atomics Rows

Atomics rows currently use `hip_moi::context`, not the publish-only sampled
context. That is intentional: source-level atomics are synchronization
semantics, and hip-moi must keep the release/acquire metadata exhaustive for
diagnostics.

The small atomics microbenchmarks mostly run around 7 to 9 microseconds
through `context` versus about 3 microseconds pass-through. The Stream-K-shaped
integration rows range from about 12.4 to 45.2 microseconds through `context`.
The current atomics resource refresh found no spills. That is a good sign:
atomics overhead is currently metadata-protocol cost, not VGPR-spill collapse.

The main atomics fast path is narrow: multi-subgroup release-capable RMWs
populate a direct-mapped producer-mask cache. This helps the four-subgroup
Stream-K-tree `atomicOr` row but is not a general solution to atomics overhead.
The latest acquire-path audit also rejected two generic local shortcuts, so the
next atomics speedup should be protocol-aware or DBI-informed.

## What The Current Data Says

Current evidence supports these claims:

* A source-level publish-only sampled path can be competitive with
  Jakub-Sampled-Loom on focused RDNA4 matmul rows.
* Diagnostic-capable source-level paths carry materially more state and should
  be judged separately from publish-only paths.
* Dense scalar LDS attention rows are useful stress tests but should not be
  treated as the only production-representative attention shape.
* Atomics support is semantically broad enough for current Stream-K-shaped
  source tests, and the current rows remain spill-free.
* The remaining atomics cost is mostly global metadata traffic, table probing,
  bounded acquire retries, and exact-shadow LDS instrumentation.
* Ping-pong measurements are only meaningful when instruction scheduling is
  separately validated.

Current evidence does not support these stronger claims:

* that `sampled_watchpoint_context` is a race detector;
* that hip-moi diagnostic paths are faster than real Loom;
* that a clean sampled run proves race freedom;
* that current source-level results automatically predict DBI overhead;
* that address-only atomics are precise enough for all protocols.

## Recommended Use

For Loom-parity performance discussion, use:

* `matmul-prod-16x8`;
* `matmul-wave-w2/w4/w8`;
* the no-score attention rows;
* high-pressure D128 attention rows when spill risk is the focus.

For synchronization semantics discussion, use:

* atomics microbenchmarks for individual release/acquire and fence patterns;
* Stream-K flag, arrival-counter, and bitmask rows for integration cases;
* the paired instrumented tests under `tests/instrumented/`.

For DBI planning, use:

* `docs/dbi_atomic_seeds.md` for atomic instruction seeds;
* ping-pong ATT validation for scheduling-sensitive kernels;
* the benchmark-local Jakub-Sampled-Loom and `sampled_watchpoint_context` rows
  as evidence about low-state publish-only metadata, not as final DBI design.

## Next Benchmark Questions

The next useful benchmark questions are:

* Can a diagnostic sampled path be made useful without scanning too many
  watchpoint slots?
* Which atomics protocol is the next real bottleneck once exact-shadow LDS
  instrumentation cost is separated from atomic metadata cost?
* Which production attention shapes from upstream projects should replace or
  supplement the current high-pressure synthetic variants?
* Can a rocjitsu DBI prototype publish the same watchpoint metadata with lower
  VGPR pressure than source-level HIP calls?
