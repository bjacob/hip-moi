<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# hip-moi Documentation

This directory is the durable documentation for hip-moi. It is written for
reviewers who need to understand exactly what the prototype records, what it
can diagnose, what it only measures, and how the benchmark results should be
interpreted.

## Reading Order

For the atomics delivery discussion, read [Atomics Model](atomics.md) first,
then [Atomics Fast-Path Notes](atomics_fast_paths.md), then the atomics rows in
the [Benchmark README](../benchmarks/README.md). The staged plan and corpus
notes are supporting material, not the main delivery narrative.

1. [Instrumentation Model](instrumentation_model.md)

   Defines the terminology, metadata records, access-time algorithms,
   diagnostic conditions, sampling mechanism, and the difference between the
   diagnostic and publish-only paths.

2. [Context Allocation](context.md)

   Explains `hip_moi::host_context`, `hip_moi::context`,
   `hip_moi::sampled_watchpoint_context`, storage allocation, sampled policy
   options, and example usage.

3. [Tutorial](tutorial/README.md)

   Shows small compilable examples. These are useful for seeing the surface API
   before reading larger tests or benchmarks.

4. [Scope And Fast Paths](scope.md)

   States the active scope boundaries, the diagnostic versus publish-only split,
   and the next semantic expansion.

5. [Ping-Pong Scheduling And `setprio`](pingpong.md)

   Records the current source survey and explains why `setprio` is a
   scheduling operation, not a memory-ordering operation, for hip-moi's current
   detector. Also records the current RDNA4 ping-pong test shapes,
   generated-code inspection workflow, and ATT trace smoke test.

6. [Atomics Model](atomics.md)

   Defines the current source-level atomics support: public API, release
   records, acquired epoch tokens, paired fences, address-only precision,
   fast-path limits, and current performance/resource interpretation.

7. [Loom And RFC Comparison](loom_rfc_comparison.md)

   Maps hip-moi concepts to the compiler-rt RFC, real HRX/Loom as summarized
   in Jakub's local materials, and the benchmark-local Jakub-Sampled-Loom
   policy.

8. [Benchmark Interpretation](benchmark_interpretation.md)

   Explains what the benchmark modes and current RDNA4 results imply for Loom,
   the RFC, atomics, ping-pong, and future DBI work.

9. [Atomics Support Plan](atomics_plan.md)

   Gives the staged implementation plan for user-kernel atomics: reference
   kernels, public API, atomic metadata, release/acquire handoffs, RMW
   operations, fence-plus-atomic patterns, Stream-K integration, and DBI seeds.

10. [Atomics Corpus](atomics_corpus.md)

   Tracks the seed kernels for the atomics plan.

11. [Atomics Fast-Path Notes](atomics_fast_paths.md)

   Interprets the Stream-K-shaped atomics benchmark rows, the implemented
   direct RMW cache, and the Stage 17 decision to stop pursuing generic
   acquire-loop trims without new evidence.

12. [DBI Atomic Instruction Seeds](dbi_atomic_seeds.md)

   Records atomic instruction seeds for the future rocjitsu DBI direction and
   keeps them separate from source-level HIP atomics semantics.

13. [Benchmark README](../benchmarks/README.md)

   Gives the current RDNA4 measurements and resource-pressure tables.

## Documentation Policy

Facts that Jakub needs for Loom, RFC, or performance discussions should be
defined here, not left only in `PLAN.md` or commit history. Worklog-style
observations should either be deleted or rewritten as precise claims with the
relevant assumptions, measurements, and file paths.
