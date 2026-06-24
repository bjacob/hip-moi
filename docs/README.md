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

5. [Benchmark README](../benchmarks/README.md)

   Gives the current RDNA4 measurements and resource-pressure tables.

## Documentation Policy

Facts that Jakub needs for Loom, RFC, or performance discussions should be
defined here, not left only in `PLAN.md` or commit history. Worklog-style
observations should either be deleted or rewritten as precise claims with the
relevant assumptions, measurements, and file paths.
