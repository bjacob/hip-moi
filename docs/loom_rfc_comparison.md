<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Loom And RFC Comparison

This document maps hip-moi concepts to three comparison points:

* the LLVM compiler-rt GPU Thread/ConcurrencySanitizer RFC;
* HRX/Loom AMDGPU TSAN as summarized in Jakub's `sanitizer-strategy` notes;
* Jakub-Sampled-Loom, the local HIP benchmark policy extracted from Jakub's
  RDNA4 matmul harness.

The goal is not to claim equivalence. The goal is to make discussion precise:
which tool records which metadata, what it can diagnose, where sampling enters,
and what false-negative sources are being accepted.

## Source Basis

The comparison is based on local materials in this workspace:

| Source | Role |
| --- | --- |
| `/home/benoit/workspace/sanitizer-strategy/rfc-thread-concurrency-sanitizer-gpus-compiler-rt.md` | Saved copy of the compiler-rt RFC text. |
| `/home/benoit/workspace/sanitizer-strategy/sanitizer_landscape.md` | Local survey of RFC, HRX/Loom, DBI, and other GPU sanitizer directions. |
| `/home/benoit/workspace/sanitizer-strategy/rdna4_matmul/README.md` | Jakub's explanation of the RDNA4 HIP harness policies. |
| `/home/benoit/workspace/sanitizer-strategy/rdna4_matmul/rdna4_matmul.hip` | Jakub's large self-contained HIP implementation. |
| `benchmarks/*benchmark.hip` | hip-moi's focused, vendored benchmark extracts. |
| `docs/instrumentation_model.md` and `docs/atomics.md` | Current hip-moi implementation model. |

The HRX submodule checkout in this workspace is not currently populated as a
normal source tree, so this document does not cite HRX source file paths. It
uses Jakub's local summaries and benchmark extraction as the grounded reference
for Loom concepts.

## Names

**RFC** means the compiler-rt proposal titled "A Thread/Concurrency Sanitizer
for GPUs in compiler-rt." It uses the familiar `-fsanitize=thread` spelling but
describes a KCSAN-like ConcurrencySanitizer approach, not CPU TSan-style
happens-before tracking.

**Real Loom** means the HRX/Loom compiler and runtime path summarized in
`sanitizer-strategy`. It inserts `sanitizer.race.access` and
`sanitizer.race.sync` observations and uses queue/runtime state supplied by the
HRX/IREE stack.

**Jakub-Sampled-Loom** means the local HIP benchmark policy in this repository's
benchmark files. It borrows the Loom epoch/generation/LDS-range idea but is a
handwritten, benchmark-local implementation. It is not upstream Loom and not a
compiler pass.

**hip-moi** means this source-level HIP instrumentation library. It sees only
operations rewritten to hip-moi API calls.

## High-Level Matrix

| Axis | RFC | Real Loom | Jakub-Sampled-Loom | hip-moi `context` | hip-moi `sampled_watchpoint_context` |
| --- | --- | --- | --- | --- | --- |
| Instrumentation level | Compiler-rt sanitizer hooks. | Loom compiler IR plus HRX/IREE runtime. | Handwritten HIP benchmark helpers. | Source-level HIP API calls. | Source-level HIP API calls. |
| Primary memory target | Global memory in the current RFC. | Workgroup/LDS memory in current AMDGPU lowering. | LDS ranges in benchmark kernels. | LDS payload races; global/LDS atomics as sync. | LDS range publication only. |
| Race owner identity | Wave-selected sampling context. | Linear workitem id. | Local workitem/lane-derived id in packed metadata. | Subgroup id. | Subgroup id. |
| Main metadata | Ephemeral watchpoints. | 64-bit shadow entry per LDS cell. | Global sampled watchpoint table. | Exact shadow, sampled watchpoints, atomic records. | Global sampled watchpoint table. |
| Synchronization model | Sampling, not race-freedom proof. | Barrier epochs from workgroup sync observations. | Barrier epochs from helper calls. | Barrier epochs plus supported atomic release/acquire tokens. | Local barrier epoch only; no diagnostics. |
| Checking time | During sampled observation windows. | At each instrumented LDS access. | At sampled accesses when report counting is enabled. | At each instrumented LDS access. | No checking. |
| Reports races? | Yes, probabilistically. | Yes, deterministically for represented LDS accesses. | Only benchmark report counters. | Yes. | No. |
| False positives | RFC claims none for represented races. | Intended deterministic exactness for represented LDS cells. | None at dword-cell granularity for checked sampled records. | Exact-shadow dword-cell granularity; address-only atomics can suppress reports, not create them. | Not applicable; no reports. |
| False negatives | Intentional sampling misses. | Unsupported code or unrepresented synchronization. | Sampling, table overwrite, reduced probes. | Unsupported raw operations, same-subgroup races, address-only atomics over-match. | All races; it is publish-only. |

## Metadata Mapping

Real Loom and hip-moi exact shadow are closest in shape. Both encode an access
record with:

* access kind;
* owner identity;
* barrier epoch;
* dispatch or launch generation;
* source site information;
* an LDS location normalized to dword-cell granularity.

The important difference is the owner. Real Loom uses a linear workitem id,
while current hip-moi uses a subgroup id. This is intentional: hip-moi is
optimized for the Loom-comparison and DBI direction where cross-subgroup LDS
sharing is the primary target and same-subgroup races are demoted.

Jakub-Sampled-Loom and hip-moi sampled watchpoints are closest in fast-path
shape. Both publish packed global records for LDS dword-cell ranges rather than
maintaining a full per-cell shadow table. Both use generation and epoch to keep
records from different launches or barrier intervals separate. Both can miss
races because publication is sampled and table slots can be overwritten.

The RFC also uses watchpoints, but its current target is global memory and its
reporting model includes a delay window and value-change detection. hip-moi's
sampled rows should not be described as an implementation of the RFC. They are
LDS-focused measurements of watchpoint-style publication/checking overhead.

## Synchronization Mapping

Real Loom's core synchronization primitive is the workgroup sync observation:
the epoch changes at a workgroup barrier, so accesses before and after the
barrier do not compare as same-epoch races.

hip-moi supports the same barrier idea through:

```c++
ctx.syncthreads();
```

and through the lower-level full-workgroup spelling:

```c++
ctx.release_fence(scope, site);
ctx.barrier(site);
ctx.acquire_fence(scope, site);
```

hip-moi adds source-level atomics modeling in `hip_moi::context`. A release
operation stores an address-scoped record containing the producer subgroup and
epoch. An acquire operation imports producer epochs for that address into the
pairwise acquired-token table. Later LDS conflict checks suppress a raw
same-epoch conflict only when the current subgroup has acquired the prior
subgroup's recorded epoch.

This atomics model is a hip-moi extension relative to the current Loom-focused
benchmark rows. It is deliberately address-scoped, not value-scoped. That makes
it closer to a DBI-friendly synchronization-object model but can create false
negatives when unrelated releases reuse the same atomic address.

## Diagnostic Capability

`hip_moi::context` is the diagnostic path. It can emit deterministic diagnostics
for represented cross-subgroup LDS conflicts, metadata saturation, and barrier
divergence. It also supports sampled-watchpoint reporting, but that mode is
currently semantic coverage rather than a performance target.

`hip_moi::sampled_watchpoint_context` is not a diagnostic path. It exists to
measure the cost of publishing sampled LDS metadata with as little context
state as possible. It is the hip-moi path most comparable to
Jakub-Sampled-Loom's publish-only performance row, but it cannot prove a race
or report one.

The RFC is diagnostic but probabilistic: a report is meaningful, while silence
is not evidence of race freedom. Real Loom is the deterministic LDS diagnostic
reference for represented accesses but requires compiler/runtime integration
that hip-moi intentionally does not assume.

## Performance Interpretation

The most important performance distinction is whether the path reports races.

Publish-only rows can keep far less state live. That is why
`sampled_watchpoint_context` can approach or beat Jakub-Sampled-Loom on narrow
matmul rows. It drops diagnostic storage, dynamic backend selection, saturation
reporting, and global subgroup epoch storage.

Diagnostic rows pay for a stronger contract. `hip_moi::context` must carry
diagnostic pointers, capacities, epoch state, exact-shadow or sampled-reporting
logic, and atomics metadata when the kernel uses atomic synchronization. That
extra live state is exactly the kind of overhead that can push production
kernels into VGPR pressure or spills.

The benchmark suite should therefore be read in two lanes:

* `sampled_watchpoint_context` versus Jakub-Sampled-Loom answers whether the
  publish-only fast path can be competitive with Jakub's sampled Loom-like
  design.
* `context` and exact-shadow rows answer how expensive deterministic or
  diagnostic-capable semantics are when carried through a source-level HIP API.

Those lanes should not be collapsed into one "hip-moi versus Loom" number.

## Implications For The RFC Discussion

The RFC and hip-moi are complementary. The RFC is a plausible low-overhead CI
path for global-memory races, with intentional sampling false negatives. hip-moi
is currently an LDS-focused prototype for explicit source instrumentation,
subgroup-scoped metadata, and atomics-aware ordering.

The key lesson hip-moi can contribute to the RFC/Loom discussion is that
generated-code pressure matters as much as algorithmic complexity. Production
RDNA4 kernels are often already close to VGPR limits. A sanitizer path that
keeps extra pointer/state values live can be slower because it spills, even if
the abstract metadata operation count looks modest.

The second lesson is that reporting semantics and publish-only metadata are
different products. A very fast sampled publisher is useful for Loom-parity
experiments and DBI planning, but it is not a sanitizer until it has a reporting
path and a clear statement of false negatives.

## Open Comparison Gaps

The next comparison work should answer these questions:

* Which real Loom source files correspond to the summarized
  `sanitizer.race.access` lowering and feedback packets once the HRX submodule
  is fully populated?
* Which exact instruction-level metadata is visible to a rocjitsu DBI pass for
  LDS addresses, barrier operations, and atomics?
* Can a DBI path reproduce the cheap publish-only watchpoint state without the
  source-level API carrying extra VGPR pressure?
* Should hip-moi keep source-level address-only atomics as the DBI bridge, or
  should value-sensitive atomics be explored once a concrete protocol demands
  the precision?
