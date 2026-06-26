<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# DBI Transition Brief

This document translates hip-moi's source-level HIP prototype into requirements
and first experiments for a future rocjitsu dynamic binary instrumentation
(DBI) effort.

It does not change hip-moi's source-level contract. hip-moi still reasons about
HIP programs in the HIP/LLVM memory model, only sees calls made through its API,
and diagnoses cross-subgroup LDS payload conflicts. The DBI effort is a
separate future track where rocjitsu observes AMDGPU instructions and the model
is the hardware execution model.

## Delivery Summary

The source-level prototype taught three useful lessons for DBI:

* The useful diagnostic payload is still LDS address ranges. Source-level
  hip-moi requires explicit LDS byte offsets, but rocjitsu should derive those
  offsets from decoded LDS instructions and effective addresses.
* Low-state publication can be fast. `sampled_watchpoint_context` shows that
  sampled LDS-range publication can be competitive when the hot path carries
  little live state.
* Generic atomics-aware diagnostics are semantically useful but expensive.
  Source-level address-scoped release records are spill-free in current RDNA4
  rows, but global metadata traffic and table probes dominate the atomics rows.

The DBI hypothesis is that rocjitsu can keep more metadata in emulator-side
state instead of source-level kernel registers. If true, it can preserve the
important recorded facts while avoiding much of the VGPR pressure and spill
risk that source-level instrumentation exposes.

## Model Boundary

Source-level hip-moi and future DBI must not be mixed into one memory model.

| Axis | Source-level hip-moi | Future rocjitsu DBI |
| --- | --- | --- |
| Program representation | HIP source rewritten to hip-moi API calls. | AMDGPU instructions observed by the emulator. |
| Memory model | HIP/LLVM concurrent memory model. | Hardware execution model to be specified from AMDGPU/rocjitsu semantics. |
| What instrumentation sees | Only explicit hip-moi calls. | Instruction opcode, operands, effective addresses, dynamic wave/lane state, and scheduling state. |
| LDS address representation | Caller-provided `lds_byte_offset`. | Effective LDS byte address computed from the instruction. |
| Valid hardware facts | Not used to justify source-level correctness. | Wave lockstep, lane masks, queue behavior, instruction ordering, and cache/scope bits may be part of the model. |
| Main comparison point | Loom/RFC source or compiler instrumentation concepts. | Hardware-level Loom-like or sanitizer instrumentation in rocjitsu. |

The DBI track can legitimately use GPU hardware facts that the source-level
HIP/LLVM model does not use. That is the main reason this brief is separate
from [`instrumentation_model.md`](instrumentation_model.md) and
[`atomics.md`](atomics.md).

## Facts To Preserve

A DBI design can change data structures, but it should preserve the observable
facts needed to explain diagnostics.

For each LDS access candidate:

* dynamic workgroup identity;
* dynamic subgroup or wave identity;
* lane mask of participating lanes, when the instruction is vector-wide;
* LDS byte range per lane or per coalesced group;
* access kind: load, store, read-modify-write, or other represented memory
  operation;
* epoch or ordering interval;
* source location when available, or instruction address when source location
  is unavailable.

For synchronization:

* full-workgroup barrier observations that split epochs;
* fence instructions and their scope bits;
* global or LDS atomic instruction address;
* atomic opcode and whether the instruction returns an old value;
* old and new values when they are needed for a protocol-specific proof;
* producer and consumer subgroup or wave identities;
* ordering evidence that explains why an apparent LDS conflict is suppressed.

For diagnostics:

* the two conflicting LDS ranges;
* the two owners, expressed in hardware terms such as wave and lane mask;
* the ordering interval in which the conflict was observed;
* the instruction addresses or source sites that created the records;
* whether the conflict was suppressed by a tracked synchronization edge.

## What Can Change At DBI Level

DBI does not need to preserve hip-moi's exact source-level storage layout.

The following source-level implementation choices are candidates for
replacement:

* `hip_moi::context::storage_ref`;
* caller-provided LDS byte offsets;
* global-memory exact shadow tables updated by kernel instructions;
* per-kernel diagnostic buffers;
* source-level `site_id` as the only call-site identity;
* generic source-level atomic-object tables.

Rocjitsu can instead keep side metadata keyed by dynamic execution state:

```text
(dispatch, workgroup, wave, instruction address, lane mask, LDS byte range)
```

That does not make the source-level prototype obsolete. It tells the DBI design
which facts were necessary and which source-level costs came from representing
those facts inside HIP code.

## LDS Address Reconstruction

Source-level hip-moi currently requires:

```c++
ctx.lds_store_at(ptr, value, /*lds_byte_offset=*/offset, HIP_MOI_SITE_ID());
```

The pointer performs the actual access; `lds_byte_offset` is the address
recorded in the shadow metadata. This is an API compromise: source-level HIP
does not give hip-moi a portable way to recover the LDS byte offset from an
arbitrary pointer without compiler help.

DBI should not need this second argument. At the instruction level, rocjitsu
should see LDS instructions and their address operands. The first DBI milestone
is therefore:

1. decode LDS load/store instructions in a minimal kernel;
2. compute the effective LDS byte address for each active lane;
3. normalize that address to the same dword-cell space used by hip-moi exact
   shadow;
4. compare the recovered ranges against the explicit offsets used by the
   matching hip-moi test.

This experiment is the bridge from `lds_load_at`/`lds_store_at` to a real
binary instrumentation path.

## Atomics Lessons For DBI

The current source-level atomics model is address-scoped:

```text
(atomic address, producer subgroup, producer epoch, launch generation)
```

Acquire operations import all producer records for the same atomic address.
This avoids keeping scalar atomic values live in the source-level
instrumentation path. It also creates false negatives when an acquire imports a
release that it did not actually observe.

For DBI, address-only metadata is the baseline, not necessarily the endpoint.
Rocjitsu can often observe more:

* the atomic opcode;
* the effective atomic address;
* the old value returned by RMW instructions;
* the value being stored or combined;
* scope/cache bits encoded on the instruction;
* surrounding fence and barrier instructions;
* lane masks and wave identity.

That makes two DBI strategies worth separating:

| Strategy | Metadata key | Benefit | Risk |
| --- | --- | --- | --- |
| Address-only | Atomic effective address. | Closest to current hip-moi model; compact; easy first implementation. | Over-imports releases for reused addresses, causing false negatives. |
| Protocol-aware | Address plus opcode/value/control-flow facts for selected protocols. | Can avoid some address-only false negatives and skip generic table scans. | Requires per-protocol proof and careful treatment of collisions or unsupported shapes. |

The source-level atomics performance audit is important here. It rejected small
generic acquire-loop trims. That suggests DBI should not spend its first effort
recreating the same generic table protocol faster. The better first target is a
specific protocol such as Stream-K arrival counters or Stream-K-tree bitmasks.

## Sampling And Watchpoints

`sampled_watchpoint_context` is the main source-level evidence for low-state
metadata publication. It is publish-only: it records sampled LDS ranges but
does not report races.

For DBI, this suggests a two-step plan:

1. Reproduce publish-only sampled LDS range publication in rocjitsu side state.
2. Add reporting only after the side-state representation and miss sources are
   explicit.

The reporting contract must state its false negatives. Silence from a sampled
tool is not proof of race freedom. A report should still explain the two LDS
accesses, their owners, and the synchronization evidence that failed to order
them.

## Performance Constraints

The source-level benchmarks identify the constraints that matter most:

* VGPR pressure is the critical warning sign for production kernels.
* SGPR pressure matters but is more negotiable on the current RDNA4 rows.
* Spills and private segment growth are more important than raw instruction
  count once a kernel is near its VGPR limit.
* Global metadata traffic can dominate atomics-aware diagnostics even when
  there are no spills.

DBI in rocjitsu may move metadata traffic into emulator-side state and avoid
source-level VGPR pressure. That is the main reason the DBI path is promising.
If a later DBI implementation injects code back into GPU execution rather than
only emulating, the VGPR and spill lessons apply again.

## First DBI Experiments

The first experiments should be small and auditable. Each experiment should
produce an uninstrumented kernel, a disassembly or decoded-instruction record,
and a statement of which metadata facts rocjitsu recovered.

1. **LDS Address Reconstruction**

   Use a tiny two-subgroup LDS handoff and one RDNA4 WMMA-shaped LDS test.
   Decode LDS instructions, compute per-lane LDS byte ranges, and compare them
   with the explicit offsets used by the corresponding hip-moi tests.

2. **Instruction Classification**

   Use [`dbi_atomic_seeds.md`](dbi_atomic_seeds.md). Start with
   `hipkittens-buffer-pk-add-bf16` because the inline assembly gives an
   explicit `buffer_atomic_pk_add_bf16` seed. The goal is to classify workload
   atomics separately from hip-moi detector atomics.

3. **Stream-K Arrival Counter**

   Use the uninstrumented counterpart of
   `rdna4-wmma-streamk-arrival-counter`. Recover the global atomic add
   instruction, the participating wave, the returned old value when available,
   and the surrounding fence/barrier structure.

4. **Stream-K-Tree Bitmask**

   Use the uninstrumented counterpart of `rdna4-wmma-streamk-tree-atomic-or`.
   Recover the `atomicOr` old-value-dependent control flow and test whether a
   protocol-aware representation can avoid the generic source-level table
   scans.

5. **Publish-Only Watchpoint Prototype**

   Implement rocjitsu-side sampled LDS range publication for a matmul or
   no-score attention row. Compare the metadata facts to
   `sampled_watchpoint_context`, but do not call it a sanitizer until reporting
   semantics are defined.

6. **First Reporting Prototype**

   Add deterministic reporting for one unsynchronized cross-subgroup LDS
   conflict. The report should name instruction addresses, LDS ranges, wave or
   subgroup owners, lane masks, and the missing synchronization evidence.

## Open Design Questions

The next DBI design discussion should answer these questions before building a
large runtime:

* What exactly is the hardware-level ordering model that rocjitsu will use for
  barriers, fences, atomics, and wave execution?
* Is subgroup identity always hardware wave identity for the target kernels, or
  does DBI need a more general owner abstraction?
* How should DBI represent lane masks and coalesced LDS ranges without losing
  the ability to explain diagnostics?
* Which protocols deserve protocol-aware synchronization metadata first:
  Stream-K arrival counters, Stream-K-tree bitmasks, ping-pong staging, or
  something from a newer production corpus?
* When does address-only atomic joining create unacceptable false negatives?
* Can value-sensitive or protocol-sensitive atomic metadata be represented
  without reintroducing the source-level VGPR-pressure problem?
* How should DBI preserve source correlation when debug info is unavailable and
  only instruction addresses are reliable?
* Which measurements replace source-level VGPR pressure for emulator-side DBI
  overhead, and which measurements remain relevant if instrumentation is later
  injected into device execution?

## Recommended Next Step

The next concrete step is the LDS address reconstruction experiment. It is the
smallest step that proves rocjitsu can remove hip-moi's explicit
`lds_byte_offset` parameter while preserving the core diagnostic payload:

```text
active lane mask + LDS byte range + dynamic owner + instruction address
```

Once that is proven, atomics and sampled watchpoints can be built on top of the
same instruction-level address representation.
