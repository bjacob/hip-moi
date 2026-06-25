<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Ping-Pong Scheduling And `setprio`

This note records the current scope assessment for ping-pong kernels and
`s_setprio`. The purpose is to decide whether this is a useful next step for
hip-moi before, or alongside, atomics.

## Definitions

**Ping-pong scheduling** means splitting a loop body into alternating
instruction clusters, then phase-shifting subgroups so that one subgroup is in a
memory-oriented cluster while another subgroup is in a compute-oriented cluster.
The intended benefit is overlap: one subgroup advances address calculations,
loads, stores, or LDS transfers while another subgroup issues matrix instructions.

**Memory cluster** means the part of the schedule dominated by memory movement
or address-update work. In the Triton AMD pass this can include global loads,
LDS stores, LDS loads, async-copy waits or commits, scalar control, and barrier
instructions.

**Dot cluster** means the part of the schedule dominated by matrix multiply
instructions, such as MFMA in CDNA-oriented examples. In our RDNA4 HIP tests the
analogous operation is WMMA.

**`s_setprio`** is the AMDGPU instruction exposed by Clang as
`__builtin_amdgcn_s_setprio`. LLVM models it as an intrinsic with side effects
but no memory access. The RDNA4 machine-readable ISA describes `S_SETPRIO` as
changing wave user priority.

**`sched_barrier`** means `llvm.amdgcn.sched.barrier`, exposed by Clang as
`__builtin_amdgcn_sched_barrier`. It constrains backend instruction scheduling.
It is not a workgroup barrier and does not synchronize memory.

## Source Survey

The clearest local sources are IREE, Triton, and LLVM/MLIR.

IREE's ROCm built-in ukernels use `pingpong_*` names and emit
`rocdl.s.setprio` around MFMA regions, for example under:

* `iree/compiler/plugins/target/ROCM/builtins/mlir_ukernel/`
* `iree/compiler/plugins/target/ROCM/builtins/specialization/`

IREE also has a lowering pattern in
`iree/compiler/src/iree/compiler/Codegen/LLVMGPU/ConvertToROCDL.cpp` that moves
some `rocdl.s.setprio` operations across MFMA operations. The local comment
states the reason directly: ping-pong scheduling wants to prevent off waves
from interrupting the MFMA region of the high-priority wave.

Triton's AMD pass is `TritonAMDGPUBlockPingpong`, implemented in the Triton
submodule at:

* `third_party/amd/lib/TritonAMDGPUTransforms/BlockPingpong.cpp`
* `third_party/amd/lib/TritonAMDGPUToLLVM/ConvertWarpPipeline.cpp`
* `third_party/amd/python/examples/gluon/`

The pass groups operations into memory and dot clusters, inserts `s_setprio`,
and uses conditional barriers to phase-shift subgroups. Its own comments say
that ping-pong is known to help only under limited conditions and that the pass
is tightly scheduling latencies. In that Triton snapshot, default enablement is
for `gfx942`, and for `gfx950` when async copy is enabled; RDNA4 is not in the
default enablement path.

LLVM/MLIR exposes the relevant primitives:

* `llvm.amdgcn.s.setprio`, with Clang builtin
  `__builtin_amdgcn_s_setprio`;
* `llvm.amdgcn.sched.barrier`, with Clang builtin
  `__builtin_amdgcn_sched_barrier`;
* `llvm.amdgcn.s.barrier`, with Clang builtin
  `__builtin_amdgcn_s_barrier`;
* split and named barrier builtins for lower-level synchronization.

The local TheRock checkout also contains RDNA4 ISA metadata for `S_SETPRIO`.
That confirms RDNA4 instruction availability, but not that RDNA4 is a tuned
performance target for the existing ping-pong transformations.

I did not find checked-out AITER, hipBLASLt, or TensileLite source trees in this
workspace. Any claims about those projects should be made only after a separate
source-mining pass over real local checkouts or pinned upstream revisions.

## Memory-Model Consequence

For hip-moi's current HIP/LLVM memory-model detector, `s_setprio` should not
advance epochs and should not create a happens-before edge.

The reason is simple: `s_setprio` is a scheduling-priority operation. It does
not perform a memory operation, it is not a fence, it is not an atomic, and it
is not a barrier. LLVM's intrinsic is marked as having side effects, so the
compiler must preserve it as an observable scheduling operation, but it is also
`IntrNoMem`.

Likewise, `sched_barrier` should not be treated as a memory synchronization
operation. It is a backend scheduling boundary.

Actual barriers are different. A full workgroup barrier, such as
`__syncthreads()` or `__builtin_amdgcn_s_barrier()` when used as a full
workgroup barrier, can remain an epoch boundary for the current detector.
Triton's `amdg.cond_barrier` lowers to a branch around `s_barrier`, so it is a
real execution barrier for a selected subset of threads. However, Triton's op
description explicitly says it does not set a memory fence. Modeling that
precisely would be a separate synchronization expansion, not a consequence of
`setprio` itself.

## Hardware-Model Consequence

For a future assembly-level tool, `s_setprio` is meaningful and should be part
of the execution model.

A conservative model is:

* each wave has a current user priority;
* `s_setprio N` changes the priority of the executing wave until another
  priority change;
* priority affects which ready wave issues instructions when waves compete for
  the same execution resource;
* priority does not guarantee a particular interleaving, and it does not make
  memory accesses ordered.

That model is intentionally weaker than a cycle-accurate scheduler. It is
strong enough to explain why ping-pong code uses `setprio`: the programmer is
trying to keep a memory-cluster wave and a compute-cluster wave from destroying
the intended overlap by letting one wave overtake the other at cluster
boundaries.

## Self-Contained HIP Reproduction

We can model useful ping-pong-shaped kernels in self-contained HIP.

The first version should be a controlled HIP test shape, not a claim of
production-equivalent scheduling:

1. Use multiple subgroups in one workgroup.
2. Use explicit LDS double or triple buffers.
3. Separate the loop body into memory and WMMA clusters.
4. Insert `__builtin_amdgcn_s_setprio(1)` and
   `__builtin_amdgcn_s_setprio(0)` around selected clusters.
5. Insert `__builtin_amdgcn_sched_barrier(0)` where we need to prevent backend
   movement across cluster boundaries.
6. Use full workgroup barriers first, so correctness and hip-moi epoch behavior
   remain easy to explain.
7. Instrument every LDS access with hip-moi.

There are two distinct LDS-sharing shapes worth keeping separate.

The first shape is private LDS staging. Each subgroup writes fragments into LDS
slots that only the same subgroup later reads. This is still useful because it
tests code that combines hip-moi instrumentation with `setprio`,
`sched_barrier`, LDS double-buffering, and WMMA, but it should not be presented
as a cooperative LDS communication test.

The second shape is cooperative LDS staging. One subgroup, or one subset of the
workgroup, writes an LDS tile that other subgroups later consume. This is where
hip-moi's subgroup-level race detector becomes semantically central: without a
real synchronization edge between the stores and the cross-subgroup loads, the
program has same-epoch conflicting LDS accesses. `setprio` and `sched_barrier`
do not fix that; an actual workgroup synchronization operation does.

These shapes are enough to answer whether hip-moi can instrument code that uses
`setprio`, whether code generation keeps the priority operations where we
expect them, and whether our overhead changes on ping-pong-like clustered
access patterns.

A more faithful Triton-like version would need asymmetric phasing: hold half of
the subgroups before the loop and reconverge them after the loop. Triton does
this with `amdg.cond_barrier`. In HIP, reproducing that exactly would require
careful use of lower-level barrier builtins or inline assembly. That should be
treated as a second step because it is easier to create invalid or deadlocking
programs while experimenting.

## RDNA4 Suitability

RDNA4 is suitable for syntax, code-generation, and instrumentation-overhead
experiments:

* `S_SETPRIO` exists in the RDNA4 ISA metadata;
* Clang exposes `__builtin_amdgcn_s_setprio`;
* hip-moi's existing RDNA4 WMMA tests can be extended into clustered
  ping-pong-shaped kernels.

RDNA4 is a weaker choice for claiming production ping-pong performance
representativeness. The strongest local producer examples are CDNA or GFX950
oriented, and Triton's default enablement does not include `gfx1201`. RDNA4
experiments should therefore be presented as controlled source-level
instrumentation experiments, not as evidence that a production RDNA4 kernel
should use ping-pong scheduling.

## Recommended hip-moi Scope

`setprio` is worth a short detour, but it should not displace atomics as the
next semantic expansion.

The recommended sequence is:

1. Add one self-contained RDNA4 HIP test with private LDS staging, `setprio`,
   `sched_barrier`, WMMA, and instrumented LDS double-buffering.
2. Add the cooperative LDS counterpart: one synchronized clean case and one
   intentionally unsynchronized diagnostic case.
3. Add a matching benchmark row only if the test shape is stable and the
   generated code visibly contains the expected `s_setprio` instructions.
4. Document that hip-moi treats `setprio` and `sched_barrier` as scheduling
   operations, not memory synchronization.
5. Return to atomics for the next real memory-model expansion.
