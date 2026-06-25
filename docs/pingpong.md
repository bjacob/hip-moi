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

## Current hip-moi Tests

The current controlled RDNA4 tests are:

* `tests/instrumented/016_rdna4_pingpong_private_lds_test.hip`
* `tests/instrumented/017_rdna4_pingpong_cooperative_lds_test.hip`

They are tests first, not benchmarks. They exist to make sure hip-moi can
instrument kernels that combine:

* multiple subgroups in one workgroup;
* LDS double-buffering;
* WMMA;
* `__builtin_amdgcn_s_setprio`;
* `__builtin_amdgcn_sched_barrier`;
* all LDS accesses routed through hip-moi.

The private test stages each subgroup's own fragments in LDS. It should not
diagnose a subgroup-level race, because the LDS producer and consumer are the
same subgroup.

The cooperative test stages an LDS tile produced by one subgroup and consumed
by other subgroups. It has two shapes:

* a synchronized clean shape, where a full workgroup barrier separates
  producers from consumers;
* an intentionally unsynchronized shape, where exact-shadow instrumentation
  must diagnose a same-epoch cross-subgroup LDS conflict.

Both tests use a small helper around `__builtin_amdgcn_s_setprio`. That helper
is marked `__forceinline__`, and it scalarizes the subgroup selector with
`__builtin_amdgcn_readfirstlane` before branching to `s_setprio`. That
scalarization is important. `s_setprio` is a scalar instruction; a branch that
the compiler if-converts with EXEC-mask manipulation can still leave both
scalar `s_setprio` arms executing in a single traced wave. In that shape, the
source appears to choose one priority per subgroup, but the machine code can
overwrite the priority immediately and defeat the intended scheduling test.

## Generated-Code Inspection

The reusable inspection script is:

```bash
benchmarks/inspect_pingpong_codegen.sh
```

It extracts the `.hip_fatbin` section from the two RDNA4 ping-pong GTest
executables, unbundles the `gfx1201` device objects, and prints kernel metadata
and instruction counts. The default build location is
`/home/benoit/workspace/hip-moi-build`; override it with
`HIP_MOI_BUILD_DIR=/path/to/build` if needed.

The script currently checks:

* LDS group segment size;
* SGPR and VGPR counts;
* SGPR and VGPR spill counts;
* private segment size;
* `s_setprio 1` and `s_setprio 0` counts;
* WMMA counts;
* `s_barrier` counts;
* `flat_load_b128` and `flat_store_b128` counts;
* scratch-memory instruction counts;
* `s_swappc_b64` call counts.

On the current unoptimized test build, the target ping-pong kernels visibly
contain `s_setprio 1`, `s_setprio 0`, and WMMA. The script also reports many
calls and scratch instructions. That is a property of inspecting GTest kernels
built in the default test configuration, not a benchmark-grade performance
claim.

One caveat is important: full workgroup barriers can appear in helper functions
called by the target kernels. In that case, the function-scoped `s_barrier`
count for a kernel can be zero even though object-level counts show real
barrier instructions in the extracted device object. A benchmark-quality
ping-pong source should be inspected at its actual optimized build settings
before interpreting barrier placement or register pressure.

`__builtin_amdgcn_sched_barrier` is also easy to misread. It constrains backend
instruction scheduling, but it does not necessarily lower to a visible machine
instruction. Absence of a named `sched_barrier` instruction in disassembly is
not, by itself, evidence that the source-level scheduling boundary was ignored.

## Optimized ATT Validation

The local TheRock tree can provide `rocprofv3` with advanced thread trace
support. In this workspace it was built by enabling profiler support:

```bash
cmake -S /home/benoit/workspace/TheRock \
      -B /home/benoit/workspace/TheRock-build \
      -DTHEROCK_ENABLE_PROFILER=ON \
      -DTHEROCK_ENABLE_ROCPROFV3=ON \
      -DTHEROCK_BUILD_TESTING=OFF
```

The resulting tool is:

```bash
/home/benoit/workspace/TheRock-build/profiler/rocprofiler-sdk/dist/bin/rocprofv3
```

In the current local build, `rocprofv3 --version` reports ROCprofiler-SDK 1.3.2
from the checked-out TheRock sources. The build may require the staged SQLite
include directory to be visible to rocprofiler-sdk compilation if the local
TheRock configuration does not propagate that sysdep include path.

The reproducible validation entry point is:

```bash
benchmarks/run_pingpong_att_validation.sh private-pass-through
benchmarks/run_pingpong_att_validation.sh private-sampled
```

The script builds `benchmarks/016_rdna4_pingpong_att_probe.hip` with optimized
`hipcc` settings and runs ROCprof ATT in serialized mode. It intentionally
launches many workgroups so the selected target CU receives traced work. The
probe has two modes:

* `private-pass-through`: same kernel shape without hip-moi instrumentation;
* `private-sampled`: the same LDS accesses through
  `hip_moi::sampled_watchpoint_context`.

The validator, `benchmarks/validate_pingpong_att.py`, prefers ROCprof's decoded
UI JSON when it is available. That decoded form is the instruction stream
reported by ROCprof after stitching the raw SQ thread-trace data. The validator
requires each traced WMMA wave to show:

* visible LDS stores and LDS loads;
* at least one `s_setprio 1` region covering visible LDS traffic;
* `s_setprio 0` as the last priority-setting instruction before each WMMA.

It can also validate an LDS-priority signature with
`--expected-lds-priority-signature`. The signature has one digit per LDS
cluster: `1` means the cluster ran under `s_setprio 1`; `0` means it ran under
`s_setprio 0`. For the four-tile probe, SIMD 0 and SIMD 1 should report
`1010`, while SIMD 2 and SIMD 3 should report `0101`.

On the current RDNA4 machine, fresh traces validated both probe modes. The
pass-through trace on SIMD 0 reported representative WMMA waves with:

```text
inst=122 setprio1=2 setprio0=7 wmma=4 lds_write=8 lds_read=8 high_prio_lds=2 lds_priority_signature=1010
```

The sampled hip-moi trace on SIMD 0 reported representative WMMA waves with:

```text
inst=1522 setprio1=2 setprio0=7 wmma=4 lds_write=8 lds_read=8 high_prio_lds=2 lds_priority_signature=1010
```

Separate fresh traces on SIMD 2 validated the complementary `0101` signature
for both pass-through and sampled hip-moi modes. Together, those traces confirm
that the optimized probe produces the intended alternating priority roles:
SIMD 0 and SIMD 1 trace waves whose high-priority LDS clusters are tiles 0 and
2, while SIMD 2 and SIMD 3 trace waves whose high-priority LDS clusters are
tiles 1 and 3.

Those counts are not benchmark numbers. They are scheduling evidence for the
controlled optimized probe: the traced waves execute LDS traffic in
priority-raised regions, reset priority before WMMA, and use complementary
priority signatures across representative SIMD selections.

The raw `.att` files remain useful, but relying only on collection success is
not enough. A focused kernel-name ATT filter was unreliable in local testing:
it could produce captured code objects without producing usable `.att` traces.
The current script therefore uses broad serialized capture and lets the
validator select the WMMA-executing waves from ROCprof's decoded output.

One limitation remains. On gfx10+ targets, ROCprof's `--att-simd-select` option
selects a single SIMD ID, not a SIMD mask. The local ATT workflow therefore
cannot capture SIMD 0/1 and SIMD 2/3 in the same run. It validates the two
complementary role schedules in separate traces. It does not directly prove,
from one timestamped trace, that both roles were simultaneously active at a
specific cycle.

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

1. Keep the current private and cooperative tests as correctness and codegen
   probes.
2. Keep the optimized ATT probe as the instruction-ordering guardrail for any
   future ping-pong benchmark source.
3. If ping-pong timing becomes important, build that benchmark separately from
   the GTest executables, with optimized compilation settings and pass-through
   plus hip-moi rows.
4. Run generated-code inspection and ATT validation on the benchmark object
   before trusting timing numbers.
5. Document that hip-moi treats `setprio` and `sched_barrier` as scheduling
   operations, not memory synchronization.
6. Return to atomics for the next real memory-model expansion.
