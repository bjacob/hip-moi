<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# DBI Atomic Instruction Seeds

This document records atomic instruction seeds for the future rocjitsu
dynamic-binary-instrumentation direction. It is intentionally separate from the
source-level HIP atomics model implemented in `hip_moi::context`.

At source level, hip-moi reasons in the HIP/LLVM memory model and diagnoses
LDS payload races. At DBI level, rocjitsu will see AMDGPU instructions,
register operands, address operands, scope bits, and hardware scheduling
behavior. The seeds below are useful because they expose concrete atomic
instruction forms that a DBI implementation would need to decode or route
around.

Do not treat this document as a request to diagnose ordinary global load/store
races in hip-moi. Global atomics remain synchronization operations for the
source-level detector.

## Seed Table

| Key | Source | Source operation | Instruction signal | Address space | Source memory-order information | DBI value |
| --- | --- | --- | --- | --- | --- | --- |
| `hipkittens-buffer-pk-add-bf16` | `/home/benoit/workspace/rocjitsu-test-corpus/corpus/fuzz-targets/third_party/hipkittens/include/common/macros.cuh:550` | Inline assembly `buffer_atomic_pk_add_bf16` | Exact inline assembly: `buffer_atomic_pk_add_bf16 a[...]` or `buffer_atomic_pk_add_bf16 v[...]` | Buffer/global through an AMDGPU buffer resource descriptor | None beyond `asm volatile(... ::: "memory")`; this is not a HIP/LLVM atomic API call | Best first DBI seed because the instruction mnemonic and operands are explicit in source. |
| `rdna4-streamk-arrival-counter` | `/home/benoit/workspace/hip-matmul/matmul_rdna4.hip:1182`, `:1366` | `atomicAdd(counter, 1)` with surrounding `__builtin_amdgcn_fence` and barriers | Current extracted hip-moi benchmark disassembly contains target workload `global_atomic_add_u32` for the user counter. hip-moi instrumentation itself adds `flat_atomic_*` metadata operations, which must be ignored as detector internals. | Global | Source comments and fences express release/acquire intent around an otherwise relaxed atomic counter helper | Good bridge seed from source-level Stream-K to DBI because it includes a real global RMW plus explicit fences. |
| `rdna4-streamk-tree-atomic-or` | `/home/benoit/workspace/hip-matmul/matmul_rdna4.hip:1587` | `atomicOr(counter, value)` with surrounding `__builtin_amdgcn_fence` and barriers | Current extracted hip-moi benchmark disassembly contains target workload `global_atomic_or_b32` for the user bitmask. hip-moi instrumentation adds separate `flat_atomic_*` metadata operations. | Global | Source uses explicit fences around the relaxed RMW helper; the returned old value drives tree control flow | Best current DBI seed for old-value-dependent RMW control flow. |
| `hip-stream-k-locks` | `/home/benoit/workspace/rocjitsu-test-corpus/corpus/fuzz-targets/third_party/hip-stream-k/include/streamk/device/device_locks.hpp:25`, `:38`, `:49` | `atomicCAS` lock, release `__hip_atomic_store`, acquire `__hip_atomic_load` | Expected instruction mix includes an atomic compare-and-swap for the lock and scoped global load/store operations for release/acquire flag publication. Confirm exact mnemonics from generated code before using as a DBI benchmark. | Global | Strong source-level release/acquire information for the flag path; lock path is less central to hip-moi's current tests | Good DBI seed for distinguishing true atomic RMW instructions from scoped non-RMW synchronization loads/stores. |
| `llama-count-equal-atomic-add` | `/home/benoit/workspace/rocjitsu-test-corpus/corpus/fuzz-targets/third_party/llama.cpp/ggml/src/ggml-cuda/count-equal.cu:25` | `atomicAdd((int*)dst, nequal)` | Expected global integer atomic add. Confirm exact generated instruction before using as an instruction-level testcase. | Global | No payload handoff; this is a reduction into an output counter | Tiny sanity seed for a global RMW with no synchronization payload. |
| `hip-fpsan-amdgcn-atomics` | `/home/benoit/workspace/rocjitsu-test-corpus/corpus/cts/third_party/hip-fpsan/include/fpsan/amdgcn_atomic.hpp` | `atomicAdd`, `atomicMin`, `atomicMax`, and `atomicCAS`-loop helpers over several scalar widths | Expected mix of global/flat atomic add, min/max, and compare-and-swap instructions depending on address-space lowering | Pointer-dependent | These are instrumentation-library atomics rather than workload synchronization operations | Useful as a width/opcode coverage seed, but not a hip-moi source-level semantic seed. |
| `tensile-buffer-cmpswap` | `/home/benoit/workspace/sanitizer-strategy/tensile_sync_audit.md:81` | Generated Tensile artifact noted as `buffer_atomic_cmpswap_b32 ... sc0` | Textual audit signal only in the current hip-moi workspace; re-run extraction against the original Tensile artifact before relying on it | Buffer/global | Generated assembly, not a HIP source memory-order API | Useful reminder that production library code may contain buffer atomics even when source-level searches are quiet. |

## What Rocjitsu Would See

For DBI, the important facts are not the same as the source-level facts that
`hip_moi::context` records today.

For an AMDGPU atomic instruction, rocjitsu can generally observe:

* the instruction mnemonic, such as `global_atomic_add_u32`,
  `global_atomic_or_b32`, `flat_atomic_cmpswap_b64`, or
  `buffer_atomic_pk_add_bf16`;
* the architectural register operands;
* whether the instruction returns an old value;
* address operands after the emulator computes the effective address;
* scope/cache bits encoded on the instruction, when present;
* dynamic wave, lane, and scheduling state.

Rocjitsu does not automatically see:

* the HIP source expression that produced the instruction;
* the C++ or HIP memory-order enum that motivated it;
* whether a surrounding fence was intended to form a source-level
  release/acquire pair;
* source-level object identity beyond the computed address.

That is why this document keeps two categories separate:

* source-level atomics tests validate hip-moi's HIP/LLVM memory-model
  approximation;
* DBI seeds identify concrete hardware instructions and operand shapes that a
  future rocjitsu implementation must handle.

## Distinguish Workload Atomics From Detector Atomics

The current hip-moi benchmark disassemblies contain two kinds of atomics:

* target workload atomics, such as the user Stream-K counter
  `global_atomic_add_u32` or bitmask `global_atomic_or_b32`;
* hip-moi metadata atomics, such as `flat_atomic_swap_b64`,
  `flat_atomic_add_u32`, `flat_atomic_max_u32`, and
  `flat_atomic_cmpswap_b64`.

For future DBI work, this distinction matters. If rocjitsu instruments an
already instrumented hip-moi benchmark, it must not mistake hip-moi's metadata
traffic for the user's synchronization protocol. The clean DBI corpus should
prefer uninstrumented kernels when the goal is to study workload atomics.

## Current Status

This inventory is complete enough for the first DBI planning discussion. The
next DBI-oriented step is described in
[`dbi_transition.md`](dbi_transition.md): start with LDS address reconstruction,
then pick one atomic seed and produce a minimal standalone uninstrumented
kernel plus disassembly record. The best first atomic-only candidate remains
`hipkittens-buffer-pk-add-bf16` because the instruction is explicit inline
assembly and does not require guessing the compiler lowering.
