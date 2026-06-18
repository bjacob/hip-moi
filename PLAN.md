<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# PLAN for hip-moi: HIP memory-ordering instrumentation library

## Goal

Build a small HIP device-side library that helps diagnose definite
workgroup-local LDS memory-ordering mistakes in manually instrumented kernels,
with near-zero false positives inside its stated contract.

The library is not a compiler pass and not a true interposer. It will not
magically intercept raw pointer dereferences, direct builtin calls, or arbitrary
HIP synchronization APIs. Users opt in by calling `hip-moi` APIs for LDS memory
accesses and synchronization.

The near-term schedule is deliberately aggressive:

* EOD: a tiny but real MVP that compiles, runs on gfx1201, and has GPU tests.
* End of tomorrow: a more credible refined MVP with better diagnostics,
  stronger tests, and at least one real-kernel idiom extracted from the matmul
  corpus.
* Later: incremental widening of the memory model, especially atomics and
  fences.

## Current status

The repository has been initialized with the planning and test-harness
foundation:

* CMake uses Ninja and builds HIP tests with GTest.
* GTest is found as a system package when available, with `FetchContent`
  fallback.
* `HIP_MOI_CTEST_PER_CASE=ON` is the default, so CTest reports one entry per
  GTest case.
* `tests/reference/mvp_reference_kernels.hip` contains the uninstrumented
  reference corpus. It is a parameterized GTest suite exposing one CTest entry
  per launched safe reference kernel.
* The reference corpus currently contains 55 safe kernels that compile, launch,
  and numerically check their outputs.
* The same source also contains compile-only diagnostic-positive and hard
  synchronization references that are intentionally not launched.
* The actual `hip-moi` implementation library and instrumented tests have not
  started yet.

The reference corpus is a map of desired coverage, not an obligation to
instrument everything immediately. The instrumented suite should grow only when
the library actually supports the corresponding behavior.

## Foundations

### Terminology

Use common HIP/AMDGPU terminology consistently:

* Thread: one logical thread in a HIP kernel, identified within a workgroup by
  `threadIdx`.
* Workgroup: the set of threads that can communicate through LDS and
  synchronize with `__syncthreads()`/`s_barrier`. In HIP source, this is the
  group indexed by `blockIdx` and sized by `blockDim`.
* Wavefront: the hardware execution grouping. Do not rely on wavefront lockstep
  for language-level correctness.
* Subgroup: a library/modeling concept for a subset of a workgroup. This is
  useful for future finer-grained epoch tracking. A subgroup may correspond to a
  wavefront in some modes, but the plan should say so explicitly when that is
  intended.

### Memory-model ground rules

The mental model for this project should be language/IR first, hardware folklore
last.

HIP device code lowers through Clang to LLVM IR. HIP APIs and Clang builtins
must ultimately be understood by what they lower to: LLVM loads/stores, atomic
operations, fences, target-specific synchronization scopes, and target-specific
intrinsics. We bring in the LLVM IR memory model, rather than only the
HIP/C++-style source memory model, because hip-moi aims to instrument Clang
builtins for operations such as fences, and those builtins are best specified by
their LLVM IR lowering.

The relevant baseline reference is LLVM LangRef, especially:

* Memory model for concurrent operations:
  https://llvm.org/docs/LangRef.html#memory-model-for-concurrent-operations
* Atomic memory ordering constraints:
  https://llvm.org/docs/LangRef.html#atomic-memory-ordering-constraints
* `fence` instruction:
  https://llvm.org/docs/LangRef.html#fence-instruction

Practical consequences:

* Do not justify correctness by saying GPU threads are special, LDS is special,
  wavefront lockstep saves us, or a barrier "probably flushes things" in a
  hardware sense. Those may matter for performance and lowering, but they are
  not the source-level correctness contract.
* Work from the generated LLVM IR when in doubt.
* Happens-before is the central relation. LLVM defines happens-before as program
  order plus synchronizes-with edges introduced by atomics and platform-specific
  mechanisms.
* LLVM IR is not word-for-word the C++ data-race model. In the plain non-atomic
  concurrent case, LLVM describes when reads may return `undef`; C++ describes
  data races as undefined behavior. For hip-moi, the actionable diagnostic
  target is still the same shape: conflicting non-atomic shared accesses that
  are not ordered by happens-before.
* A release fence and an acquire fence do not, by themselves, create an
  inter-thread synchronizes-with edge. In LLVM, release/acquire fences
  synchronize only through an appropriate atomic object/protocol, or through
  some other platform-specific mechanism. A bare
  `fence_release(); fence_acquire();` pair around LDS accesses is therefore not
  enough to prove the LDS accesses ordered.
* `__syncthreads()`-like APIs conflate two ideas that we eventually want to
  model separately: a control barrier and memory-ordering fences. The MVP treats
  `ctx.syncthreads()` as one full-workgroup epoch boundary. Later phases can
  split this into lower-level primitives once their IR-level meaning is clear.

## MVP

The MVP is the smallest useful library that can be built and tested today.

### MVP contract

The MVP diagnoses only this model:

* Memory scope: LDS/shared memory only.
* Threading scope: one workgroup only.
* Accesses: explicit calls to `hip-moi` templated LDS load/store APIs.
* Synchronization: uniform full-workgroup `ctx.syncthreads()` only.
* Race condition of interest: two instrumented LDS accesses overlap in byte
  range, are from different threads, are in the same synchronization epoch, and
  at least one is a write.
* Non-goals: instrumented user global-memory accesses, inter-workgroup races,
  atomics, warp/wavefront lockstep assumptions, raw memory accesses, divergent
  real barriers, and fence-only synchronization.

Within this contract:

* The library should diagnose all definite conflicting same-epoch LDS accesses.
* The library should not diagnose non-overlapping accesses.
* The library should not diagnose accesses separated by a uniform
  `ctx.syncthreads()`.
* The library should preserve the program's actual synchronization behavior by
  still executing the real HIP/Clang synchronization operations.

This intentionally accepts many false negatives outside the model. The key MVP
quality bar is that any diagnostic it does emit should be trustworthy.

### MVP API

Use an explicit user-provided context from day one:

```c++
namespace hip_moi {

struct config {
  // Number of threads participating in this context.
  int thread_count;
  // Number of threads in each full subgroup; the final subgroup may be partial.
  int threads_per_subgroup;
  // Number of subgroups represented in this context.
  int subgroup_count;
};

struct access_record;
struct diagnostic;
struct subgroup_state;

// Non-owning view of detector metadata buffers. Buffers may live in global
// memory or in __shared__ memory.
struct context_storage_ref {
  access_record* access_records;
  int access_record_capacity;
  diagnostic* diagnostics;
  int diagnostic_capacity;
  subgroup_state* subgroup_states;
  int subgroup_capacity;
};

// Optional fixed-capacity storage helper for users who want static storage.
template <int AccessCapacity, int DiagnosticCapacity, int SubgroupCapacity = 1>
struct static_context_storage {
  __device__ context_storage_ref ref();
};

class context {
 public:
  // Binds the context to caller-provided storage and runtime shape metadata.
  __device__ context(context_storage_ref storage, config cfg);

  // Initializes shared metadata for this workgroup before instrumented use.
  __device__ void init_workgroup();

  // Records and performs an LDS load.
  template <typename T>
  __device__ T lds_load(const T* ptr);

  // Records and performs an LDS store.
  template <typename T>
  __device__ void lds_store(T* ptr, T value);

  // Performs a real full-workgroup synchronization and advances the epoch.
  __device__ void syncthreads();

  // Queries whether this context has recorded diagnostics.
  __device__ bool has_error() const;
  __device__ int error_count() const;
};

}  // namespace hip_moi
```

Notes:

* First implementation target: `include/hip_moi/hip_moi.hpp`, as a header-only
  library target exported by CMake.
* `context_storage_ref` is a non-owning view of caller-provided metadata
  buffers. Those buffers may be in global memory, which avoids consuming scarce
  LDS in kernels that already use nearly all available shared memory.
* `static_context_storage` is only a convenience helper for users who do want
  fixed-size storage, for example in `__shared__` memory. Its capacities are
  template parameters because that is how the helper embeds fixed-size arrays.
  The main `context` type is not templated on capacities.
* Prefer this storage-view design over virtual device-side polymorphism for the
  MVP. It gives both global-memory and static-storage users the same context
  API, avoids relying on devirtualization for zero overhead, and still leaves
  room for a future storage-policy specialization if measurements show it is
  needed.
* Do not assume `thread_count == threads_per_subgroup * subgroup_count` unless
  a specific mode explicitly requires an even subgroup partition. The MVP can use
  `threads_per_subgroup == thread_count` and `subgroup_count == 1`. Future
  subgroup modes may allow a partial final subgroup, where
  `threads_per_subgroup * subgroup_count >= thread_count`.
* LDS load/store APIs should be templated immediately. Internally, every memory
  access becomes a byte range: base address plus `sizeof(T)` bytes.
* The first API can require trivially copyable `T`.
* A pointer wrapper can come later, but member functions on the context are the
  easiest way to make real progress today.
* The namespace should avoid claiming to replace HIP builtins. It is a separate
  diagnostic API.
* The API should eventually support labels/source locations, but MVP diagnostics
  can start with compact numeric records.

### Instrumented kernel shape

An uninstrumented reference kernel:

```c++
__global__ void kernel(int* out) {
  __shared__ int lds[32];

  int tid = threadIdx.x;
  lds[tid] = tid;
  __syncthreads();
  out[tid] = lds[(tid + 1) & 31];
}
```

The MVP instrumented version should look like:

```c++
__global__ void kernel(int* out, hip_moi::context_storage_ref storage) {
  __shared__ int lds[32];

  hip_moi::config cfg{
      static_cast<int>(blockDim.x),
      static_cast<int>(blockDim.x),
      1,
  };
  hip_moi::context ctx(storage, cfg);

  ctx.init_workgroup();
  ctx.syncthreads();

  int tid = threadIdx.x;
  ctx.lds_store(&lds[tid], tid);
  ctx.syncthreads();

  int value = ctx.lds_load(&lds[(tid + 1) & 31]);
  out[tid] = value;
}
```

MVP contract for instrumented kernels:

* Every LDS access that should participate in diagnostics goes through
  `ctx.lds_load` or `ctx.lds_store`.
* Any raw LDS access is invisible to hip-moi and outside the diagnostic
  contract.
* Synchronization that should advance the shadow epoch goes through
  `ctx.syncthreads()`.
* `ctx.syncthreads()` performs the real full-workgroup synchronization and
  advances the detector's shadow epoch.
* Tests should start with one workgroup per launch where possible, then broaden
  only when storage/result handling supports multiple workgroups cleanly.

### MVP implementation model

Start with a workgroup epoch model.

Each instrumented LDS access records:

* address or LDS-relative byte offset,
* byte size,
* access kind: load or store,
* thread id,
* subgroup id,
* epoch id,
* optional source id,
* whether the slot is valid.

On every `lds_load<T>` or `lds_store<T>`:

1. Compute the byte range.
2. Record the new access in the active epoch.
3. Perform the real load or store.

Conflict checking may happen immediately or be deferred. The preferred MVP
direction is deferred checking from per-thread access logs, so the common access
path does not need detector-created cross-thread synchronization. Whenever
checking runs, if byte ranges overlap, thread ids differ, and either access is a
write, record a diagnostic.

On `ctx.syncthreads()`:

1. Execute the real full-workgroup synchronization.
2. Check the epoch that just ended for conflicts, or preserve enough records to
   check it later.
3. Advance the epoch.
4. Clear or logically invalidate access records from the previous epoch.
5. Ensure detector metadata updates are complete before returning, using
   detector-internal synchronization if needed.

The actual call should preserve HIP semantics. If `ctx.syncthreads()` delegates
to `__syncthreads()`, verify what HIP emits for gfx1201. If it is implemented
using lower-level Clang builtins, verify the emitted LLVM IR.

### MVP metadata correctness

The detector's own metadata is shared state. It must not have uncontrolled races
that make the detector nondeterministic. At the same time, detector
synchronization must not become part of the program being diagnosed.

Core rule:

* hip-moi maintains a shadow happens-before model for the user's program. Any
  atomics, locks, fences, or barriers used internally by hip-moi are out of band
  and must never be treated as user-program synchronization by that shadow
  model.

Important ordering rule:

* Run hip-moi metadata bookkeeping before performing the user's actual LDS load
  or store.

That keeps any synchronization used internally by the detector from accidentally
placing the user's LDS access inside a detector-created critical section. The
tool may serialize its bookkeeping, but it should not rely on that serialization
to make the user's memory access correct.

Preferred implementation direction:

* Give each thread ownership of a small append-only region of the access log.
  This avoids cross-thread detector synchronization on the common
  `lds_load<T>`/`lds_store<T>` path.
* Compare records and emit diagnostics at modeled synchronization points, such
  as `ctx.syncthreads()`, and at explicit test/finalization points.
* If a shared diagnostic counter needs atomic updates, treat those atomics as
  detector-internal bookkeeping only. Use the weakest ordering sufficient for
  detector correctness.

Fallback/debug implementation:

1. Acquire detector metadata lock.
2. Compare the new access with prior access records in the current epoch.
3. Append the new access record.
4. Release detector metadata lock.
5. Perform the user's real LDS load or store.

This fallback may perturb scheduling and may introduce real synchronization in
the instrumented executable, but diagnostics must still be computed from the
shadow model that ignores detector-created happens-before edges.

This intentionally diagnoses unordered same-epoch conflicting accesses based on
the program order expressed through hip-moi calls, not on a particular hardware
interleaving observed during one run.

Instrumentation may perturb the original bug. That is acceptable if the shadow
model still deterministically reports the conflicting accesses that were present
in the instrumented API trace. The primary value proposition is to turn racy
patterns into deterministic diagnostics, not to preserve the original race. The
bad failure mode is the conjunction of perturbing or hiding the original race and
also failing to report it in the shadow diagnostic.

### MVP subgroup-ready shape

Even though the MVP has only one full-workgroup epoch, do not bake that into
type names or storage layout. Design the metadata as:

```c++
epoch[subgroup_id]
```

with `subgroup_count == 1` for the MVP.

This keeps a path open for kernels where independent subgroups within one
workgroup operate on different shared tiles or synchronize at different
granularities. Do not implement subgroup semantics in the MVP; just avoid
closing the door.

### MVP diagnostics

Make diagnostics testable before making them pretty.

Recommended first diagnostic record:

```c++
struct diagnostic {
  uint32_t kind;
  uint32_t epoch;
  uint32_t writer_or_first_thread_id;
  uint32_t reader_or_second_thread_id;
  uintptr_t first_addr;
  uintptr_t second_addr;
  uint32_t first_size;
  uint32_t second_size;
};
```

Host-side tests should be able to copy back a small result buffer and assert:

* number of diagnostics,
* diagnostic kind,
* participating threads,
* whether the test passed/failed as expected.

Avoid relying on device `printf` for pass/fail. It is useful for debugging, but
not a stable test oracle.

### Cases to avoid in the MVP

Do not begin with tests where only some threads execute a real
`__syncthreads()`/`s_barrier`. Those can hang or become undefined before the
diagnostic path gets a clean chance to report.

Instead, stage them as later negative tests in one of two ways:

* simulation-only synchronization API that records intent without executing a
  real barrier, or
* process-level timeout/expected-hang tests, if they prove useful.

The MVP should prefer kernels that always terminate.

### EOD MVP implementation steps

The goal for EOD is a tiny but real library with GPU tests that compile and run.

1. Create the CMake skeleton.
   * Header-only library target.
   * Tests built with TheRock Clang:
     `/home/benoit/workspace/TheRock-build/dist/rocm/llvm/bin/clang++`.
     CMake rejects the `hipcc` wrapper as `CMAKE_HIP_COMPILER`.
   * Default target `gfx1201`.
   * GTest from system package, fallback to FetchContent.

2. Add the first public header.
   * `include/hip_moi/hip_moi.hpp`
   * `context_storage_ref`
   * `static_context_storage`
   * `context`
   * `lds_load<T>`
   * `lds_store<T>`
   * `syncthreads()`

3. Implement single-epoch workgroup metadata.
   * Runtime-capacity metadata buffers addressed through `context_storage_ref`.
   * At least one global-memory-backed storage path for tests.
   * Optional `static_context_storage` helper for fixed-size storage.
   * Fixed-capacity access log.
   * Fixed-capacity diagnostics.
   * Shared lock or atomic reservation for detector metadata updates.
   * Bookkeeping before the real LDS access.
   * Single workgroup per test kernel is acceptable initially.
   * One subgroup epoch slot only, but use names/layout that allow more.

4. Add a minimal host/device test harness.
   * Launch tiny kernels.
   * Copy diagnostic counters/records back.
   * Assert with GTest.
   * Use a parameterized GTest suite so CTest can report one entry per
     instrumented kernel when `HIP_MOI_CTEST_PER_CASE=ON`.

5. Add EOD test cases.
   * Same thread store then load: no diagnostic.
   * Different threads access non-overlapping LDS addresses: no diagnostic.
   * Different threads read same LDS address: no diagnostic.
   * Different threads write/read same LDS address in same epoch: diagnostic.
   * Different threads write/write same LDS address in same epoch: diagnostic.
   * Different threads write/read same LDS address separated by uniform
     `ctx.syncthreads()`: no diagnostic.
   * Templated `float`, `int`, and a small vector/struct type if time permits.

6. Verify.
   * Configure with CMake.
   * Build with Ninja.
   * Run the GTest binary.
   * If possible, inspect one simple generated LLVM IR file for the
     synchronization lowering.

### MVP test corpus

The current `tests/reference/` corpus should remain ahead of the instrumented
suite. It keeps concrete, compiling kernel shapes available even before the
library can diagnose them.

The future `tests/instrumented/` suite should be contractual: an instrumented
test should exist only for behavior the library claims to support today. Do not
bulk-instrument the whole reference corpus at once. Instead, each broadening of
library capability earns one or a few matching instrumented tests.

Proposed instrumented test layout:

```text
tests/instrumented/
  safe_mvp_test.hip
  race_mvp_test.hip
  test_support.hpp
```

`safe_mvp_test.hip` should assert both numerical outputs and zero diagnostics.
`race_mvp_test.hip` should assert deterministic diagnostics; for racy kernels,
numerical output is not the oracle.

Incremental instrumented test growth:

1. Add the smallest safe instrumented kernel when `ctx.lds_load` and
   `ctx.lds_store` exist.
2. Add the smallest same-epoch write/read diagnostic when overlap detection
   exists.
3. Add write/write diagnostics.
4. Add `ctx.syncthreads()` separation tests when epoch advancement exists.
5. Add all-thread array cases when per-thread metadata and byte-range tracking
   are solid.
6. Add loops when repeated epochs are solid.
7. Add tiled and matmul-like LDS cases when the basic machinery has survived
   enough pressure.

Layer 1: toy deterministic kernels.

Positive/no-diagnostic tests:

* one thread only,
* same address read/read across threads,
* non-overlapping addresses,
* adjacent addresses,
* write/read separated by uniform `ctx.syncthreads()`,
* write/write separated by uniform `ctx.syncthreads()`,
* multiple epochs with repeated reuse of the same LDS address.

Negative/diagnostic tests:

* same-address write/read in one epoch,
* same-address read/write in one epoch,
* same-address write/write in one epoch,
* partially overlapping byte ranges,
* conflicting access after metadata log gets near capacity,
* missing `ctx.syncthreads()` in a simple producer/consumer pattern.

Layer 2: API behavior tests.

* templated scalar loads/stores,
* trivially copyable struct loads/stores,
* alignment assumptions,
* metadata initialization,
* diagnostic overflow behavior,
* multiple calls to `ctx.syncthreads()`,
* multiple workgroups if supported.

## Refined MVP by end of tomorrow

The goal for tomorrow is not broad generality. It is making the MVP feel like a
credible tool rather than a toy.

1. Improve address/range tracking.
   * Use byte ranges consistently.
   * Handle partially overlapping typed accesses.
   * Add tests for adjacent vs partially overlapping ranges.

2. Improve diagnostics.
   * Add source/location IDs or stringless labels.
   * Add first-conflict preservation so later conflicts do not hide the useful
     one.
   * Add overflow diagnostics for metadata capacity exhaustion.

3. Make context initialization robust.
   * Clear shared metadata exactly once per workgroup.
   * Ensure all threads see initialized metadata before use.
   * Test multiple workgroups if the storage/result protocol supports it.

4. Start extracting real-kernel idioms.
   * Read `/home/benoit/workspace/hip-matmul/matmul_rdna4.hip`.
   * Extract the simplest LDS write/syncthreads/read pattern.
   * Convert only the minimum number of loads/stores to `ctx.lds_*`.
   * Preserve the original control structure closely.

5. Add synchronization lowering notes.
   * Compile tiny examples using `ctx.syncthreads()`, raw `__syncthreads()`,
     `__builtin_amdgcn_s_barrier`, and `__builtin_amdgcn_fence`.
   * Save or document their LLVM IR shape.
   * Identify which combinations produce LLVM `fence` instructions, target
     syncscopes, and/or target-specific barrier intrinsics.
   * Do not infer cross-thread synchronization from naked fences.

## Plan beyond the MVP

After the refined MVP, widen scope in small semantic steps. Each step should add
one new kind of happens-before edge or one new class of instrumented access, with
tests that prove both positive and negative cases.

### Step 1: Real-kernel LDS coverage

Broaden the corpus before broadening the memory model too much.

* Extract more LDS tiling patterns from
  `/home/benoit/workspace/hip-matmul/matmul_rdna4.hip`.
* Cover vectorized LDS accesses.
* Cover double-buffered LDS.
* Cover repeated tile loops with multiple synchronization phases.
* Keep atomics out of these tests until the atomic model exists.

### Step 2: Hard synchronization negative tests

Add a way to test bad synchronization patterns that might otherwise hang.

* Add simulation mode for synchronization APIs, where the library records
  barrier intent without executing a real barrier.
* Use simulation mode to diagnose missing participation in a barrier.
* Keep real-barrier divergent tests separate, probably with process-level
  timeout handling if they are ever useful.

### Step 3: Subgroup epochs

Allow a workgroup to contain multiple independently tracked subgroups.

* Add multiple epoch counters per workgroup.
* Let users provide or compute a subgroup id.
* Test subgroup A and subgroup B operating independently on non-overlapping LDS
  ranges.
* Test same-address communication within one subgroup.
* Test data crossing subgroup boundaries only after an explicit full-workgroup
  synchronization.
* Do not equate subgroup with wavefront unless that mode explicitly says so.

### Step 4: Atomic access instrumentation, without new ordering

First teach the library to see atomic operations as operations, without yet
using them to create happens-before edges.

* Add `ctx.atomic_load`, `ctx.atomic_store`, and eventually
  `ctx.atomic_compare_exchange`/RMW wrappers for the smallest useful set of
  types.
* Record atomic object address, byte width, order, scope, thread id, and epoch.
* Diagnose unsupported mixes conservatively, such as non-atomic LDS access racing
  with an atomic access to the same byte range if the model cannot prove the
  program well-defined.
* Add tests showing that merely calling relaxed atomics does not order unrelated
  LDS accesses.

### Step 5: Acquire/release atomics as synchronization

Next model the common flag handoff without explicit fences.

Example shape:

```c++
// Producer thread
ctx.lds_store(ptr, value);
ctx.atomic_store(flag, 1, memory_order_release);

// Consumer thread
while (ctx.atomic_load(flag, memory_order_acquire) != 1) {}
auto value = ctx.lds_load(ptr);
```

Required model:

* Track per-atomic-object writes and the value/version they publish.
* When an acquire load observes a release store, add a synchronizes-with edge.
* Use that edge to order prior LDS accesses in the producer before later LDS
  accesses in the consumer.
* Add negative tests where the acquire load does not observe the release store.
* Add negative tests where the operations are relaxed and therefore do not
  create the needed edge.

### Step 6: Explicit fences paired with atomic handoff

Only after Step 5, add explicit fence modeling. This is the incremental version
of "fences + atomics"; naked fences remain negative cases.

Example shape:

```c++
// Producer thread
ctx.lds_store(ptr, value);
ctx.fence_release();
ctx.atomic_store(flag, 1, memory_order_relaxed);

// Consumer thread
while (ctx.atomic_load(flag, memory_order_relaxed) != 1) {}
ctx.fence_acquire();
auto value = ctx.lds_load(ptr);
```

Required model:

* Record a thread-local "pending release fence" state.
* Associate that pending release state with the following atomic write.
* Record that an atomic read observed a particular atomic write/version.
* Associate that observed write with the following acquire fence.
* Add the happens-before edge only when the LLVM fence rule is satisfied:
  release fence before atomic write, atomic read observes that write, acquire
  fence after the read.
* Add negative tests for release/acquire fences with no atomic handoff.
* Add negative tests where the acquire fence occurs before the observing load.
* Add negative tests where the load observes the wrong atomic write.

### Step 7: RMWs and release sequences

Then extend the atomic handoff model to read-modify-write operations and release
sequences.

* Model atomic RMWs as both reads and writes of an atomic object.
* Preserve enough per-object modification order to know which write a load or
  RMW observed.
* Support release sequences headed by a release store.
* Add tests where an acquire load observes a value produced through a release
  sequence.
* Add tests where the release sequence is broken and no synchronizes-with edge
  should be created.

### Step 8: Sync scopes and stronger orderings

Only after the simple acquire/release cases work, widen to target-specific
details.

* Model relevant LLVM/HIP sync scopes.
* Distinguish workgroup-scope, agent/device-scope, and system-scope operations
  when they appear in generated IR.
* Decide how much `seq_cst` support is worth implementing.
* Add tests that compare source-level HIP calls with the LLVM IR they produce.

### Step 9: Ergonomics and runtime support

Improve usability after the core model is trustworthy.

* Add `lds_ptr<T>` for kernels where replacing every access with member
  functions is too noisy.
* Keep context member functions as the lowest-risk API.
* Add larger diagnostic buffers.
* Add host-side diagnostic decoding.
* Consider optional binary/runtime support if header-only storage becomes too
  limiting.

### Post-MVP test corpus

Keep the corpus layered:

* Synchronization lowering tests: compile `ctx.syncthreads()`, raw
  `__syncthreads()`, `__builtin_amdgcn_s_barrier`, and
  `__builtin_amdgcn_fence`; inspect/record emitted LLVM IR.
* Real-kernel idioms: keep kernels small enough to understand in one screen,
  preserve original LDS access patterns, and replace only relevant LDS accesses
  with `ctx.lds_*`.
* Future hard cases: divergent barriers, fence-only snippets without atomic
  handoff, atomics forming release/acquire chains, subgroup-only
  synchronization, wavefront lockstep assumptions, global memory communication,
  and cross-workgroup communication.

## Development notes

* Keep the implementation warning-clean in HIP C++17 mode.
* Run `clang-format` on modified source files once code exists.
* Prefer simple fixed-capacity data structures initially. Overflow should be a
  diagnostic, not silent corruption.
* Keep every emitted diagnostic conservative. If the tool cannot prove a
  conflict inside its model, prefer no diagnostic.
* Document every unsupported case as a false-negative risk, not as a bug in the
  MVP.
