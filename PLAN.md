<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# PLAN for hip-moi: HIP memory-ordering instrumentation library

## Goal

Build a small HIP device-side library that helps diagnose definite
workgroup-local LDS memory-ordering mistakes in manually instrumented kernels,
with near-zero false positives inside its stated contract.

The project now has two distinct purposes:

1. Provide a real HIP-language diagnostic tool. In this purpose, hip-moi judges
   instrumented HIP programs by the HIP/LLVM memory model. GPU hardware
   execution folklore is not a correctness argument at this level.
2. Serve as a stepping stone toward a separate future assembly-level
   instrumentation effort. In that future effort, the relevant memory model is
   the hardware's model. Wavefront lockstep, issue behavior, and the shape of
   load/store queues may then become semantic facts rather than folklore.

These purposes must not be conflated. This repository still implements
HIP-language instrumentation, and diagnostics emitted by this library must be
principled in the HIP/LLVM model. The assembly-level purpose changes the
short-term priorities because it tells us which abstractions are most worth
learning from now: multi-subgroup workgroups, cross-subgroup communication, and
instrumentation modes that can intentionally ignore intra-subgroup races.

The library is not a compiler pass and not a true interposer. It will not
magically intercept raw pointer dereferences, direct builtin calls, or arbitrary
HIP synchronization APIs. Users opt in by calling `hip-moi` APIs for LDS memory
accesses and synchronization.

The initial single-subgroup MVP exists. The next priority is to generalize it
toward multi-subgroup workgroups and to design a second, lower-overhead
`subgroup-level` instrumentation mode before spending much more effort on
diagnostic polish.

## Current status

The repository has been initialized with the planning and test-harness
foundation:

* CMake uses Ninja and builds HIP tests with GTest.
* CMake exports a header-only `hip_moi::hip_moi` library target.
* On Clang/GCC-compatible compilers, hip-moi's own test translation units
  compile with `-Wall -Wextra -Werror`. These warning flags are private to the
  tests and are not propagated through the public `hip_moi::hip_moi` target.
* GTest is found as a system package when available, with `FetchContent`
  fallback.
* `HIP_MOI_CTEST_PER_CASE=ON` is the default, so CTest reports one entry per
  GTest case.
* RDNA4 WMMA tests are registered only when the configured HIP offload
  architectures are gfx12-family targets.
* `docs/tutorial/` contains self-contained numbered HIP example programs, a
  README, and a CMake subtree that builds the examples and registers them as
  CTests. The tutorial CTests include both a passing synchronized example,
  expected-failing diagnosed examples, and a gfx12-gated real RDNA4
  data-tiled WMMA matmul example. The README is the tutorial's primary content:
  each example is explained as a plain HIP kernel first, then as the
  corresponding hip-moi instrumented kernel, while the standalone `.hip`
  programs serve as compiled companions. The matmul tutorial presents the
  data-tiled layout through per-lane vector fragment loads and stores.
* `tests/reference/mvp_reference_kernels.hip` contains the uninstrumented
  reference corpus. It is a parameterized GTest suite exposing one CTest entry
  per launched safe reference kernel.
* The reference corpus currently contains 55 safe kernels that compile, launch,
  and numerically check their outputs.
* The same source also contains compile-only diagnostic-positive and hard
  synchronization references that are intentionally not launched.
* `include/hip_moi/hip_moi.hpp` exists with the first public API skeleton:
  `config`, `access_record`, `diagnostic`, `subgroup_state`,
  `context_storage_ref`, `static_context_storage`, and `context`.
* The current `context` implementation is intentionally only a pass-through
  skeleton in synchronization scope, but now has real same-epoch access
  logging: `init_workgroup()` initializes counters, epoch storage, and access
  record validity; `lds_load<T>` and `lds_store<T>` record the LDS byte range
  before performing the real access; and `syncthreads()` executes a real barrier
  while advancing the first epoch slot. Detector metadata initialization and
  epoch increments use detector-internal device fences before releasing the
  workgroup so global-memory metadata is visible to the participating threads.
* `tests/instrumented/001_safe_mvp_test.hip` contains the first instrumented
  kernel test. It passes caller-provided global-memory metadata storage into a
  kernel, performs a same-thread instrumented LDS store/load, checks the
  numerical result, asserts two logged accesses, and asserts zero diagnostics.
* `tests/instrumented/002_race_mvp_test.hip` contains the first
  diagnostic-positive instrumented kernel. It checks that a same-epoch LDS
  write/read byte-range conflict from two different threads produces one
  deterministic diagnostic.
* `tests/instrumented/003_host_context_test.hip` contains the first
  user-facing host diagnostics tests. It exercises `hip_moi::host_context`,
  `HIP_MOI_CHECK`, explicit nonfatal diagnostic consumption, the default
  scope-based destructor handling of unconsumed diagnostics, and the destructor
  reporting/abort opt-outs. It now also checks stderr diagnostic text and abort
  behavior using GTest stderr capture and death tests.
* `tests/instrumented/004_basic_conflict_predicate_test.hip` broadens the raw
  detector-contract coverage for same-epoch byte ranges: read/read same address,
  write/write same address, non-overlapping writes, adjacent byte ranges, and an
  overlapping full-object/subobject write.
* `tests/instrumented/005_epoch_boundary_test.hip` exercises uniform
  `ctx.syncthreads()` as the MVP epoch boundary. It checks that same-address
  accesses separated by a barrier do not report, repeated reuse across epochs
  does not report, and a new same-epoch conflict after a barrier reports in the
  new epoch.
* `tests/instrumented/006_all_thread_array_test.hip` starts the all-thread
  ladder step. It covers independent per-thread LDS writes, own-slot reads after
  a barrier, neighbor reads after a barrier, and an intentionally missing-barrier
  neighbor-read diagnostic case.
* `tests/instrumented/007_metadata_capacity_test.hip` covers access-log
  overflow, diagnostic counters that exceed stored diagnostic capacity, and the
  host-facing stderr report for truncated diagnostic buffers.
* `tests/instrumented/008_loop_epoch_test.hip` covers looped epoch patterns:
  safe scalar producer/consumer loops, all-thread own-slot loops, repeated
  missing-barrier diagnostics, and diagnostic epoch numbering across loop
  iterations.
* `tests/instrumented/009_tiled_lds_test.hip` covers 2D tiled LDS idioms:
  row-major copy, transpose, skewed stride, blocked layout, diagonal gather,
  striped load/store, and an unsynchronized transpose diagnostic case.
* `tests/instrumented/010_matmul_like_test.hip` covers small cooperative LDS
  matmul idioms: simple 2x2 and 4x4 tiles, a chunked K loop, and a scalar
  missing-barrier diagnostic. The numerical tests use explicit small integer
  input matrices and compare GPU outputs against a host-side reference matmul.
* `tests/instrumented/011_epoch_log_lifetime_test.hip` verifies that access-log
  storage is reused at epoch boundaries, so long synchronized loops can run with
  capacity sized for one epoch rather than the whole kernel.
* `tests/instrumented/012_matmul_pipeline_test.hip` covers double-buffered and
  pipeline-like matmul LDS idioms: safe ping-pong buffering plus
  diagnostic-positive buffer reuse and partial tile overwrite cases. The safe
  output cases use explicit small integer inputs and a host-side reference
  matmul oracle.
* `tests/instrumented/013_rdna4_wmma_row_major_test.hip` is a gfx12-gated real
  RDNA4 WMMA smoke test using
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`, all 32 threads,
  conventional row-major LDS tiles, single-buffer and double-buffer safe cases,
  and a diagnostic-positive row overwrite. The safe cases now use non-uniform
  small integer-valued `_Float16` inputs and exact host-reference outputs.
* `tests/instrumented/014_rdna4_wmma_data_tiled_test.hip` is the matching
  gfx12-gated packed-layout test. It uses the same WMMA intrinsic, but each
  thread's A/B fragment is a contiguous 16-byte object at byte offset
  `lane * 16`, and each thread's C accumulator fragment is a contiguous
  32-byte object at byte offset `lane * 32`, stored with one `f32x8_t` vector
  store. The test includes a diagnostic-positive neighbor-fragment overwrite.
  The packed A/B/C fragments are generated from logical tiles and checked
  against the same exact host-reference matmul.
* `tests/instrumented/015_thread_level_subgroup_test.hip` starts
  multi-subgroup `thread-level` coverage. It uses a 64-thread workgroup split
  into two 32-thread subgroups, checks the `context` thread/subgroup/rank
  helpers, verifies subgroup ids recorded in access records, and asserts
  same-subgroup and cross-subgroup same-epoch diagnostics.
* The current detector uses atomic reservation for access-log and diagnostic-log
  slots. Access records are published with a valid bit before scanning, avoiding
  the wavefront-divergent spinlock deadlock that a device-side metadata lock
  would risk. These atomics and fences are detector-internal and must not be
  treated as user-program synchronization by the shadow model.
* Access logging, basic conflict diagnostics, host reporting, byte-range edge
  cases, epoch-boundary tests, first all-thread array cases, metadata capacity
  tests, looped epoch tests, tiled LDS tests, simple matmul-like tests,
  pipeline-like matmul tests, RDNA4 WMMA row-major/data-tiled tests, and first
  multi-subgroup `thread-level` tests exist. Epoch-local access-log lifetime now
  exists. `subgroup-level` mode, diagnostic quality work, and low-overhead
  per-thread logs are still future work.

The reference corpus is a map of desired coverage, not an obligation to
instrument everything immediately. The instrumented suite should grow only when
the library actually supports the corresponding behavior.

Next implementation slice: `subgroup-level` mode design and first prototype.
The immediate goal is to decide how `thread-level` HIP mode and lower-overhead
`subgroup-level` mode coexist without compromising either contract.

## Foundations

### Terminology

Use common HIP/AMDGPU terminology consistently:

* Thread: one logical thread in a HIP kernel, identified within a workgroup by
  `threadIdx`.
* Workgroup: the set of threads that can communicate through LDS and
  synchronize with `__syncthreads()`/`s_barrier`. In HIP source, this is the
  group indexed by `blockIdx` and sized by `blockDim`.
* Wavefront: the hardware execution grouping. Do not rely on wavefront lockstep
  for HIP-language correctness. For the future assembly-level effort, wavefront
  behavior is part of the hardware model and must be studied directly.
* Subgroup: a library/modeling concept for a subset of a workgroup. This is
  useful for future finer-grained epoch tracking. A subgroup may correspond to a
  wavefront in some modes, especially assembly-oriented modes, but the plan
  should say so explicitly when that is intended.

### Memory-model ground rules

For the HIP-language instrumentation purpose, the mental model is language/IR
first, hardware folklore last.

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

For the assembly-level future effort, the ground rules are different. The
program being instrumented would no longer be judged as HIP source or LLVM IR;
it would be judged by the hardware model. In that setting, wavefront lockstep,
instruction issue, load/store queue behavior, and other hardware details are
not folklore. They are candidates for the actual model. This is why hip-moi
should learn early how to represent multiple subgroups in one workgroup and how
to focus on races between subgroups. Within-subgroup races remain possible
research questions, but they are not the assembly-oriented MVP center of mass.

## MVP

The MVP is the smallest useful library that can be built and tested today.

### MVP contract

The existing MVP diagnoses only this `thread-level` HIP-language model:

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

The planned `subgroup-level` mode will have a different, explicitly narrower
contract:

* It is not a full HIP-language race detector.
* It focuses on conflicting accesses by different subgroups within one
  workgroup.
* It intentionally ignores races among threads in the same subgroup.
* It is motivated by lower-overhead HIP experimentation and by the future
  assembly-level effort, where cross-subgroup races are expected to be the main
  MVP target.
* Any same-subgroup false negative in this mode is expected behavior and must be
  documented as part of the mode contract.

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

enum class instrumentation_mode {
  thread_level,
  subgroup_level,
};

// Non-owning view of detector metadata buffers. Buffers may live in global
// memory or in __shared__ memory.
struct context_storage_ref {
  access_record* access_records;
  int access_record_capacity;
  diagnostic* diagnostics;
  int diagnostic_capacity;
  subgroup_state* subgroup_states;
  int subgroup_capacity;
  int* access_count;
  int* epoch_access_count;
  int* diagnostic_count;
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

struct host_context_options {
  int access_record_capacity;
  int diagnostic_capacity;
  int subgroup_capacity;

  bool destructor_reports;
  bool destructor_aborts;
  FILE* diagnostic_stream;
};

class host_context {
 public:
  // Owns global-memory detector metadata and hands a non-owning view to kernels.
  explicit host_context(host_context_options options = {});

  context_storage_ref device_ref();

  // Synchronizes, copies diagnostics to the host, optionally prints them, and
  // marks them consumed so the destructor will not report/abort on the same
  // diagnostics.
  bool check(FILE* stream = stderr);

  void disable_destructor_reporting();
  void disable_destructor_abort();
  void disable_destructor_check();
};

#define HIP_MOI_CHECK(context) ...

}  // namespace hip_moi
```

Notes:

* First implementation target: `include/hip_moi/hip_moi.hpp`, as a header-only
  library target exported by CMake.
* `context_storage_ref` is a non-owning view of caller-provided metadata
  buffers. Those buffers may be in global memory, which avoids consuming scarce
  LDS in kernels that already use nearly all available shared memory.
* The current implementation uses detector-internal atomics and fences to
  reserve and publish metadata slots. These operations must not be modeled as
  user-program synchronization.
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
* The next API design pass needs an explicit instrumentation-mode story. The
  mode names should be `thread_level` and `subgroup_level` in code, and
  `thread-level` and `subgroup-level` in prose. The `thread-level` mode lets
  every participating thread log accesses, and records need enough identity to
  distinguish threads. The `subgroup-level` mode should be able to use subgroup
  identity instead of thread identity and may eventually use
  subgroup-representative instrumentation, such as only the 0-th thread of each
  subgroup recording a subgroup-level access summary.
* Do not silently make the existing per-thread `lds_load`/`lds_store` wrappers
  mean "only subgroup leader logs" until the contract is clear. Arbitrary
  per-thread accesses are not necessarily summarized by the leader's address.
  A lower-overhead mode may need separate subgroup-level APIs or a different
  record path for accesses that are known to be collective/tile-shaped.
* LDS load/store APIs should be templated immediately. Internally, every memory
  access becomes a byte range: base address plus `sizeof(T)` bytes.
* The first API can require trivially copyable `T`.
* A pointer wrapper can come later, but member functions on the context are the
  easiest way to make real progress today.
* The namespace should avoid claiming to replace HIP builtins. It is a separate
  diagnostic API.
* The API should eventually support labels/source locations, but MVP diagnostics
  can start with compact numeric records.
* End users should not have to manually copy `diagnostic_count` or the raw
  diagnostic buffer. There are two first-class host-side consumption patterns.
  The explicit pattern is `hip_moi::host_context` plus `HIP_MOI_CHECK(moi)`,
  which reports diagnostics to `stderr` and aborts at a precise host source
  location. The scope-based pattern lets `host_context`'s destructor consume any
  still-unconsumed diagnostics; by default, it reports them to `stderr` and
  aborts the process. Advanced users can independently opt out of destructor
  reporting and destructor aborting.

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
      /*thread_count=*/static_cast<int>(blockDim.x),
      /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
      /*subgroup_count=*/1,
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
* thread id in `thread-level` mode,
* subgroup id,
* epoch id,
* optional source id,
* whether the slot is valid.

`subgroup-level` mode should have a compact record form that can omit thread id
if diagnostics never distinguish threads within one subgroup.

On every `lds_load<T>` or `lds_store<T>`:

1. Compute the byte range.
2. Record the new access in the active epoch.
3. Perform the real load or store.

Conflict checking may happen immediately or be deferred. The preferred MVP
direction is deferred checking from per-thread access logs, so the common access
path does not need detector-created cross-thread synchronization. Whenever
checking runs in `thread-level` mode, if byte ranges overlap, thread ids
differ, and either access is a write, record a diagnostic. In `subgroup-level`
mode, the corresponding predicate uses subgroup ids instead of thread ids and
ignores same-subgroup conflicts by contract.

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

Current fallback/debug implementation:

1. Atomically reserve an access-record slot.
2. Publish the access record with a valid bit.
3. Compare the new access with valid access records in the current epoch.
4. Atomically reserve diagnostic slots for any conflicts.
5. Perform the user's real LDS load or store.

This fallback may perturb scheduling and may introduce extra ordering in the
instrumented executable, but diagnostics must still be computed from the shadow
model that ignores detector-created happens-before edges.

This intentionally diagnoses unordered same-epoch conflicting accesses based on
the program order expressed through hip-moi calls, not on a particular hardware
interleaving observed during one run.

Instrumentation may perturb the original bug. That is acceptable if the shadow
model still deterministically reports the conflicting accesses that were present
in the instrumented API trace. The primary value proposition is to turn racy
patterns into deterministic diagnostics, not to preserve the original race. The
bad failure mode is the conjunction of perturbing or hiding the original race and
also failing to report it in the shadow diagnostic.

### Immediate multi-subgroup generalization

The first MVP used one full-workgroup epoch. The next milestone is to make
subgroup identity real in the implementation and tests. Keep metadata shaped as:

```c++
epoch[subgroup_id]
```

with `subgroup_count == 1` remaining a valid degenerate case.

This supports kernels where independent subgroups within one workgroup operate
on different shared tiles or synchronize at different granularities. The first
multi-subgroup implementation can still treat `ctx.syncthreads()` as a
full-workgroup operation that advances all subgroup epochs together. Later
subgroup-local synchronization can advance only the participating subgroup.

Required early tests:

* a 64-thread workgroup split into two 32-thread subgroups,
* a cross-subgroup same-epoch LDS conflict that reports,
* a same-subgroup same-epoch LDS conflict that reports in `thread-level` mode,
* the same same-subgroup conflict intentionally not reporting in
  `subgroup-level` mode,
* cross-subgroup accesses separated by `ctx.syncthreads()` not reporting.

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

### Initial MVP implementation steps

These steps describe the original tiny-but-real MVP target: a library with GPU
tests that compile and run.

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

5. Add initial test cases.
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
  001_safe_mvp_test.hip
  002_race_mvp_test.hip
  003_host_context_test.hip
  004_basic_conflict_predicate_test.hip
  005_epoch_boundary_test.hip
  006_all_thread_array_test.hip
  007_metadata_capacity_test.hip
  008_loop_epoch_test.hip
  009_tiled_lds_test.hip
  010_matmul_like_test.hip
  011_epoch_log_lifetime_test.hip
  012_matmul_pipeline_test.hip
  013_rdna4_wmma_row_major_test.hip
  014_rdna4_wmma_data_tiled_test.hip
  015_thread_level_subgroup_test.hip
  test_support.hpp
```

`001_safe_mvp_test.hip` should assert both numerical outputs and zero
diagnostics. `002_race_mvp_test.hip` should assert deterministic diagnostics;
for racy kernels, numerical output is not the oracle.
`003_host_context_test.hip` asserts end-user behavior: explicit checks,
stderr/fatal policy, and scope-based destructor handling.
`004_basic_conflict_predicate_test.hip` asserts the basic MVP predicate around
same-epoch byte ranges before the suite moves on to epoch-boundary behavior.
`005_epoch_boundary_test.hip` asserts the MVP epoch-boundary behavior of
uniform `ctx.syncthreads()`.
`006_all_thread_array_test.hip` asserts first all-thread LDS array behavior,
including a missing-barrier diagnostic case.
`007_metadata_capacity_test.hip` asserts access-log overflow and diagnostic
buffer truncation behavior, including the user-facing host report.
`008_loop_epoch_test.hip` asserts looped epoch behavior, including repeated
safe barriers and repeated missing-barrier diagnostics.
`009_tiled_lds_test.hip` asserts 2D tile layouts, tiled gathers, and an
unsynchronized transpose diagnostic.
`010_matmul_like_test.hip` asserts cooperative LDS matmul-like access patterns
using explicit small integer inputs, host-reference output checks, and a scalar
missing-barrier diagnostic.
`011_epoch_log_lifetime_test.hip` asserts that access-log capacity is scoped to
the active epoch rather than the cumulative kernel trace.
`012_matmul_pipeline_test.hip` asserts double-buffered and pipeline-like matmul
LDS patterns with host-reference output checks, including diagnostic-positive
buffer reuse cases.
`013_rdna4_wmma_row_major_test.hip` asserts RDNA4/gfx12 WMMA intrinsic coverage
using all 32 threads, conventional row-major LDS tiles, non-uniform exact
inputs, and host-reference output checks.
`014_rdna4_wmma_data_tiled_test.hip` asserts the matching RDNA4/gfx12 WMMA
coverage for packed A/B/C fragments laid out at `lane * fragment_size` byte
offsets, with packed data generated from logical tiles and checked against a
host-reference matmul.
`015_thread_level_subgroup_test.hip` asserts first multi-subgroup
`thread-level` behavior: helper-derived subgroup identity, per-record subgroup
ids, same-subgroup diagnostics, cross-subgroup diagnostics, and full-workgroup
barrier separation across subgroups.

Tutorial examples live under `docs/tutorial/`. They are not a coverage corpus;
they are executable documentation for the user-facing workflow. The README may
quote short excerpts, but the quoted code should be periodically swept against
the actual compiled examples:

```text
docs/tutorial/
  README.md
  001_passing_syncthreads.hip
  002_failing_same_epoch_race.hip
  003_destructor_fallback.hip
  004_rdna4_wmma_data_tiled_matmul.hip
```

Incremental instrumented test growth:

1. Add the smallest safe instrumented kernel when `ctx.lds_load` and
   `ctx.lds_store` exist. Done.
2. Add the smallest same-epoch write/read diagnostic when overlap detection
   exists. Done.
3. Add the host-side `HIP_MOI_CHECK` path and scope-based destructor handling
   for unconsumed diagnostics. Done.
4. Add write/write diagnostics and basic byte-range edge cases. Done.
5. Add `ctx.syncthreads()` separation tests when epoch advancement exists. Done.
6. Add all-thread array cases when per-thread metadata and byte-range tracking
   are solid. Done.
7. Add metadata capacity and diagnostic truncation tests. Done.
8. Add loops when repeated epochs are solid. Done.
9. Add tiled LDS cases when the basic machinery has survived enough pressure.
   Done.
10. Add matmul-like LDS cases. Done.
11. Add epoch-local access-log lifetime so long synchronized loops do not fill
    the log with obsolete epochs. Done.
12. Add double-buffered and pipeline-like matmul LDS cases, including more
    diagnostic-positive matmul-shaped races. Done.
13. Add RDNA4-gated real WMMA tests for both conventional row-major LDS tiles
    and data-tiled packed fragments. Done.
14. Add `thread-level` multi-subgroup tests: at least two subgroups in one
    workgroup, cross-subgroup conflicts, same-subgroup conflicts, and
    full-workgroup synchronization separating subgroups. Done.
15. Add `subgroup-level` mode tests: cross-subgroup conflicts still report,
    same-subgroup conflicts intentionally do not report, and the mode contract
    is visible in diagnostic metadata and host reports.
16. Prototype lower-overhead subgroup-representative logging for tile-shaped
    accesses if the design is clear enough; otherwise document the blockers and
    keep using all-thread logging filtered by subgroup id.
17. Improve diagnostic quality with labels/source locations and first-conflict
    preservation.

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

## Revised Near-Term Milestone

The refined MVP is now defined less by diagnostic polish and more by whether the
library teaches us the right subgroup abstractions.

1. Make subgroup identity operational. Done for the first `thread-level` slice.
   * Compute `subgroup_id` and thread rank within subgroup from runtime config.
   * Support `subgroup_count > 1` in storage initialization and epoch state.
   * Keep `ctx.syncthreads()` as a full-workgroup operation initially, advancing
     all subgroup epochs together.
   * Add two-subgroup tests using one workgroup.

2. Preserve the existing `thread-level` HIP mode. Done for first
   multi-subgroup diagnostics.
   * Continue detecting same-epoch conflicts between any two different threads.
   * Include same-subgroup and cross-subgroup diagnostic-positive tests.
   * Keep this mode principled in the HIP/LLVM memory model.

3. Design the `subgroup-level` mode.
   * Define the mode as intentionally ignoring same-subgroup conflicts.
   * Decide whether mode selection is a runtime enum, a separate context type,
     a storage policy, or some other zero-overhead-in-practice shape.
   * Avoid virtual dispatch in device code.
   * Decide whether the first implementation logs all thread accesses and
     filters by subgroup id, or introduces a subgroup-representative API.

4. Prototype the `subgroup-level` mode.
   * Produce deterministic diagnostics for cross-subgroup conflicts.
   * Produce no diagnostic for same-subgroup conflicts by contract.
   * Measure or at least inspect the metadata footprint difference from
     `thread-level` mode.
   * Investigate whether thread id can be removed from the hot access record in
     this mode.

5. Explore subgroup-representative instrumentation.
   * Candidate direction: only the 0-th thread of each subgroup records a
     subgroup-level access summary.
   * Do this only for access patterns where the representative can describe the
     whole subgroup's memory footprint, such as tile-shaped or fragment-shaped
     operations.
   * Do not pretend this summarizes arbitrary per-thread pointer accesses.
   * Determine whether this can reduce atomics or eliminate them from the common
     access path.

6. Then resume diagnostic quality work.
   * Add source/location IDs or stringless labels.
   * Add first-conflict preservation so later conflicts do not hide the useful
     one.
   * Add clearer mode-aware host diagnostics, especially for
     `subgroup-level` false negatives by design.

7. Keep synchronization lowering notes on the horizon.
   * Compile tiny examples using `ctx.syncthreads()`, raw `__syncthreads()`,
     `__builtin_amdgcn_s_barrier`, and `__builtin_amdgcn_fence`.
   * Save or document their LLVM IR shape for the HIP-language model.
   * Separately note which hardware facts would matter for the future
     assembly-level model.
   * Do not infer HIP-level cross-thread synchronization from naked fences.

## Plan beyond the MVP

After the refined MVP, widen scope in small semantic steps. Each step should add
one new kind of happens-before edge or one new class of instrumented access, with
tests that prove both positive and negative cases.

### Step 1: Multi-subgroup `thread-level` HIP mode

Allow a workgroup to contain multiple tracked subgroups without changing the
HIP-language diagnostic contract.

* Add multiple epoch counters per workgroup.
* Compute subgroup id and thread rank within subgroup from `config`.
* Preserve `thread-level` diagnostics within and across subgroups.
* Treat `ctx.syncthreads()` as a full-workgroup epoch boundary that advances all
  subgroup epochs.
* Test subgroup A and subgroup B operating independently on non-overlapping LDS
  ranges.
* Test same-address communication within one subgroup.
* Test data crossing subgroup boundaries only after an explicit full-workgroup
  synchronization.

### Step 2: `subgroup-level` mode design

Design a second instrumentation mode that intentionally ignores conflicts among
threads in the same subgroup.

* Use `subgroup_level` as the code-facing mode name and `subgroup-level` in
  prose.
* Document its false-negative contract.
* Decide how users select the mode.
* Decide whether the mode shares `context` with `thread-level` mode or uses a
  separate context/storage wrapper.
* Decide whether mode selection must be compile-time for the hot path or can be
  a runtime value that clang optimizes away in local use.
* Avoid virtual dispatch in device code.
* Design a compact access record that tracks subgroup id but can omit thread id.
* Decide how diagnostics should report "subgroup A vs subgroup B" without
  pretending to identify exact threads.

### Step 3: `subgroup-level` mode prototype

Build the narrow mode before pursuing more esoteric synchronization semantics.

* First implementation may log all thread accesses and filter same-subgroup
  conflicts out of the shadow model.
* Then investigate subgroup-representative logging where only the 0-th thread
  of each subgroup records a subgroup-level access summary.
* Use representative logging only for access patterns whose byte ranges can be
  summarized by a subgroup leader, such as cooperative tile or fragment
  operations.
* Measure or inspect metadata size, access-record fields, atomic usage, and
  generated code for both modes.
* Add paired tests showing the same kernel reports in `thread-level` mode but
  intentionally does not report same-subgroup conflicts in `subgroup-level`
  mode.
* Add matmul-shaped cross-subgroup conflicts, because this is the main bridge to
  the future assembly-level effort.

### Step 4: Real-kernel LDS coverage under both modes

Broaden the corpus before broadening the memory model too much.

* Extract more LDS tiling patterns from
  `/home/benoit/workspace/hip-matmul/matmul_rdna4.hip`.
* Cover vectorized LDS accesses.
* Cover double-buffered LDS.
* Cover repeated tile loops with multiple synchronization phases.
* Keep RDNA4/gfx12-specific WMMA coverage split by input layout: conventional
  row-major tiles and data-tiled packed fragments.
* For each useful real-kernel idiom, decide whether it belongs in
  `thread-level` mode, `subgroup-level` mode, or both.
* Keep atomics out of these tests until the atomic model exists.

### Step 5: Hard synchronization negative tests

Add a way to test bad synchronization patterns that might otherwise hang.

* Add simulation mode for synchronization APIs, where the library records
  barrier intent without executing a real barrier.
* Use simulation mode to diagnose missing participation in a barrier.
* Keep real-barrier divergent tests separate, probably with process-level
  timeout handling if they are ever useful.

### Step 6: Atomic access instrumentation, without new ordering

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

### Step 7: Acquire/release atomics as synchronization

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

### Step 8: Explicit fences paired with atomic handoff

Only after Step 7, add explicit fence modeling. This is the incremental version
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

### Step 9: RMWs and release sequences

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

### Step 10: Sync scopes and stronger orderings

Only after the simple acquire/release cases work, widen to target-specific
details.

* Model relevant LLVM/HIP sync scopes.
* Distinguish workgroup-scope, agent/device-scope, and system-scope operations
  when they appear in generated IR.
* Decide how much `seq_cst` support is worth implementing.
* Add tests that compare source-level HIP calls with the LLVM IR they produce.

### Step 11: Ergonomics and runtime support

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
* Multi-subgroup mode tests: use the same logical kernels in `thread-level` and
  `subgroup-level` modes when possible, so the mode contract is visible in the
  expected diagnostics.
* Cross-subgroup matmul idioms: include cases where different subgroups in one
  workgroup cooperate through LDS tiles, because these are the strongest bridge
  to the future assembly-level project.
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
* Keep instrumentation modes explicit. A false negative that is unacceptable in
  `thread-level` HIP mode may be the intended contract in `subgroup-level`
  mode.
