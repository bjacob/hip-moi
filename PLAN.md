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

The initial single-subgroup MVP exists, first multi-subgroup tests exist, and
`subgroup-level` now exists as a separate mode. The next priority is to keep
meeting ordinary HIP code where it is: users should instrument the LDS accesses
they actually wrote, while the library improves diagnostics, multi-subgroup
coverage, and synchronization modeling around that access-level API.

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
  expected-failing diagnosed examples, a subgroup-level cross-subgroup
  diagnostic example, a subgroup-level coalescing opt-in example, and a
  gfx12-gated real RDNA4 data-tiled WMMA matmul example. The README is the
  tutorial's primary content: it introduces `thread-level` and `subgroup-level`
  as peer instrumentation modes, then explains each example as a plain HIP
  kernel first and as the corresponding hip-moi instrumented kernel second. The
  standalone `.hip` programs serve as compiled companions. The matmul tutorial
  presents the data-tiled layout through per-lane vector fragment loads and
  stores.
* `docs/coalescing.md` explains the current coalescing model: users opt in with
  nonzero `site_id` values, default zero-site accesses remain exact-only, the
  current implementation emits subgroup-level summaries at epoch boundaries,
  thread-level mode remains exact-record based, subgroup-level summaries
  participate in epoch-close conflict detection, subgroup-level mode proves
  summaries from a separate optional lane-based coalescing access log, and the
  detected patterns are currently limited to simple contiguous or fixed-stride
  lane-to-address shapes. For subgroup-level opted-in accesses,
  `coalescing_access_record` is also the hot-path access representation when
  storage is available; ordinary exact `access_record` logging is used only for
  default-site accesses and coalescing-access overflow/fallback. The doc now
  specifies the subgroup-level coalescing counters, including
  `coalescing_fallback_count` for opted-in accesses that fell back to exact
  records.
* `docs/context.md` describes The context allocation plan. Host-owned contexts
  now take a byte budget, defaulting to 16 MiB of device global memory, and
  partition one HIP allocation into the typed buffers needed by the selected
  mode. Fine-grained typed capacities remain as advanced/test overrides, while
  `storage_ref` and `static_context_storage` remain available for users who
  deliberately provide their own storage.
* `tests/reference/mvp_reference_kernels.hip` contains the uninstrumented
  reference corpus. It is a parameterized GTest suite exposing one CTest entry
  per launched safe reference kernel.
* The reference corpus currently contains 55 safe kernels that compile, launch,
  and numerically check their outputs.
* The same source also contains compile-only diagnostic-positive and hard
  synchronization references that are intentionally not launched.
* The public headers are split by role. `hip_moi/common.hpp` contains shared
  enums, `subgroup_state`, and small helper routines.
  `hip_moi/thread_level_context.hpp` contains `thread_level_context`.
  `hip_moi/subgroup_level_context.hpp` contains `subgroup_level_context`.
  `hip_moi/host_context.hpp` contains host ownership and reporting, and
  `hip_moi/hip_moi.hpp` is the umbrella include.
* The current `thread_level_context` and `subgroup_level_context`
  implementations now have separate public type families: each context owns its
  own `config`, `access_record`, `diagnostic`, `storage_ref`, and
  `static_context_storage` types. `thread_level_context::access_record` keeps
  `thread_id`; `subgroup_level_context::access_record` intentionally does not.
  `thread_level_context::diagnostic` reports thread ids, while
  `subgroup_level_context::diagnostic` reports subgroup ids. Both contexts still
  have real same-epoch access logging: `init_workgroup()` initializes counters,
  epoch storage, and access record validity; `lds_load<T>` and `lds_store<T>`
  record the LDS byte range before performing the real access; and
  `syncthreads()` executes a real barrier while advancing epoch slots. Detector
  metadata initialization and epoch increments use detector-internal device
  fences before releasing the workgroup so global-memory metadata is visible to
  the participating threads. `subgroup_level_context` now checks conflicts at
  epoch close instead of scanning prior records on every access: `lds_load<T>`
  and `lds_store<T>` append exact records, `ctx.syncthreads()` closes the
  just-ended epoch, and a final uniform `ctx.finish()`/`ctx.close_epoch()` is
  available for kernels whose last epoch is not followed by another barrier.
* Instrumented tests that are intentionally limited to a one-subgroup workgroup
  now use `single_subgroup` near the start of the file name.
  Diagnostic-positive single-subgroup race tests have subgroup-level companion
  checks asserting zero diagnostics, making the mode contract explicit: a
  single-subgroup workgroup has no cross-subgroup race for subgroup-level mode
  to report. Purely diagnostic-free single-subgroup tests do not need redundant
  subgroup-level mirrors.
* `tests/instrumented/001_single_subgroup_safe_mvp_test.hip` contains the first instrumented
  kernel test. It passes caller-provided global-memory metadata storage into a
  kernel, performs a same-thread instrumented LDS store/load, checks the
  numerical result, asserts two logged accesses, and asserts zero diagnostics.
* `tests/instrumented/002_single_subgroup_race_mvp_test.hip` contains the first
  diagnostic-positive instrumented kernel. It checks that a same-epoch LDS
  write/read byte-range conflict from two different threads produces one
  deterministic thread-level diagnostic, and that the same one-subgroup
  conflict is intentionally not reported by subgroup-level mode.
* `tests/instrumented/003_host_context_test.hip` contains the first
  user-facing host diagnostics tests. It exercises `hip_moi::host_context`,
  `HIP_MOI_CHECK`, explicit nonfatal diagnostic consumption, the default
  scope-based destructor handling of unconsumed diagnostics, and the destructor
  reporting/abort opt-outs. It also exercises
  `hip_moi::subgroup_level_host_context`, whose diagnostics print subgroup ids
  instead of representative or scaled thread ids. The tests check stderr
  diagnostic text and abort behavior using GTest stderr capture and death tests.
* `tests/instrumented/004_single_subgroup_basic_conflict_predicate_test.hip` broadens the raw
  detector-contract coverage for same-epoch byte ranges: read/read same address,
  write/write same address, non-overlapping writes, adjacent byte ranges, and an
  overlapping full-object/subobject write.
* `tests/instrumented/005_single_subgroup_epoch_boundary_test.hip` exercises uniform
  `ctx.syncthreads()` as the MVP epoch boundary. It checks that same-address
  accesses separated by a barrier do not report, repeated reuse across epochs
  does not report, and a new same-epoch conflict after a barrier reports in the
  new epoch.
* `tests/instrumented/006_single_subgroup_all_thread_array_test.hip` starts the all-thread
  ladder step. It covers independent per-thread LDS writes, own-slot reads after
  a barrier, neighbor reads after a barrier, and an intentionally missing-barrier
  neighbor-read diagnostic case.
* `tests/instrumented/007_single_subgroup_metadata_capacity_test.hip` covers access-log
  overflow, diagnostic counters that exceed stored diagnostic capacity, and the
  host-facing stderr report for truncated diagnostic buffers.
* `tests/instrumented/008_single_subgroup_loop_epoch_test.hip` covers looped epoch patterns:
  safe scalar producer/consumer loops, all-thread own-slot loops, repeated
  missing-barrier diagnostics, and diagnostic epoch numbering across loop
  iterations.
* `tests/instrumented/009_single_subgroup_tiled_lds_test.hip` covers 2D tiled LDS idioms:
  row-major copy, transpose, skewed stride, blocked layout, diagonal gather,
  striped load/store, and an unsynchronized transpose diagnostic case.
* `tests/instrumented/010_single_subgroup_matmul_like_test.hip` covers small cooperative LDS
  matmul idioms: simple 2x2 and 4x4 tiles, a chunked K loop, and a scalar
  missing-barrier diagnostic. The numerical tests use explicit small integer
  input matrices and compare GPU outputs against a host-side reference matmul.
* `tests/instrumented/011_single_subgroup_epoch_log_lifetime_test.hip` verifies that access-log
  storage is reused at epoch boundaries, so long synchronized loops can run with
  capacity sized for one epoch rather than the whole kernel.
* `tests/instrumented/012_single_subgroup_matmul_pipeline_test.hip` covers double-buffered and
  pipeline-like matmul LDS idioms: safe ping-pong buffering plus
  diagnostic-positive buffer reuse and partial tile overwrite cases. The safe
  output cases use explicit small integer inputs and a host-side reference
  matmul oracle.
* `tests/instrumented/013_single_subgroup_rdna4_wmma_row_major_test.hip` is a gfx12-gated real
  RDNA4 WMMA smoke test using
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`, all 32 threads,
  conventional row-major LDS tiles, single-buffer and double-buffer safe cases,
  and a diagnostic-positive row overwrite. The safe cases now use non-uniform
  small integer-valued `_Float16` inputs and exact host-reference outputs. The
  diagnostic-positive one-subgroup overwrite has a subgroup-level companion
  check asserting zero diagnostics.
* `tests/instrumented/014_single_subgroup_rdna4_wmma_data_tiled_test.hip` is the matching
  gfx12-gated packed-layout test. It uses the same WMMA intrinsic, but each
  thread's A/B fragment is a contiguous 16-byte object at byte offset
  `lane * 16`, and each thread's C accumulator fragment is a contiguous
  32-byte object at byte offset `lane * 32`, stored with one `f32x8_t` vector
  store. The test includes a diagnostic-positive neighbor-fragment overwrite.
  The packed A/B/C fragments are generated from logical tiles and checked
  against the same exact host-reference matmul. The one-subgroup overwrite also
  has a subgroup-level companion check asserting zero diagnostics.
* `tests/instrumented/015_thread_level_subgroup_test.hip` starts
  multi-subgroup `thread-level` coverage. It uses a 64-thread workgroup split
  into two 32-thread subgroups, checks the `thread_level_context`
  thread/subgroup/lane helpers, verifies subgroup ids recorded in access
  records, and asserts same-subgroup and cross-subgroup same-epoch diagnostics.
* `tests/instrumented/016_subgroup_level_bootstrap_test.hip` starts
  `subgroup-level` coverage. It uses `subgroup_level_context` as a separate
  type, uses subgroup-specific access records and diagnostics, reports
  cross-subgroup same-epoch conflicts, and intentionally ignores same-subgroup
  conflicts. It also verifies that subgroup-level conflicts are deferred until
  `ctx.finish()` when there is no intervening `ctx.syncthreads()`.
* `tests/instrumented/017_subgroup_level_multisubgroup_test.hip` broadens
  `subgroup-level` coverage to a 128-thread, four-subgroup workgroup. It covers
  independent subgroup LDS slots, cross-subgroup array conflicts, barriers that
  separate cross-subgroup communication, looped conflict epochs, tiled row
  layouts, tiled row collisions, and matmul-shaped cooperative tile sharing with
  both safe and missing-barrier cases.
* `tests/instrumented/018_rdna4_multisubgroup_wmma_data_tiled_test.hip` is a
  gfx12-gated real WMMA bridge test under both modes. It uses a 64-thread
  workgroup split into two 32-thread subgroups, data-tiled vector fragments,
  double-buffered two-tile LDS staging, exact host-reference output checks, and
  a missing-barrier cross-subgroup diagnostic case checked in both
  `thread-level` and `subgroup-level` modes.
* `tests/instrumented/019_rdna4_multisubgroup_wmma_row_major_test.hip` is the
  matching gfx12-gated row-major WMMA bridge test under both modes. It uses a
  64-thread workgroup split into two 32-thread subgroups, conventional
  row-major A/B/C tiles, double-buffered two-tile LDS staging, exact
  host-reference output checks, and a missing-barrier cross-subgroup diagnostic
  case checked in both `thread-level` and `subgroup-level` modes.
* `tests/instrumented/020_site_id_test.hip` covers exact source-site id
  plumbing. It checks that `HIP_MOI_SITE_ID()` produces nonzero compile-time
  ids, default accesses still record `site_id == 0`, explicit site ids are
  stored in access records, and both thread-level and subgroup-level
  diagnostics carry site ids without changing the detector's exact behavior.
* `tests/instrumented/022_subgroup_level_coalescing_test.hip` covers
  subgroup-level coalescing access logs and summaries. It checks that default
  site ids do not write coalescing access records, opted-in sites write one
  coalescing access record per lane when storage exists, opted-in sites fall
  back to exact records when coalescing access storage is absent, contiguous and
  fixed-stride lane patterns summarize, repeated lanes are rejected, and
  independent subgroups produce separate summaries.
* `tests/instrumented/023_subgroup_level_coalesced_conflict_test.hip` starts
  using subgroup-level coalesced summaries in conflict detection. It checks
  summary-vs-summary diagnostics, summary-vs-exact diagnostics,
  summary-vs-unsummarized coalescing diagnostics, coalescing-storage overflow
  and absent-storage exact fallback, read/read silence for coalesced and
  unsummarized coalescing accesses, disjoint summaries, and fixed-stride gaps
  whose enclosing spans overlap but whose represented per-lane byte ranges do
  not.
* The current detector uses atomic reservation for access-log and diagnostic-log
  slots. Access records are published with a valid bit before scanning, avoiding
  the wavefront-divergent spinlock deadlock that a device-side metadata lock
  would risk. These atomics and fences are detector-internal and must not be
  treated as user-program synchronization by the shadow model.
* Access logging, basic conflict diagnostics, host reporting, byte-range edge
  cases, epoch-boundary tests, first all-thread array cases, metadata capacity
  tests, looped epoch tests, tiled LDS tests, simple matmul-like tests,
  pipeline-like matmul tests, RDNA4 WMMA row-major/data-tiled tests, and first
  multi-subgroup `thread-level` and `subgroup-level` tests exist. Epoch-local
  access-log lifetime now exists. Dissociated `subgroup-level` records and
  diagnostics now exist. Mode-aware subgroup-level host reporting now exists.
  The tutorial now presents thread-level and subgroup-level modes as peers, and
  the single-subgroup instrumented tests now mark themselves in their file names
  while checking subgroup-level silence for representative diagnostic-positive
  intra-subgroup races. Real multi-subgroup RDNA4 WMMA bridge tests now exist
  for both data-tiled and row-major layouts under both modes. Diagnostic quality
  work, access-level multi-subgroup refinement, and synchronization lowering
  notes are still future work. Exact `site_id` plumbing now exists: callers may
  pass an explicit `hip_moi::site_id`, `HIP_MOI_SITE_ID()` builds nonzero site
  ids by compile-time hashing, and access records plus diagnostics carry compact
  numeric site ids. Thread-level coalescing has been dropped for now; thread-level
  mode remains exact-record based because its contract requires per-thread race
  visibility. The subgroup-level coalescing-access pass remains: nonzero-site subgroup
  accesses write lane/address `coalescing_access_record`s instead of ordinary
  exact `access_record`s when coalescing access storage is available, and
  `ctx.syncthreads()` summarizes proven contiguous or fixed-stride
  lane-to-address patterns. Subgroup-level conflict detection is now deferred to
  epoch close: newly emitted summaries are compared with other summaries and
  with unsummarized exact or coalescing access records, then remaining
  unsummarized exact and coalescing access records are compared with each other.
  The subgroup-level summary
  builder now avoids per-lane coalescing-access-log rescans for each candidate
  group by accumulating lane masks and endpoints in one scan and validating the
  fixed-stride pattern in one additional scan. Subgroup-level mode also has
  optional epoch-close grouping scratch storage: when supplied, it builds one
  group record per subgroup/site/kind/size key in an open-addressed scratch
  table before summary validation, avoiding the older prior-record leader scan
  across many distinct sites. Subgroup-level storage also tracks
  `coalescing_fallback_count`, the number of opted-in accesses that could not
  use the coalescing access log and therefore fell back to ordinary exact
  records.

The reference corpus is a map of desired coverage, not an obligation to
instrument everything immediately. The instrumented suite should grow only when
the library actually supports the corresponding behavior.

Next implementation slice: improve diagnostics on top of the access-level
instrumentation path. The near-term work should preserve first-conflict
information, make host reports more useful now that site ids exist, keep
expanding multi-subgroup tests that replace real LDS accesses with
`ctx.lds_load`/`ctx.lds_store`, reduce remaining subgroup-level coalescing costs
where that can be done without weakening correctness, and start recording
lowering notes for `ctx.syncthreads()` and lower-level builtins.

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
* Lane: a thread's local position within a subgroup. When a modeled subgroup
  corresponds to a wavefront, this is the usual AMDGPU lane concept. For now the
  implementation computes it from `thread_id % threads_per_subgroup`; future
  code should prefer names such as `lane_in_subgroup` over "rank".

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

struct subgroup_state;

class thread_level_context {
 public:
  struct config {
    // Number of threads participating in this context.
    int thread_count;
    // Number of threads in each full subgroup; the final subgroup may be partial.
    int threads_per_subgroup;
    // Number of subgroups represented in this context.
    int subgroup_count;
  };

  struct access_record {
    uintptr_t address;
    uint32_t byte_count;
    uint32_t thread_id;
    uint32_t subgroup_id;
    uint32_t epoch;
    uint32_t kind;
    uint32_t valid;
    uint64_t site_id;
  };

  struct diagnostic {
    uint32_t kind;
    uint32_t epoch;
    uint32_t first_thread_id;
    uint32_t second_thread_id;
    uintptr_t first_addr;
    uintptr_t second_addr;
    uint32_t first_size;
    uint32_t second_size;
    uint64_t first_site_id;
    uint64_t second_site_id;
  };

  // Non-owning view of thread-level detector metadata buffers.
  struct storage_ref {
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
    __device__ storage_ref ref();
  };

  // Binds the context to caller-provided storage and runtime shape metadata.
  __device__ thread_level_context(storage_ref storage, config cfg);

  // Thread-level diagnostics report conflicts between distinct threads.
  // Provides init_workgroup, templated lds_load/lds_store, syncthreads,
  // has_error/error_count, and thread/subgroup identity helpers.
};

class subgroup_level_context {
 public:
  struct config {
    int thread_count;
    int threads_per_subgroup;
    int subgroup_count;
  };

  struct access_record {
    uintptr_t address;
    uint32_t byte_count;
    uint32_t subgroup_id;
    uint32_t epoch;
    uint32_t kind;
    uint32_t valid;
    uint64_t site_id;
  };

  struct coalesced_access_record {
    uintptr_t first_address;
    uint32_t byte_count;
    uint32_t span_byte_count;
    uint32_t first_lane;
    uint32_t subgroup_id;
    uint32_t epoch;
    uint32_t kind;
    uint32_t participant_count;
    uint32_t valid;
    int64_t address_stride;
    uint64_t site_id;
  };

  struct coalescing_access_record {
    uintptr_t address;
    uint32_t byte_count;
    uint32_t lane;
    uint32_t subgroup_id;
    uint32_t epoch;
    uint32_t kind;
    uint32_t valid;
    uint64_t site_id;
  };

  struct coalescing_group_record;

  struct diagnostic {
    uint32_t kind;
    uint32_t epoch;
    uint32_t first_subgroup_id;
    uint32_t second_subgroup_id;
    uintptr_t first_addr;
    uintptr_t second_addr;
    uint32_t first_size;
    uint32_t second_size;
    uint64_t first_site_id;
    uint64_t second_site_id;
  };

  // Non-owning view of subgroup-level detector metadata buffers.
  struct storage_ref {
    access_record* access_records;
    int access_record_capacity;
    diagnostic* diagnostics;
    int diagnostic_capacity;
    subgroup_state* subgroup_states;
    int subgroup_capacity;
    int* access_count;
    int* epoch_access_count;
    int* diagnostic_count;
    coalesced_access_record* coalesced_access_records;
    int coalesced_access_record_capacity;
    int* coalesced_access_count;
    coalescing_access_record* coalescing_access_records;
    int coalescing_access_record_capacity;
    int* coalescing_access_count;
    int* epoch_coalescing_access_count;
    int* coalescing_fallback_count;
    coalescing_group_record* coalescing_group_records;
    int coalescing_group_record_capacity;
    int* coalescing_group_count;
  };

  template <int AccessCapacity, int DiagnosticCapacity, int SubgroupCapacity = 1>
  struct static_context_storage {
    __device__ storage_ref ref();
  };

  // Binds the context to caller-provided storage and runtime shape metadata.
  __device__ subgroup_level_context(storage_ref storage, config cfg);

  // Subgroup-level diagnostics report conflicts between distinct subgroups and
  // intentionally ignore same-subgroup conflicts.
  // Provides the same user-facing operations as thread_level_context.
};

class site_id {
 public:
  constexpr site_id() = default;
  explicit constexpr site_id(uint64_t value);
  constexpr uint64_t value() const;
  constexpr bool allows_coalescing() const;
};

inline constexpr site_id no_site_id{};

// Temporary compatibility alias while older examples migrate.
using context = thread_level_context;

struct host_context_options {
  size_t storage_bytes;  // Default: 16 MiB.

  // Advanced/test overrides. Negative means "auto from storage_bytes";
  // zero disables optional coalescing buffers.
  int access_record_capacity;
  int coalesced_access_record_capacity;
  int coalescing_access_record_capacity;
  int coalescing_group_record_capacity;
  int diagnostic_capacity;

  // Number of subgroups represented in this context.
  int subgroup_capacity;

  bool destructor_reports;
  bool destructor_aborts;
  FILE* diagnostic_stream;
};

class host_context {
 public:
  // Owns thread-level global-memory detector metadata and hands a non-owning view
  // to kernels.
  explicit host_context(host_context_options options = {});

  thread_level_context::storage_ref device_ref();

  // Synchronizes, copies diagnostics to the host, optionally prints them, and
  // marks them consumed so the destructor will not report/abort on the same
  // diagnostics.
  bool check(FILE* stream = stderr);

  void disable_destructor_reporting();
  void disable_destructor_abort();
  void disable_destructor_check();
};

#define HIP_MOI_CHECK(context) ...

#define HIP_MOI_SITE_ID() ...

}  // namespace hip_moi
```

Notes:

* First implementation target: `include/hip_moi/hip_moi.hpp`, as a header-only
  library target exported by CMake.
* Each context's `storage_ref` is a non-owning view of caller-provided metadata
  buffers. Those buffers may be in global memory, which avoids consuming scarce
  LDS in kernels that already use nearly all available shared memory.
* The host-owned context path follows The context allocation plan: ordinary
  users provide `host_context_options::storage_bytes`, defaulting to 16 MiB, and
  the host context partitions one device global-memory allocation into aligned
  typed slices. Fine-grained typed capacities are advanced overrides and test
  controls, not the primary user-facing allocation model.
* The current implementation uses detector-internal atomics and fences to
  reserve and publish metadata slots. These operations must not be modeled as
  user-program synchronization.
* Each context's `static_context_storage` is only a convenience helper for users
  who do want fixed-size storage, for example in `__shared__` memory. Its
  capacities are template parameters because that is how the helper embeds
  fixed-size arrays. The main context types are not templated on capacities.
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
* The mode names are `thread_level` and `subgroup_level` in code, and
  `thread-level` and `subgroup-level` in prose. Users select a mode by choosing
  `thread_level_context` or `subgroup_level_context`, not by setting a runtime
  mode field.
* `lds_load` and `lds_store` should take an optional `site_id` argument with
  default value `hip_moi::no_site_id`. The default `site_id{0}` means exact
  logging only and does not allow coalescing. In subgroup-level mode, a nonzero
  site id opts that access site into possible automatic coalescing when
  coalescing storage is supplied, while preserving exact diagnostic semantics as
  the fallback. In thread-level mode, site ids are diagnostic metadata only.
* `site_id` should be a tiny explicit wrapper around `uint64_t`, not a naked
  integer in the public API. The explicit constructor and getter prevent
  accidental argument-order mixups, especially in `lds_store(ptr, value, site)`.
* Provide one source-site macro, `HIP_MOI_SITE_ID()`, that expands to a
  `hip_moi::site_id`. Do not hide the whole `lds_load` or `lds_store` operation
  behind macros. A user should write, for example,
  `ctx.lds_load(&lds[i], HIP_MOI_SITE_ID())`.
* `HIP_MOI_SITE_ID()` should be implemented with Clang/GCC-compatible
  compile-time source-location data. Prefer Clang's source-location builtins
  where available, especially `__builtin_LINE()`, `__builtin_COLUMN()`, and
  `__COUNTER__`, with `__FILE__` hashed into the id for cross-file
  discrimination. Use a pattern such as
  `detail::site_id_constant<detail::make_site_id(...)>::value` so
  `make_site_id` is required to constant-evaluate; filename hashing must not
  become a runtime device-side string loop.
* The `thread-level` mode lets every participating thread log accesses, and
  records keep enough identity to distinguish threads. The `subgroup-level`
  mode uses subgroup identity instead of thread identity in its hot record and
  diagnostic types. Its primary user model is still access-level
  instrumentation: users replace the LDS loads and stores they wrote with
  `ctx.lds_load` and `ctx.lds_store`.
* Do not silently make the existing per-thread `lds_load`/`lds_store` wrappers
  mean "only subgroup leader logs". Arbitrary per-thread accesses are not
  necessarily summarized by the leader's address. Any lower-overhead path should
  be automatic, implicit, and conservative: the user still instruments actual
  HIP LDS accesses, while the detector may coalesce records only when it proves
  a regularity pattern for a nonzero `site_id`. The active coalescing path is
  subgroup-level only; the earlier thread-level summary experiment was removed
  because it did not serve a clear optimization goal.
* The target design is plain dissociation of mode-specific hot metadata:
  `thread-level` keeps per-thread identity in its records, while
  `subgroup-level` uses records and diagnostics shaped around
  subgroup-to-subgroup conflicts and omits per-thread identity when the mode
  contract allows it.
* Do not default to a templated "one detector parameterized by mode policy"
  design. Use templates only for small leaf helpers or storage conveniences
  where they clearly simplify generated code without forcing the two modes into
  the same data layout.
* Share code only at boundaries that do not constrain either mode: byte-range
  overlap predicates, subgroup arithmetic, diagnostics formatting, host
  ownership, and simple allocation/storage helpers.
* LDS load/store APIs should be templated immediately. Internally, every memory
  access becomes a byte range: base address plus `sizeof(T)` bytes.
* The first API can require trivially copyable `T`.
* A pointer wrapper can come later, but member functions on the context are the
  easiest way to make real progress today.
* The namespace should avoid claiming to replace HIP builtins. It is a separate
  diagnostic API.
* The API should eventually support labels/source locations. The first concrete
  step is the compact numeric `site_id`, which can support both diagnostics and
  later automatic access-log coalescing.
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
__global__ void kernel(int* out, hip_moi::thread_level_context::storage_ref storage) {
  __shared__ int lds[32];

  hip_moi::thread_level_context::config cfg{
      /*thread_count=*/static_cast<int>(blockDim.x),
      /*threads_per_subgroup=*/static_cast<int>(blockDim.x),
      /*subgroup_count=*/1,
  };
  hip_moi::thread_level_context ctx(storage, cfg);

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
* optional compact `site_id`,
* whether the slot is valid.

`subgroup-level` mode uses a distinct compact record that tracks subgroup ids
and omits thread id, because diagnostics intentionally do not distinguish
threads within one subgroup. The current implementation still records each
instrumented access call. That is intentional for now because it keeps the API
close to ordinary HIP code: users instrument the LDS accesses they actually
wrote.

Automatic coalescing is currently subgroup-level only. A possible later
thread-level optimization would need a fresh design, not a mechanical mirror of
the subgroup-level path, because thread-level diagnostics require per-thread
race visibility. In thread-level mode, `site_id` remains useful diagnostic
metadata but does not trigger summary construction.

`subgroup_level_context` has separate summary storage and now produces summaries
through a separate coalescing access path. Because subgroup-level ordinary exact
records intentionally omit thread id, subgroup-level coalescing does not
reintroduce per-thread identity into the subgroup-level `access_record`. That
preserves the "pay only for what you use" invariant: default-site subgroup-level
diagnostics keep compact subgroup-shaped access records, while nonzero-site
accesses use lane-level `coalescing_access_record`s when that storage is
available.

The subgroup-level coalescing-access path:

* Keep `subgroup_level_context::access_record` as the ordinary exact diagnostic
  log for default-site accesses and fallback. It records address, byte size,
  subgroup id, epoch, kind, validity, and site id, but not thread id or lane.
* Use a separate optional `coalescing_access_record` log, used only when
  `site_id != 0` and the caller supplied coalescing access storage. A
  coalescing access record stores the same grouping key plus `lane` and address:
  `(epoch, subgroup_id, site_id, kind, byte_count, lane, address)`.
* If the coalescing access log is absent or full, fall back to publishing an
  ordinary exact `access_record` so instrumentation remains conservative.
* Compute `lane` transiently in the executing thread from
  `thread_id() % threads_per_subgroup` at first. This does not put thread id or
  lane into the subgroup-level diagnostic record; it only populates optional
  coalescing metadata.
* At `ctx.syncthreads()`, scan coalescing access records by
  `(epoch, subgroup_id, site_id, kind, byte_count)`. Build a `lane_mask` and a
  `repeated_lane_mask` from those coalescing access records. For 32- or 64-lane subgroups,
  a `uint64_t` mask is enough; larger modeled subgroups can conservatively
  disable coalescing until a multiword mask exists.
* Emit a subgroup-level coalesced summary only when there is no repeated lane,
  the lane mask is contiguous, and the addresses follow a fixed signed stride
  over lane order with non-overlapping individual accesses.
* Reset the epoch coalescing-access count at the same modeled epoch boundary as
  the exact access log.

This design keeps the layers separate: `access_record` is the ordinary exact
fallback log, `coalescing_access_record` is the lane-carrying opted-in access
log, and `coalesced_access_record` is the compact summary emitted when the
regularity check succeeds. In subgroup-level mode, those summaries now
participate in conflict detection at epoch close: newly emitted summaries are
compared with each other and with exact or coalescing access records that were
not themselves summarized. The overlap test walks the represented per-lane byte
ranges, so a fixed-stride summary with gaps does not report merely because its
enclosing span overlaps another summary's
span.

On every `lds_load<T>` or `lds_store<T>`:

1. Compute the byte range.
2. Record the new access in the active epoch, including `site_id.value()`.
3. Perform the real load or store.

Conflict checking is mode-specific. `thread-level` mode currently checks the
new exact record against prior exact records immediately. `subgroup-level` mode
defers conflict checking until epoch close, so the common access path only
publishes exact records and optional nonzero-site coalescing access records. Whenever
checking runs in `thread-level` mode, if byte ranges overlap, thread ids differ,
and either access is a write, record a diagnostic. In `subgroup-level` mode, the
corresponding predicate uses subgroup ids instead of thread ids and ignores
same-subgroup conflicts by contract.

On `ctx.syncthreads()` in `subgroup-level` mode:

1. Execute the real full-workgroup synchronization.
2. Summarize nonzero-site coalescing access records for the epoch that just ended.
3. Check newly emitted summaries against other summaries and unsummarized exact
   records.
4. Check remaining unsummarized exact records against each other.
5. Advance the epoch.
6. Clear or logically invalidate access/coalescing access records from the previous epoch.
7. Ensure detector metadata updates are complete before returning, using
   detector-internal synchronization if needed.

`ctx.finish()`/`ctx.close_epoch()` performs the same subgroup-level epoch-close
work for a final epoch that is not followed by another real user barrier. It
must be called uniformly by all threads in the workgroup.

On `ctx.syncthreads()` in `thread-level` mode:

1. Execute the real full-workgroup synchronization.
2. Advance the epoch.
3. Clear or logically invalidate access records from the previous epoch.
4. Ensure detector metadata updates are complete before returning, using
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

Current `thread-level` fallback/debug implementation:

1. Atomically reserve an access-record slot.
2. Publish the access record with a valid bit.
3. Compare the new access with valid access records in the current epoch.
4. Atomically reserve diagnostic slots for any conflicts.
5. Perform the user's real LDS load or store.

Current `subgroup-level` implementation:

1. For default-site accesses, atomically reserve an exact access-record slot and
   publish the compact subgroup-level access record with a valid bit.
2. For nonzero-site accesses with coalescing access storage, publish one
   `coalescing_access_record` carrying lane and address.
3. If coalescing access storage is absent or full, fall back to publishing an
   ordinary exact access record.
4. Perform the user's real LDS load or store.
5. At epoch close, check summaries, unsummarized exact records, and
   unsummarized coalescing access records for cross-subgroup conflicts.

These implementations may perturb scheduling and may introduce extra ordering
in the instrumented executable, but diagnostics must still be computed from the
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

Thread-level diagnostic record:

```c++
struct diagnostic {
  uint32_t kind;
  uint32_t epoch;
  uint32_t first_thread_id;
  uint32_t second_thread_id;
  uintptr_t first_addr;
  uintptr_t second_addr;
  uint32_t first_size;
  uint32_t second_size;
};
```

Subgroup-level diagnostic record:

```c++
struct diagnostic {
  uint32_t kind;
  uint32_t epoch;
  uint32_t first_subgroup_id;
  uint32_t second_subgroup_id;
  uintptr_t first_addr;
  uintptr_t second_addr;
  uint32_t first_size;
  uint32_t second_size;
};
```

Host-side tests should be able to copy back a small result buffer and assert:

* number of diagnostics,
* diagnostic kind,
* participating threads or subgroups, depending on the selected context type,
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
   * `thread_level_context::storage_ref`
   * `thread_level_context::static_context_storage`
   * `thread_level_context`
   * `lds_load<T>`
   * `lds_store<T>`
   * `syncthreads()`

3. Implement single-epoch workgroup metadata.
   * Runtime-capacity metadata buffers addressed through
     `thread_level_context::storage_ref`.
   * At least one global-memory-backed storage path for tests.
   * Optional `thread_level_context::static_context_storage` helper for
     fixed-size storage.
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
  001_single_subgroup_safe_mvp_test.hip
  002_single_subgroup_race_mvp_test.hip
  003_host_context_test.hip
  004_single_subgroup_basic_conflict_predicate_test.hip
  005_single_subgroup_epoch_boundary_test.hip
  006_single_subgroup_all_thread_array_test.hip
  007_single_subgroup_metadata_capacity_test.hip
  008_single_subgroup_loop_epoch_test.hip
  009_single_subgroup_tiled_lds_test.hip
  010_single_subgroup_matmul_like_test.hip
  011_single_subgroup_epoch_log_lifetime_test.hip
  012_single_subgroup_matmul_pipeline_test.hip
  013_single_subgroup_rdna4_wmma_row_major_test.hip
  014_single_subgroup_rdna4_wmma_data_tiled_test.hip
  015_thread_level_subgroup_test.hip
  016_subgroup_level_bootstrap_test.hip
  017_subgroup_level_multisubgroup_test.hip
  018_rdna4_multisubgroup_wmma_data_tiled_test.hip
  test_support.hpp
```

Files with `single_subgroup` in their names are deliberately one-subgroup
workgroups. When such a file has deterministic thread-level race diagnostics,
it should also include subgroup-level checks showing that intra-subgroup races
are outside subgroup-level mode's reporting contract. Diagnostic-free
single-subgroup tests generally should not add subgroup-level mirrors unless
they exercise a subgroup-level behavior not already covered elsewhere.

`001_single_subgroup_safe_mvp_test.hip` should assert both numerical outputs and zero
diagnostics. `002_single_subgroup_race_mvp_test.hip` should assert deterministic diagnostics;
for racy kernels, numerical output is not the oracle.
`003_host_context_test.hip` asserts end-user behavior: explicit checks,
stderr/fatal policy, and scope-based destructor handling.
`004_single_subgroup_basic_conflict_predicate_test.hip` asserts the basic MVP predicate around
same-epoch byte ranges before the suite moves on to epoch-boundary behavior.
`005_single_subgroup_epoch_boundary_test.hip` asserts the MVP epoch-boundary behavior of
uniform `ctx.syncthreads()`.
`006_single_subgroup_all_thread_array_test.hip` asserts first all-thread LDS array behavior,
including a missing-barrier diagnostic case.
`007_single_subgroup_metadata_capacity_test.hip` asserts access-log overflow and diagnostic
buffer truncation behavior, including the user-facing host report.
`008_single_subgroup_loop_epoch_test.hip` asserts looped epoch behavior, including repeated
safe barriers and repeated missing-barrier diagnostics.
`009_single_subgroup_tiled_lds_test.hip` asserts 2D tile layouts, tiled gathers, and an
unsynchronized transpose diagnostic.
`010_single_subgroup_matmul_like_test.hip` asserts cooperative LDS matmul-like access patterns
using explicit small integer inputs, host-reference output checks, and a scalar
missing-barrier diagnostic.
`011_single_subgroup_epoch_log_lifetime_test.hip` asserts that access-log capacity is scoped to
the active epoch rather than the cumulative kernel trace.
`012_single_subgroup_matmul_pipeline_test.hip` asserts double-buffered and pipeline-like matmul
LDS patterns with host-reference output checks, including diagnostic-positive
buffer reuse cases.
`013_single_subgroup_rdna4_wmma_row_major_test.hip` asserts RDNA4/gfx12 WMMA intrinsic coverage
using all 32 threads, conventional row-major LDS tiles, non-uniform exact
inputs, and host-reference output checks.
`014_single_subgroup_rdna4_wmma_data_tiled_test.hip` asserts the matching RDNA4/gfx12 WMMA
coverage for packed A/B/C fragments laid out at `lane * fragment_size` byte
offsets, with packed data generated from logical tiles and checked against a
host-reference matmul.
`015_thread_level_subgroup_test.hip` asserts first multi-subgroup
`thread-level` behavior: helper-derived subgroup identity, per-record subgroup
ids, same-subgroup diagnostics, cross-subgroup diagnostics, and full-workgroup
barrier separation across subgroups.
`016_subgroup_level_bootstrap_test.hip` asserts the first `subgroup-level`
behavior: cross-subgroup conflicts report, same-subgroup conflicts intentionally
do not report, and the explicit `subgroup_level_context` surface is exercised.
`017_subgroup_level_multisubgroup_test.hip` asserts richer subgroup-level
multi-subgroup behavior: array, loop, tiled, and matmul-shaped cross-subgroup
LDS sharing across a four-subgroup workgroup.
`018_rdna4_multisubgroup_wmma_data_tiled_test.hip` asserts real RDNA4/gfx12
multi-subgroup WMMA coverage under both modes: data-tiled vector fragment LDS
staging, double-buffered two-tile loops, exact host-reference output checks, and
cross-subgroup missing-barrier diagnostics.

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
  005_subgroup_level_cross_subgroup_race.hip
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
    same-subgroup conflicts intentionally do not report, and the explicit
    `subgroup_level_context` surface is exercised. Done for the semantic
    bootstrap.
16. Introduce dissociated `subgroup-level` records and storage, removing thread
    id from the hot metadata when the mode contract does not need it. Done.
17. Add mode-aware host support for `subgroup-level` mode, so reports can
    identify subgroup-to-subgroup conflicts without pretending to identify exact
    causative threads. Done.
18. Add richer multi-subgroup `subgroup-level` tests covering arrays, loops,
    tiled layouts, and matmul-shaped cross-subgroup LDS sharing. Done.
19. Update tutorial examples to cover thread-level and subgroup-level usage as
    peer modes, with compiled examples for both user-facing paths. Done for the
    first passing example and a cross-subgroup diagnostic example.
20. Rename single-subgroup instrumented tests with a fixed-width numeric prefix
    followed by `single_subgroup`, and add subgroup-level companion checks for
    deterministic diagnostic-positive single-subgroup race cases. Done.
21. Add real multi-subgroup RDNA4 WMMA coverage under both `thread-level` and
    `subgroup-level` modes. Done for data-tiled and row-major double-buffered
    fragment-staging slices.
22. Add the `site_id` wrapper, optional `site_id` arguments to `lds_load` and
    `lds_store`, and the `HIP_MOI_SITE_ID()` macro. Store compact site ids in
    access records and diagnostics without changing exact detector behavior.
    Done.
23. Improve diagnostic quality with site ids, labels/source locations, and
    first-conflict preservation.
24. Keep broadening ordinary access-level multi-subgroup coverage where it
    exercises real HIP code shapes.
25. Start synchronization-lowering notes for `ctx.syncthreads()` and lower-level
    builtins, while keeping naked fences paired with atomics as a later
    memory-model widening step.
26. Explore automatic coalescing of access records only after site ids exist.
    Coalescing must be conservative, implicit, and keyed by nonzero site ids; if
    a site has repeated dynamic instances in one epoch or no recognized
    regularity pattern, keep exact records. Done for the subgroup-level path.
    The first thread-level summary-only experiment was removed because it did
    not provide a clear optimization win and would require a separate design to
    preserve per-thread race visibility.
27. Design subgroup-level coalescing access metadata and only then consider
    hot-path compression. The first path should be a separate optional
    coalescing access log keyed by subgroup/epoch/site/kind/byte size and
    carrying lane plus address. It should build `lane_mask` and
    `repeated_lane_mask` at epoch boundaries, separate from the subgroup-level
    access record, so that the ordinary diagnostic record can keep omitting
    thread id and lane. Done.
28. Add `lane_in_subgroup()` as the preferred helper name, migrate tests and
    docs toward lane terminology, and keep the existing rank-named helper only
    as a temporary compatibility spelling if needed. Done for the public helper
    and new tests/docs; older tests may still use the compatibility spelling.
29. Use subgroup-level coalesced summaries in conflict detection without yet
    changing the hot access-log path. Done: summaries are compared against other
    summaries and against unsummarized exact records at epoch close.
30. Decide whether coalesced summaries should start driving actual access-log
    compression or exact-scan suppression. Done for subgroup-level opted-in
    accesses: nonzero-site subgroup accesses now write
    `coalescing_access_record`s instead of ordinary exact `access_record`s when
    coalescing access storage is available. Conflict detection is deferred to
    epoch close, summaries are checked first, and unsummarized coalescing access
    records participate as exact fallback accesses. Thread-level remains
    immediate/exact for now.
31. Optimize subgroup-level summary construction. The current summary builder
    may still spend too much epoch-close work grouping coalescing access
    records. Done for the first per-group pass: one accumulation scan builds
    lane masks and endpoint records, and one validation scan checks the
    fixed-stride address relation, avoiding the previous per-lane
    coalescing-access-log rescans for full-subgroup sites.
32. Decide whether subgroup-level summary construction needs explicit
    epoch-close grouping scratch storage. Done: `subgroup_level_context` now has
    optional `coalescing_group_record` storage. When supplied, the collector
    builds one group record per subgroup/epoch/site/kind/byte-size key and emits
    summaries from those groups. If group scratch is absent or full, it falls
    back to the older coalescing-access scan.
33. Decide whether subgroup-level group lookup needs a hash/direct-indexed
    structure. Done: subgroup-level group scratch now uses the
    `coalescing_group_record` array as an open-addressed table keyed by
    subgroup/epoch/site/kind/byte-size. If the table fills or probing cannot
    find a slot, the collector falls back to the older coalescing-access scan.
34. Reduce the remaining subgroup-level coalescing costs. The next costs are
    per-group coalescing-access-log validation scans and hot-path traffic: one
    coalescing access record is still written per opted-in participating lane.
35. Add subgroup-level coalescing fallback observability and exact-fallback
    coverage. Done: subgroup-level storage now has `coalescing_fallback_count`
    for opted-in accesses that fall back to exact records, the tests cover
    absent coalescing access storage, coalescing access overflow, repeated-lane
    summary-vs-unsummarized conflicts, and unsummarized read/read silence, and
    the tutorial has a compiled subgroup-level coalescing opt-in example.
36. Drop the thread-level coalescing experiment. Done:
    `thread_level_context` no longer owns coalesced summary storage or scans
    epochs for regular access patterns, and the dedicated thread-level
    coalescing-opportunity test target was removed. Thread-level site ids remain
    as exact-record and diagnostic metadata.
37. Implement The context allocation plan for host-owned contexts. Done for the
    first slice: `host_context_options` now has a primary `storage_bytes` byte
    budget, defaulting to 16 MiB; host contexts use one HIP device allocation
    and carve it into aligned typed slices for the selected mode; typed
    capacities remain available as advanced/test overrides; `docs/context.md`
    documents default global-memory storage, manual storage, and saturation
    behavior.
38. Keep improving saturation behavior and observability.
    * Add direct tests for too-small `storage_bytes` death behavior.
    * Consider public host-side introspection for computed capacities and
      storage layout, if users need to tune without reading implementation
      internals.
    * Preserve the invariant that storage exhaustion produces diagnostics or
      conservative fallback, not silent corruption.

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
   * Compute `subgroup_id` and lane within subgroup from runtime config.
   * Support `subgroup_count > 1` in storage initialization and epoch state.
   * Keep `ctx.syncthreads()` as a full-workgroup operation initially, advancing
     all subgroup epochs together.
   * Add two-subgroup tests using one workgroup.

2. Preserve the existing `thread-level` HIP mode. Done for first
   multi-subgroup diagnostics.
   * Continue detecting same-epoch conflicts between any two different threads.
   * Include same-subgroup and cross-subgroup diagnostic-positive tests.
   * Keep this mode principled in the HIP/LLVM memory model.

3. Design the `subgroup-level` mode. Done for the initial separated type
   families.
   * Define the mode as intentionally ignoring same-subgroup conflicts.
   * Use a separate `subgroup_level_context` type rather than a runtime mode
     field on `config`.
   * Keep `using context = thread_level_context` only as a temporary
     compatibility alias.
   * Use separate `subgroup-level` records and diagnostics instead of scaling or
     repurposing thread ids.
   * Avoid virtual dispatch in device code.
   * Avoid a large templated detector-policy design unless a very small helper
     boundary emerges naturally.

4. Prototype the `subgroup-level` mode. Done for the current all-thread logging
   implementation.
   * Produce deterministic diagnostics for cross-subgroup conflicts.
   * Produce no diagnostic for same-subgroup conflicts by contract.
   * The current version uses subgroup-specific records but still logs each
     instrumented access call.

5. Introduce a dissociated `subgroup-level` record path. Done.
   * Remove thread id from the hot metadata when diagnostics only distinguish
     subgroups.
   * Keep byte-range overlap, subgroup arithmetic, host ownership, and
     diagnostic formatting as shared leaf helpers where that remains natural.
   * Measure or inspect metadata footprint, atomic usage, and generated code for
     both modes.

6. Add mode-aware host support for `subgroup-level` diagnostics. Done.
   * Provide a host ownership/reporting path for `subgroup_level_context`.
   * Print subgroup ids for subgroup-level diagnostics instead of representative
     or scaled thread ids.
   * Keep the default `host_context` name thread-level for compatibility unless a
     clearer naming split is introduced.

7. Add richer `subgroup-level` multi-subgroup coverage. Done for the first
   arrays/loops/tiles/matmul-shaped slice.
   * Exercise more than two subgroups in one workgroup.
   * Include no-diagnostic independent subgroup slots and barrier-separated
     communication.
   * Include diagnostic-positive cross-subgroup array, tiled, looped-epoch, and
     matmul-shaped cases.

8. Resume diagnostic quality work.
   * Add source/location IDs or stringless labels.
   * Add first-conflict preservation so later conflicts do not hide the useful
     one.
   * Add clearer mode-aware host diagnostics, especially for
     `subgroup-level` false negatives by design.

9. Keep synchronization lowering notes on the horizon.
   * Compile tiny examples using `ctx.syncthreads()`, raw `__syncthreads()`,
     `__builtin_amdgcn_s_barrier`, and `__builtin_amdgcn_fence`.
   * Save or document their LLVM IR shape for the HIP-language model.
   * Separately note which hardware facts would matter for the future
     assembly-level model.
   * Do not infer HIP-level cross-thread synchronization from naked fences.

10. Keep automatic access coalescing conservative and implicit.
    * The user-facing API remains access-level: users call `ctx.lds_load` and
      `ctx.lds_store` at the places where HIP code actually loads and stores
      LDS.
    * A nonzero `site_id` marks an access site as eligible for possible
      coalescing in subgroup-level mode. `site_id{0}` means exact logging only.
    * Coalescing is allowed only when the detector proves a regularity pattern
      for records sharing epoch, subgroup, access kind, byte size, and site id.
      Candidate patterns include contiguous, fixed-stride, and eventually
      tile-shaped lane-to-address mappings.
    * If a static site executes multiple dynamic instances in one epoch, leave
      it uncoalesced for that epoch. This keeps the initial model safe without
      solving dynamic instance identity.
    * The first thread-level summary experiment was removed. Thread-level mode
      stays exact-record based for now because its diagnostics need per-thread
      race visibility. Done.
    * Subgroup-level coalescing access metadata and summary-based conflict
      detection now exist. Done.
    * Subgroup-level exact pairwise work is now skipped for exact records covered
      by a coalesced summary. Done.
    * Subgroup-level summary construction now avoids per-lane
      coalescing-access-log rescans within one candidate group. Done.
    * Subgroup-level summary construction now has optional epoch-close grouping
      scratch storage across many distinct sites. Done.
    * Subgroup-level group scratch now uses open addressing instead of scanning
      the existing groups for every coalescing access record. Done.
    * Subgroup-level opted-in accesses now use `coalescing_access_record` as the
      hot-path access log when storage is available, falling back to ordinary
      exact records when it is not. Done.
    * Subgroup-level coalescing fallback is now observable through
      `coalescing_fallback_count`, and tests cover both absent storage and
      full-buffer fallback. Done.
    * Further hot-path compression can come later, after the remaining
      epoch-close costs are small enough to make coalescing a credible
      optimization for broad kernels.

11. Make host-owned context storage usable without per-buffer tuning. First
    slice done.
    * Treat `storage_bytes` as the primary host allocation control.
    * Default to a large global-memory budget because most real kernels cannot
      spare LDS for detector metadata.
    * Keep manual typed `storage_ref` and `static_context_storage` available for
      tests, generated experiments, and users who deliberately want static
      storage.
    * Keep saturation behavior explicit: exact metadata overflow should report
      `metadata_full` when possible, diagnostic buffers may truncate stored
      records while preserving counts, and coalescing overflow should fall back
      to exact logging.

## Plan beyond the MVP

After the refined MVP, widen scope in small semantic steps. Each step should add
one new kind of happens-before edge or one new class of instrumented access, with
tests that prove both positive and negative cases.

### Step 1: Multi-subgroup `thread-level` HIP mode

Allow a workgroup to contain multiple tracked subgroups without changing the
HIP-language diagnostic contract.

* Add multiple epoch counters per workgroup.
* Compute subgroup id and lane within subgroup from `config`.
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
* Users select the mode by choosing the context type:
  `thread_level_context` or `subgroup_level_context`.
* Prefer separate hot metadata structures for the two modes. The public context
  surface may remain similar, but `subgroup-level` must not be constrained to
  reuse `thread-level` access records.
* Keep mode selection compile-time by construction: the class type carries the
  contract.
* Avoid virtual dispatch in device code.
* Use a compact `subgroup-level` access record that tracks subgroup id but omits
  thread id. Done.
* Report "subgroup A vs subgroup B" diagnostics without pretending to identify
  exact threads. Done in device/test diagnostics and host reporting.
* Share only leaf helpers that are naturally common, such as byte-range overlap,
  subgroup-index arithmetic, host-side diagnostic formatting, and storage
  ownership.
* Use templatization sparingly; do not make a mode-policy template hierarchy the
  default answer to sharing code between the two modes.

### Step 3: `subgroup-level` mode prototype

Build the narrow mode before pursuing more esoteric synchronization semantics.

* First implementation logs all instrumented access calls but stores them in the
  `subgroup-level` record shape, so the hot record omits thread id. Done.
* Host ownership/reporting for the `subgroup-level` diagnostic shape exists.
* Continue exercising ordinary access-level instrumentation under this mode:
  users still replace real LDS accesses with `ctx.lds_load` and
  `ctx.lds_store`.
* Measure or inspect metadata size, access-record fields, atomic usage, and
  generated code for both modes.
* Add paired tests showing the same kernel reports in `thread-level` mode but
  intentionally does not report same-subgroup conflicts in `subgroup-level`
  mode.
* Add matmul-shaped cross-subgroup conflicts, because this is the main bridge to
  the future assembly-level effort. Done for the first non-WMMA subgroup-level
  slice.
* Later, consider automatic coalescing of records from regular access sites.
  This should be keyed by explicit nonzero `site_id` values but should not
  require a separate user-facing summary API.

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
* Prefer structural dissociation between instrumentation modes when optimization
  pressure exists. This project is partly a prototype for a future assembly
  effort, so letting `subgroup-level` do the right thing for itself matters more
  than minimizing every line of duplicated code.
* Keep every emitted diagnostic conservative. If the tool cannot prove a
  conflict inside its model, prefer no diagnostic.
* Document every unsupported case as a false-negative risk, not as a bug in the
  MVP.
* Keep instrumentation modes explicit. A false negative that is unacceptable in
  `thread-level` HIP mode may be the intended contract in `subgroup-level`
  mode.
