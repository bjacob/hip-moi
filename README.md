<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# hip-moi

hip-moi is a source-level HIP instrumentation prototype for studying LDS
memory-ordering diagnostics and their overhead on architecture-specific HIP
workloads.

The immediate audience is Jakub. The goal is to make the results useful for
improving Loom, informing the LLVM GPU ThreadSanitizer RFC discussion, and
deciding what a practical HIP-level instrumentation API should look like.

## Current Scope

hip-moi instruments explicit LDS accesses written through its API:

```c++
ctx.lds_store_at(ptr, value, /*lds_byte_offset=*/offset, HIP_MOI_SITE_ID());
value = ctx.lds_load_at(ptr, /*lds_byte_offset=*/offset, HIP_MOI_SITE_ID());
```

The active detector is subgroup-scoped. It diagnoses, or models, conflicts
between different subgroups in the same workgroup. It intentionally does not try
to diagnose races wholly inside one subgroup; that choice follows the current
Loom-comparison priority.

The project has three important execution paths:

| Path | Purpose |
| --- | --- |
| `hip_moi::context` + `backend_kind::exact_shadow` | Diagnostic-capable exact shadow table. |
| `hip_moi::context` + `backend_kind::sampled_watchpoint` | Diagnostic-capable sampled watchpoint backend, or publish-only mode through the general context. |
| `hip_moi::sampled_watchpoint_context` | Narrow publish-only fast path for Loom-style performance experiments. It does not report races. |

The third path exists because performance work showed that keeping the full
diagnostic context live in production-shaped kernels creates substantial VGPR,
spill, and code-size overhead. It is not a replacement for the diagnostic API.

## Key Definitions

* **Workgroup**: a HIP block.
* **Subgroup**: the fixed-size thread group used as the instrumentation owner.
  The current architecture-specific GPU tests use 32-thread waves.
* **Lane**: a thread's index inside its subgroup.
* **Epoch**: a logical interval between instrumented full-workgroup barriers.
  Current diagnostic paths treat same-epoch conflicting accesses from different
  subgroups as a reportable condition.
* **Site id**: a compile-time identifier produced by `HIP_MOI_SITE_ID()`. It
  distinguishes source instrumentation sites in diagnostics and sampling
  decisions.
* **Watchpoint**: hip-moi's software metadata record for one sampled LDS dword
  range. It is not a hardware watchpoint.

## What To Read

* [docs/instrumentation_model.md](docs/instrumentation_model.md): precise
  description of recorded metadata, access-time algorithms, diagnostics, and
  sampling.
* [docs/atomics.md](docs/atomics.md): source-level atomics model, precision
  trade-offs, paired fences, and current atomics performance interpretation.
* [docs/loom_rfc_comparison.md](docs/loom_rfc_comparison.md): mapping between
  hip-moi, the compiler-rt RFC, real Loom, and Jakub-Sampled-Loom.
* [docs/benchmark_interpretation.md](docs/benchmark_interpretation.md): how to
  read the benchmark modes, results, and resource-pressure signals.
* [docs/dbi_transition.md](docs/dbi_transition.md): transition brief from
  source-level hip-moi lessons to future rocjitsu DBI requirements and first
  experiments.
* [docs/context.md](docs/context.md): host/device context allocation and usage.
* [benchmarks/README.md](benchmarks/README.md): current architecture-specific
  benchmark suite, resource pressure, validation status, and latency results.
* [docs/tutorial/README.md](docs/tutorial/README.md): small compilable examples.
* [tests/instrumented/README.md](tests/instrumented/README.md): correctness test
  corpus.

`PLAN.md` remains a project planning document. New durable facts should move
into `docs/` as they become stable enough to use in discussions.

## Current Takeaway

On the production-shaped matmul extraction, the narrow
`sampled_watchpoint_context` path is faster than the local Jakub-Sampled-Loom
comparison row. On attention-shaped workloads the picture is more nuanced:
dense scalar LDS scratch remains expensive, while no-score/register-handoff
variants are much closer to the pass-through baseline.

The benchmark README records the current numbers and the intermediate resource
metrics, including LDS usage, VGPR pressure, spills, and private segment state.

## Architecture Support

The portable scalar LDS, atomics, and tutorial examples build on any HIP target
supported by the configured toolchain. The architecture-specific rows are gated
by `CMAKE_HIP_ARCHITECTURES`:

| Architecture family | CMake architecture match | Coverage |
| --- | --- | --- |
| RDNA4 | `gfx120*` | RDNA4 WMMA tests, benchmarks, reference matmul corpus, and the RDNA4 WMMA tutorial. |
| CDNA4/gfx950 | `gfx950*` | CDNA4 MFMA ports of the RDNA4 attention, ping-pong, and Stream-K WMMA rows. |
| gfx1250 | `gfx1250` | gfx1250 WMMA ports of the RDNA4 attention, ping-pong, Stream-K, and standalone matmul benchmark rows. |

The gfx1250 rows are build- and smoke-validated, but this repository does not
claim hardware performance numbers for them until physical gfx1250 hardware is
available.

## Build And Test

Use an out-of-tree CMake build directory. For example:

```sh
cmake -S . -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build build
ctest --test-dir build --output-on-failure
```

Benchmarks are not enabled in the default build. See
[benchmarks/README.md](benchmarks/README.md) for the benchmark targets and
standalone scripts.
