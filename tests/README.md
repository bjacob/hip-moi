<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# hip-moi tests

The test tree is intentionally split into two layers.

* `reference/` contains ordinary uninstrumented HIP kernels. These are concrete
  source-level shapes for instrumented tests and benchmarks. They make explicit
  which threads run which accesses, what the workgroup size is, and which cases
  are expected to be diagnostic-positive or diagnostic-free.
* `instrumented/` contains the active `hip_moi::context` tests. The suite is
  intentionally narrower than the reference corpus and now focuses on
  subgroup-scoped exact-shadow and sampled-watchpoint behavior.

All tests are registered through GTest. CMake first tries `find_package(GTest)`
and falls back to `FetchContent` if no system package is available.
By default, `HIP_MOI_CTEST_PER_CASE=ON` registers each GTest `TEST()` as a
separate CTest entry. Set `-DHIP_MOI_CTEST_PER_CASE=OFF` to register one CTest
entry per test binary, which is faster for local HIP runs.
The reference corpus uses a parameterized GTest suite, so the default CTest view
has one named entry per launched reference kernel.

The reference GTest binary launches only well-defined kernels. Kernels that
intentionally contain races or divergent barriers are compiled, but not launched
by the reference self-test. They are there to prevent the corpus from being only
comments while avoiding undefined behavior in the reference run.

The detector contract tested by the instrumented suite is documented in
[`../docs/instrumentation_model.md`](../docs/instrumentation_model.md).

Configure/build example:

```sh
cmake -E env CCACHE_DISABLE=1 cmake -G Ninja -S . -B build \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake -E env CCACHE_DISABLE=1 cmake --build build --target hip_moi_reference_selftest
ctest --test-dir build --output-on-failure
```

When using a TheRock SDK, point CMake at the SDK's underlying ROCm Clang and
ROCm prefix rather than using `hipcc` as `CMAKE_HIP_COMPILER`.
