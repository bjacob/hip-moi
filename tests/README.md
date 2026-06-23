<!--
Copyright (c) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# hip-moi tests

The test tree is intentionally split into two layers.

* `reference/` contains ordinary uninstrumented HIP kernels. These are concrete
  source-level reference cases for instrumented tests and benchmark shapes. They make
  it explicit which threads run which accesses, what the workgroup size is, and
  which cases are expected to be diagnostic-positive or diagnostic-free.
* `instrumented/` contains the active `hip_moi::context` tests. The suite is
  intentionally narrower than the reference corpus and now focuses on
  subgroup-level exact-shadow and sampled-watchpoint behavior.

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

Configure/build example for this machine:

```sh
cmake -E env CCACHE_DISABLE=1 cmake -G Ninja -S hip-moi -B hip-moi-build \
  -DCMAKE_CXX_COMPILER=/home/benoit/workspace/TheRock-build/dist/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/home/benoit/workspace/TheRock-build/dist/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/home/benoit/workspace/TheRock-build/dist/rocm \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/home/benoit/workspace/TheRock-build/dist/rocm
cmake -E env CCACHE_DISABLE=1 cmake --build hip-moi-build --target hip_moi_reference_selftest
ctest --test-dir hip-moi-build --output-on-failure
```

CMake 4.2 rejects `hipcc` as `CMAKE_HIP_COMPILER`; use the underlying TheRock
Clang for CMake builds.
