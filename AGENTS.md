# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# hip-moi agent instructions

This repository is a HIP memory-ordering instrumentation library prototype.

## Session workflow

At the end of every substantive work session:

1. Update `PLAN.md` so it closely reflects the current project state and next
   steps.
2. Ensure all tests pass.
3. Run `git clang-format`.
4. Commit the session changes with git.

If `git clang-format` changes source files, rebuild and rerun the affected tests
before committing.

## Build and test

Use Ninja. The canonical build directory in this workspace is:

```sh
/home/benoit/workspace/hip-moi-build
```

Configure with TheRock Clang, not `hipcc`, for CMake HIP builds:

```sh
cmake -E env CCACHE_DISABLE=1 cmake -G Ninja \
  -S /home/benoit/workspace/hip-moi \
  -B /home/benoit/workspace/hip-moi-build \
  -DCMAKE_CXX_COMPILER=/home/benoit/workspace/TheRock-build/dist/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/home/benoit/workspace/TheRock-build/dist/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/home/benoit/workspace/TheRock-build/dist/rocm \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/home/benoit/workspace/TheRock-build/dist/rocm
```

Build and test:

```sh
cmake -E env CCACHE_DISABLE=1 cmake --build /home/benoit/workspace/hip-moi-build
ctest --test-dir /home/benoit/workspace/hip-moi-build --output-on-failure
```

GPU tests may require running CTest with permissions that allow ROCm device
access.
