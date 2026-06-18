# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# tests/CMakeLists.txt
#
find_package(GTest QUIET)
if(NOT TARGET GTest::gtest_main)
  include(FetchContent)
  set(INSTALL_GTEST
      OFF
      CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
endif()

# Per-case (ON) registers every TEST() as its own ctest entry via
# gtest_discover_tests. Per-binary (OFF) registers each executable as a single
# ctest entry, which is faster for local runs with HIP runtime initialization.
option(HIP_MOI_CTEST_PER_CASE
       "Register each gtest TEST() as its own ctest entry (vs one entry per binary)"
       ON)

add_executable(hip_moi_reference_selftest
  reference/mvp_reference_kernels.hip
)

target_compile_features(hip_moi_reference_selftest PRIVATE cxx_std_17)
target_link_libraries(hip_moi_reference_selftest PRIVATE GTest::gtest_main)

set_target_properties(hip_moi_reference_selftest PROPERTIES
  HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
)

if(HIP_MOI_ROCM_ROOT)
  set_target_properties(hip_moi_reference_selftest PROPERTIES
    BUILD_RPATH "${HIP_MOI_ROCM_ROOT}/lib"
  )
endif()

include(GoogleTest)
set(_hip_moi_reference_test_env)
if(HIP_MOI_ROCM_ROOT)
  set(_hip_moi_reference_test_env
      PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${HIP_MOI_ROCM_ROOT}/lib")
endif()

if(HIP_MOI_CTEST_PER_CASE)
  gtest_discover_tests(hip_moi_reference_selftest
                       NO_PRETTY_VALUES
                       ${_hip_moi_reference_test_env})
else()
  add_test(NAME hip_moi_reference_selftest
           COMMAND hip_moi_reference_selftest)
  if(_hip_moi_reference_test_env)
    set_tests_properties(hip_moi_reference_selftest
                         ${_hip_moi_reference_test_env})
  endif()
endif()
