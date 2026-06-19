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

include(GoogleTest)
set(_hip_moi_reference_test_env)
if(HIP_MOI_ROCM_ROOT)
  set(_hip_moi_reference_test_env
      PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${HIP_MOI_ROCM_ROOT}/lib")
endif()

function(hip_moi_add_gtest name)
  add_executable(${name} ${ARGN})
  target_compile_features(${name} PRIVATE cxx_std_17)
  target_link_libraries(${name} PRIVATE hip_moi::hip_moi GTest::gtest_main)
  target_compile_options(
    ${name}
    PRIVATE
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall>
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wextra>
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Werror>
      $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-Wall>
      $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-Wextra>
      $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-Werror>)

  set_target_properties(${name} PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
  )

  if(HIP_MOI_ROCM_ROOT)
    set_target_properties(${name} PROPERTIES
      BUILD_RPATH "${HIP_MOI_ROCM_ROOT}/lib"
    )
  endif()

  if(HIP_MOI_CTEST_PER_CASE)
    gtest_discover_tests(${name}
                         NO_PRETTY_VALUES
                         ${_hip_moi_reference_test_env})
  else()
    add_test(NAME ${name}
             COMMAND ${name})
    if(_hip_moi_reference_test_env)
      set_tests_properties(${name}
                           ${_hip_moi_reference_test_env})
    endif()
  endif()
endfunction()

set(_hip_moi_build_rdna4_tests TRUE)
foreach(_hip_moi_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
  if(NOT _hip_moi_arch MATCHES "^gfx12")
    set(_hip_moi_build_rdna4_tests FALSE)
  endif()
endforeach()

hip_moi_add_gtest(hip_moi_reference_selftest
                  reference/mvp_reference_kernels.hip)
hip_moi_add_gtest(hip_moi_instrumented_single_subgroup_safe_mvp_test
                  instrumented/001_single_subgroup_safe_mvp_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_single_subgroup_race_mvp_test
                  instrumented/002_single_subgroup_race_mvp_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_host_context_test
                  instrumented/003_host_context_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_single_subgroup_basic_conflict_predicate_test
                  instrumented/004_single_subgroup_basic_conflict_predicate_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_single_subgroup_epoch_boundary_test
                  instrumented/005_single_subgroup_epoch_boundary_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_single_subgroup_all_thread_array_test
                  instrumented/006_single_subgroup_all_thread_array_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_single_subgroup_metadata_capacity_test
                  instrumented/007_single_subgroup_metadata_capacity_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_single_subgroup_loop_epoch_test
                  instrumented/008_single_subgroup_loop_epoch_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_single_subgroup_tiled_lds_test
                  instrumented/009_single_subgroup_tiled_lds_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_single_subgroup_matmul_like_test
                  instrumented/010_single_subgroup_matmul_like_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_single_subgroup_epoch_log_lifetime_test
                  instrumented/011_single_subgroup_epoch_log_lifetime_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_single_subgroup_matmul_pipeline_test
                  instrumented/012_single_subgroup_matmul_pipeline_test.hip)
if(_hip_moi_build_rdna4_tests)
  hip_moi_add_gtest(hip_moi_instrumented_single_subgroup_rdna4_wmma_row_major_test
                    instrumented/013_single_subgroup_rdna4_wmma_row_major_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_single_subgroup_rdna4_wmma_data_tiled_test
                    instrumented/014_single_subgroup_rdna4_wmma_data_tiled_test.hip)
endif()
hip_moi_add_gtest(hip_moi_instrumented_thread_level_subgroup_test
                  instrumented/015_thread_level_subgroup_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_subgroup_level_bootstrap_test
                  instrumented/016_subgroup_level_bootstrap_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_subgroup_level_multisubgroup_test
                  instrumented/017_subgroup_level_multisubgroup_test.hip)
if(_hip_moi_build_rdna4_tests)
  hip_moi_add_gtest(hip_moi_instrumented_rdna4_multisubgroup_wmma_data_tiled_test
                    instrumented/018_rdna4_multisubgroup_wmma_data_tiled_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_rdna4_multisubgroup_wmma_row_major_test
                    instrumented/019_rdna4_multisubgroup_wmma_row_major_test.hip)
endif()
