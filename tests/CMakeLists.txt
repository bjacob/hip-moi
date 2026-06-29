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
  if(NOT _hip_moi_arch MATCHES "^gfx120")
    set(_hip_moi_build_rdna4_tests FALSE)
  endif()
endforeach()

set(_hip_moi_build_gfx1250_tests TRUE)
foreach(_hip_moi_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
  if(NOT _hip_moi_arch MATCHES "^gfx1250$")
    set(_hip_moi_build_gfx1250_tests FALSE)
  endif()
endforeach()

set(_hip_moi_build_cdna4_tests TRUE)
foreach(_hip_moi_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
  if(NOT _hip_moi_arch MATCHES "^gfx950")
    set(_hip_moi_build_cdna4_tests FALSE)
  endif()
endforeach()

hip_moi_add_gtest(hip_moi_reference_selftest
                  reference/mvp_reference_kernels.hip)
hip_moi_add_gtest(hip_moi_reference_atomic
                  reference/atomic_reference_kernels.hip)
if(_hip_moi_build_rdna4_tests)
  hip_moi_add_gtest(hip_moi_reference_rdna4_jakub_matmul
                    reference/rdna4_jakub_matmul_reference.hip)
endif()
hip_moi_add_gtest(hip_moi_instrumented_single_subgroup_safe_mvp_test
                  instrumented/001_single_subgroup_safe_mvp_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_host_context_test
                  instrumented/002_host_context_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_site_id_test
                  instrumented/003_site_id_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_hard_synchronization_negative_test
                  instrumented/004_hard_synchronization_negative_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_shadow_abi_test
                  instrumented/005_shadow_abi_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_lds_offset_api_test
                  instrumented/006_lds_offset_api_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_exact_shadow_backend_test
                  instrumented/007_exact_shadow_backend_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_sampled_watchpoint_backend_test
                  instrumented/008_sampled_watchpoint_backend_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_attention_block_test
                  instrumented/009_attention_block_test.hip)
if(_hip_moi_build_rdna4_tests)
  hip_moi_add_gtest(hip_moi_instrumented_rdna4_wmma_attention_block_test
                    instrumented/010_rdna4_wmma_attention_block_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_rdna4_d128_attention_block_test
                    instrumented/011_rdna4_d128_attention_block_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_rdna4_d128_attention_pressure_test
                    instrumented/012_rdna4_d128_attention_pressure_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_rdna4_wmma_register_handoff_test
                    instrumented/013_rdna4_wmma_register_handoff_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_rdna4_wmma_no_score_lds_attention_test
                    instrumented/014_rdna4_wmma_no_score_lds_attention_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_rdna4_d128_no_score_lds_attention_test
                    instrumented/015_rdna4_d128_no_score_lds_attention_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_rdna4_pingpong_private_lds_test
                    instrumented/016_rdna4_pingpong_private_lds_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_rdna4_pingpong_cooperative_lds_test
                    instrumented/017_rdna4_pingpong_cooperative_lds_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_rdna4_pingpong_wide_cooperative_lds_test
                    instrumented/018_rdna4_pingpong_wide_cooperative_lds_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_rdna4_wmma_streamk_arrival_counter_test
                    instrumented/028_rdna4_wmma_streamk_arrival_counter_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_rdna4_wmma_streamk_tree_atomic_or_test
                    instrumented/029_rdna4_wmma_streamk_tree_atomic_or_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_rdna4_wmma_attention_lds_alias_handoff_test
                    instrumented/036_rdna4_wmma_attention_lds_alias_handoff_test.hip)
endif()
if(_hip_moi_build_gfx1250_tests)
  hip_moi_add_gtest(hip_moi_instrumented_gfx1250_wmma_register_handoff_test
                    instrumented/013_gfx1250_wmma_register_handoff_test.hip)
endif()
if(_hip_moi_build_cdna4_tests)
  hip_moi_add_gtest(hip_moi_instrumented_cdna4_mfma_attention_block_test
                    instrumented/010_cdna4_mfma_attention_block_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_cdna4_d128_attention_block_test
                    instrumented/011_cdna4_d128_attention_block_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_cdna4_d128_attention_pressure_test
                    instrumented/012_cdna4_d128_attention_pressure_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_cdna4_mfma_register_handoff_test
                    instrumented/013_cdna4_mfma_register_handoff_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_cdna4_mfma_no_score_lds_attention_test
                    instrumented/014_cdna4_mfma_no_score_lds_attention_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_cdna4_d128_no_score_lds_attention_test
                    instrumented/015_cdna4_d128_no_score_lds_attention_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_cdna4_pingpong_private_lds_test
                    instrumented/016_cdna4_pingpong_private_lds_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_cdna4_pingpong_cooperative_lds_test
                    instrumented/017_cdna4_pingpong_cooperative_lds_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_cdna4_pingpong_wide_cooperative_lds_test
                    instrumented/018_cdna4_pingpong_wide_cooperative_lds_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_cdna4_mfma_streamk_arrival_counter_test
                    instrumented/028_cdna4_mfma_streamk_arrival_counter_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_cdna4_mfma_streamk_tree_atomic_or_test
                    instrumented/029_cdna4_mfma_streamk_tree_atomic_or_test.hip)
  hip_moi_add_gtest(hip_moi_instrumented_cdna4_mfma_attention_lds_alias_handoff_test
                    instrumented/036_cdna4_mfma_attention_lds_alias_handoff_test.hip)
endif()
hip_moi_add_gtest(hip_moi_instrumented_atomic_api_test
                  instrumented/019_atomic_api_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_atomic_metadata_test
                  instrumented/020_atomic_metadata_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_atomic_happens_before_test
                  instrumented/021_atomic_happens_before_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_atomic_fast_path_test
                  instrumented/022_atomic_fast_path_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_atomic_rmw_happens_before_test
                  instrumented/023_atomic_rmw_happens_before_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_atomic_or_bitmask_happens_before_test
                  instrumented/024_atomic_or_bitmask_happens_before_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_atomic_fence_happens_before_test
                  instrumented/025_atomic_fence_happens_before_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_streamk_flag_protocol_test
                  instrumented/026_streamk_flag_protocol_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_streamk_two_tile_flag_protocol_test
                  instrumented/027_streamk_two_tile_flag_protocol_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_atomic_exchange_compare_exchange_test
                  instrumented/030_atomic_exchange_compare_exchange_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_atomic_fence_rmw_happens_before_test
                  instrumented/031_atomic_fence_rmw_happens_before_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_workgroup_barrier_fence_test
                  instrumented/032_workgroup_barrier_fence_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_atomic_fence_extended_test
                  instrumented/033_atomic_fence_extended_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_atomic_bitwise_happens_before_test
                  instrumented/034_atomic_bitwise_happens_before_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_attention_lds_alias_handoff_test
                  instrumented/035_attention_lds_alias_handoff_test.hip)
hip_moi_add_gtest(hip_moi_instrumented_streamk_two_level_reduction_test
                  instrumented/037_streamk_two_level_reduction_test.hip)
