# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# docs/tutorial/CMakeLists.txt
#
# Build and exercise the tutorial programs so examples do not drift.

function(hip_moi_add_tutorial name source)
  add_executable(${name} ${source})
  target_compile_features(${name} PRIVATE cxx_std_17)
  target_link_libraries(${name} PRIVATE hip_moi::hip_moi)
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
endfunction()

function(hip_moi_add_passing_tutorial_test test_name target_name)
  add_test(NAME ${test_name}
           COMMAND ${target_name})
  if(HIP_MOI_ROCM_ROOT)
    set_tests_properties(${test_name}
                         PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${HIP_MOI_ROCM_ROOT}/lib")
  endif()
endfunction()

function(hip_moi_add_failing_tutorial_test test_name target_name)
  set(command
      ${CMAKE_COMMAND}
      "-DEXPECTED_PROGRAM=$<TARGET_FILE:${target_name}>")

  set(index 0)
  foreach(expected_substring IN LISTS ARGN)
    list(APPEND command "-DEXPECTED_SUBSTRING_${index}=${expected_substring}")
    math(EXPR index "${index} + 1")
  endforeach()

  list(APPEND command -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake")

  add_test(NAME ${test_name}
           COMMAND ${command})
  if(HIP_MOI_ROCM_ROOT)
    set_tests_properties(${test_name}
                         PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${HIP_MOI_ROCM_ROOT}/lib")
  endif()
endfunction()

set(_hip_moi_build_rdna4_tutorials TRUE)
foreach(_hip_moi_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
  if(NOT _hip_moi_arch MATCHES "^gfx12")
    set(_hip_moi_build_rdna4_tutorials FALSE)
  endif()
endforeach()

hip_moi_add_tutorial(hip_moi_tutorial_001_passing_syncthreads
                     001_passing_syncthreads.hip)
hip_moi_add_tutorial(hip_moi_tutorial_002_failing_cross_subgroup_race
                     002_failing_cross_subgroup_race.hip)
hip_moi_add_tutorial(hip_moi_tutorial_003_destructor_fallback
                     003_destructor_fallback.hip)
hip_moi_add_tutorial(hip_moi_tutorial_005_context_cross_subgroup_race
                     005_context_cross_subgroup_race.hip)

hip_moi_add_passing_tutorial_test(HipMoiTutorial.001PassingSyncthreads
                                  hip_moi_tutorial_001_passing_syncthreads)
hip_moi_add_failing_tutorial_test(HipMoiTutorial.002FailingCrossSubgroupRace
                                  hip_moi_tutorial_002_failing_cross_subgroup_race
                                  "kind=access_conflict"
                                  "HIP_MOI_CHECK failed")
hip_moi_add_failing_tutorial_test(HipMoiTutorial.003DestructorFallback
                                  hip_moi_tutorial_003_destructor_fallback
                                  "kind=access_conflict"
                                  "unconsumed diagnostics")
hip_moi_add_failing_tutorial_test(HipMoiTutorial.005ContextCrossSubgroupRace
                                  hip_moi_tutorial_005_context_cross_subgroup_race
                                  "kind=access_conflict"
                                  "first_subgroup="
                                  "second_subgroup="
                                  "HIP_MOI_CHECK failed")

if(_hip_moi_build_rdna4_tutorials)
  hip_moi_add_tutorial(hip_moi_tutorial_004_rdna4_wmma_data_tiled_matmul
                       004_rdna4_wmma_data_tiled_matmul.hip)
  hip_moi_add_passing_tutorial_test(
    HipMoiTutorial.004Rdna4WmmaDataTiledMatmul
    hip_moi_tutorial_004_rdna4_wmma_data_tiled_matmul)
endif()
