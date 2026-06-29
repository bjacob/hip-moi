# Copyright 2026
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# benchmarks/CMakeLists.txt
#

option(HIP_MOI_BUILD_BENCHMARKS "Build hip-moi benchmark executables" OFF)

if(NOT HIP_MOI_BUILD_BENCHMARKS)
  return()
endif()

set(_hip_moi_build_rdna4_benchmarks TRUE)
foreach(_hip_moi_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
  if(NOT _hip_moi_arch MATCHES "^gfx12")
    set(_hip_moi_build_rdna4_benchmarks FALSE)
  endif()
endforeach()

set(_hip_moi_build_cdna4_benchmarks TRUE)
foreach(_hip_moi_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
  if(NOT _hip_moi_arch MATCHES "^gfx950")
    set(_hip_moi_build_cdna4_benchmarks FALSE)
  endif()
endforeach()

if(NOT _hip_moi_build_rdna4_benchmarks AND NOT _hip_moi_build_cdna4_benchmarks)
  message(STATUS "Skipping hip-moi architecture-specific benchmarks for CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
  return()
endif()

function(hip_moi_add_benchmark name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE hip_moi::hip_moi)
  target_compile_features(${name} PRIVATE cxx_std_20)
  target_compile_options(
    ${name}
    PRIVATE
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall>
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wextra>
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Werror>
      $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-O3>
      $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-ffast-math>
      $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-Wall>
      $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-Wextra>
      $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-Werror>
      $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-Wno-unused-command-line-argument>)
  set_target_properties(${name} PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON)
  if(HIP_MOI_ROCM_ROOT)
    set_target_properties(${name} PROPERTIES
      BUILD_RPATH "${HIP_MOI_ROCM_ROOT}/lib")
  endif()
endfunction()

if(_hip_moi_build_rdna4_benchmarks)
  hip_moi_add_benchmark(hip_moi_benchmark_w2_2x4 w2_2x4_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_w4_4x16 w2_2x4_benchmark.hip)
  target_compile_definitions(hip_moi_benchmark_w4_4x16 PRIVATE
    HIP_MOI_BENCH_WAVES=4
    HIP_MOI_BENCH_MTILES=4
    HIP_MOI_BENCH_NTILES=16
    HIP_MOI_BENCH_WAVE_TILE_M=4
    HIP_MOI_BENCH_WAVE_TILE_N=4)

  hip_moi_add_benchmark(hip_moi_benchmark_w8_16x8 w2_2x4_benchmark.hip)
  target_compile_definitions(hip_moi_benchmark_w8_16x8 PRIVATE
    HIP_MOI_BENCH_WAVES=8
    HIP_MOI_BENCH_MTILES=16
    HIP_MOI_BENCH_NTILES=8
    HIP_MOI_BENCH_WAVE_TILE_M=4
    HIP_MOI_BENCH_WAVE_TILE_N=4
    HIP_MOI_BENCH_STORAGE_BYTES=4194304)

  hip_moi_add_benchmark(hip_moi_benchmark_prod_16x8 prod_16x8_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_attention_block
                              010_rdna4_wmma_attention_block_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_attention_d128
                              011_rdna4_d128_attention_block_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_attention_d128_pressure
                              012_rdna4_d128_attention_pressure_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_attention_no_score_lds
                              014_rdna4_wmma_no_score_lds_attention_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_attention_d128_no_score_lds
                              015_rdna4_d128_no_score_lds_attention_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_pingpong_private_lds
                              016_rdna4_pingpong_att_probe.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_pingpong_wide_cooperative_lds
                              018_rdna4_pingpong_wide_cooperative_lds_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_atomic_flag_handoff
                              019_atomic_flag_handoff_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_atomic_metadata
                              020_atomic_metadata_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_atomic_happens_before
                              021_atomic_happens_before_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_atomic_rmw_happens_before
                              023_atomic_rmw_happens_before_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_atomic_or_bitmask_happens_before
                              024_atomic_or_bitmask_happens_before_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_atomic_fence_happens_before
                              025_atomic_fence_happens_before_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_streamk_flag_protocol
                              026_streamk_flag_protocol_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_streamk_two_tile_flag_protocol
                              027_streamk_two_tile_flag_protocol_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_rdna4_wmma_streamk_arrival_counter
                              028_rdna4_wmma_streamk_arrival_counter_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_rdna4_wmma_streamk_tree_atomic_or
                              029_rdna4_wmma_streamk_tree_atomic_or_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_atomic_exchange_compare_exchange
                              030_atomic_exchange_compare_exchange_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_atomic_fence_rmw_happens_before
                              031_atomic_fence_rmw_happens_before_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_atomic_fence_extended
                              033_atomic_fence_extended_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_atomic_bitwise_happens_before
                              034_atomic_bitwise_happens_before_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_rdna4_wmma_attention_lds_alias_handoff
                              036_rdna4_wmma_attention_lds_alias_handoff_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_streamk_two_level_reduction
                              037_streamk_two_level_reduction_benchmark.hip)
endif()

if(_hip_moi_build_cdna4_benchmarks)
  hip_moi_add_benchmark(hip_moi_benchmark_cdna4_mfma_streamk_arrival_counter
                        028_cdna4_mfma_streamk_arrival_counter_benchmark.hip)
  hip_moi_add_benchmark(hip_moi_benchmark_cdna4_mfma_streamk_tree_atomic_or
                        029_cdna4_mfma_streamk_tree_atomic_or_benchmark.hip)
endif()
