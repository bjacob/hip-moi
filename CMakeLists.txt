# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# CMakeLists.txt
#
# Build system for the hip-moi HIP memory-ordering instrumentation library.
#
cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES
      gfx1201
      CACHE STRING "HIP GPU architecture to build for")
endif()

project(hip_moi LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(NOT DEFINED HIP_MOI_ROCM_ROOT AND DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT)
  set(HIP_MOI_ROCM_ROOT
      "${CMAKE_HIP_COMPILER_ROCM_ROOT}"
      CACHE PATH "ROCm root used for hip-moi test runtime libraries")
endif()

add_library(hip_moi INTERFACE)
add_library(hip_moi::hip_moi ALIAS hip_moi)
target_include_directories(hip_moi
                           INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(hip_moi INTERFACE cxx_std_17)
target_compile_options(
  hip_moi
  INTERFACE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wextra>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Werror>
    $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-Wall>
    $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-Wextra>
    $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-Werror>)

enable_testing()

add_subdirectory(tests)
