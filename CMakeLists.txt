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

enable_testing()

add_subdirectory(tests)
