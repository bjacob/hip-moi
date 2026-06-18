# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# docs/tutorial/expect_failure.cmake
#
# CTest helper for tutorial programs that intentionally abort after printing a
# hip-moi diagnostic.

if(NOT DEFINED EXPECTED_PROGRAM)
  message(FATAL_ERROR "EXPECTED_PROGRAM is required")
endif()

execute_process(
  COMMAND "${EXPECTED_PROGRAM}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
)

set(combined_output "${output}\n${error}")

if(result STREQUAL "0")
  message(FATAL_ERROR
          "Expected ${EXPECTED_PROGRAM} to fail, but it exited successfully.\n"
          "${combined_output}")
endif()

set(index 0)
while(index LESS 10)
  set(var "EXPECTED_SUBSTRING_${index}")
  if(DEFINED ${var})
    string(FIND "${combined_output}" "${${var}}" found_at)
    if(found_at EQUAL -1)
      message(FATAL_ERROR
              "Expected ${EXPECTED_PROGRAM} output to contain '${${var}}'.\n"
              "Process result: ${result}\n"
              "${combined_output}")
    endif()
  endif()
  math(EXPR index "${index} + 1")
endwhile()

message(STATUS "Observed expected failure from ${EXPECTED_PROGRAM}: ${result}")
