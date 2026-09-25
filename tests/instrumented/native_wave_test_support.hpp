// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef HIP_MOI_TESTS_INSTRUMENTED_NATIVE_WAVE_TEST_SUPPORT_HPP
#define HIP_MOI_TESTS_INSTRUMENTED_NATIVE_WAVE_TEST_SUPPORT_HPP

#include "test_support.hpp"
#include <stdexcept>

namespace hip_moi::test
{
    inline int native_wave_size()
    {
        int        device = 0;
        int        size   = 0;
        hipError_t status = hipGetDevice(&device);
        if(status == hipSuccess)
        {
            status = hipDeviceGetAttribute(&size, hipDeviceAttributeWarpSize, device);
        }
        if(status != hipSuccess)
        {
            throw std::runtime_error(hipGetErrorString(status));
        }
        return size;
    }
} // namespace hip_moi::test
#endif
