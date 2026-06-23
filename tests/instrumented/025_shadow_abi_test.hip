// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/instrumented/025_shadow_abi_test.hip
//
// Tests for Loom-style compact shadow/watchpoint metadata helpers.
#include "hip_moi/hip_moi.hpp"

#include <gtest/gtest.h>

namespace
{
    using hip_moi::shadow_access_kind;
    using hip_moi::detail::decode_exact_shadow_entry;
    using hip_moi::detail::decode_sampled_watchpoint_entry;
    using hip_moi::detail::exact_shadow_entries_conflict;
    using hip_moi::detail::pack_exact_shadow_entry;
    using hip_moi::detail::pack_sampled_watchpoint_entry;
    using hip_moi::detail::sampled_watchpoints_conflict;

    static_assert(hip_moi::detail::exact_shadow::granule_bytes == 4);
    static_assert(hip_moi::detail::exact_shadow::site_shift
                      + hip_moi::detail::exact_shadow::site_bits
                  == 64);
    static_assert(hip_moi::detail::sampled_watchpoint::granule_bytes == 4);
    static_assert(hip_moi::detail::sampled_watchpoint::count_shift
                      + hip_moi::detail::sampled_watchpoint::count_bits
                  == 64);
    static_assert(hip_moi::detail::shadow_kind_from_access_kind(hip_moi::access_kind::load)
                  == shadow_access_kind::read);
    static_assert(hip_moi::detail::shadow_kind_from_access_kind(hip_moi::access_kind::store)
                  == shadow_access_kind::write);
    static_assert(hip_moi::detail::shadow_kind_conflicts(shadow_access_kind::write,
                                                         shadow_access_kind::read));
    static_assert(!hip_moi::detail::shadow_kind_conflicts(shadow_access_kind::read,
                                                          shadow_access_kind::read));

    TEST(HipMoiShadowAbi, ExactShadowEntryRoundTripsMaskedFields)
    {
        constexpr uint64_t packed = pack_exact_shadow_entry(shadow_access_kind::write,
                                                            /*owner_id=*/0x413,
                                                            /*epoch=*/0x477,
                                                            /*generation=*/0x1f1234,
                                                            /*site_id=*/0x9abcdef);
        constexpr hip_moi::detail::exact_shadow_entry decoded = decode_exact_shadow_entry(packed);

        EXPECT_EQ(decoded.kind, shadow_access_kind::write);
        EXPECT_EQ(decoded.owner_id, 0x13u);
        EXPECT_EQ(decoded.epoch, 0x77u);
        EXPECT_EQ(decoded.generation, 0xf1234u);
        EXPECT_EQ(decoded.site_id, 0xbcdefu);
    }

    TEST(HipMoiShadowAbi, ExactShadowConflictPredicateMatchesSubgroupContract)
    {
        constexpr hip_moi::detail::exact_shadow_entry current{
            shadow_access_kind::read,
            /*owner_id=*/2,
            /*epoch=*/17,
            /*generation=*/99,
            /*site_id=*/101,
        };
        constexpr hip_moi::detail::exact_shadow_entry prior_write{
            shadow_access_kind::write,
            /*owner_id=*/1,
            /*epoch=*/17,
            /*generation=*/99,
            /*site_id=*/202,
        };
        constexpr hip_moi::detail::exact_shadow_entry same_owner{
            shadow_access_kind::write,
            /*owner_id=*/2,
            /*epoch=*/17,
            /*generation=*/99,
            /*site_id=*/202,
        };
        constexpr hip_moi::detail::exact_shadow_entry old_epoch{
            shadow_access_kind::write,
            /*owner_id=*/1,
            /*epoch=*/16,
            /*generation=*/99,
            /*site_id=*/202,
        };
        constexpr hip_moi::detail::exact_shadow_entry prior_read{
            shadow_access_kind::read,
            /*owner_id=*/1,
            /*epoch=*/17,
            /*generation=*/99,
            /*site_id=*/202,
        };

        EXPECT_TRUE(exact_shadow_entries_conflict(current, prior_write));
        EXPECT_FALSE(exact_shadow_entries_conflict(current, same_owner));
        EXPECT_FALSE(exact_shadow_entries_conflict(current, old_epoch));
        EXPECT_FALSE(exact_shadow_entries_conflict(current, prior_read));
    }

    TEST(HipMoiShadowAbi, SampledWatchpointRoundTripsRangeFields)
    {
        constexpr uint64_t packed = pack_sampled_watchpoint_entry(shadow_access_kind::read,
                                                                  /*owner_id=*/0x512,
                                                                  /*epoch=*/0x4aa,
                                                                  /*generation=*/0x1abcde,
                                                                  /*start_cell=*/0x4567,
                                                                  /*cell_count=*/32,
                                                                  /*consumed=*/true);
        constexpr hip_moi::detail::sampled_watchpoint_entry decoded
            = decode_sampled_watchpoint_entry(packed);

        EXPECT_TRUE(decoded.valid);
        EXPECT_TRUE(decoded.consumed);
        EXPECT_EQ(decoded.kind, shadow_access_kind::read);
        EXPECT_EQ(decoded.owner_id, 0x112u);
        EXPECT_EQ(decoded.epoch, 0xaau);
        EXPECT_EQ(decoded.generation, 0xabcdeu);
        EXPECT_EQ(decoded.start_cell, 0x567u);
        EXPECT_EQ(decoded.cell_count, 32u);
    }

    TEST(HipMoiShadowAbi, SampledWatchpointConflictRequiresExactRangeOverlap)
    {
        constexpr hip_moi::detail::sampled_watchpoint_entry current{
            /*valid=*/true,
            /*consumed=*/false,
            shadow_access_kind::write,
            /*owner_id=*/3,
            /*epoch=*/11,
            /*generation=*/42,
            /*start_cell=*/20,
            /*cell_count=*/4,
        };
        constexpr hip_moi::detail::sampled_watchpoint_entry overlapping_read{
            /*valid=*/true,
            /*consumed=*/false,
            shadow_access_kind::read,
            /*owner_id=*/1,
            /*epoch=*/11,
            /*generation=*/42,
            /*start_cell=*/23,
            /*cell_count=*/3,
        };
        constexpr hip_moi::detail::sampled_watchpoint_entry adjacent_read{
            /*valid=*/true,
            /*consumed=*/false,
            shadow_access_kind::read,
            /*owner_id=*/1,
            /*epoch=*/11,
            /*generation=*/42,
            /*start_cell=*/24,
            /*cell_count=*/2,
        };
        constexpr hip_moi::detail::sampled_watchpoint_entry consumed_read{
            /*valid=*/true,
            /*consumed=*/true,
            shadow_access_kind::read,
            /*owner_id=*/1,
            /*epoch=*/11,
            /*generation=*/42,
            /*start_cell=*/23,
            /*cell_count=*/3,
        };
        constexpr hip_moi::detail::sampled_watchpoint_entry old_generation{
            /*valid=*/true,
            /*consumed=*/false,
            shadow_access_kind::read,
            /*owner_id=*/1,
            /*epoch=*/11,
            /*generation=*/41,
            /*start_cell=*/23,
            /*cell_count=*/3,
        };

        EXPECT_TRUE(sampled_watchpoints_conflict(current, overlapping_read));
        EXPECT_FALSE(sampled_watchpoints_conflict(current, adjacent_read));
        EXPECT_FALSE(sampled_watchpoints_conflict(current, consumed_read));
        EXPECT_FALSE(sampled_watchpoints_conflict(current, old_generation));
    }
} // namespace
