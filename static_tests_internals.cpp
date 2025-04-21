/*
 * MIT License
 * Copyright (c) 2025 Youcef Lemsafer
 * See LICENSE file for more details.
 * Creation date: April 2025.
 */
#include "lfp.hpp"

constexpr lfp::details::bitmask_impl<uint64_t, 7> mask_7;

static_assert(mask_7.size() == 56);
static_assert(mask_7.offset(49) == 0);
static_assert(mask_7.offset(77) == 7);
static_assert(mask_7.offset(91) == 11);
static_assert(mask_7.offset(119) == 18);
static_assert(mask_7.offset(133) == 22);
static_assert(mask_7.offset(259) == 0);
static_assert(mask_7.word_at(0) ==
  //   0      7  11     18  22     29          41 44          56     63
  //   v      v   v      v   v      v           v  v           v      v
     0b0111111011101111110111011111101111111111101101111111111101111110);
static_assert(mask_7.word_at(26) ==
     0b1110111111111110110111111111110111111011101111110111011111101111);
static_assert(mask_7.word_at(55) ==
     0b1011111101110111111011101111110111111111110110111111111110111111);



