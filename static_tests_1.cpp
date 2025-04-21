/*
 * MIT License
 * Copyright (c) 2025 Youcef Lemsafer
 * See LICENSE file for more details.
 * Creation date: February 2025.
 */
#include "lfp.hpp"


using lfp::sieve_to_vector;

static_assert(sieve_to_vector<uint32_t>(int16_t(1), int16_t(2)) == std::vector<uint32_t>{});
static_assert(sieve_to_vector<uint32_t>(int16_t(0), int16_t(6)) == std::vector<uint32_t>{2,3,5});
static_assert(sieve_to_vector<uint16_t>(int16_t(4), int16_t(17)) == std::vector<uint16_t>{5,7,11,13});
static_assert(sieve_to_vector<uint16_t>(int16_t(250), int16_t(260)) == std::vector<uint16_t>{251,257});
static_assert(sieve_to_vector<uint16_t>(int16_t(240), int16_t(400)) == std::vector<uint16_t>{241, 251, 257, 263, 269, 271,
	    277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373,
	    379, 383, 389, 397});
static_assert(sieve_to_vector<uint16_t>(int16_t(250), int16_t(259)) == std::vector<uint16_t>{251,257});
static_assert(sieve_to_vector<uint32_t>(uint16_t(251), uint16_t(252)) == std::vector<uint32_t>{251});
static_assert(sieve_to_vector<uint32_t>(uint16_t(251), uint16_t(253)) == std::vector<uint32_t>{251});
static_assert(sieve_to_vector<uint32_t>(uint16_t(251), uint16_t(254)) == std::vector<uint32_t>{251});
static_assert(sieve_to_vector<uint32_t>(uint16_t(251), uint16_t(255)) == std::vector<uint32_t>{251});
static_assert(sieve_to_vector<uint16_t>(uint16_t(250), uint16_t(259)) == std::vector<uint16_t>{251,257});
static_assert(sieve_to_vector<uint32_t>(uint16_t(251), uint16_t(252)) == std::vector<uint32_t>{251});
static_assert(sieve_to_vector<uint32_t>(uint16_t(251), uint16_t(253)) == std::vector<uint32_t>{251});
static_assert(sieve_to_vector<uint32_t>(uint16_t(251), uint16_t(254)) == std::vector<uint32_t>{251});
static_assert(sieve_to_vector<uint32_t>(uint16_t(251), uint16_t(255)) == std::vector<uint32_t>{251});
static_assert(sieve_to_vector<uint32_t>(uint16_t(251), uint16_t(256)) == std::vector<uint32_t>{251});
static_assert(sieve_to_vector<uint32_t>(uint16_t(251), uint16_t(257)) == std::vector<uint32_t>{251});
static_assert(sieve_to_vector<uint32_t>(uint16_t(251), uint16_t(258)) == std::vector<uint32_t>{251,257});
static_assert(sieve_to_vector<uint16_t>(uint16_t(256), uint16_t(277)) == std::vector<uint16_t>{257, 263, 269, 271});
static_assert(sieve_to_vector<uint16_t>(uint16_t(498), uint16_t(525)) == std::vector<uint16_t>{499, 503, 509, 521, 523});
static_assert(sieve_to_vector<uint16_t>(uint16_t(1202), uint16_t(1279)) == std::vector<uint16_t>{1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259,
	    1277});
static_assert(sieve_to_vector<uint16_t>(uint16_t(3300), uint16_t(3391)) == std::vector<uint16_t>{3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343,
	    3347, 3359, 3361, 3371, 3373, 3389});

