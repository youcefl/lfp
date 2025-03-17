/*
 * MIT License
 * Copyright (c) 2025 Youcef Lemsafer
 * See LICENSE file for more details.
 * Creation date: February 2025.
 */
#include "lfp.hpp"

#ifndef DISABLE_STATIC_ASSERT_TESTS

using lfp::sieve16;
using lfp::sieve32;
using lfp::sieve;

static_assert(sieve16<uint32_t>(1,2) == std::vector<uint32_t>{});
static_assert(sieve16<uint32_t>(0, 6) == std::vector<uint32_t>{2,3,5});
static_assert(sieve16<uint16_t>(4, 17) == std::vector<uint16_t>{5,7,11,13});
static_assert(sieve16<uint16_t>(250,260) == std::vector<uint16_t>{251,257});
static_assert(sieve16<uint16_t>(240,400) == std::vector<uint16_t>{241, 251, 257, 263, 269, 271,
	    277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373,
	    379, 383, 389, 397});
static_assert(sieve16<uint16_t>(250,259) == std::vector<uint16_t>{251,257});
static_assert(sieve16<uint32_t>(251,252) == std::vector<uint32_t>{251});
static_assert(sieve16<uint32_t>(251,253) == std::vector<uint32_t>{251});
static_assert(sieve16<uint32_t>(251,254) == std::vector<uint32_t>{251});
static_assert(sieve16<uint32_t>(251,255) == std::vector<uint32_t>{251});
static_assert(sieve16<uint16_t>(250,259) == std::vector<uint16_t>{251,257});
static_assert(sieve16<uint32_t>(251,252) == std::vector<uint32_t>{251});
static_assert(sieve16<uint32_t>(251,253) == std::vector<uint32_t>{251});
static_assert(sieve16<uint32_t>(251,254) == std::vector<uint32_t>{251});
static_assert(sieve16<uint32_t>(251,255) == std::vector<uint32_t>{251});
static_assert(sieve16<uint32_t>(251,256) == std::vector<uint32_t>{251});
static_assert(sieve16<uint32_t>(251,257) == std::vector<uint32_t>{251});
static_assert(sieve16<uint32_t>(251,258) == std::vector<uint32_t>{251,257});
static_assert(sieve16<uint16_t>(256,277) == std::vector<uint16_t>{257, 263, 269, 271});
static_assert(sieve16<uint16_t>(498,525) == std::vector<uint16_t>{499, 503, 509, 521, 523});
static_assert(sieve16<uint16_t>(1202,1279) == std::vector<uint16_t>{1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259,
	    1277});
static_assert(sieve16<uint16_t>(3300,3391) == std::vector<uint16_t>{3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343,
	    3347, 3359, 3361, 3371, 3373, 3389});
static_assert(sieve16<uint16_t>(8192,8193) == std::vector<uint16_t>{});
static_assert(sieve16<uint16_t>(8190,8191) == std::vector<uint16_t>{});
static_assert(sieve16<uint16_t>(8190,8192) == std::vector<uint16_t>{8191});
static_assert(sieve16<uint16_t>(32767,32802) == std::vector<uint16_t>{32771, 32779, 32783, 32789, 32797, 32801});
static_assert(sieve16<uint16_t>(48300,48403) == std::vector<uint16_t>{48311, 48313, 48337, 48341, 48353, 48371,
	    48383, 48397});
static_assert(sieve16<uint16_t>(65470,65535) == std::vector<uint16_t>{65479, 65497, 65519, 65521});
static_assert(sieve16<uint16_t>(65534,65535) == std::vector<uint16_t>{});
static_assert(sieve16<uint16_t>(65533,65535) == std::vector<uint16_t>{});
static_assert(sieve16<uint16_t>(65532,65535) == std::vector<uint16_t>{});
static_assert(sieve16<uint16_t>(0,65535).size() == 6542);

static_assert(sieve32<uint32_t>(0, 3) == std::vector<uint32_t>{2});
static_assert(sieve32<uint32_t>(262121, 262144) == std::vector<uint32_t>{262121, 262127, 262133, 262139});
static_assert(sieve32<uint32_t>(1048576, 1048700) == std::vector<uint32_t>{1048583, 1048589, 1048601, 1048609,
	    1048613, 1048627, 1048633, 1048661, 1048681});
static_assert(sieve32<uint32_t>(61075016, 61075116)  == std::vector<uint32_t>{61075019, 61075037, 61075057, 61075061,
	    61075087, 61075099, 61075103, 61075109, 61075111});
static_assert(sieve32<uint32_t>(1074041825, 1074041924) == std::vector<uint32_t>{1074041849, 1074041869});
static_assert(sieve32<uint32_t>(4294967196, 4294967295) == std::vector<uint32_t>{4294967197, 4294967231,
	    4294967279, 4294967291});
static_assert(sieve32<uint32_t>(4294967293, 4294967294) == std::vector<uint32_t>{});
static_assert(sieve32<uint32_t>(4294967294, 4294967295) == std::vector<uint32_t>{});
static_assert(sieve32<uint32_t>(2147483548, 2147483648) == std::vector<uint32_t>{2147483549, 2147483563,
	    2147483579, 2147483587, 2147483629, 2147483647});
static_assert(sieve32<uint32_t>(3221225472, 3221225672) == std::vector<uint32_t>{3221225473, 3221225479,
	    3221225533, 3221225549, 3221225551, 3221225561, 3221225563, 3221225599, 3221225617, 3221225641,
	    3221225653, 3221225659, 3221225669});

#endif


