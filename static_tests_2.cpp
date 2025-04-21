/*
 * MIT License
 * Copyright (c) 2025 Youcef Lemsafer
 * See LICENSE file for more details.
 * Creation date: February 2025.
 */
#include "lfp.hpp"


using lfp::sieve_to_vector;

static_assert(sieve_to_vector<uint16_t>(uint16_t(8192), uint16_t(8193)) == std::vector<uint16_t>{});
static_assert(sieve_to_vector<uint16_t>(uint16_t(8190), uint16_t(8191)) == std::vector<uint16_t>{});
static_assert(sieve_to_vector<uint16_t>(uint16_t(8190), uint16_t(8192)) == std::vector<uint16_t>{8191});
static_assert(sieve_to_vector<uint16_t>(uint16_t(32767), uint16_t(32802)) == std::vector<uint16_t>{32771, 32779, 32783, 32789, 32797, 32801});
static_assert(sieve_to_vector<uint16_t>(uint16_t(48300), uint16_t(48403)) == std::vector<uint16_t>{48311, 48313, 48337, 48341, 48353, 48371,
	    48383, 48397});
static_assert(sieve_to_vector<uint16_t>(uint16_t(65470), uint16_t(65535)) == std::vector<uint16_t>{65479, 65497, 65519, 65521});
static_assert(sieve_to_vector<uint16_t>(uint16_t(65534), uint16_t(65535)) == std::vector<uint16_t>{});
static_assert(sieve_to_vector<uint16_t>(uint16_t(65533), uint16_t(65535)) == std::vector<uint16_t>{});
static_assert(sieve_to_vector<uint16_t>(uint16_t(65532), uint16_t(65535)) == std::vector<uint16_t>{});
static_assert(sieve_to_vector<uint16_t>(uint16_t(0), uint16_t(65535)).size() == 6542);

static_assert(sieve_to_vector<uint32_t>(uint32_t(0), uint32_t(3)) == std::vector<uint32_t>{2});
static_assert(sieve_to_vector<uint32_t>(uint32_t(262121), uint32_t(262144)) == std::vector<uint32_t>{262121, 262127, 262133, 262139});
static_assert(sieve_to_vector<uint32_t>(uint32_t(1048576), uint32_t(1048700)) == std::vector<uint32_t>{1048583, 1048589, 1048601, 1048609,
	    1048613, 1048627, 1048633, 1048661, 1048681});
static_assert(sieve_to_vector<uint32_t>(uint32_t(61075016), uint32_t(61075116))  == std::vector<uint32_t>{61075019, 61075037, 61075057, 61075061,
	    61075087, 61075099, 61075103, 61075109, 61075111});
static_assert(sieve_to_vector<int64_t>(uint64_t(1000000), uint64_t(1000010)) == std::vector<int64_t>{1000003});

// The following tests make g++-12.2 stall while g++-13.3 has no issue compiling this file.
// They are active by default, this allows to deactivate them.
#ifndef LFP_TESTS_DISABLE_STATIC_ASSERTS_IN_HIGHER_RANGES

static_assert(sieve_to_vector<uint32_t>(uint32_t(1074041825), uint32_t(1074041924)) == std::vector<uint32_t>{1074041849, 1074041869});
static_assert(sieve_to_vector<uint32_t>(uint32_t(4294967196), uint32_t(4294967295)) == std::vector<uint32_t>{4294967197, 4294967231,
	    4294967279, 4294967291});
static_assert(sieve_to_vector<uint32_t>(uint32_t(4294967293), uint32_t(4294967294)) == std::vector<uint32_t>{});
static_assert(sieve_to_vector<uint32_t>(uint32_t(4294967294), uint32_t(4294967295)) == std::vector<uint32_t>{});
static_assert(sieve_to_vector<uint32_t>(uint32_t(2147483548), uint32_t(2147483648)) == std::vector<uint32_t>{2147483549, 2147483563,
	    2147483579, 2147483587, 2147483629, 2147483647});
static_assert(sieve_to_vector<uint32_t>(uint32_t(3221225472), uint32_t(3221225672)) == std::vector<uint32_t>{3221225473, 3221225479,
	    3221225533, 3221225549, 3221225551, 3221225561, 3221225563, 3221225599, 3221225617, 3221225641,
	    3221225653, 3221225659, 3221225669});

#endif // LFP_TESTS_DISABLE_STATIC_ASSERTS_IN_HIGHER_RANGES

