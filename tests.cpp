/*
 * MIT License
 * Copyright (c) 2025 Youcef Lemsafer
 * See LICENSE file for more details.
 * Creation date: February 2025.
 */
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"

#include "lfp.hpp"
#include "lfp_tests.hpp"

using lfp::count_primes;
using lfp::sieve_to_vector;
using Catch::Matchers::Equals;



TEST_CASE("Sieve of Eratosthenes - small primes - #1") {
    CHECK_THAT(count_primes(0u, 1u), equals(0));
    CHECK_THAT(count_primes(0u, 2u), equals(0));
    CHECK_THAT(count_primes(0u, 3u), equals(1));
    CHECK_THAT(count_primes(0u, 4u), equals(2));
    CHECK_THAT(count_primes(0u, 5u), equals(2));
    CHECK_THAT(count_primes(0u, 6u), equals(3));
}


TEST_CASE("Sieve of Eratosthenes - small primes - #2") {
    CHECK_THAT(count_primes(10u, 23u), equals(4));
    CHECK_THAT(count_primes(0u, 101u), equals(25));
    CHECK_THAT(count_primes(13u, 50u), equals(10));
    CHECK_THAT(count_primes(100u, 131u), equals(6));
}


TEST_CASE("Sieve of Eratosthenes - larger primes - #1") {
    CHECK_THAT(count_primes(0u, 1000u), equals(168));
    CHECK_THAT(count_primes(0u, 10000u), equals(1229));
    CHECK_THAT(count_primes(0u, 65536u), equals(6542));
    CHECK_THAT(count_primes(65536u, 131072u), equals(5709));
}


TEST_CASE("Sieve of Eratosthenes - larger primes - #2") {
    CHECK_THAT(count_primes(131070u, 141709u), equals(909));
    CHECK_THAT(count_primes(312307u, 313409u), equals(91));
    CHECK_THAT(count_primes(524287u, 524287u), equals(0));
    CHECK_THAT(count_primes(524287u, 524288u), equals(1));
}


TEST_CASE("Sieve of Eratosthenes - even larger primes - #1") {
    CHECK_THAT(count_primes(3141592u, 3142001u), equals(22));
    CHECK_THAT(count_primes(8388608u, 8389009u), equals(25));
    CHECK_THAT(count_primes(4458763u, 4459763u), equals(70));
    CHECK_THAT(count_primes(24033746u, 24043746u), equals(620));
}


TEST_CASE("Sieve of Eratosthenes - even larger primes - #2") {
    CHECK_THAT(count_primes(17310383u, 52187113u), equals(2014451));
    CHECK_THAT(count_primes(57931727u, 83251653u), equals(1401439));
    CHECK_THAT(count_primes(100'000'000u, 101'000'000u), equals(54208));
    CHECK_THAT(count_primes(500'000'000u, 501'000'000u), equals(49918));
}


TEST_CASE("Sieve of Eratosthenes - even larger primes - #3") {
    CHECK_THAT(count_primes(2039522013u, 2039622967u), equals(4656));
    CHECK_THAT(count_primes(3550634754u, 3561957981u), equals(514735));
    CHECK_THAT(count_primes(3343233271u, 3394567417u), equals(2339267));
    CHECK_THAT(count_primes(4132050211u, 4208912531u), equals(3469201));
}


TEST_CASE("Sieve of Eratosthenes - limits - #1") {
    CHECK_THAT(count_primes(4294967200u, 4294967295u), equals(3));
}

TEST_CASE("Sieve of Eratosthenes - limits - #2") {
    CHECK_THAT(count_primes(uint32_t(0), uint32_t(4294967295u)), equals(203280221));
}

TEST_CASE("Sieve of Eratosthenes - list primes - #1") {
    CHECK_THAT(sieve_to_vector<int32_t>(uint16_t(0), uint16_t(2)), Equals(std::vector<int32_t>{}));
    CHECK_THAT(sieve_to_vector<int32_t>(uint16_t(0), uint16_t(2)), Equals(primes_by_division<int32_t>(0, 2)));
    CHECK_THAT(sieve_to_vector<int32_t>(uint16_t(0), uint16_t(3)), Equals(std::vector<int32_t>{{2}}));
    CHECK_THAT(sieve_to_vector<int32_t>(uint16_t(0), uint16_t(3)), Equals(primes_by_division<int32_t>(0, 3)));
    CHECK_THAT(sieve_to_vector<int32_t>(uint16_t(0), uint16_t(20)), Equals(primes_by_division<int32_t>(0, 20)));
    CHECK_THAT(sieve_to_vector<int32_t>(uint16_t(0), uint16_t(20)), Equals(std::vector<int32_t>{{2, 3, 5, 7, 11, 13, 17, 19}}));
}


TEST_CASE("Sieve of Eratosthenes - list primes - #2") {
    CHECK_THAT(sieve_to_vector<int32_t>(uint16_t(0), uint16_t(100)), Equals(primes_by_division<int32_t>(0, 100)));
    CHECK_THAT(sieve_to_vector<int32_t>(uint16_t(1), uint16_t(101)), Equals(primes_by_division<int32_t>(1, 101)));
    CHECK_THAT(sieve_to_vector<int32_t>(uint16_t(9000), uint16_t(10000)), Equals(primes_by_division<int32_t>(9000, 10000)));
    CHECK_THAT(sieve_to_vector<int32_t>(uint16_t(65500), uint16_t(65535)), Equals(primes_by_division<int32_t>(65500, 65535)));
}


TEST_CASE("Sieve of Eratosthenes - list primes - #3") {
    CHECK_THAT(sieve_to_vector<int32_t>(131000, 131500), Equals(primes_by_division<int32_t>(131000, 131500)));
    CHECK_THAT(sieve_to_vector<int32_t>(640191, 703411), Equals(primes_by_division<int32_t>(640191, 703411)));
    CHECK_THAT(sieve_to_vector<int32_t>(1'350'209, 1'358'907), Equals(primes_by_division<int32_t>(1'350'209, 1'358'907)));
    CHECK_THAT(sieve_to_vector<int32_t>(2'147'483'548, 2'147'483'647), Equals(primes_by_division<int32_t>(2'147'483'548, 2'147'483'647)));
}


TEST_CASE("Sieve of Eratosthenes - list primes - #4") {
    CHECK_THAT(sieve_to_vector<uint32_t>(8093*8093, 8191*8191+1), Equals(primes_by_division<uint32_t>(8093*8093, 8191*8191+1)));
    CHECK_THAT(sieve_to_vector<uint32_t>(4'294'967'200, 4'294'967'295), Equals(primes_by_division<uint32_t>(4'294'967'200, 4'294'967'295)));
}


TEST_CASE("Sieve of Eratosthenes - primes iterator and range") {
    using namespace lfp;
    using namespace lfp::details;
    {
        bitmap<uint32_t> bmp;
	sieve_data<uint32_t> sievdat{.bitmap_ = &bmp};
        inner_sieve<uint32_t>(u8primes<uint8_t>, 300u, 400u, [](auto, auto, auto) {}, sievdat);
        primes_iterator<uint32_t, uint32_t> it{&bmp}, ite{&bmp, true};
        CHECK_THAT(std::vector<uint32_t>(it, ite), Equals(primes_by_division<uint32_t>(300, 400)));
    }
    {
	bitmap<uint32_t> bmp;
	sieve_data<uint32_t> sievdat{.bitmap_ = &bmp};
	inner_sieve<int32_t>(u16primes<uint16_t>, 10000u, 12000u, [](auto, auto, auto) {}, sievdat);
        primes_iterator<int32_t, uint32_t> it{&bmp}, ite{&bmp, true};
        CHECK_THAT(std::vector<int32_t>(it, ite), Equals(primes_by_division<int32_t>(10000, 12000)));
    }
    {
	std::vector<int32_t> primes;
	for(auto p : sieve<int32_t>(uint32_t(100000), uint32_t(101000))) {
	    primes.push_back(p);
	}
	CHECK_THAT(primes, Equals(primes_by_division<int32_t>(100000, 101000)));
    }
    {
	auto n0 = 1234567890, n1 = 1234667890;
	auto sieveres = sieve<int32_t>(uint32_t(n0), uint32_t(n1));
	std::vector<int32_t> primes;
	primes.reserve(sieveres.count());
	for(auto p : sieveres) {
	    primes.push_back(p);
	}
	CHECK_THAT(primes, Equals(primes_by_division<int32_t>(n0, n1)));
    }
}

TEST_CASE("Sieve of Eratosthenes - above 2^32 - #1") {
    CHECK_THAT(lfp::sieve_to_vector<int64_t>(uint64_t(300), uint64_t(400)), Equals(primes_by_division<int64_t>(uint64_t(300), uint64_t(400))));
    CHECK_THAT(lfp::sieve_to_vector<int64_t>(uint64_t(1) << 37, (uint64_t(1) << 37) + 100),
		Equals(std::vector<int64_t>{137438953481, 137438953501, 137438953513, 137438953541, 137438953567}));
    CHECK_THAT(lfp::sieve_to_vector<int64_t>(uint64_t(1) << 43, (uint64_t(1) << 43) + 200),
		Equals(primes_by_division<int64_t>(uint64_t(1) << 43, (uint64_t(1) << 43) + 200)));
    CHECK_THAT(lfp::sieve_to_vector<int64_t>(uint64_t(1) << 44, (uint64_t(1) << 44) + 400),
	    Equals(std::vector<int64_t>{17592186044423, 17592186044437, 17592186044443, 17592186044471, 17592186044591,
		    17592186044611, 17592186044651, 17592186044659, 17592186044747, 17592186044767, 17592186044813}));
    CHECK_THAT(lfp::sieve_to_vector<int64_t>(uint64_t(1) << 50, (uint64_t(1) << 50) + 1000),
		    Equals(primes_by_division<int64_t>(uint64_t(1) << 50, (uint64_t(1) << 50) + 1000)));
}

TEST_CASE("Sieve of Eratosthenes - above 2^32 - #2") {
    CHECK_THAT(lfp::sieve<int64_t>((uint64_t(1) << 33) - (uint64_t(1) << 30), uint64_t(1) << 33).count(), equals(47076888));
    CHECK_THAT(lfp::sieve_to_vector<uint64_t>(uint64_t(10000000000000000000u), uint64_t(10000000000000000100u)),
		    Equals(std::vector<uint64_t>{uint64_t(10000000000000000051u), uint64_t(10000000000000000087u), uint64_t(10000000000000000091u),
			    uint64_t(10000000000000000097u), uint64_t(10000000000000000099u)}));
    CHECK_THAT(lfp::sieve_to_vector<uint64_t>(uint64_t(18446744073709551516u), uint64_t(18446744073709551615u)),
		    Equals(std::vector<uint64_t>{uint64_t(18446744073709551521u), uint64_t(18446744073709551533u), uint64_t(18446744073709551557u)}));
}

TEST_CASE("Sieve of Eratosthenes - multithreaded sieve") {
    CHECK_THAT(lfp::sieve<int64_t>(uint64_t(0), uint64_t(1000000), lfp::threads{1}).count(), equals(78498));
    CHECK_THAT(lfp::sieve<int64_t>(uint64_t(0), uint64_t(1000000), lfp::threads{2}).count(), equals(78498));
    CHECK_THAT(lfp::sieve<int64_t>(uint64_t(0), uint64_t(1'000'000'000), lfp::threads{4}).count(), equals(50847534));
    CHECK_THAT(lfp::sieve<int64_t>(uint64_t(641 * 641), uint64_t(8191 * 8191 + 1), lfp::threads{4}).count(), equals(3922190));
}

// g++-12.2 is unable to compile the following test while g++-13.3 compiles it fine,
// any call to begin(sieve_results<int128_t>&) or end(sieve_results<int128_t>&) makes the compiler hang...
// So I deactivated the test unless g++ major >= 13.
#if LFP_HAS_UINT128
#  if defined(LFP_DISABLE_TESTS_ABOVE_64_BITS)
#    pragma message "Tests for ranges above 2^64 are disabled."
#  else

TEST_CASE("Sieve of Eratosthenes - above 2^64 - #1") {
    lfp::uint128_t n0 = lfp::uint128_t(1) << 64;
    auto res = lfp::sieve<lfp::int128_t>(n0, n0 + 1000);
    CHECK_THAT(res.count(), equals(25));
    std::vector<lfp::int128_t> primes{begin(res), end(res)};
    CHECK_THAT(primes.front(), equals((lfp::uint128_t(1) << 64) + 13));
    CHECK_THAT(primes[3], equals((lfp::uint128_t(1) << 64) + 81));
    CHECK_THAT(primes[21], equals((lfp::uint128_t(1) << 64) + 925));
    CHECK_THAT(primes.back(), equals((lfp::uint128_t(1) << 64) + 997));
}

TEST_CASE("Sieve of Eratosthenes - above 2^64 - #2") {
    lfp::uint128_t p = lfp::uint128_t(4294967311u);
    auto p2 = p * p, n0 = p2 - 21;
    auto res = lfp::sieve<lfp::uint128_t>(n0, n0 + 100);
    CHECK_THAT(res.count(), equals(5));
    std::vector<lfp::uint128_t> primes{begin(res), end(res)};
    std::vector<lfp::uint128_t> expectedPrimes{n0 + 33, n0 + 39, n0 + 57, n0 + 79, n0 + 91};
    CHECK_THAT(primes, Equals(expectedPrimes));
}

TEST_CASE("Sieve of Eratosthenes - above 2^64 - #3") {
    lfp::uint128_t p = lfp::uint128_t(4394967299u), q = lfp::uint128_t(4394967301u);
    auto n0 = p * q - 99;
    auto res = lfp::sieve<lfp::uint128_t>(n0, n0 + 100);
    CHECK_THAT(res.count(), equals(4));
    std::vector<lfp::uint128_t> primes{begin(res), end(res)};
    std::vector<lfp::uint128_t> expectedPrimes{n0 + 11, n0 + 57, n0 + 69, n0 + 89};
    CHECK_THAT(primes, Equals(expectedPrimes));
}

#  endif // LFP_DISABLE_TESTS_ABOVE_64_BITS
#endif // LFP_HAS_UINT128

TEST_CASE("Sieve of Eratosthenes - misc - #1") {
    CHECK_THAT(lfp::count_primes(uint64_t(1'005'000'000'000),  uint64_t(1'006'250'000'000)), equals(45228966));
}

TEST_CASE("Sieve of Eratosthenes - misc - #2") {
    auto primes = lfp::sieve_to_vector<int64_t>(10'000'000'000ull, 10'000'140'000ull);
    CHECK_THAT(primes.size(), equals(6073));
    CHECK_THAT(primes.back(), equals(10'000'139'969));
    CHECK_THAT(primes[primes.size() - 2], equals(10'000'139'953));
    CHECK_THAT(primes.front(), equals(10'000'000'019));
    // 3037th prime after 10^10
    CHECK_THAT(primes[primes.size() / 2], equals(10'000'070'131));
}

TEST_CASE("Sieve of Eratosthenes - misc - #3") {
    auto primes = lfp::sieve_to_vector<int32_t>(2147483548u, 2147483648u);
    CHECK_THAT(primes.size(), equals(6));
    CHECK_THAT(primes, Equals(std::vector<int32_t>{2147483549, 2147483563, 2147483579, 2147483587, 2147483629, 2147483647}));
}


TEST_CASE("Sieve of Eratosthenes - misc - #4") {
    uint64_t n0 = uint64_t{1} << 32;
    auto primes = lfp::sieve_to_vector<int64_t>(n0, n0 + 100);
    CHECK_THAT(primes.size(), equals(6));
    CHECK_THAT(primes, Equals(std::vector<int64_t>{4294967311, 4294967357, 4294967371, 4294967377, 4294967387, 4294967389}));
}



