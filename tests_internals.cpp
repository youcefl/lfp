/*
 * MIT License
 * Copyright (c) 2025 Youcef Lemsafer
 * See LICENSE file for more details.
 * Creation date: May 2025.
 */
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "lfp.hpp"
#include "lfp_tests.hpp"


namespace lfpd = lfp::details;
using Catch::Matchers::Equals;


TEST_CASE("Sieve of Eratosthenes - internals - bitmap - #1")
{
    using bitmap = lfpd::bitmap_impl<uint64_t, uint64_t>;
    bitmap bmp{7, 27};

    CHECK_THAT(bmp.size(), equals(27));
    CHECK_THAT(bmp.first_value(), equals(7));
    CHECK_THAT(bmp.last_value(), equals(103));
    CHECK_THAT(bmp.value_at(0), equals(7));
    CHECK_THAT(bmp.value_at(8), equals(37));
    CHECK_THAT(bmp.value_at(16), equals(67));
    CHECK_THAT(bmp.value_at(24), equals(97));
    CHECK_THAT(bmp.value_at(bmp.index_of(103)), equals(103));
}

TEST_CASE("Sieve of Eratosthenes - internals - bitmap - #2")
{
    using bitmap_u16 = lfpd::bitmap_impl<uint16_t, uint64_t>;
    bitmap_u16 bmp{71 * 71, 321};

    CHECK_THAT(bmp.size(), equals(321));
    CHECK_THAT(bmp.first_value(), equals(5041));
    CHECK_THAT(bmp.last_value(), equals(6241));
    CHECK_THAT(bmp.value_at(0), equals(5041));
    CHECK_THAT(bmp.value_at(320), equals(6241));
}

TEST_CASE("Sieve of Eratosthenes - internals - bitmask application - #1")
{
    lfpd::bitmap<uint32_t> bmp1{71 * 71, 320}, bmp2{bmp1};
    lfpd::bitmask_pack<uint64_t, 7, 11> pck;
    bmp1.apply(pck);
    for(auto i = 0; i < bmp2.size(); ++i) {
	auto v = bmp2.value_at(i);
	if(!(v % 7) || !(v % 11)) {
	    bmp2.reset(i);
	}
    }

    for(auto i = 0; i < bmp1.size(); ++i) {
	if(bmp1.at(i) != bmp2.at(i)) {
	    std::ostringstream ostr;
	    ostr << "Difference found at offset " << i << ", masked: " << int(bmp1.at(i))
		    << ", reset at multiples of 7 and 11: " << int(bmp2.at(i));
	    CHECK(nullptr == ostr.str().data());
	}
    }
}

TEST_CASE("Sieve of Eratosthenes - internals - bitmask application - #2")
{
    lfpd::bitmap<uint32_t> bmp{7, (6241 - 7) / 30 * 8 + 7};
    lfpd::bitmask_pack<uint64_t, 7, 11, 13, 17, 19, 23, 29, 31> pck1;
    bmp.apply(pck1);
    lfpd::bitmask_pack<uint64_t, 37, 41, 43, 47, 53, 59, 61, 67> pck2;
    bmp.apply(pck2);
    lfpd::bitmask_pack<uint64_t, 71> pck3;
    bmp.apply(pck3);
    lfpd::bitmask_pack<uint64_t, 73> pck4;
    bmp.apply(pck4);

    std::vector<uint32_t> primes;
    bmp.foreach_setbit([&primes](auto idx, auto val) { primes.push_back(val); });
    auto expectedPrimes = primes_by_division<uint32_t>(7, 6241);

    CHECK_THAT(bmp.last_value(), equals(6239));
    CHECK_THAT(primes, Equals(expectedPrimes));
}


template <typename UInt>
void testSmallRangeBitmasking(UInt n0, std::size_t size)
{
    lfpd::bitmap<UInt> bmp{n0, size};
    lfpd::bitmask_pack<uint64_t, 7, 11, 13, 17, 19, 23, 29, 31> pck;
    bmp.apply(pck);

    std::vector<std::pair<UInt, bool>> actual;
    for(auto i = 0; i < bmp.size(); ++i) {
	actual.emplace_back(bmp.value_at(i), bmp.at(i));
    }
    std::vector<std::pair<UInt, bool>> expected;
    for(auto i = 0; i < bmp.size(); ++i) {
	auto v = bmp.value_at(i);
	expected.emplace_back(v, (v % 7) && (v % 11) && (v % 13)
			&& (v % 17) && (v % 19) && (v % 23) && (v % 29) && (v % 31));
    }

    CHECK_THAT(actual, Equals(expected));
}


TEST_CASE("Sieve of Eratosthenes - internals - bitmask application - #3")
{
    testSmallRangeBitmasking((uint64_t(1) << 32) + 1, 27);
}

#if LFP_HAS_UINT128
TEST_CASE("Sieve of Eratosthenes - internals - bitmask application - 128-bit - #1")
{
    testSmallRangeBitmasking<lfp::uint128_t>((uint64_t(1) << 32) + 1, 27);
}

TEST_CASE("Sieve of Eratosthenes - internals - bitmask application - 128-bit - #2")
{
    testSmallRangeBitmasking<lfp::uint128_t>((lfp::uint128_t{1} << 64) + 1, 27);
}

TEST_CASE("Sieve of Eratosthenes - internal - 128-bit integers output - #1")
{
    CHECK_THAT(lfp::to_string(lfp::uint128_t{}), Equals("0"));
    lfp::uint128_t x{2199023255579ull};
    CHECK_THAT(lfp::to_string(x * x - 24), Equals("4835703278577263954625217"));
    CHECK_THAT(lfp::to_string(x * x + 48), Equals("4835703278577263954625289"));
    CHECK_THAT(lfp::to_string(std::numeric_limits<lfp::uint128_t>::max()),
		             Equals("340282366920938463463374607431768211455"));
    lfp::uint128_t ten_19{10'000'000'000'000'000'000ull};
    CHECK_THAT(lfp::to_string(ten_19 * ten_19), 
			     Equals("100000000000000000000000000000000000000"));
}
#endif

