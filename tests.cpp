/*
 * MIT License
 * Copyright (c) 2025 Youcef Lemsafer
 * See LICENSE file for more details.
 * Creation date: February 2025.
 */
#include "catch2/catch_test_macros.hpp"
#include "lfp.hpp"

using lfp::count_primes;
using lfp::sieve16;
using lfp::sieve32;

TEST_CASE("Sieve of Erathostenes - small primes 1") {
    REQUIRE(count_primes(0, 1) == 0);
    REQUIRE(count_primes(0, 2) == 0);
    REQUIRE(count_primes(0, 3) == 1);
    REQUIRE(count_primes(0, 4) == 2);
    REQUIRE(count_primes(0, 5) == 2);
    REQUIRE(count_primes(0, 6) == 3);
}


TEST_CASE("Sieve of Erathostenes - small primes 2") {
    REQUIRE(count_primes(10, 23) == 4);
    REQUIRE(count_primes(0, 101) == 25);
    REQUIRE(count_primes(13, 50) == 10);
    REQUIRE(count_primes(100, 131) == 6);
}


TEST_CASE("Sieve of Erathostenes - larger primes 1") {
    REQUIRE(count_primes(0, 1000) == 168);
    REQUIRE(count_primes(0, 10000) == 1229);
    REQUIRE(count_primes(0, 65536) == 6542);
    REQUIRE(count_primes(65536, 131072) == 5709);
}


TEST_CASE("Sieve of Erathostenes - larger primes 2") {
    REQUIRE(count_primes(131070, 141709) == 909);
    REQUIRE(count_primes(312307, 313409) == 91);
    REQUIRE(count_primes(524287, 524287) == 0);
    REQUIRE(count_primes(524287, 524288) == 1);
}


TEST_CASE("Sieve of Erathostenes - even larger primes 1") {
    REQUIRE(count_primes(3141592, 3142001) == 22);
    REQUIRE(count_primes(8388608, 8389009) == 25);
    REQUIRE(count_primes(4458763, 4459763) == 70);
    REQUIRE(count_primes(24033746, 24043746) == 620);
}


TEST_CASE("Sieve of Erathostenes - even larger primes 2") {
    REQUIRE(count_primes(17310383, 52187113) == 2014451);
    REQUIRE(count_primes(57931727, 83251653) == 1401439);
    REQUIRE(count_primes(100'000'000, 101'000'000) == 54208);
    REQUIRE(count_primes(500'000'000, 501'000'000) == 49918);
}


TEST_CASE("Sieve of Erathostenes - even larger primes 3") {
    REQUIRE(count_primes(2039522013, 2039622967) == 4656);
    REQUIRE(count_primes(3550634754, 3561957981) == 514735);
    REQUIRE(count_primes(3343233271, 3394567417) == 2339267);
    REQUIRE(count_primes(4132050211, 4208912531) == 3469201);
}


TEST_CASE("Sieve of Erathostenes - limits") {
    REQUIRE(count_primes(4294967200, 4294967295));
}


template <typename T>
std::vector<T>
primes_by_division(uint64_t a, uint64_t b)
{
    std::vector<T> results;
    if(a <= 2) {
	if(b > 2) {
            results.push_back(2);
	}
	a = 3;
    }
    for(auto c = a | 1; c < b; c += 2) {
	if(!(c % 3) && (c != 3)) {
	    continue;
	}
	bool isPrime = true;
	int w[] = {2,4}, i = 0;
        for(uint64_t p = 5; p * p <= c; p += w[i], i^=1) {
	    if(!(c % p)) {
                isPrime = false;
		break;
	    }
	}
	if(isPrime) {
	    results.push_back(c);
	}
    }
    return results;
}


TEST_CASE("Sieve of Erathostenes - list primes - 1") {
    REQUIRE(sieve16<int32_t>(0, 2) == std::vector<int32_t>{});
    REQUIRE(sieve16<int32_t>(0, 2) == primes_by_division<int32_t>(0, 2));
    REQUIRE(sieve16<int32_t>(0, 3) == std::vector<int32_t>{{2}});
    REQUIRE(sieve16<int32_t>(0, 3) == primes_by_division<int32_t>(0, 3));
    REQUIRE(sieve16<int32_t>(0, 20) == primes_by_division<int32_t>(0, 20));
    REQUIRE(sieve16<int32_t>(0, 20) == std::vector<int32_t>{{2, 3, 5, 7, 11, 13, 17, 19}});
}


TEST_CASE("Sieve of Erathostenes - list primes - 2") {
    REQUIRE(sieve16<int32_t>(0, 100) == primes_by_division<int32_t>(0, 100));
    REQUIRE(sieve16<int32_t>(1, 101) == primes_by_division<int32_t>(1, 101));
    REQUIRE(sieve16<int32_t>(9000, 10000) == primes_by_division<int32_t>(9000, 10000));
    REQUIRE(sieve16<int32_t>(65500, 65535) == primes_by_division<int32_t>(65500, 65535));
}


TEST_CASE("Sieve of Erathostenes - list primes - 3") {
    REQUIRE(sieve32<int32_t>(131000, 131500) == primes_by_division<int32_t>(131000, 131500));
    REQUIRE(sieve32<int32_t>(640191, 703411) == primes_by_division<int32_t>(640191, 703411));
    REQUIRE(sieve32<int32_t>(1'350'209, 1'358'907) == primes_by_division<int32_t>(1'350'209, 1'358'907));
    REQUIRE(sieve32<int32_t>(2'147'483'548, 2'147'483'647) == primes_by_division<int32_t>(2'147'483'548, 2'147'483'647));
}


TEST_CASE("Sieve of Erathostenes - list primes - 4") {
    REQUIRE(sieve32<uint32_t>(4'294'967'200, 4'294'967'295) == primes_by_division<uint32_t>(4'294'967'200, 4'294'967'295));
}


TEST_CASE("Primes iterator") {
    using namespace lfp;
    using namespace lfp::details;
    {
        Bitmap bmp;
        inner_sieve<uint32_t>(u8primes<uint8_t>(), 300u, 400u, [](auto, auto, auto) {}, bmp);
        PrimesIterator<uint32_t> it{&bmp}, ite{&bmp, true};
        REQUIRE(std::vector<uint32_t>{it, ite} == primes_by_division<uint32_t>(300, 400));
    }
    {
	Bitmap bmp;
	inner_sieve<int32_t>(u16primes, 10000u, 12000u, [](auto, auto, auto) {}, bmp);
        PrimesIterator<int32_t> it{&bmp}, ite{&bmp, true};
        REQUIRE(std::vector<int32_t>{it, ite} == primes_by_division<int32_t>(10000, 12000));	
    }
}

TEST_CASE("Sieve of Ertathostenes - above 2^32 - 1") {
    REQUIRE(lfp::sieve_to_vector<int64_t>(300ull, 400ull) == primes_by_division<int64_t>(300ull, 400ull));
    REQUIRE(lfp::sieve_to_vector<int64_t>(1ull << 37, (1ull << 37) + 100) ==
		    std::vector<int64_t>{137438953481, 137438953501, 137438953513, 137438953541, 137438953567});
    REQUIRE(lfp::sieve_to_vector<int64_t>(1ull << 43, (1ull << 43) + 200) ==
		    primes_by_division<int64_t>(1ull << 43, (1ull << 43) + 200));
    REQUIRE(lfp::sieve_to_vector<int64_t>(1ull << 44, (1ull << 44) + 400) ==
	    std::vector<int64_t>{17592186044423, 17592186044437, 17592186044443, 17592186044471, 17592186044591,
		    17592186044611, 17592186044651, 17592186044659, 17592186044747, 17592186044767, 17592186044813});
    REQUIRE(lfp::sieve_to_vector<int64_t>(1ull << 50, (1ull << 50) + 1000) ==
		    primes_by_division<int64_t>(1ull << 50, (1ull << 50) + 1000)); 
}



