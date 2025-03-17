/*
 * MIT License
 * Copyright (c) 2025 Youcef Lemsafer
 * See LICENSE file for more details.
 * Creation date: February 2025.
 */
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"

#include "lfp.hpp"

using lfp::count_primes;
using lfp::sieve16;
using lfp::sieve32;
using Catch::Matchers::Equals;


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

template <typename T>
struct EqualsMatcher : Catch::Matchers::MatcherBase<T>
{
    EqualsMatcher(T const & target) : target_(target) {}
    bool match(T const & actual) const override {
	return actual == target_;
    }
    std::string describe() const override {
	return "equals " + Catch::Detail::stringify(target_);
    }
private:
    T target_;
};

template <typename T>
EqualsMatcher<T> equals(T const & target)
{
    return EqualsMatcher<T>(target);
}


TEST_CASE("Sieve of Erathostenes - small primes 1") {
    CHECK_THAT(count_primes(0, 1), equals(0));
    CHECK_THAT(count_primes(0, 2), equals(0));
    CHECK_THAT(count_primes(0, 3), equals(1));
    CHECK_THAT(count_primes(0, 4), equals(2));
    CHECK_THAT(count_primes(0, 5), equals(2));
    CHECK_THAT(count_primes(0, 6), equals(3));
}


TEST_CASE("Sieve of Erathostenes - small primes 2") {
    CHECK_THAT(count_primes(10, 23), equals(4));
    CHECK_THAT(count_primes(0, 101), equals(25));
    CHECK_THAT(count_primes(13, 50), equals(10));
    CHECK_THAT(count_primes(100, 131), equals(6));
}


TEST_CASE("Sieve of Erathostenes - larger primes 1") {
    CHECK_THAT(count_primes(0, 1000), equals(168));
    CHECK_THAT(count_primes(0, 10000), equals(1229));
    CHECK_THAT(count_primes(0, 65536), equals(6542));
    CHECK_THAT(count_primes(65536, 131072), equals(5709));
}


TEST_CASE("Sieve of Erathostenes - larger primes 2") {
    CHECK_THAT(count_primes(131070, 141709), equals(909));
    CHECK_THAT(count_primes(312307, 313409), equals(91));
    CHECK_THAT(count_primes(524287, 524287), equals(0));
    CHECK_THAT(count_primes(524287, 524288), equals(1));
}


TEST_CASE("Sieve of Erathostenes - even larger primes 1") {
    CHECK_THAT(count_primes(3141592, 3142001), equals(22));
    CHECK_THAT(count_primes(8388608, 8389009), equals(25));
    CHECK_THAT(count_primes(4458763, 4459763), equals(70));
    CHECK_THAT(count_primes(24033746, 24043746), equals(620));
}


TEST_CASE("Sieve of Erathostenes - even larger primes 2") {
    CHECK_THAT(count_primes(17310383, 52187113), equals(2014451));
    CHECK_THAT(count_primes(57931727, 83251653), equals(1401439));
    CHECK_THAT(count_primes(100'000'000, 101'000'000), equals(54208));
    CHECK_THAT(count_primes(500'000'000, 501'000'000), equals(49918));
}


TEST_CASE("Sieve of Erathostenes - even larger primes 3") {
    CHECK_THAT(count_primes(2039522013, 2039622967), equals(4656));
    CHECK_THAT(count_primes(3550634754, 3561957981), equals(514735));
    CHECK_THAT(count_primes(3343233271, 3394567417), equals(2339267));
    CHECK_THAT(count_primes(4132050211, 4208912531), equals(3469201));
}


TEST_CASE("Sieve of Erathostenes - limits") {
    CHECK_THAT(count_primes(4294967200, 4294967295), equals(3));
}


TEST_CASE("Sieve of Erathostenes - list primes - 1") {
    CHECK_THAT(sieve16<int32_t>(0, 2), Equals(std::vector<int32_t>{}));
    CHECK_THAT(sieve16<int32_t>(0, 2), Equals(primes_by_division<int32_t>(0, 2)));
    CHECK_THAT(sieve16<int32_t>(0, 3), Equals(std::vector<int32_t>{{2}}));
    CHECK_THAT(sieve16<int32_t>(0, 3), Equals(primes_by_division<int32_t>(0, 3)));
    CHECK_THAT(sieve16<int32_t>(0, 20), Equals(primes_by_division<int32_t>(0, 20)));
    CHECK_THAT(sieve16<int32_t>(0, 20), Equals(std::vector<int32_t>{{2, 3, 5, 7, 11, 13, 17, 19}}));
}


TEST_CASE("Sieve of Erathostenes - list primes - 2") {
    CHECK_THAT(sieve16<int32_t>(0, 100), Equals(primes_by_division<int32_t>(0, 100)));
    CHECK_THAT(sieve16<int32_t>(1, 101), Equals(primes_by_division<int32_t>(1, 101)));
    CHECK_THAT(sieve16<int32_t>(9000, 10000), Equals(primes_by_division<int32_t>(9000, 10000)));
    CHECK_THAT(sieve16<int32_t>(65500, 65535), Equals(primes_by_division<int32_t>(65500, 65535)));
}


TEST_CASE("Sieve of Erathostenes - list primes - 3") {
    CHECK_THAT(sieve32<int32_t>(131000, 131500), Equals(primes_by_division<int32_t>(131000, 131500)));
    CHECK_THAT(sieve32<int32_t>(640191, 703411), Equals(primes_by_division<int32_t>(640191, 703411)));
    CHECK_THAT(sieve32<int32_t>(1'350'209, 1'358'907), Equals(primes_by_division<int32_t>(1'350'209, 1'358'907)));
    CHECK_THAT(sieve32<int32_t>(2'147'483'548, 2'147'483'647), Equals(primes_by_division<int32_t>(2'147'483'548, 2'147'483'647)));
}


TEST_CASE("Sieve of Erathostenes - list primes - 4") {
    CHECK_THAT(sieve32<uint32_t>(4'294'967'200, 4'294'967'295), Equals(primes_by_division<uint32_t>(4'294'967'200, 4'294'967'295)));
}


TEST_CASE("Sieve of Erathostenes - primes iterator") {
    using namespace lfp;
    using namespace lfp::details;
    {
        Bitmap bmp;
        inner_sieve<uint32_t>(u8primes<uint8_t>(), 300u, 400u, [](auto, auto, auto) {}, bmp);
        PrimesIterator<uint32_t> it{&bmp}, ite{&bmp, true};
        CHECK_THAT(std::vector<uint32_t>(it, ite), Equals(primes_by_division<uint32_t>(300, 400)));
    }
    {
	Bitmap bmp;
	inner_sieve<int32_t>(u16primes, 10000u, 12000u, [](auto, auto, auto) {}, bmp);
        PrimesIterator<int32_t> it{&bmp}, ite{&bmp, true};
        CHECK_THAT(std::vector<int32_t>(it, ite), Equals(primes_by_division<int32_t>(10000, 12000)));
    }
}

TEST_CASE("Sieve of Ertathostenes - above 2^32 - 1") {
    CHECK_THAT(lfp::sieve_to_vector<int64_t>(300ull, 400ull), Equals(primes_by_division<int64_t>(300ull, 400ull)));
    CHECK_THAT(lfp::sieve_to_vector<int64_t>(1ull << 37, (1ull << 37) + 100),
		Equals(std::vector<int64_t>{137438953481, 137438953501, 137438953513, 137438953541, 137438953567}));
    CHECK_THAT(lfp::sieve_to_vector<int64_t>(1ull << 43, (1ull << 43) + 200),
		Equals(primes_by_division<int64_t>(1ull << 43, (1ull << 43) + 200)));
    CHECK_THAT(lfp::sieve_to_vector<int64_t>(1ull << 44, (1ull << 44) + 400),
	    Equals(std::vector<int64_t>{17592186044423, 17592186044437, 17592186044443, 17592186044471, 17592186044591,
		    17592186044611, 17592186044651, 17592186044659, 17592186044747, 17592186044767, 17592186044813}));
    CHECK_THAT(lfp::sieve_to_vector<int64_t>(1ull << 50, (1ull << 50) + 1000),
		    Equals(primes_by_division<int64_t>(1ull << 50, (1ull << 50) + 1000)));
}

TEST_CASE("Sieve of Ertathostenes - above 2^32 - 2") {
    CHECK_THAT(lfp::sieve<int64_t>((1ull << 33) - (1ull << 30), 1ull << 33).count(), equals(47076888));
}


