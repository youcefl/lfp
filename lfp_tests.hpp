/*
 * MIT License
 * Copyright (c) 2025 Youcef Lemsafer
 * See LICENSE file for more details.
 * Creation date: May 2025.
 */
#pragma once

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"

template <typename T>
inline std::vector<T>
primes_by_division(uint64_t a, uint64_t b)
{
    std::vector<T> results;
    if(a <= 2) {
	if(b > 2) {
            results.push_back(2);
	}
	a = 3;
    }
    constexpr auto cmax = std::numeric_limits<decltype(b)>::max();
    for(auto c = a | 1; c < b; c = (c > cmax - 2) ? cmax : c + 2) {
	if(!(c % 3) && (c != 3)) {
	    continue;
	}
	bool isPrime = true;
	int w[] = {2,4}, i = 0;
        for(uint64_t p = 5; (p * p <= c) && (p <= std::numeric_limits<uint32_t>::max()); p += w[i], i^=1) {
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
inline EqualsMatcher<T> equals(T const & target)
{
    return EqualsMatcher<T>(target);
}


