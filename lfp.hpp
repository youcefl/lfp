/*
 * MIT License
 * Copyright (c) 2024 Youcef Lemsafer
 * See LICENSE file for more details.
 * Creation date: december 2024.
 */
#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <iostream>
#include <limits>
#include <numeric>
#include <algorithm>
#include <map>
#include <bit>
#include <ranges>
#include <sstream>
#include <chrono>
#include <thread>
#include <future>



class Bitmap
{
public:
    constexpr Bitmap() = default;
    explicit constexpr Bitmap(uint64_t n0, std::size_t size);
    constexpr void assign(uint64_t n0, std::size_t size);
    constexpr std::size_t size() const;
    constexpr std::size_t indexOf(uint64_t val) const;
    constexpr void reset(std::size_t index);
    constexpr uint64_t popcount() const;
    void check() const;
    constexpr uint8_t at(std::size_t index) const;
    template <typename Func>
    constexpr void foreach_setbit(Func ff) const;

private:
    std::vector<uint64_t> vec_;
    std::size_t size_;
    uint64_t n0_;
    static constexpr std::array<int,30> d_{1,0,5,4,3,2,1,0,3,2,1,0,1,0,3,2,1,0,1,0,3,2,1,0,5,4,3,2,1,0};
    using ElemType = decltype(vec_)::value_type;
    using NumLim = std::numeric_limits<ElemType>;
};

constexpr
Bitmap::Bitmap(uint64_t n0, std::size_t size)
  : vec_((size + NumLim::digits - 1)/NumLim::digits,
            ~decltype(vec_)::value_type{})
  , size_(size)
  , n0_(n0 + d_[n0 % 30])
{
    if(size % NumLim::digits) {
        vec_.back() &= (ElemType{1} << (size % NumLim::digits)) - 1;
    }
}

constexpr
void
Bitmap::assign(uint64_t n0, std::size_t size)
{
    vec_.assign((size + NumLim::digits - 1)/NumLim::digits,
            ~decltype(vec_)::value_type{});
    size_ = size;
    n0_ = n0 + d_[n0 % 30];

    if(size % NumLim::digits) {
        vec_.back() &= (ElemType{1} << (size % NumLim::digits)) - 1;
    }
}

constexpr
std::size_t
Bitmap::size() const
{
    return size_;
}

constexpr
std::size_t
Bitmap::indexOf(uint64_t val) const
{
    auto valmod30idx = (val % 30) * 4 / 15;
    auto n0mod30idx = (n0_ % 30) * 4 /15;
    return (val - n0_) / 30 * 8 + ((valmod30idx >= n0mod30idx) ? valmod30idx - n0mod30idx : 8 + valmod30idx - n0mod30idx);
}

constexpr
void
Bitmap::reset(std::size_t index)
{
    vec_[index / NumLim::digits] &= ~(ElemType{1} << (index % NumLim::digits));
}

constexpr
uint64_t
Bitmap::popcount() const
{
    return std::transform_reduce(std::begin(vec_), std::end(vec_), 0, std::plus{}, [](auto v){ return std::popcount(v);});
}

template <typename Func>
constexpr
void
Bitmap::foreach_setbit(Func ff) const
{
    constexpr int rez[] = {1,7,11,13,17,19,23,29};
    constexpr int d[] = {6,4,2,4,2,4,6,2};
    auto k = 0;
    auto c = n0_;
    for(auto i = std::distance(std::begin(rez), std::find(std::begin(rez), std::end(rez), n0_ % 30));
        k < size();
        c += d[i], i = (i+1)%8, ++k) {
	if(at(k)) {
	    ff(k, c);
	}
    }
}

void
Bitmap::check() const
{
    int rez[] = {1,7,11,13,17,19,23,29};
    int d[] = {6,4,2,4,2,4,6,2};
    auto k = 0;
    auto c = n0_;
    for(auto i = std::distance(std::begin(rez), std::find(std::begin(rez), std::end(rez),n0_%30));
        k < size_ ;
        c += d[i], i = (i+1)%8, ++k) {
        bool is_prime = true;
        for(auto p : {2,3,5,7,11,13,17,19,21,23,29, 31, 37, 41,
                43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
                97, 101, 103, 107, 109, 113, 127, 131, 137,
                139, 149, 151, 157, 163, 167, 173, 179, 181,
                191, 193, 197, 199, 211, 223, 227, 229, 233,
                239, 241, 251}) {
            if((c != p) && ((c % p) == 0)) {
                is_prime = false;
                break;
            }
            if(p*p > c) {
                break;
            }
        }
        if(k != indexOf(c)) {
            std::cout << "bad index: " << k << " for " << c << std::endl;
        }
        if(is_prime != ((vec_[k / NumLim::digits] & (1ull << (k % NumLim::digits))) != 0)) {
            std::cout << "is_prime=" << is_prime << ", " << ((vec_[k / NumLim::digits] & (1ull << (k % NumLim::digits))) != 0) << " | ";
            std::cout << c << " is" << (is_prime ? "" : " not") << " a prime." << std::endl;
        }
    }
}

constexpr
uint8_t Bitmap::at(std::size_t index) const
{
    return (vec_[index / NumLim::digits] & (ElemType{1} << (index % NumLim::digits))) ? 1 : 0;
}

constexpr int8_t adjt[8][14] = {
{0, 4, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 4, 2},
{0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 4, 0, 4, 2},
{0, 4, 4, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 2},
{0, 2, 2, 0, 4, 0, 0, 2, 0, 0, 2, 0, 4, 2},
{0, 2, 4, 0, 2, 0, 0, 2, 0, 0, 4, 0, 2, 2},
{0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 4, 4},
{0, 2, 4, 0, 4, 0, 0, 2, 0, 0, 2, 0, 2, 2},
{0, 2, 4, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 4}};

constexpr uint8_t wheel[8][8] = {
{6,4,2,4,2,4,6,2},
{4,2,4,2,4,6,2,6},
{2,4,2,4,6,2,6,4},
{4,2,4,6,2,6,4,2},
{2,4,6,2,6,4,2,4},
{4,6,2,6,4,2,4,2},
{6,2,6,4,2,4,2,4},
{2,6,4,2,4,2,4,6}
};

constexpr uint8_t whoffs[8][8] = {
{0, 1, 2, 3, 4, 5, 6, 7},
{2, 7, 5, 4, 1, 0, 6, 3},
{0, 2, 6, 4, 7, 5, 1, 3},
{6, 2, 1, 5, 4, 0, 7, 3},
{2, 6, 7, 3, 4, 0, 1, 5},
{0, 6, 2, 4, 1, 3, 7, 5},
{6, 1, 3, 4, 7, 0, 2, 5},
{0, 7, 6, 5, 4, 3, 2, 1}    
};

template <typename T>
constexpr std::array<T,54> u8primes()
{
    return std::array<T,54> {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
        179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
        233, 239, 241, 251};
}

// Returns the smallest integer coprime to 30 and greater than or equal to @param n0
template <typename Ret, typename U>
constexpr
Ret compute_gte_coprime(U n0)
{
    constexpr uint8_t dn0[30] = {1,0,5,4,3,2,1,0,3,2,1,0,1,0,3,2,1,0,1,0,3,2,1,0,5,4,3,2,1,0};
    return Ret{n0} + dn0[n0 % 30];
}

// Returns the largest ineteger coprime to 30 and less than @param n1
template <typename Ret, typename U>
constexpr
Ret compute_lt_coprime(U n1)
{
    constexpr uint8_t dn[30] = {1,2,1,2,3,4,5,6,1,2,3,4,1,2,1,2,3,4,1,2,1,2,3,4,1,2,3,4,5,6};
    return Ret{n1} - dn[n1 % 30];
}

template <typename T, typename SP, typename U, typename Func>
constexpr auto 
inner_sieve(SP const & smallPrimes, U n0, U n1, Func ff, Bitmap & bmp)
{
    if((n0 >= n1) || (n0 > std::numeric_limits<U>::max() - 2)) {
        return ff(std::begin(smallPrimes), std::begin(smallPrimes), nullptr);
    }

    auto it0 = std::lower_bound(std::begin(smallPrimes), std::end(smallPrimes), n0);
    auto it1 = std::lower_bound(std::begin(smallPrimes), std::end(smallPrimes), n1);
    if(it0 != std::end(smallPrimes)) {
	if(it1 != std::end(smallPrimes)) {
	    return ff(it0, it1, nullptr);
	} else {
	    if(n1 <= smallPrimes.back() + 2) {
		return ff(it0, it1, nullptr);
	    }
	    n0 = smallPrimes.back() + 2;
	}
    }
    n0 = compute_gte_coprime<U>(n0);
    auto ne = compute_lt_coprime<std::size_t>(n1);
    if(n0 > ne) {
        return ff(it0, it0, nullptr);
    }
    bmp.assign(n0, (ne - n0)/30 * 8 + ((ne % 30) >= (n0 % 30) ? (ne%30)*4/15 - (n0%30)*4/15 : 8 - (n0%30)*4/15 + (ne%30)*4/15) + 1);

    for(auto p : smallPrimes | std::views::drop(3)) {
	auto p2 = U{p} * p;
	if(p2 > ne) {
	    break;
	}
        constexpr auto c_max = std::numeric_limits<decltype(p2)>::max();
	decltype(p2) dp2;
        auto c = (p2 >= n0) ? p2 : (((dp2 = (n0 - p2 + 2 * p - 1)/(2 * p)*(2 * p)), c_max - p2 < dp2) ? 0 : p2 + dp2);
	if(!c) {
	    continue;
	}
        auto cmod30 = c % 30;
        switch(cmod30) {
            case 3: case 5: case 9: case 15: case 21: case 25: case 27: {
		auto dc = adjt[(p % 30) * 4 / 15][cmod30 >> 1] * p;
		c = (c_max - c < dc) ? 0 : c + dc;
		cmod30 = c % 30;
		}
		break;
            default:;
	    }
	if(!c) {
	    continue;
	}

	auto prevIdx = 0;
	auto count = 0;
	int32_t firstIndex = -1;
	std::array<int,8> deltas{};
	for(auto j = whoffs[(p%30)*4/15][cmod30*4/15]; c <= ne;
	    c = ((c_max - c < wheel[(p%30)*4/15][j]*p) ? ne + 1 : c + wheel[(p%30)*4/15][j]*p), j = (j+1)%8) {
	    auto currIdx = bmp.indexOf(c);
	    if(!count) {
		firstIndex = currIdx;
	    }

	    if(count > 0 && count < 9) {
		    deltas[count - 1] = currIdx - prevIdx;
	    }

	    if(count == 8) {
		break;
	    }
	    ++count;
	    prevIdx = currIdx;
	}

	for(std::size_t i = ((firstIndex >= 0) ? firstIndex : bmp.size()), j = 0; i < bmp.size(); i += deltas[j], j = (j + 1) % 8) {
	    bmp.reset(i);
	    if(!deltas[j]) {
		    break;
	    }
	}
    }
    return ff(it0, std::end(smallPrimes), &bmp);
}

template <typename T, typename It>
constexpr std::vector<T>
collectSieveResults(It first, It last, Bitmap const * bmp)
{
    auto count = std::distance(first, last) + (bmp ? bmp->popcount() : 0);
    if(!count) {
	return std::vector<T>{};
    }
    std::vector<T> res;
    res.reserve(count);
    std::copy(first, last, std::back_inserter(res));
    if(bmp) {
        bmp->foreach_setbit([&res](auto, T p) { res.push_back(p); });
    }

    return res;
}

template <typename T>
constexpr std::vector<T>
sieve16(uint16_t n0, uint16_t n1)
{
    constexpr auto smallPrimes = u8primes<uint16_t>();
    Bitmap bmp;
    return inner_sieve<T>(smallPrimes, n0, n1, collectSieveResults<T, decltype(std::begin(smallPrimes))>, bmp);
}


constexpr auto u16primes = []() {
    std::array<uint16_t, sieve16<uint16_t>(0,65535).size()> u16primes;
    std::ranges::copy(sieve16<uint16_t>(0,65535), std::begin(u16primes));
    return u16primes;
  }();

template <typename T>
constexpr std::vector<T>
sieve32(uint32_t n0, uint32_t n1)
{
    Bitmap bmp;
    return inner_sieve<T>(u16primes, n0, n1, collectSieveResults<T, decltype(std::begin(u16primes))>, bmp);
}

int32_t count_primes(uint32_t n0, uint32_t n1)
{
    Bitmap bmp;
    int32_t count = 0;
    constexpr auto rangeSize = 24*1024*1024;
    constexpr auto maxn = std::numeric_limits<uint32_t>::max();
    for(auto a0 = n0, a1 = std::min(n1, (maxn - rangeSize < n0) ? maxn : n0 + rangeSize);
	a0 < n1;
        a0 = (maxn - rangeSize < a0) ? maxn : a0 + rangeSize, 
	  a1 = std::min(n1, maxn - rangeSize < a0 ? maxn : a0 + rangeSize)) {
        count += inner_sieve<int32_t>(u16primes, a0, a1,
	[](auto it0, auto it1, Bitmap const * zbmp) {
	    return int32_t(std::distance(it0, it1)) + (zbmp ? int32_t(zbmp->popcount()) : 0);
	    }, bmp);
    }
    return count; 
}


int32_t threaded_count_primes(int32_t numThreads, uint32_t n0, uint32_t n1)
{
    if(n0 >= n1) {
	return 0;
    }
    if(numThreads == 0) {
        numThreads = int32_t(std::thread::hardware_concurrency());
    }
    if(!numThreads) {
        std::cerr << "Could not determine the number of concurrent threads supported, defaulting to 1." << std::endl;
	numThreads = 1;
    }
    if(n1 - n0 < numThreads) {
        return count_primes(n0, n1);
    }
    std::vector<std::future<int32_t>> results;
    for(uint32_t k = n0, dk = (n1 - n0) / numThreads, ek = (n1 - n0) % numThreads; k < n1; ek = ek ? ek - 1 : 0) {
	auto kmax = k + dk + (ek ? 1 : 0);
	results.emplace_back(std::async(std::launch::async, count_primes, k, kmax));
	k = kmax;
    }
    return std::accumulate(std::begin(results), std::end(results),
		    int32_t{}, [](auto x, auto & y) { return x + y.get(); });
}

