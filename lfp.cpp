/*
 * MIT License
 * Copyright (c) 2024 Youcef Lemsafer
 * See LICENSE file for more details.
 * Creation date: december 2024.
 */


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


void gentable()
{
    int rz[] = {1,7,11,13,17,19,23,29};
    int ps[] = {31,7,11,13,17,19,23,29};
    uint8_t wheel[8][30]={};
    for(auto p : ps) {
        std::cout << "{";
        auto j = 0;
        for(auto n0 = p*p, n = n0; n - n0 < 30*p; n += 2*p) {
            if(std::find(std::begin(rz), std::end(rz), std::gcd(n,30)) == std::end(rz)) {
                continue;
            }
            //std::cout << " /*" << n % 30 << "*/ ";
            wheel[(p%30)*4/15][n%30] = j++;
            auto k = 2;
            for(; std::find(std::begin(rz), std::end(rz), std::gcd(n + k*p,30)) == std::end(rz); k += 2) {
            }
            std::cout << k << ",";
        }
        std::cout << "}" << std::endl;
    }
    std::cout << std::endl;
    for(auto i = 0; i < 8; ++i) {
        std::cout << "{";
        for(auto j : {1,7,11,13,17,19,23,29}) {
            std::cout << int(wheel[i][j]) << ((j==29)?"":", ");
        }
        std::cout << ((i==7)?"}":"},") << std::endl;
    }
}

void work()
{
    /*for(int i = 0; i < 30; ++i) {
        if(std::gcd(i, 30) == 1) {
            std::cout << ", " << i;
        }
    }
    std::cout << std::endl;*/
    std::array<int, 8> rz{1, 7, 11, 13, 17, 19, 23, 29};
    std::map<int,int> tabl, atabl;
    int z = 0;
    for(int i = 30; i < 30000; i += 30) {
        for(auto r : rz) {
            atabl[z] = i + r;
            tabl[i + r] = z++;
        }
    }
/*    for (auto kv : tabl) {
        std::cout << kv.first << " <-> " << kv.second << std::endl;
    }*/
    for(auto p : {7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
                47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113}) {
        std::cout << p << ": " << (p % 30) << " ";
        auto c = p * p;
        std::vector<int> delta, dk;
        for(auto i = 0, j = 0, k = 0; i <= 90; i += 2) {
            if(std::find(std::begin(rz), std::end(rz), (c + i * p) % 30) != std::end(rz)) {
                //std::cout << " " << "[" << i << ", " << i - j << "]" << c + i * p;
                std::cout << " [" << tabl[c + i * p] << "]:" << c + i*p
                        << "(" << ((c + i * p) % 30) << ")"; 
                if(i) {
                    delta.push_back(tabl[c + i * p]-j);
                    dk.push_back(i - k);
                }
                
                j = tabl[c + i * p];
                k = i;
            }
        }
        std::cout << std::endl << p << ":" << (p < 10 ? " ": "") << "  ";
        for(auto d : delta) {
            std::cout << " " << d;
        }
        std::cout << std::endl << p << ":" << (p < 10 ? " ": "") << "  ";
        for(auto d : dk) {
            std::cout << " " << d;
        }
        std::cout << std::endl << p << ":" << (p < 10 ? " ": "") << "* ";
        auto s = p * p;
        //std::cout << " [" << (s == atabl[(s - 30) * 8 /30]) << "]";
        for(auto d : dk) {
            //std::cout << " [" << (atabl[(s + d*p - 30) * 8 /30] == s + d*p) << "]";
            static const std::array<int,30> dst{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7};
            std::cout << " " << (s % 30 + d * p)*4/15 - 4*(s % 30)/15 << ":" << (s%30+d*p)*4/15 - dst[s % 30];
            s += d * p;
        }
        std::cout << "\n" << std::endl;
        delta.clear();
        //std::cout << std::endl;
    }
    std::cout << "\n" << std::endl;
}


class Bitmap
{
public:
    explicit constexpr Bitmap(uint64_t n0, std::size_t size);
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

/*
1  {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29}
7  {19, 3, 17, 1, 15, 29, 13, 27, 11, 25, 9, 23, 7, 21, 5}
11 {1, 23, 15, 7, 29, 21, 13, 5, 27, 19, 11, 3, 25, 17, 9}
13 {19, 15, 11, 7, 3, 29, 25, 21, 17, 13, 9, 5, 1, 27, 23}
17 {19, 23, 27, 1, 5, 9, 13, 17, 21, 25, 29, 3, 7, 11, 15}
19 {1, 9, 17, 25, 3, 11, 19, 27, 5, 13, 21, 29, 7, 15, 23}
23 {19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 1, 17, 3}
29 {1, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3}
*/

uint8_t jt[8][30] =
{
    // 1   3   5   7   9  11  13  15  17  19  21  23  25  27  29
    {0,6,0,4,0,2,0,4,0,2,0,2,0,4,0,2,0,2,0,4,0,2,0,6,0,4,0,2,0,2},
    {0,4,0,2,0,2,0,6,0,2,0,6,0,4,0,2,0,2,0,4,0,4,0,2,0,4,0,2,0,2},
    {0,2,0,4,0,4,0,2,0,2,0,6,0,6,0,2,0,4,0,2,0,2,0,4,0,2,0,2,0,4},
    {0,4,0,2,0,2,0,4,0,4,0,2,0,6,0,2,0,2,0,4,0,2,0,2,0,4,0,2,0,6},
    {0,6,0,2,0,4,0,2,0,2,0,4,0,2,0,2,0,6,0,2,0,4,0,4,0,2,0,2,0,4},
    {0,4,0,2,0,2,0,4,0,2,0,2,0,4,0,2,0,6,0,6,0,2,0,2,0,4,0,4,0,2},
    {0,2,0,2,0,4,0,2,0,4,0,4,0,2,0,2,0,4,0,6,0,2,0,6,0,2,0,2,0,4},
    {0,2,0,2,0,4,0,6,0,2,0,4,0,2,0,2,0,4,0,2,0,2,0,4,0,2,0,4,0,6}
};

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

void shortjt() {
    for(int i = 0; i < 8; ++i) {
        std::cout << "{";
        for(auto j : {0,3,5,6,9,10,12,15,16,18,21,22,25,27}) {
            std::cout << int(jt[i][j]) << ((j==27)?"":", ");
        }
        std::cout << "}," << std::endl;
    }
}

void tbitmap()
{
    Bitmap bmp{6,32768};
    auto numFailures = 0;
    auto k = 0;
    for(auto i = 0; i < 6000; i+=30) {
        for(auto j : {1,7,11,13,17,19,23,29}) {
            if(i+j < 6) {
                continue;
            }
            auto idx = bmp.indexOf(i + j);
            if(idx != k) {
                std::cerr << "Failure for " << i+j 
                    << ", expected index: " << k << ", actual: " << idx << std::endl;
                ++numFailures;
            }
            ++k;
        }
    }
    std::cout << numFailures << " failure(s)" << std::endl;
}

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


template <typename T, typename SP, typename U, typename Func>
constexpr auto 
inner_sieve(SP const & smallPrimes, U n0, U n1, Func ff)
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
    constexpr uint8_t dn0[30] = {1,0,5,4,3,2,1,0,3,2,1,0,1,0,3,2,1,0,1,0,3,2,1,0,5,4,3,2,1,0};
    n0 = n0 + dn0[n0 % 30];
    constexpr uint8_t dn[30] = {1,2,1,2,3,4,5,6,1,2,3,4,1,2,1,2,3,4,1,2,1,2,3,4,1,2,3,4,5,6};
    std::size_t ne = std::size_t(n1) - dn[n1 % 30];
    if(n0 > ne) {
        return ff(it0, it0, nullptr);
    }
    Bitmap bmp{n0, (ne - n0)/30 * 8 + ((ne % 30) >= (n0 % 30) ? (ne%30)*4/15 - (n0%30)*4/15 : 8 - (n0%30)*4/15 + (ne%30)*4/15) + 1};

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
	for(auto j = whoffs[(p%30)*4/15][cmod30*4/15]; c <= ne;
	    c = ((c_max - c < wheel[(p%30)*4/15][j]*p) ? ne + 1 : c + wheel[(p%30)*4/15][j]*p), j = (j+1)%8) {
	    bmp.reset(bmp.indexOf(c));
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
    return inner_sieve<T>(smallPrimes, n0, n1, collectSieveResults<T, decltype(std::begin(smallPrimes))>);
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
    return inner_sieve<T>(u16primes, n0, n1, collectSieveResults<T, decltype(std::begin(u16primes))>);
}

int32_t count_primes(uint32_t n0, uint32_t n1)
{
    return inner_sieve<int32_t>(u16primes, n0, n1,
	[](auto it0, auto it1, Bitmap const * bmp) {
	    return int32_t(std::distance(it0, it1)) + (bmp ? int32_t(bmp->popcount()) : 0);
        });
}

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

void displayUsage()
{
    std::cerr << "Usage:\n\tlfp [-t num_threads] n0 n1" << std::endl;
}

int main(int argc, char** argv)
{
    if((argc != 3) && (argc != 5)) {
	displayUsage();
	return 1;
    }
    int n0idx = 1, n1idx = 2;
    int32_t numThreads{};
    if(argc == 5) {
	if(argv[1] != std::string{"-t"}) {
	    displayUsage();
	    return 1;
	}
	std::istringstream istr{argv[2]};
	istr >> numThreads;
	n0idx += 2;
	n1idx += 2;
    }
    uint32_t n0, n1;
    std::istringstream istr0{argv[n0idx]};
    istr0 >> n0;
    std::istringstream istr1{argv[n1idx]};
    istr1 >> n1;

    auto const startt = std::chrono::steady_clock::now();
    auto numPrimes = threaded_count_primes(numThreads, n0, n1);
    auto const endt = std::chrono::steady_clock::now();
    
    std::cout << "The number of prime numbers in range [" << n0 << ", " << n1 << "[ is "
	   << numPrimes << "." << std::endl;
    std::cout << "Took " << std::chrono::duration<double>(endt - startt) << std::endl;
    return 0;
}

