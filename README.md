[![CI](https://github.com/youcefl/lfp/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/youcefl/lfp/actions/workflows/c-cpp.yml)

# LFP

## A constexpr implementation of the sieve of Erathostenes

LFP is a constexpr implementation of the sieve of Erathostenes, it consists of a single header file, namely lfp.hpp.
Note that you will need a C++20 capable compiler to be able to use LFP. 

Here are some examples of use:

```c++
#include "lfp.hpp"

// Construct an array containing the prime numbers below 256 as 8-bit unsigned integers
constexpr auto primesBelow256 = []() {
    auto vecPrimesBelow256 = lfp::sieve_to_vector<uint8_t>(0, 256);
    constexpr auto size = lfp::sieve_to_vector<uint8_t>(0, 256).size();
    std::array<uint8_t, size> arr;
    std::ranges::copy(vecPrimesBelow256, std::begin(arr));
    return arr;
  }();

// Some checks on the resulting array...
static_assert(primesBelow256.size() == 54);
static_assert(primesBelow256[53] == 251);
static_assert(primesBelow256[25] == 101);

// Check the number of primes between 0 and 10^5
static_assert(lfp::count_primes(0, 100'000) == 9592);


// Sieve range [0, 10^7) using up to 8 concurrent threads and put the resulting primes in a vector
auto sieveRes = sieve<int32_t>(0, 10000000, lfp::Threads{8});
std::vector<int32_t> primes;
primes.reserve(sieveRes.count());
for(auto p : sieveRes) {
    primes.push_back(p);
}

// shorter version:
auto sieveRes = sieve<int32_t>(0, 10000000, lfp::Threads{8});
auto rng = sieveRes.range();
std::vector<int32_t> primes{rng.begin(), rng.end()};

// Sieve range [10^8 and 10^8+10^7) using the default number of concurrent threads, then
// iterate over the resulting primes
for(auto p : sieve<uint32_t>(100000000, 110000000, lfp::Threads{})) {
}

```

## Public API

The public API is as follows:

```c++

namespace lfp {

// Forward declarations
struct Threads;
template <typename T> class SieveResults;

/// Returns the result of sieving the range [n0, n1).
/// T is the type of the resuling prime numbers
template <typename T, typename U>
constexpr SieveResults<T> sieve(U n0, U n1);

/// Returns the result of sieving the range [n0, n1).
/// The sieving is performed using at most threads.count() concurrent threads.
template <typename T, typename U>
SieveResults<T> sieve(U n0, U n1, Threads const & threads);

/// Returns a vector containing the prime numbers in range [n0, n1)
template <typename T, typename U>
constexpr std::vector<T> sieve_to_vector(U n0, U n1);

/// Returns the number of prime numbers in range [n0, n1)
template <typename U>
std::size_t count_primes(U n0, U n1);

/// Returns the number of prime numbers in range [n0, n1)
/// the sieving is performed using at most threads.count() ooncurrent threads.
template <typename U>
std::size_t count_primes(U n0, U n1, Threads const & threads);

/// @struct Holding concurrency information
struct Threads
{
    /// Constructs an instance x such that x.count() == std::thread::hardware_concurrency().
    /// If std::thread::hardware_concurrency() == 0, then x.count() is equal to 1.
    Threads();
    /// Constructs an instance x such that x.count() == c
    explicit Threads(unsigned int c);
    /// Returns the maximum number of concurrent threads to use during sieving.
    unsigned int count() const;
// ...Private part omitted...
};

/// Class holding sieve results
template<typename T>
class SieveResults
{
public:
    using range_type = ....;
    constexpr SieveResults(std::vector<T>&& prefix, std::vector<details::Bitmap>&& bitmaps);
    /// Returns a range suitable for iterating over the prime numbers resulting from the sieve
    constexpr auto range();
    constexpr operator range_type ();
    /// Returns the number of prime numbers found by the sieve.
    constexpr std::size_t count();

    friend constexpr auto begin(SieveResults & rng);
    friend constexpr auto end(SieveResults & rng);
// ... Private part omitted...
};

} // namespace lfp

```



## Performances

This sieve leverages several optimization techniques for performance and scalability:
 - numbers that are not coprime to 30 i.e. divisible by 2, 3 or 5 are not considered, this is a well known way to speed-up a sieve called wheel factorization.
 - a range $R = [n_0, n_1[$ to be sieved is represented by a sequence of bits, called a bitmap, to each bit corresponds one and only one of the integers coprime to 30 in $R$, thus we only consume about $\frac{4}{15}\cdot(n_1 - n_0)$ bits of memory to represent R.
 - when crossing out the multiples of a prime $p$, we start at $p^{2}$ and, since we are using a modulo 30 wheel, we only consider the multiples of $p$ of the form $p^{2} + 2kp$ ($k$ being an integer >= 0) that are coprime to 30.
 - for each prime $p$ below a certain threshold (currently 104), we precompute a bitmask of length $8p$ at compile time. These bitmasks are applied in batches—e.g., in one pass, we cross out multiples of {7, 11, ..., 31}. The threshold 104 was chosen to group primes into three sets of eight (optimized for batch processing) and empirically showed better performance than higher values (though future tuning may optimize this further).
 - the sieve is segmented i.e. if the size of the range R to sieve exceeds a certain threshold S, R is split into segments of size at most S.
 - the sieve is multithreaded, we allocate N threads and each thread deals with part of the range to sieve.
 - the memory allocated for a bitmap is 64-byte aligned.

The following table gives an idea of the performances to expect from the sieve (all durations are in seconds):

| Range \ Threads | 1 | 4 | 8 | 16 | 32 | 48 | 64 | Number of primes |
|-----------------|---|---|---|----|----|----|----|------------------|
| $\left[0, 10^{9}\right[$ | 0.211 | 0.057 | 0.030 | 0.018 | 0.015 | 0.016 | 0.015 | **50847534** |
| $\left[0, 2^{32}-1\right[$ | 0.979 | 0.263 | 0.133 | 0.070 | 0.043 | 0.036 | 0.037 | **203280221** |
| $\left[10^{12}, 10^{12}+10^{10}\right[$ | 4.643 | 1.175 | 0.592 | 0.301 | 0.158 | 0.120 | 0.104 | **361840208** |
| $\left[10^{15}, 10^{15}+10^{10}\right[$ | 26.475 | 6.620 | 3.324 | 1.670 | 0.863 | 0.603 | 0.521 | **289531946** |
| $\left[10^{18}, 10^{18}+10^{10}\right[$ | 392.079 | 98.792 | 49.344 | 24.668 | 12.874 | 8.944 | 7.640 | **241272176** |
| $\left[2^{64}-10^{10}, 2^{64}-1\right[$ | 1478.033 | 374.467 | 186.686 | 93.948 | 49.188 | 34.535 | 29.364 | **225402976** |


These timings were measured on an AMD EPYC 9R14, the compilation flags used are "-std=c++20 -O3 -march=native -mtune=native" (OS: Debian 12, compiler: g++ version 12.2).


