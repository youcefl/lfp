[![CI](https://github.com/youcefl/lfp/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/youcefl/lfp/actions/workflows/c-cpp.yml)

# LFP

## A constexpr implementation of the sieve of Eratosthenes

LFP is a constexpr implementation of the sieve of Eratosthenes, it consists of a single header file, namely `lfp.hpp`.
It was born out of my willingness to learn and also give back to the community. After several attempts over the years that did not yield satisfying results, LFP, which started in late 2024, finally achieved my goals thanks to recent C++ language features.
LFP distinguishes itself by maintaining the following key properties:
- standard C++, any standard complying compiler should be able to compile the code.
- constexpr, the user should be able to sieve any "reasonable" range for primes at compile time.
- header only, the sieve should be usable by including a single header file.
- ability to use user defined arithmetic types provided that they satisfy a given set of conditions.
- ability to sieve above 2^64.

To maximize adoption I limited the implementation to C++20 features (developement was done using g++ 12.2 with `-std=c++20`).
To ensure this remained a fully original work, I deliberately avoided looking into existing implementations. The only external assistance came from discussions with DeepSeek and ChatGPT.


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
auto sieveRes = lfp::sieve<int32_t>(0, 10000000, lfp::threads{8});
std::vector<int32_t> primes;
primes.reserve(sieveRes.count());
for(auto p : sieveRes) {
    primes.push_back(p);
}

// shorter version:
auto sieveRes = lfp::sieve<int32_t>(0, 10000000, lfp::threads{8});
auto rng = sieveRes.range();
std::vector<int32_t> primes{rng.begin(), rng.end()};

// Sieve range [10^8 and 10^8+10^7) using the default number of concurrent threads, then
// iterate over the resulting primes
for(auto p : lfp::sieve<uint32_t>(100000000, 110000000, lfp::threads{})) {
    ....
}

```

## Public API

The public API is as follows:

```c++

namespace lfp {

// Forward declarations
struct threads;
template <typename T, typename U> class sieve_results;
template <typename T> struct to_unsigned;
template <typename T> using to_unsigned_t = typename to_unsigned<T>::type;

/// Returns the result of sieving the range [n0, n1).
/// ResultInt is the type of the resuling prime numbers
template <typename ResultInt, typename Int>
constexpr sieve_results<ResultInt, to_unsigned_t<Int>> sieve(Int n0, Int n1);

/// Returns the result of sieving the range [n0, n1).
/// The sieving is performed using at most threads.count() concurrent threads.
template <typename ResultInt, typename Int>
sieve_results<ResultInt, to_unsigned_t<Int>> sieve(Int n0, Int n1, threads const & threads);

/// Returns a vector containing the prime numbers in range [n0, n1)
template <typename ResultInt, typename Int>
constexpr std::vector<ResultInt> sieve_to_vector(Int n0, Int n1);

/// Returns the number of prime numbers in range [n0, n1)
template <typename Int>
constexpr std::size_t count_primes(Int n0, Int n1);

/// Returns the number of prime numbers in range [n0, n1)
/// the sieving is performed using at most threads.count() ooncurrent threads.
template <typename Int>
std::size_t count_primes(Int n0, Int n1, threads const & threads);

/// @struct Holding concurrency information
struct threads
{
    /// Constructs an instance x such that x.count() == std::thread::hardware_concurrency().
    /// If std::thread::hardware_concurrency() == 0, then x.count() is equal to 1.
    threads();
    /// Constructs an instance x such that x.count() == c
    explicit threads(unsigned int c);
    /// Returns the maximum number of concurrent threads to use during sieving.
    unsigned int count() const;
private:
    unsigned int count_;
    static unsigned int default_count();
};

/// Class holding sieve results
template<typename T, typename U>
class sieve_results
{
    // ... Private part omitted ...
public:
    /// Returns a range suitable for iterating over the prime numbers resulting from the sieve
    constexpr auto range();
    /// Implicit cast to a range
    constexpr operator range_type ();
    /// Returns the number of prime numbers found by the sieve.
    constexpr std::size_t count();

    friend constexpr auto begin(sieve_results & rng);
    friend constexpr auto end(sieve_results & rng);
};

} // namespace lfp

```


## Performances

This sieve leverages several optimization techniques for performance and scalability:
 - numbers that are not coprime to 30 i.e. divisible by 2, 3 or 5 are not considered, this is a well known way to speed-up a sieve called wheel factorization.
 - a range $R = [n_0, n_1[$ to be sieved is represented by a sequence of bits, called a bitmap, to each bit corresponds one and only one of the integers coprime to 30 in $R$, thus we only consume about $\frac{4}{15}\cdot(n_1 - n_0)$ bits of memory to represent R.
 - when crossing out the multiples of a prime $p$, we start at $p^{2}$ and, since we are using a modulo 30 wheel, we only consider the multiples of $p$ of the form $p^{2} + 2kp$ ($k$ being an integer >= 0) that are coprime to 30.
 - for each prime $p$ below a certain threshold (currently 104), we precompute a bitmask of length $8p$ at compile time. These bitmasks are applied in batches—e.g., in one pass, we cross out multiples of {7, 11, ..., 31}. The threshold 104 was chosen to group primes into three sets of eight (optimized for batch processing) and empirically showed better performance than higher values (though future tuning may optimize this further).
 - the sieve employs bucket sieving for primes above 204800—a technique that groups large primes into cache-friendly batches. This threshold was chosen to balance overhead and gains but may be optimized further in future releases.
 - the sieve is segmented i.e. if the size of the range R to sieve exceeds a certain threshold S, R is split into segments of size at most S.
 - the sieve is multithreaded, we allocate N threads and each thread deals with part of the range to sieve.
 - the memory allocated for a bitmap is 64-byte aligned.

The following table gives an idea of the performances to expect from the sieve (all durations are in seconds):

| Range \ Threads | 1 | 4 | 8 | 16 | 32 | 48 | 64 | Number of primes |
|-----------------|---|---|---|----|----|----|----|------------------|
| $\left[0, 10^{9}\right[$ | 0.228 | 0.079 | 0.050 | 0.035 | 0.034 | 0.036 | 0.037 | **50847534** |
| $\left[0, 2^{32}-1\right[$ | 1.077 | 0.362 | 0.224 | 0.146 | 0.104 | 0.092 | 0.088 | **203280221** |
| $\left[10^{12}, 10^{12}+10^{10}\right[$ | 4.421 | 1.610 | 1.058 | 0.741 | 0.530 | 0.453 | 0.421 | **361840208** |
| $\left[10^{15}, 10^{15}+10^{10}\right[$ | 16.800 | 6.148 | 4.052 | 2.878 | 2.308 | 1.983 | 1.888 | **289531946** |
| $\left[10^{18}, 10^{18}+10^{10}\right[$ | 69.846 | 25.593 | 16.562 | 11.224 | 7.947 | 6.384 | 6.199 | **241272176** |
| $\left[2^{64}-10^{10}, 2^{64}-1\right[$ | 200.867 | 71.364 | 45.911 | 30.781 | 21.050 | 16.361 | 16.497 | **225402976** |

These timings were measured on an AMD EPYC 9R14, the compilation flags used are "-std=c++20 -O3 -march=native -mtune=native -DNDEBUG" (OS: Debian 12, compiler: g++ version 12.2).


