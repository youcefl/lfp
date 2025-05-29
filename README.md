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

## Testing and benchmarking

To benchmark the library and/or run the existing tests download the repository and run make.
To compile the command line tool:
```
$ cd lfp
$ make lfp
```
To run the existing tests:
```
$ make checks
```
The lfp executable can be used to sieve a range [n_0, n_1[ for primes, use option -t/--threads to specify a number of threads and -p/--primes to output the primes found e.g.
```
# Sieve range [0, 10^10[ using up to 8 concurrent threads and output the number of primes found in the range
$ lfp -t 8 0 10000000000

# Output all primes in the range [100000, 101000[
$ lfp -p 100000 101000
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
## Validation

LFP’s results were rigorously verified through:
- Dynamic unit tests: Runtime checks across edge cases and arbitrary ranges.
- Static assertions: Compile-time validation of constexpr outputs.
- Cross-referencing:
    - For ranges below 2<sup>64</sup>: automated comparison against primesieve and WolframAlpha.
    - For ranges above 2<sup>64</sup>: manual verification using WolframAlpha as an authoritative source.

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
|-----------------|---|---|---|---|---|---|---|-------------------|
| $\left[0, 10^{9}\right[$ | .280 | .096 | .062 | .042 | .035 | .037 | .037 | **50847534** |
| $\left[0, 2^{32}-1\right[$ | 1.301 | .444 | .276 | .180 | .125 | .110 | .103 | **203280221** |
| $\left[10^{12}, 10^{12}+10^{10}\right[$ | 5.648 | 2.061 | 1.341 | .926 | .655 | .552 | .504 | **361840208** |
| $\left[10^{15}, 10^{15}+10^{10}\right[$ | 18.091 | 6.614 | 4.338 | 3.057 | 2.447 | 2.061 | 1.959 | **289531946** |
| $\left[10^{18}, 10^{18}+10^{10}\right[$ | 72.772 | 26.614 | 17.191 | 11.681 | 8.297 | 6.601 | 6.372 | **241272176** |
| $\left[2^{64}-10^{10}, 2^{64}-1\right[$ | 208.423 | 74.847 | 48.160 | 32.267 | 22.064 | 17.044 | 17.016 | **225402976** |

These timings were measured on an AMD EPYC 9R14, the compilation flags used are "-std=c++20 -O3 -march=native -mtune=native -DNDEBUG" (OS: Debian 12, compiler: g++ version 12.2).

Performance optimization was halted once LFP reached usable speeds. While it cannot match primesieve (a highly optimized tool developed over 15 years), surpassing it was never the objective—LFP focuses on compile-time flexibility and standards compliance.

## Sieving above ${2}^{64}$

LFP supports prime generation beyond 64-bit limits by accepting arbitrary unsigned integer types (e.g., `unsigned __int128` or user-provided big-integer types). When available, `unsigned __int128` is automatically enabled, allowing direct sieving above $2^{64}$. Example: to count primes in $[2^{72}, 2^{72} + 10^{10}[$ using 48 threads:
```
$ ./lfp -t 48 4722366482869645213696 4722366482879645213696
```
### Validation
Some verification were done through:
 - WolframAlpha Cross-Checks: manual validation of arbitrary ranges e.g. $[2^{70}, 2^{70} + 10^3[$:
```
~$ ./lfp -t 4 -p 1180591620717411303424 1180591620717411304424
The number of prime numbers in range [1180591620717411303424, 1180591620717411304424[ is 23.
Took 50.6176s
Primes:
1180591620717411303449
1180591620717411303491
1180591620717411303503
1180591620717411303529
1180591620717411303539
1180591620717411303613
1180591620717411303619
1180591620717411303659
1180591620717411303727
1180591620717411303763
1180591620717411303809
1180591620717411303829
1180591620717411303839
1180591620717411303883
1180591620717411303911
1180591620717411303949
1180591620717411304069
1180591620717411304109
1180591620717411304117
1180591620717411304151
1180591620717411304223
1180591620717411304313
1180591620717411304393
```
- Targeted Stress Tests:
Ranges centered on $p^2$ and $p \cdot q$ (where $q$ is the smallest prime $> p$) for $p > 2^{32}$.<br>
Purpose: Ensures correct crossing-out of multiples for large base primes.<br>
Example: Verified $p^2 = 18446744202558570721$ for $p = 2^{32} + 15$ (a 33-bit prime):<br>
```
$ ./lfp -p 18446744202558570700 18446744202558570800
The number of prime numbers in range [18446744202558570700, 18446744202558570800[ is 5.
Took 6.31544s
Primes:
18446744202558570733
18446744202558570739
18446744202558570757
18446744202558570779
18446744202558570791
```

### Performance Notes

Trade-offs: Operations on large integers are inherently slower due to arbitrary-precision arithmetic.<br>
Parallelism: Multi-threading (-t N) mitigates latency for very large ranges.
