[![CI](https://github.com/youcefl/lfp/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/youcefl/lfp/actions/workflows/c-cpp.yml)

# LFP

A constexpr implementation of the sieve of Erathostenes

LFP is a constexpr implementation of the sieve of Erathostenes, it consists of a single header file, namely lfp.hpp,
to be included by the user.

Here is an example of use:

```c++
#include "lfp.hpp"

constexpr auto primesBelow256 = lfp::sieve<int8_t>(0, 256);
static_assert(primesBelow256.size() == 54);

// Check the number of primes between 10^7 and 10^7+10^6
static_assert(lfp::count_primes(10'000'000, 11'000'000) == 61938);

```

The sieve is optimized in the following ways:
 - numbers that are not coprime to 30 i.e. divisible by 2, 3 or 5 are not considered, this is a well known way to speed-up a sieve called wheel factorization.
 - the sieve is segmented i.e. if the size of the range R to sieve exceeds a certain threshold S, R is split into segments of size at most S.
 - the sieve is multithreaded we allocate N threads with each of the threads deals with part of the range to sieve.

The following table gives an idea of the performances to expect from the sieve:

| Range        | Threads | Sieve time (in seconds) |
|--------------|---------|-------------------------|
| [0, 10^9[    | 1       | 0.59s                   | 
| [0, 2*10^9[  | 1       | 1.22s                   | 
| [0, 4*10^9[  | 1       | 2.5s                    |

These timings were measured on a Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz which was otherwise idle.


