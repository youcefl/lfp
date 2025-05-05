/*
 * MIT License
 * Copyright (c) 2024-2025 Youcef Lemsafer
 * See LICENSE file for more details.
 * Creation date: december 2024.
 */
/// @file lfp.hpp
/// @brief Header defining the lfp library
///
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
#include <variant>
#include <cassert>
#include <span>


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
constexpr std::size_t count_primes(U n0, U n1);

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
private:
    unsigned int count_;
    static unsigned int defaultCount();
};

} // namespace lfp



#ifdef NDEBUG
# define LFP_ASSERT(...) ((void)0);
#else
# ifndef LFP_ASSERT
#   define LFP_ASSERT(...) assert(__VA_ARGS__)
# endif
#endif

#define LFP_CONCAT_IMPL(x, y) x##y
#define LFP_CONCAT(x, y) LFP_CONCAT_IMPL(x, y)

#ifdef LFP_ACTIVATE_LOG
#  define LFP_LOG(msg) \
    do { \
        if(!std::is_constant_evaluated()) { \
           ::lfp::details::log() << msg << "\n";\
       } \
      } while(0)

#  define LFP_LOG_S(msg, msgAtEos) \
	LFP_LOG(msg);\
	auto LFP_CONCAT(blockStartTime, __LINE__) = !std::is_constant_evaluated()\
               ? std::chrono::high_resolution_clock::now()\
	       : decltype(std::chrono::high_resolution_clock::now()){}; \
	auto LFP_CONCAT(logAtEob, __LINE__) = [&](){\
	    if(!std::is_constant_evaluated()) {\
                return std::variant<details::log_impl::log_at_eos, int>{\
		    details::make_log_at_eos([&](){\
                        auto duration = std::chrono::duration<double>{\
			        std::chrono::high_resolution_clock::now()\
				  - LFP_CONCAT(blockStartTime, __LINE__)};\
			::lfp::details::log() << msgAtEos << " (duration: " << duration << ")\n";\
		    })\
		};\
	    }\
	    return std::variant<details::log_impl::log_at_eos, int>{0};\
          }();
#else
#  define LFP_LOG(msg) ((void)0)
#  define LFP_LOG_S(msg, msgAtEob) ((void)0)
#endif


namespace lfp { 
namespace details {

class log_impl;

/// Returns the unique logger
inline log_impl & log();

/// Logging class
class log_impl
{
    struct appender;
public:
    log_impl(std::ostream & stream);
    template <typename Arg>
    appender & operator <<(Arg&& arg);
    struct log_at_eos {
	log_at_eos(log_at_eos const&) = delete;
	log_at_eos(log_at_eos&& other) noexcept
	{
	    func_ = std::move(other.func_);
	    other.func_ = [](){};
	}
	log_at_eos(std::function<void()> func);
	~log_at_eos();
	std::function<void()> func_;
    };

private:
    class appender {
    public:
	template <typename Arg>
	appender & operator <<(Arg&& arg);
    private:
	appender() = default;
	log_impl * parent_;
	friend class log_impl;
    };
    std::ostream & stream_;
    appender appender_;
};


inline
log_impl::log_impl(std::ostream & stream)
    : stream_(stream)
{
    appender_.parent_ = this;
}

template <typename Arg>
inline
log_impl::appender & log_impl::operator<<(Arg&& arg)
{
    auto ts = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
       ).count();
    stream_ << ts << " | " << std::forward<Arg>(arg);
    return appender_;
}

template <typename Arg>
inline
log_impl::appender & log_impl::appender::operator <<(Arg&& arg)
{
    parent_->stream_ << std::forward<Arg>(arg);
    return *this;
}

log_impl::log_at_eos::log_at_eos(std::function<void()> func)
    : func_(func)
{}

inline log_impl::log_at_eos::~log_at_eos()
{
    func_();
}

template <typename Func>
auto make_log_at_eos(Func func)
{
    return log_impl::log_at_eos{func};
}

inline log_impl & log()
{
    static log_impl inst{std::cout};
    return inst;
}


/// Constants
inline constexpr std::size_t bucket_sieve_threshold = 65535;
inline constexpr std::size_t initial_bucket_capacity = 32768;

/// Value meaning no offset i.e. the requested value is not present in the sequence
inline constexpr std::size_t noffs = (std::numeric_limits<std::size_t>::max)();

/// True if and only if T is one of the types listed after it
template <typename T, typename... Rest>
inline constexpr bool is_one_of_v = sizeof...(Rest) && ((std::is_same_v<T, Rest> || ...));

/// Returns the address of an object as an integer value
template <typename T>
inline auto obj_addr(T const* ptr)
{
    return reinterpret_cast<std::uintptr_t>(ptr);
}	

/// Residues modulo 30 that are coprime to 30
template <typename T>
inline constexpr auto cp30_residues = std::to_array<T>({1, 7, 11, 13, 17, 19, 23, 29});

/// All the primes below 2^8 in ascending order
template <typename T>
inline constexpr auto u8primes = std::to_array<T>({
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251
});

/// The really tiny primes, all numbers represented in a bitmap are coprime to
/// these, but they can still be returned by a sieve e.g. if the user requests
/// the primes in a range containing one of them.
template <typename T>
inline constexpr std::array<T,3> primesBelowSix = {2, 3, 5};


// Some two dimensional wheels to speed-up the localization of the multiples of a prime
inline constexpr uint8_t wheel[8][8] = {
    {6, 4, 2, 4, 2, 4, 6, 2},
    {4, 2, 4, 2, 4, 6, 2, 6},
    {2, 4, 2, 4, 6, 2, 6, 4},
    {4, 2, 4, 6, 2, 6, 4, 2},
    {2, 4, 6, 2, 6, 4, 2, 4},
    {4, 6, 2, 6, 4, 2, 4, 2},
    {6, 2, 6, 4, 2, 4, 2, 4},
    {2, 6, 4, 2, 4, 2, 4, 6}
};

inline constexpr uint8_t whoffs[8][8] = {
    {0, 1, 2, 3, 4, 5, 6, 7},
    {2, 7, 5, 4, 1, 0, 6, 3},
    {0, 2, 6, 4, 7, 5, 1, 3},
    {6, 2, 1, 5, 4, 0, 7, 3},
    {2, 6, 7, 3, 4, 0, 1, 5},
    {0, 6, 2, 4, 1, 3, 7, 5},
    {6, 1, 3, 4, 7, 0, 2, 5},
    {0, 7, 6, 5, 4, 3, 2, 1}    
};

/// "Adjustment" table used for finding the smallest multiple of a prime p
/// above max(p^2, n0) for a given n0.
/// More precisely, let p be a prime, we start to cross out the multiples of p
/// at p^2 but since we are using a mod 30 wheel we need those multiples to be
/// coprime to 30. What this table does is, once a multiple c of p of the form
/// p^2 + 2kp is found, it adds to c what needs to be added to make it coprime
/// to 30.
inline constexpr int8_t adjt[8][14] = {
    {0, 4, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 4, 2},
    {0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 4, 0, 4, 2},
    {0, 4, 4, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 2},
    {0, 2, 2, 0, 4, 0, 0, 2, 0, 0, 2, 0, 4, 2},
    {0, 2, 4, 0, 2, 0, 0, 2, 0, 0, 4, 0, 2, 2},
    {0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 4, 4},
    {0, 2, 4, 0, 4, 0, 0, 2, 0, 0, 2, 0, 2, 2},
    {0, 2, 4, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 4}
};

template <typename UInt>
constexpr auto cp30res_to_idx(UInt coprimeTo30Residue)
{
    return (coprimeTo30Residue * 17) >> 6;
}


struct ComputeSquareOfP {};
struct NoUpperBoundCond {};
struct ResiduesNotNeeded {};

template <typename U, typename V, typename  SquareOfPComputer = ComputeSquareOfP, typename UpperBoundCond = NoUpperBoundCond, typename ResiduesReceiver = ResiduesNotNeeded>
struct first_multiple_finder
{
    constexpr auto operator()(U p, V n0);

    SquareOfPComputer sqr_of_p_comp_;
    UpperBoundCond upper_bound_cond_;
    ResiduesReceiver residues_receiver_;
};

template <typename U, typename V, typename SquareOfPComputer, typename UpperBoundCond, typename ResiduesReceiver>
constexpr auto
first_multiple_finder<U, V, SquareOfPComputer, UpperBoundCond, ResiduesReceiver>::operator()(U p, V n0)
{
    auto const p2 = [&](){
	if constexpr (std::is_same_v<SquareOfPComputer, ComputeSquareOfP>) {
	   return p * p; 
        } else {
	   return sqr_of_p_comp_();
        }
      }();
    constexpr auto c_max = std::numeric_limits<V>::max();
    V dp2;
    auto c = (p2 >= n0) ? p2 
                        : (((dp2 = (n0 - p2 + 2 * p - 1)/(2 * p)*(2 * p)),
                            c_max - p2 < dp2) ? 0 
                                                 : p2 + dp2);
    if(!c) {
	return decltype(c){};
    }
    if constexpr(!std::is_same_v<UpperBoundCond, NoUpperBoundCond>) {
       if(!upper_bound_cond_(c)) {
	   return decltype(c){};
       }
    }
    constexpr auto callerNeedsResidues = !std::is_same_v<ResiduesReceiver, ResiduesNotNeeded>;
    auto cmod30 = c % 30;
    switch(cmod30) {
        case 3: case 5: case 9: case 15: case 21: case 25: case 27: {
            auto dc = adjt[(p % 30) * 4 / 15][cmod30 >> 1] * p;
            c = (c_max - c < dc) ? 0 : c + dc;
	    if constexpr(callerNeedsResidues) {
            	cmod30 = c % 30;
	    }
        }
        break;
        default:;
    }
    if constexpr(!std::is_same_v<UpperBoundCond, NoUpperBoundCond>) {
       if(!upper_bound_cond_(c)) {
	   return decltype(c){};
       }
    }
    if constexpr(callerNeedsResidues) {
	residues_receiver_(p % 30, cmod30);
    }
    return c;
}

template <typename U, typename V>
constexpr auto find_first_multiple_impl(U p, V n0)
{
    return first_multiple_finder<U, V>{}(p, n0);
}

template <typename U, typename V, typename SquareOfPComputer, typename UpperBoundCond, typename ResiduesReceiver>
constexpr auto find_first_multiple_impl(U p, V n0, SquareOfPComputer sqrOfPComp, UpperBoundCond upperBoundCond, ResiduesReceiver residuesReceiver)
{
    auto finder = first_multiple_finder<U, V, SquareOfPComputer, UpperBoundCond, ResiduesReceiver>{
	  sqrOfPComp,
	  upperBoundCond,
	  residuesReceiver
      };
    return finder(p, n0);
}

template <typename U, typename V>
constexpr auto
find_first_multiple_ae_b(U p, V n0, V n1, U pSquared, uint8_t & pmod30, uint8_t & cmod30)
{
   return find_first_multiple_impl(p, n0,
	      [pSquared](){ return pSquared; },
	      [n1](auto c){ return c < n1; },
		   [&](uint8_t pm30, uint8_t cm30){
		       pmod30 = pm30;
		       cmod30 = cm30;
		   });
}

/// Computes the smallest multiple m of p such that m >= max(p^2, n0) and
/// gcd(m, 30) = 1.
/// @param p the prime a multiple of which is requested
/// @param n0 lower bound for returned multiple of p
/// @return 0 when such a multiple does not fit in the return type.
template <typename U, typename V>
constexpr auto
find_first_multiple_ae(U p, V n0)
{
    return find_first_multiple_impl(p, n0);
}

// Returns the smallest integer coprime to 30 and greater than or equal to @param n0
template <typename Ret, typename U>
constexpr
Ret compute_gte_coprime(U n0)
{
    constexpr uint8_t dn[30] = {
        1, 0, 5, 4, 3, 2, 1, 0, 3, 2, 1, 0, 1, 0, 3,
        2, 1, 0, 1, 0, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0
      };

    return Ret{n0} + dn[n0 % 30];
}


/// Returns the largest integer coprime to 30 and < @param n1
template <typename Ret, typename U>
constexpr
Ret compute_lt_coprime(U n1)
{
    constexpr uint8_t dn[30] = {
        1, 2, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 1, 2, 1,
        2, 3, 4, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6
      };
 
    return Ret{n1} - dn[n1 % 30];
}


template <typename U, typename V>
constexpr std::size_t index_of(U m, uint8_t mmod30Idx,  V m0)
{
    LFP_ASSERT((m >= m0) && (std::gcd(m, 30) == 1)
	      && (mmod30Idx == cp30res_to_idx(m % 30))  && (std::gcd(m0, 30) == 1));

    auto m0_idx = cp30res_to_idx(m0 % 30);
    return (m - m0) / 30 * 8 + ((mmod30Idx >= m0_idx) ? mmod30Idx - m0_idx : 8 + mmod30Idx - m0_idx);
}

/// Returns the index of m relative to the index of m0.
/// @pre both m and m0 are coptime to 30 and m >= m0
template <typename U, typename V>
constexpr std::size_t index_of(U m, V m0)
{
    LFP_ASSERT((m >= m0) && (std::gcd(m, 30) == 1) && (std::gcd(m0, 30) == 1));

    auto m_idx = cp30res_to_idx(m % 30);
    auto m0_idx = cp30res_to_idx(m0 % 30);
    return (m - m0) / 30 * 8 + ((m_idx >= m0_idx) ? m_idx - m0_idx : 8 + m_idx - m0_idx);
}

/// Returns the value at offset offs relative to the offset of n0 i.e. the value v such that offs == index_of(v, n0)
/// @pre n0 is coprime to 30
template <typename V>
constexpr V value_at(V n0, std::size_t offs)
{
    LFP_ASSERT(std::gcd(n0, 30) == 1);

    auto n0Idx = cp30res_to_idx(n0 % 30);
    auto nxIdx = (n0Idx + offs) % 8;
    return n0 + (offs / 8) * 30 
	      + ((nxIdx >= n0Idx) 
		 ? cp30_residues<V>[nxIdx] - cp30_residues<V>[n0Idx]
	         : 30 + cp30_residues<V>[nxIdx] - cp30_residues<V>[n0Idx]);
}



/// Computes the index of the first multiple of prime p in range [n0, ne] relative to the index of n0
/// as well as the offsets of the following multiples of p relative to firstMultIdx.
/// @pre p is a prime number > 6, n0 <= n1, gcd(n0, 30) = 1 and gcd(ne, 30) = 1.
/// @post firstMultIdx is the index of the first multiple of p in [n0, n1[ or noffs if no such multiple,
/// offsets is filled with the offsets of the next multiples of p relative to firstMultIdx,
/// @return the number of offsets written to offs
template <typename U, typename V>
constexpr std::size_t compute_offsets(std::size_t & firstMultIdx,
    std::array<std::size_t, 7> & offsets,
    U p,
    V n0,
    V ne,
    V c
    )
{
    firstMultIdx = noffs;
    
#if 0
    if(c < n0) { //@todo: remove. Used for debugging
        c = find_first_multiple_ae(p, n0);
	if(!std::is_constant_evaluated()) {
	    std::cout << "Bad multiple, got " << c << " where a value > " << n0 << " was requested." << std::endl;
	}
    }
#endif
    constexpr auto c_max = std::numeric_limits<U>::max();
    bool isFirstIndex = true;
    std::size_t offsCount = 0;
    auto const iw = cp30res_to_idx(p % 30);
    auto const jw = cp30res_to_idx(c % 30);

    for(auto j = whoffs[iw][jw];
        c <= ne;
        c = ((c_max - c < wheel[iw][j] * p)
         ? ne + 1
         : c + wheel[iw][j] * p), j = (j + 1) % 8) {
        auto currIdx = index_of(c, n0);
        if(isFirstIndex) {
            firstMultIdx = currIdx;
            isFirstIndex = false;
            continue;
        }
        offsets[offsCount++] = currIdx - firstMultIdx;
        if(offsCount == offsets.size()) {
            break;
        }
    }
    return offsCount;
}

template <typename U, typename V>
constexpr std::size_t compute_remaining_offsets(std::size_t firstMultIdx,
    std::array<std::size_t, 7> & offsets,
    U p,
    V n0,
    V ne,
    V c,
    uint8_t whpos
    )
{
    constexpr auto c_max = std::numeric_limits<U>::max();
    bool isFirstIndex = true;
    std::size_t offsCount = 0;
    auto const iw = whpos >> 3;
    auto const jw = whpos & 7;
    for(auto j = whoffs[iw][jw];
        c <= ne;
        c = ((c_max - c < wheel[iw][j] * p)
          ? ne + 1
          : c + wheel[iw][j] * p), j = (j + 1) % 8) {
        if(isFirstIndex) {
            isFirstIndex = false;
            continue;
        }
        auto currIdx = index_of(c, n0);
        offsets[offsCount++] = currIdx - firstMultIdx;
        if(offsCount == offsets.size()) {
            break;
        }
    }
    return offsCount;
}


/// Applies given mask to value starting at bit value + offs.
/// @pre  0 <= offs < std::numeric_limits<U>::digits
template <typename U, typename V>
constexpr void mask_at(U & value, V offs, U mask)
{
    value &= ((mask >> offs) | (offs ? (~U{} << (std::numeric_limits<U>::digits - offs)) : 0));
}

/// Applies to value, starting at bit 0 i.e. the msb, the part of mask starting at bit mask + offs.
/// @pre  0 <= offs < std::numeric_limits<U>::digits
template <typename U, typename V>
constexpr void mask(U & value, U mask, V offs)
{
    value &= (offs ? (mask << (std::numeric_limits<U>::digits - offs)) : U{}) | (~U{} >> offs);
}


/// @class allocator
/// An allocator class used when specific alignment is needed.
template <typename U, std::size_t Alignment = 64>
struct allocator
{
    static_assert(Alignment >= alignof(U));
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two.");

    using value_type = U;

    template <typename V>
    struct rebind {
	using other =  allocator<V, Alignment>;
    };

    static constexpr std::align_val_t alignment {Alignment};

    constexpr allocator() = default;

    template <typename V>
    constexpr allocator(allocator<V, Alignment> const &) noexcept;

    [[nodiscard]] constexpr U* allocate(std::size_t n);

    constexpr void deallocate(U* ptr, std::size_t /*n*/) noexcept;

    friend bool operator==(allocator const &, allocator const &) noexcept { return true; }
    friend bool operator!=(allocator const &, allocator const &) noexcept { return false; }
};


template <typename U, std::size_t Alignment>
template <typename V>
constexpr
allocator<U, Alignment>::allocator(allocator<V, Alignment> const &) noexcept
{}

template <typename U, std::size_t Alignment>
[[nodiscard]] constexpr U*
allocator<U, Alignment>::allocate(std::size_t n)
{
    if(std::is_constant_evaluated()) {
        return new U[n];
    }
    if(n > std::numeric_limits<std::size_t>::max() / sizeof(U))
    {
        throw std::bad_array_new_length{};
    }
    auto ptr = ::operator new(n * sizeof(U), alignment);
    if(ptr) {
        return static_cast<U*>(ptr);
    }
    throw std::bad_alloc{};
}


template <typename U, std::size_t Alignment>
constexpr void
allocator<U, Alignment>::deallocate(U* ptr, std::size_t /*n*/) noexcept
{
    if(std::is_constant_evaluated()) {
        delete [] ptr;
        return;
    }
    ::operator delete(ptr, alignment);

}

/// Useful when instantiating class bitmask_impl e.g. bitmask_impl<..., use_dictionary_t{true}, ...>
using use_dictionary_t = bool;

/// @class bitmask_impl
/// Holds the bitmask to be applied to a bitmap in order to reset the bits corresponding to the multiples of a prime P.
/// @param U the limb type
/// @param P the prime number
/// @param UseDictionary whether to precompute and store the masks for all possible indices
/// @param Alignment alignment to use for the array holding the mask(s)
template <typename U, std::size_t P, use_dictionary_t UseDictionary = P < 128, std::size_t Alignment = 64>
class bitmask_impl
{
public:
    constexpr bitmask_impl() = default;

    /// Returns the prime corresponding to this bitmask.
    constexpr std::size_t prime() const;
    /// Returns the size of the bitmask
    constexpr std::size_t size() const;
    /// Returns the word at offset idx.
    /// The bitmask is cyclic: if (idx + digits_) > size() the returned word is
    // the concatenation of the bits in the ranges [idx, size()[ and [0, idx[.
    /// @pre idx < size()
    constexpr U word_at(std::size_t idx) const;
    /// Returns the offset in the bitmask corresponding to the composite c
    /// @pre c is coprime to 30 and of the form p^2 + 2kp (k being an integer >= 0).
    constexpr auto offset(std::size_t c) const;

private:
    static constexpr U word_at(U const * data, std::size_t idx);

    /// Loops on numbers that are coprime to 30 in range [p^2, p^2 + 30p[
    /// calling f(i, j) each time i is of the form p^2+2kp, j being the index of i
    /// relative to the index of p^2 which is always zero. 
    template <typename Func>
    static constexpr void for_each_coprime_to_30(Func f);
    static constexpr auto compute_offsets();  
    static constexpr auto compute_data();

    static constexpr auto digits_ = std::numeric_limits<U>::digits;
    static constexpr auto size_ = 8 * P;
    static constexpr std::size_t data_size_ = UseDictionary 
                                      ? size_ : 1 + (size_ + digits_ - 1) / digits_;
    alignas(Alignment) static constexpr std::array<U, data_size_> data_ = compute_data();
    static constexpr std::array<uint16_t, 8> offs_ = compute_offsets();
};


template <typename U, std::size_t P, use_dictionary_t UseDictionary, std::size_t Alignment>
constexpr std::size_t
bitmask_impl<U, P, UseDictionary, Alignment>::prime() const
{
    return P;
}


template <typename U, std::size_t P, use_dictionary_t UseDictionary, std::size_t Alignment>
constexpr std::size_t
bitmask_impl<U, P, UseDictionary, Alignment>::size() const
{
    return size_;
}


template <typename U, std::size_t P, use_dictionary_t UseDictionary, std::size_t Alignment>
constexpr U
bitmask_impl<U, P, UseDictionary, Alignment>::word_at(std::size_t idx) const
{
    if constexpr(UseDictionary) {
        return data_[idx];            
    } else {
        return word_at(&data_[0], idx);
    }
}


template <typename U, std::size_t P, use_dictionary_t UseDictionary, std::size_t Alignment>
constexpr auto
bitmask_impl<U, P, UseDictionary, Alignment>::offset(std::size_t c) const
{
    return offs_[cp30res_to_idx(c % 30)];
}


template <typename U, std::size_t P, use_dictionary_t UseDictionary, std::size_t Alignment>
constexpr U
bitmask_impl<U, P, UseDictionary, Alignment>::word_at(U const * data, std::size_t idx)
{
    auto widx = idx /digits_;
    auto shift = idx % digits_;
    return shift ? (data[widx] << shift) | (data[widx + 1] >> (digits_ - shift))
                : data[widx];
}


template <typename U, std::size_t P, use_dictionary_t UseDictionary, std::size_t Alignment>
template <typename Func>
constexpr void
bitmask_impl<U, P, UseDictionary, Alignment>::for_each_coprime_to_30(Func f)
{
    constexpr std::array<int, 8> residues{1, 7, 11, 13, 17, 19, 23, 29};
    constexpr std::array<int, 8> wheel   {6, 4,  2,  4,  2,  4,  6,  2};
    for(int i0 = P * P, i = i0, wi = ((i % 30 == 1) ? 0 : 5), j = 0;
        i < i0 + 30 * P;
        i += wheel[wi], wi = (wi + 1) % 8, ++j) {
        if((i - i0) % (2 * P)) {
            continue;
        }
        f(i, j);
    }
}


template <typename U, std::size_t P, use_dictionary_t UseDictionary, std::size_t Alignment>
constexpr auto
bitmask_impl<U, P, UseDictionary, Alignment>::compute_offsets()
{
    std::remove_cv_t<decltype(offs_)> offsets;
    for_each_coprime_to_30([&offsets](auto i, auto j) {
        offsets[cp30res_to_idx(i % 30)] = j;
    });
    return offsets;
} 


template <typename U, std::size_t P, use_dictionary_t UseDictionary, std::size_t Alignment>
constexpr auto
bitmask_impl<U, P, UseDictionary, Alignment>::compute_data()
{
    std::array<U, 1 + (size_ + digits_ - 1) / digits_> bits;
    std::ranges::fill(bits, ~U{});
    for_each_coprime_to_30([&bits](auto, auto j) {
        bits[j / digits_] &= ~(U{1} << (digits_ - 1 - (j % digits_)));
    });

    // Fill remaining bits by copying from the beginning.
    if(size_ % digits_) {
        std::size_t d = 0, t = size_;
        for(; t < bits.size() * digits_;) {
            auto dsh = d % digits_;
            auto tsh = t % digits_;
            auto shm = size_ - d < digits_ ? size_ - d : digits_ - dsh;
            bits[t / digits_] &= (( (bits[d / digits_]  << dsh)
                                | (shm >= digits_ ? U{} : ~U{} >> shm)
                                ) >> tsh) | (tsh ? ~U{} << (digits_ - tsh) : U{});
            auto srcBits = (std::min)(digits_ - d % digits_, size_ - d);
            auto dstBits = digits_ - t % digits_;
            auto writtenBits = (std::min)(srcBits, dstBits);
            d = (d + writtenBits) % size_;
            t += writtenBits;
        }
    } else {
        bits.back() = bits[0];
    }
    if constexpr(UseDictionary) {
        std::array<U, data_size_> dictionary;
        for(std::size_t i = 0; i < size_; ++i) {
            dictionary[i] = word_at(&bits[0], i);
        }
        return dictionary;
    } else {
        return bits;
    }
}


template <typename U, uint8_t... SmallPrimes>
class bitmask_pack
{
    static constexpr bool is_prime(uint8_t n);
public:
    static_assert(sizeof...(SmallPrimes) != 0, "Expecting at least one prime!");
    static_assert((is_prime(SmallPrimes) && ...), "Expecting prime numbers below 255!");

    constexpr bitmask_pack() = default;
    /// Returns a comma separated list of the SmallPrimes as a string
    static constexpr std::string_view primes_as_string();
    /// Returns the number of bitmask objects in this pack
    static constexpr auto size();
    /// Returns a tuple containing all bitmask objects in this pack
    constexpr auto const & get_all() const;
    /// Gets the bitmask object at index I
    template <std::size_t I>
    constexpr auto const & get() const;
    /// Returns the largest prime in SmallPrimes
    static constexpr std::uint8_t max_prime();
    /// Returns the combination of the masks at the given offsets
    template <std::size_t N>
    constexpr U combined_masks(std::size_t const (&offsets) [N]) const;

private:
    using primes_sequence = std::integer_sequence<std::size_t, SmallPrimes...>;

    static constexpr auto primes_as_string(primes_sequence primesSeq);

    template <std::size_t P, std::size_t... I>
    static constexpr bool is_in_sequence(std::integer_sequence<std::size_t, I...>);

    static constexpr auto make_masks_tuple();

    template <std::size_t N, std::size_t... I>
    constexpr U combined_masks_impl(std::size_t const (&offs) [N], std::index_sequence<I...>) const;

    /// Largest prime in SmallPrimes
    static constexpr std::uint8_t upper_bound_ = []() {
        constexpr auto smallPrimes = std::to_array<std::size_t>({SmallPrimes...});
        return *std::max_element(std::begin(smallPrimes), std::end(smallPrimes));
      } ();

    /// The masks
    decltype(make_masks_tuple()) masks_;
    // 3 digits per prime, 2 characters for the separator ", ", and the final '\0':
    // 3 * sizeof...(SmallPrimes) + 2 * (sizeof...(SmallPrimes) - 1) + 1
    static constexpr std::array<char, 5 * sizeof...(SmallPrimes) - 1> str_primes_ = primes_as_string(primes_sequence{});
    static constexpr std::string_view str_primes_view_{str_primes_.cbegin(), std::find(str_primes_.cbegin(), str_primes_.cend(), '\0')};

};


template <typename U, uint8_t... SmallPrimes>
constexpr auto bitmask_pack<U, SmallPrimes...>::size()
{
    return sizeof...(SmallPrimes);
}

template <typename U, uint8_t... SmallPrimes>
constexpr std::string_view bitmask_pack<U, SmallPrimes...>::primes_as_string()
{
//    return	std::string_view{str_primes_.cbegin(), std::find(str_primes_.cbegin(), str_primes_.cend(), '\0')};
    return str_primes_view_;
}

template <typename U, uint8_t... SmallPrimes>
constexpr auto bitmask_pack<U, SmallPrimes...>::primes_as_string(primes_sequence)
{
    // all of this because std::to_string is not constexpr in C++20
    std::array<char, str_primes_.size()> ret{};
    int i = 0;
    char * sep = nullptr;
    for(auto p : std::to_array<uint8_t>({SmallPrimes...})) {
	if(i) {
	    ret[i++] = ',';
	    ret[i++] = ' ';
	}
	char strp[8] = {};
	int j = 7;
	for(;p ; p /= 10) {
	    strp[j--] = '0' + (p % 10);
	}
	std::copy(strp + j + 1, strp + sizeof(strp), &ret[i]);
	i += sizeof(strp) - j - 1;
    }
    return ret;
}

template <typename U, uint8_t... SmallPrimes>
constexpr auto const & bitmask_pack<U, SmallPrimes...>::get_all() const
{
    return masks_;
}

template <typename U, uint8_t... SmallPrimes>
template <std::size_t I>
constexpr auto const & bitmask_pack<U, SmallPrimes...>::get() const
{
    return std::get<I>(masks_);
}

template <typename U, uint8_t... SmallPrimes>
constexpr std::uint8_t bitmask_pack<U, SmallPrimes...>::max_prime()
{
    return upper_bound_;
}

template <typename U, uint8_t... SmallPrimes>
template <std::size_t N>
constexpr U bitmask_pack<U, SmallPrimes...>::combined_masks(std::size_t const (&offs) [N]) const
{
    static_assert(N == sizeof...(SmallPrimes),
        "The number of offsets must be equal to the number of bitmasks in the pack.");
    return combined_masks_impl(offs, std::make_index_sequence<N>{});
}

template <typename U, uint8_t... SmallPrimes>
template <std::size_t P, std::size_t... I>
constexpr bool bitmask_pack<U, SmallPrimes...>::is_in_sequence(std::integer_sequence<std::size_t, I...>)
{
    return ((P == I) || ...);
}

template <typename U, uint8_t... SmallPrimes>
constexpr auto bitmask_pack<U, SmallPrimes...>::make_masks_tuple()
{
    return std::make_tuple(bitmask_impl<U, SmallPrimes>{}...);
}

template <typename U, uint8_t... SmallPrimes>
template <std::size_t N, std::size_t... I>
constexpr U bitmask_pack<U, SmallPrimes...>::combined_masks_impl(std::size_t const (&offs) [N], std::index_sequence<I...>) const
{
    return (std::get<I>(masks_).word_at(offs[I]) & ...);
}

template <typename U, uint8_t... SmallPrimes>
constexpr bool bitmask_pack<U, SmallPrimes...>::is_prime(uint8_t n)
{
    return std::find(std::begin(u8primes<uint8_t>), std::end(u8primes<uint8_t>),
                     n) != std::end(u8primes<uint8_t>);
}

template <std::size_t prime>
using bitmask = bitmask_impl<uint64_t, prime>;


template <typename PrimeT>
class bucket
{
public:
    // Should be sufficient because if not, it means someone created 500MiB bitmap
    // i.e. a bitmap representing a range of length 2^32/8*30 = 16'106'127'360 which
    // is way too much.
    using offset_type = std::uint32_t;
    using offsets_type = std::vector<offset_type, allocator<offset_type, 64>>;

    constexpr bucket(offsets_type & scratchOffsets, std::size_t initialCapacity);
    constexpr std::size_t size() const;
    constexpr bool is_empty() const;
    constexpr void add(PrimeT prime, offset_type cOffs, uint8_t whpos);
    /// Returns the primes in this bucket
    constexpr std::vector<PrimeT> const & primes() const;
    template <typename V>
    constexpr offsets_type const & compute_offsets(V n0, V ne, std::size_t bmpSize);
    constexpr void clear();
private:
    std::vector<PrimeT> primes_;
    offsets_type first_multiples_offsets_;
    std::vector<uint8_t> whpos_;
    offsets_type & scratch_offsets_;
};

template <typename PrimeT>
constexpr
bucket<PrimeT>::bucket(offsets_type & scratchOffsets, std::size_t initialCapacity)
    : scratch_offsets_(scratchOffsets)
{
    primes_.reserve(initialCapacity);
    first_multiples_offsets_.reserve(initialCapacity);
    whpos_.reserve(initialCapacity);
}

template <typename PrimeT>
constexpr void
bucket<PrimeT>::clear()
{
    primes_.clear();
    first_multiples_offsets_.clear();
    whpos_.clear();
}

template <typename PrimeT>
constexpr std::size_t
bucket<PrimeT>::size() const
{
    return primes_.size();
}

template <typename PrimeT>
constexpr bool
bucket<PrimeT>::is_empty() const
{
    return primes_.empty();
}

template <typename PrimeT>
constexpr void
bucket<PrimeT>::add(PrimeT prime, offset_type cOffs, uint8_t whpos)
{
    primes_.push_back(prime);
    first_multiples_offsets_.push_back(cOffs);
    whpos_.push_back(whpos);
}

template <typename PrimeT>
constexpr std::vector<PrimeT> const &
bucket<PrimeT>::primes() const
{
    return primes_;
}


template <typename PrimeT>
template <typename V>
constexpr bucket<PrimeT>::offsets_type const &
bucket<PrimeT>::compute_offsets(V n0, V ne, std::size_t bmpSize)
{
    LFP_LOG("bucket " << reinterpret_cast<intptr_t>(this)
		    << ": computing offsets, number of primes: " << primes_.size()
		    << ", first prime: " << (!primes_.empty() ? primes_.front() : 0)
		    << ", last prime: " << (!primes_.empty() ? primes_.back() : 0));

    scratch_offsets_.clear();

    for(std::size_t k{}, kmax{primes_.size()}; k < kmax; ++k) {
	auto p = primes_[k];
	auto start = first_multiples_offsets_[k];
	LFP_ASSERT(start < bmpSize);
	auto whpos = whpos_[k];
	auto c = value_at(n0, start);
        std::array<std::size_t, 7> offsets;
	auto offsCount = compute_remaining_offsets(start, offsets, p, n0, ne, c, whpos);
         // If things are done correctly the following condition is true most of the time
        // because buckets are for big primes (someone like 509 has nothing to do here,
        // whereas 12503 is welcome).
        if(!offsCount) {
	    scratch_offsets_.push_back(start);
	    continue;
	}	
        /// @todo: this is duplicated code, have a look at the end of inner_sieve
        auto i = start;
        if(offsCount == offsets.size()) {
            // offsets are periodic, period is 8p
            // @todo: can there be an overflow when adding 8*p?
            for(; i + 8 * p < bmpSize; i += 8 * p) {
                scratch_offsets_.push_back(i);
                scratch_offsets_.push_back(i + offsets[0]);
                scratch_offsets_.push_back(i + offsets[1]);
                scratch_offsets_.push_back(i + offsets[2]);
                scratch_offsets_.push_back(i + offsets[3]);
                scratch_offsets_.push_back(i + offsets[4]);
                scratch_offsets_.push_back(i + offsets[5]);
                scratch_offsets_.push_back(i + offsets[6]);
            }
        }
        const auto i0 = i;
        for(auto j = 0; i < bmpSize; i = i0 + offsets[j], ++j) {
            scratch_offsets_.push_back(i);
            if(j == offsCount) {
                break;
            }
        }
    }

    LFP_LOG("bucket " << reinterpret_cast<intptr_t>(this) << ": found " << scratch_offsets_.size() << " offsets");

    return scratch_offsets_;
}


template <typename T> class PrimesIterator;


class Bitmap
{
public:
    using value_type = uint64_t; // temporary, will become U when the class becomes a template

    constexpr Bitmap();
    explicit constexpr Bitmap(uint64_t n0, std::size_t size);
    constexpr void assign(uint64_t n0, std::size_t size);
    constexpr std::size_t size() const;
    constexpr std::size_t indexOf(uint64_t val) const;
    constexpr value_type first_value() const;
    constexpr value_type last_value() const;
    constexpr value_type value_at(std::size_t idx) const;
    constexpr void reset(std::size_t index);
    template <std::size_t Prime>
    constexpr void apply(bitmask_impl<uint64_t, Prime> const & mask);
    template <uint8_t... Primes>
    constexpr void apply(bitmask_pack<uint64_t, Primes...> const & maskPack);
    template <typename PrimeT>
    constexpr void apply_and_clear(bucket<PrimeT> & bkt);
    template <typename PrimeT>
    constexpr void apply(bucket<PrimeT> & bkt);
    constexpr uint64_t popcount() const;
    void check() const;
    constexpr uint8_t at(std::size_t index) const;
    template <typename Func>
    constexpr void foreach_setbit(Func ff) const;

private:
    template <typename Int> static constexpr std::size_t indexInResidues(Int);
    static constexpr value_type value_at_impl(value_type n0, std::size_t offs);

    struct mask_application_data {
        std::size_t first_composite_;
        std::size_t first_composite_index_;
        std::size_t current_mask_offset_;
        std::size_t current_bitmap_index_;
    };
    template <std::size_t Prime>
    constexpr mask_application_data compute_mask_application_data(bitmask_impl<uint64_t, Prime> const & bmk) const;
    template <std::size_t Prime>
    constexpr void apply(bitmask_impl<uint64_t, Prime> const & bmk, mask_application_data & mappData, std::size_t endOffset);

    std::vector<uint64_t, allocator<uint64_t, 64>> vec_;
    std::size_t size_;
    /// First value
    value_type n0_;
    /// Last value
    value_type ne_;
    static constexpr std::array<int,30> d_{1,0,5,4,3,2,1,0,3,2,1,0,1,0,3,2,1,0,1,0,3,2,1,0,5,4,3,2,1,0};
    static constexpr std::array<int,8> residues_{1,7,11,13,17,19,23,29};
    static constexpr std::array<int,8> deltas_{6,4,2,4,2,4,6,2};
    static constexpr auto nopos_ = (std::numeric_limits<std::size_t>::max)();
    using ElemType = decltype(vec_)::value_type;
    using NumLim = std::numeric_limits<ElemType>;
    static constexpr std::size_t digits_{std::numeric_limits<ElemType>::digits};

    template <typename T> friend class PrimesIterator;
};

constexpr
Bitmap::Bitmap()
  : size_(0)
  , n0_(0)
  , ne_(0)
{}

constexpr
Bitmap::Bitmap(uint64_t n0, std::size_t size)
  : vec_((size + NumLim::digits - 1)/NumLim::digits,
            ~decltype(vec_)::value_type{})
  , size_(size)
  , n0_(n0 + d_[n0 % 30])
  , ne_(size_ ? value_at_impl(n0_, size_ - 1) : n0_)
{
    LFP_ASSERT(size_ >= 1);

    if(size % NumLim::digits) {
        vec_.back() &= ~ElemType{} << (NumLim::digits - size % NumLim::digits);
    }

    LFP_LOG("constructed bitmap " << obj_addr(this) << ", [" << n0_ << ", " << ne_ << "], size = " << size_);
}

constexpr
void
Bitmap::assign(uint64_t n0, std::size_t size)
{
    LFP_LOG("assigning bitmap " << obj_addr(this));

    vec_.assign((size + NumLim::digits - 1)/NumLim::digits,
            ~decltype(vec_)::value_type{});
    size_ = size;
    n0_ = n0 + d_[n0 % 30];
    ne_ = size_ ? value_at_impl(n0_, size_ - 1) : n0_;

    if(size % NumLim::digits) {
        vec_.back() &= ~ElemType{} << (NumLim::digits - size % NumLim::digits);
    }

    LFP_LOG("assigned bitmap " << obj_addr(this) << ": [" << n0_ << ", " << ne_ << "], size = " << size_);
}

constexpr
std::size_t
Bitmap::size() const
{
    return size_;
}

constexpr
std::size_t
Bitmap::indexOf(value_type val) const
{
    return index_of(val, n0_);
}

constexpr
Bitmap::value_type
Bitmap::first_value() const
{
    return n0_;
}

constexpr
Bitmap::value_type
Bitmap::last_value() const
{
    return ne_;
}

constexpr
Bitmap::value_type
Bitmap::value_at(std::size_t idx) const
{
    return value_at_impl(n0_, idx);
}

constexpr
Bitmap::value_type
Bitmap::value_at_impl(value_type n0, size_t idx)
{
    auto n0res = n0 % 30;
    auto nxres = residues_[(n0res * 4 / 15 + idx) % 8]; 
    return n0 + (idx / 8) * 30 + ((nxres >= n0res) ? nxres - n0res : 30 + nxres - n0res );
}

constexpr
void
Bitmap::reset(std::size_t index)
{
    vec_[index / NumLim::digits] &= ~(ElemType{1} << (NumLim::digits - 1 - index % NumLim::digits));
}

constexpr
uint64_t
Bitmap::popcount() const
{
    return std::transform_reduce(std::begin(vec_), std::end(vec_), 0, std::plus{},
		    [](auto v){ return std::popcount(v);});
}

template <typename Func>
constexpr
void
Bitmap::foreach_setbit(Func ff) const
{
    auto k = 0;
    auto c = n0_;
    for(auto i = indexInResidues(n0_);
        k < size();
        c += deltas_[i], i = (i + 1) % 8, ++k) {
	if(at(k)) {
	    ff(k, c);
	}
    }
}

template <typename Int>
constexpr
std::size_t
Bitmap::indexInResidues(Int x)
{
    return std::distance(std::begin(residues_), std::find(std::begin(residues_), std::end(residues_), x % 30));
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
        if(is_prime != (at(k) != 0)) {
            std::cout << "is_prime=" << is_prime << ", " << (at(k) != 0) << " | ";
            std::cout << c << " is" << (is_prime ? "" : " not") << " a prime." << std::endl;
        }
    }
}

constexpr
uint8_t Bitmap::at(std::size_t index) const
{
    return (vec_[index / NumLim::digits] >> (NumLim::digits - 1 - index % NumLim::digits)) & 1;
}

template <std::size_t Prime>
constexpr Bitmap::mask_application_data
Bitmap::compute_mask_application_data(bitmask_impl<uint64_t, Prime> const & bmk) const
{
    mask_application_data mappData{0, nopos_, nopos_, nopos_};

    const auto p = bmk.prime();
    auto c = find_first_multiple_ae(p, n0_);
    if(!c) {
        return mappData;
    }
    auto cOffs = indexOf(c);
    if(cOffs >= size_) {
        return mappData;
    }
    mappData.first_composite_ = c;
    mappData.first_composite_index_ = cOffs;
    mappData.current_mask_offset_ = bmk.offset(c);
    mappData.current_bitmap_index_ = cOffs;
    return mappData;
}


template <std::size_t Prime>
constexpr
void Bitmap::apply(bitmask_impl<uint64_t, Prime> const & bmk)
{
    LFP_LOG("bitmap " << obj_addr(this) << ": about to apply precomputed bitmask (prime = " << bmk.prime() << ")");

    auto mappData = compute_mask_application_data(bmk);
    apply(bmk, mappData, size());
 
    LFP_LOG("bitmap " << obj_addr(this) << ": applied precomputed bitmask (prime = " << bmk.prime() << ")");
}


template <std::size_t Prime>
constexpr
void Bitmap::apply(bitmask_impl<uint64_t, Prime> const & bmk, mask_application_data & mappData, std::size_t endOffset)
{
    using U = uint64_t; //@todo: remove this once the class is templated
    auto cOffs = mappData.first_composite_index_;
    if((cOffs >= endOffset) || (cOffs > size())) {
	return;
    }
    auto cOffsBmk = mappData.current_mask_offset_;
    const auto bmk_size = bmk.size();
    if(cOffs % digits_) {
	auto mask = bmk.word_at(cOffsBmk);
	auto isEndBeforeWordEnd = (endOffset - cOffs) + (cOffs % digits_) < digits_;
	mask = isEndBeforeWordEnd
		? mask | (~U{} >> (cOffs % digits_ + endOffset - cOffs))
		: mask;
	details::mask_at(vec_[cOffs / digits_], cOffs % digits_, mask);
	auto delta = isEndBeforeWordEnd ? endOffset - cOffs : digits_ - (cOffs % digits_);
	cOffs += delta;
	cOffsBmk = (cOffsBmk + delta) % bmk_size;
 	if(isEndBeforeWordEnd) {
	    mappData.current_bitmap_index_ = endOffset;
	    mappData.current_mask_offset_ = cOffsBmk;
	    return;
	}
    }
    std::size_t i0 = cOffs / digits_;
    std::size_t i = i0;
    auto const cOffsBmk0 = cOffsBmk;
    std::size_t const imax = endOffset / digits_;
    for(; i < imax; ++i, cOffsBmk = (cOffsBmk + digits_) % bmk_size) {
        vec_[i] &= bmk.word_at(cOffsBmk);
    }
    auto delta = (i - i0) * digits_;
    cOffs += delta;
    cOffsBmk = (cOffsBmk0 + delta) % bmk_size;
    if((cOffs < size()) && (endOffset % digits_)) {
        auto mask = bmk.word_at(cOffsBmk);
	details::mask_at(vec_[cOffs / digits_], cOffs % digits_, mask);
	delta = endOffset - cOffs;
	cOffs += delta;
	cOffsBmk = (cOffsBmk + delta) % bmk_size;
    }
    mappData.current_bitmap_index_ = cOffs;
    mappData.current_mask_offset_ = cOffsBmk;
}


template <uint8_t... Primes>
constexpr
void Bitmap::apply(bitmask_pack<uint64_t, Primes...> const & maskPack)
{
    LFP_LOG_S("bitmap " << obj_addr(this) << ": about to apply pack of bitmasks for primes {"
		        << maskPack.primes_as_string() << "}",
	      "bitmap " << obj_addr(this) << ": pack of bitmasks applied");

    constexpr auto masksCount = maskPack.size();
    // We apply each mask individualy until we reach an offset where they can be applied
    // all together.
    mask_application_data mappData[masksCount];
    [&]<std::size_t... I>(std::index_sequence<I...>){
        ((mappData[I] = compute_mask_application_data(maskPack.template get<I>())),...);
    }(std::make_index_sequence<maskPack.size()>{});

    auto commonStart = std::max_element(std::begin(mappData), std::end(mappData),
        [](auto const & x, auto const & y){
            return x.first_composite_index_ < y.first_composite_index_;
        })->first_composite_index_;

    if(commonStart == nopos_) {
	// At least one prime has no multiple to cross out in the bitmap,
	// in such cases we apply the applicable masks one by one.
	[&]<std::size_t... I>(std::index_sequence<I...>) {
	    ((apply(maskPack.template get<I>(), mappData[I], size())),...);
	}(std::make_index_sequence<masksCount>{});

	return;
    }

    // Apply each mask till commonStart and then perform a combined application of the masks
    commonStart = (commonStart % digits_) ? commonStart + (digits_ - commonStart % digits_) : commonStart;
    [&]<std::size_t... I>(std::index_sequence<I...>){
	((apply(maskPack.template get<I>(), mappData[I], commonStart)),...);
    }(std::make_index_sequence<masksCount>{});

    // Apply all masks combined from commonStart on
    alignas(64) std::size_t offsets[masksCount];
    std::ranges::copy(mappData | std::views::transform([&](auto const & dat){
			      return dat.current_mask_offset_; }), std::begin(offsets));
    auto incOffsetsImpl = [&]<std::size_t... I>(std::index_sequence<I...>){
	(((offsets[I] += digits_ % maskPack.template get<I>().size()),
	(offsets[I] = (offsets[I] >= maskPack.template get<I>().size())
	                 ? offsets[I] - maskPack.template get<I>().size() : offsets[I])),...);
      };
    auto incOffsets = [&](){ incOffsetsImpl(std::make_index_sequence<masksCount>{}); };

    for(std::size_t i = mappData[0].current_bitmap_index_ / digits_; i < vec_.size(); ++i) {
	auto mask = maskPack.combined_masks(offsets);
	vec_[i] &= mask;
	incOffsets();
    }
}

template <typename PrimeT>
constexpr void
Bitmap::apply_and_clear(bucket<PrimeT> & bkt)
{
    apply(bkt);
    bkt.clear();
}


template <typename PrimeT>
constexpr void
Bitmap::apply(bucket<PrimeT> & bkt)
{
    /// @todo: Do we really have to have this static assert? It seems logical because
    /// base primes ought to be smaller than the ones being hunted...
    static_assert(sizeof(PrimeT) <= sizeof(value_type));

    if(bkt.is_empty()) {
	return;
    }

    LFP_LOG("bitmap " << obj_addr(this) << ": about to apply bucket " << obj_addr(&bkt));

    auto const & offsets = bkt.compute_offsets(n0_, ne_, size_);

    LFP_LOG("bitmap " << obj_addr(this) << ": reseting " << offsets.size() << " bit(s)");

    for(auto offsPtr = offsets.data(), offsPtrEnd = offsPtr + offsets.size();
        offsPtr != offsPtrEnd;
	++offsPtr) {
	reset(*offsPtr);
    }

    LFP_LOG("bitmap " << obj_addr(this) << ": bucket " << obj_addr(&bkt) << " applied");
}


struct sieve_data
{
   Bitmap * bitmap_ = nullptr;
   bool have_to_initialize_bitmap_ = true;
   bool have_to_ignore_bucketable_primes_ = false;
};


template <typename T, typename BP, typename U, typename Func>
constexpr auto 
inner_sieve(BP const & basePrimes, U n0, U n1, Func ff, sieve_data const & sievdat)
{
    auto const & tinyPrimes = primesBelowSix<T>;
    if((n0 >= n1) || (n0 > std::numeric_limits<U>::max() - 2)) {
        return ff(std::begin(tinyPrimes), std::begin(tinyPrimes), nullptr);
    }

    auto it0 = std::lower_bound(std::begin(tinyPrimes), std::end(tinyPrimes), n0);
    auto it1 = std::lower_bound(std::begin(tinyPrimes), std::end(tinyPrimes), n1);
    if(it0 != std::end(tinyPrimes)) {
	if(it1 != std::end(tinyPrimes)) {
	    return ff(it0, it1, nullptr);
	} else {
	    if(n1 <= tinyPrimes.back() + 2) {
		return ff(it0, it1, nullptr);
	    }
	    n0 = tinyPrimes.back() + 2;
	}
    }
    n0 = compute_gte_coprime<U>(n0);
    auto ne = compute_lt_coprime<U>(n1);
    if(n0 > ne) {
        return ff(it0, it0, nullptr);
    }
    if(sievdat.have_to_initialize_bitmap_) {
        sievdat.bitmap_->assign(n0, (ne - n0)/30 * 8
			+ ((ne % 30) >= (n0 % 30) 
			  ? (ne % 30) * 4 / 15 - (n0 % 30) * 4 / 15
			  : 8 - (n0 % 30) * 4 / 15 + (ne % 30) * 4 / 15) + 1);
    }
    auto & bmp = *sievdat.bitmap_;

    // Primes below a certain threshold are dealt with by applying precomputed masks to the bitmap
    constexpr unsigned int lastSmallPrime = 103;
    constexpr unsigned int smallPrimesThreshold = lastSmallPrime + 1;
    if(std::ranges::distance(basePrimes 
			    | std::views::drop_while([](auto p) { return p < 7; }) 
			    | std::views::take_while([](auto p) { return p < smallPrimesThreshold; })
			    )) {
        bitmask_pack<typename std::remove_cvref_t<decltype(bmp)>::value_type,
                 7, 11, 13, 17, 19, 23, 29, 31> bitmasks_1;
        bitmask_pack<typename std::remove_cvref_t<decltype(bmp)>::value_type,
                 37, 41, 43, 47, 53, 59, 61, 67> bitmasks_2;
        bitmask_pack<typename std::remove_cvref_t<decltype(bmp)>::value_type,
                 71, 73, 79, 83, 89, 97, 101, lastSmallPrime> bitmasks_3;
        bmp.apply(bitmasks_1);
        bmp.apply(bitmasks_2);
        bmp.apply(bitmasks_3);
    }

    for(auto p : basePrimes
		    | std::views::drop_while([smallPrimesThreshold](auto p) { return p < smallPrimesThreshold; })
		    | std::views::take_while([&sievdat](auto p){
			    return !(sievdat.have_to_ignore_bucketable_primes_ && (p >= bucket_sieve_threshold)); })) {
        auto p2 = U{p} * p;
        if(p2 > ne) {
            break;
        }
        // Early continue if p has no multiple in the current segment. The first condition is to avoid paying the cost of
        // a modulo when there is a higher probability of p having a multiple in the current segment.
        // This doesn't worsen things too much in the lower ranges while improving the performances in the higher ranges.
        if(p > ne - n0) {
                if(auto n0_mod_p = n0 % p; n0_mod_p && (p - n0_mod_p > ne - n0)) {
                continue;
                }
        }
        U c = find_first_multiple_ae(decltype(p2){p}, n0);
        if(!c) {
            continue;
        }
            
        std::size_t startOffs{noffs};
        std::array<std::size_t, 7> offsets;
        auto offsetsCount = compute_offsets(startOffs, offsets, p, n0, ne, c);
        
        if(startOffs == noffs) {
            // No multiple of p greater or equal to  p^2 (and coprime to 30) in the current segment
            continue;
        }
        // Unroll the loop when possible i.e. when we have a full period
	std::size_t i = startOffs;
        if(offsetsCount == offsets.size()) {
            for(; i + 8 * p < bmp.size(); i += 8 * p) {
                bmp.reset(i);
                bmp.reset(i + offsets[0]);
                bmp.reset(i + offsets[1]);
                bmp.reset(i + offsets[2]);
                bmp.reset(i + offsets[3]);
                bmp.reset(i + offsets[4]);
                bmp.reset(i + offsets[5]);
                bmp.reset(i + offsets[6]);
            }
        }
        // Deal with the remaining offsets (or the partial period)
        const auto i0 = i;
        for(auto j = 0; i < bmp.size(); i = i0 + offsets[j], ++j) {
            bmp.reset(i);
            if(j == offsetsCount) {
                break;
            }
        }
    }

    return ff(it0, std::end(tinyPrimes), &bmp);
}


template <typename T, typename It = decltype(std::array<T,3>{}.cbegin())>
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

// Proxy iterator iterating over a Bitmap instance allowing enumeration of the corresponding prime numbers 
template <typename T>
class PrimesIterator
{
public:
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using reference = value_type;
    using const_reference = const reference;
    using pointer = void;
    using iterator_category = std::input_iterator_tag;

    explicit constexpr PrimesIterator(details::Bitmap * bmp, bool isEnd = false);
    constexpr PrimesIterator() =  default;
    constexpr T operator*() const;
    constexpr T operator*();
    constexpr PrimesIterator& operator++();
    constexpr PrimesIterator operator++(int);
    constexpr bool operator==(PrimesIterator const &) const;
    constexpr bool operator!=(PrimesIterator const &) const;

private:
    constexpr void next();

    Bitmap * bmp_;
    std::size_t index_;
    T current_value_;
    std::size_t i_;
    bool is_first_;
};

template <typename T>
constexpr
PrimesIterator<T>::PrimesIterator(details::Bitmap * bmp, bool isEnd)
    : bmp_(bmp)
    , index_(0)
    , current_value_(bmp->n0_)
    , i_(details::Bitmap::indexInResidues(current_value_))
    , is_first_(true)
{
    if(!isEnd) {
	if(bmp_->size()) {
            next();
	}
    } else {
	index_ = bmp_->size();
    }
}

template <typename T>
constexpr void
PrimesIterator<T>::next()
{
    if(!is_first_ && (index_ < bmp_->size())) {
	current_value_ += details::Bitmap::deltas_[i_],
	  i_ = (i_ + 1) % 8, ++index_;
    }
    for(; index_ < bmp_->size();
        current_value_ += details::Bitmap::deltas_[i_],
	  i_ = (i_ + 1) % 8, ++index_) {
	if(bmp_->at(index_)) {
	    is_first_ = false;
	    break;
	}
    }
}

template <typename T>
constexpr
T PrimesIterator<T>::operator*() const
{
    return current_value_;
}

template <typename T>
constexpr
T PrimesIterator<T>::operator*()
{
    return current_value_;
}


template <typename T>
constexpr
PrimesIterator<T> & PrimesIterator<T>::operator++()
{
    next();
    return *this;
}

template <typename T>
constexpr
PrimesIterator<T> PrimesIterator<T>::operator++(int)
{
    PrimesIterator<T> tmp{*this};
    ++(*this);
    return tmp;
}

template <typename T>
constexpr
bool PrimesIterator<T>::operator==(PrimesIterator<T> const & other) const
{
    return (bmp_ == other.bmp_) && (index_ == other.index_);
}

template <typename T>
constexpr
bool PrimesIterator<T>::operator!=(PrimesIterator<T> const & other) const
{
    return !(*this == other);
}

// Iterator wrapper, this exist solely because C++20 does not have std::views::concat.
// It wraps different kind of iterators in order to be able to call std::views::join
template <typename T>
struct IterW
{
    using iterator_category = std::input_iterator_tag;
    using reference = T;
    using const_reference = T;
    using pointer = void;
    using difference_type = std::ptrdiff_t;
    using value_type = T;

    constexpr IterW() = default;
    constexpr explicit IterW(typename std::vector<T>::iterator);
    constexpr explicit IterW(PrimesIterator<T>);
 
    constexpr T operator*();
    constexpr T operator*() const;
    constexpr IterW& operator++();
    constexpr IterW operator++(int);
    constexpr bool operator==(IterW const& other) const;
    constexpr bool operator!=(IterW const& other) const;

private:
    std::variant<typename std::vector<T>::iterator, PrimesIterator<T>> it_;
};


template <typename T>
constexpr
IterW<T>::IterW(typename std::vector<T>::iterator vecIter)
    : it_(vecIter)
{}

template <typename T>
constexpr
IterW<T>::IterW(PrimesIterator<T> myIter)
    : it_(myIter)
{}

template <typename T>
constexpr
T IterW<T>::operator*()
{
    return std::visit([](auto&& z) -> T { return *z; }, it_);
}

template <typename T>
constexpr
T IterW<T>::operator*() const
{
    return std::visit([](auto&& z) -> T { return *z; }, it_);
}

template <typename T>
constexpr
IterW<T>& IterW<T>::operator++()
{
    std::visit([](auto&& z){ ++z; }, it_);
    return *this;
}

template <typename T>
constexpr
IterW<T> IterW<T>::operator++(int)
{
    auto tmp = *this;
    ++(*this);
    return tmp;
}

template <typename T>
constexpr
bool IterW<T>::operator==(IterW<T> const& other) const
{
    return it_ == other.it_;
}

template <typename T>
constexpr
bool IterW<T>::operator!=(IterW<T> const& other) const
{
    return it_ != other.it_;
}

} // namespace details

/// Class holding sieve results
template<typename T>
class SieveResults
{
    std::vector<T> prefix_;
    std::vector<details::Bitmap> bmps_;
    // For the fields below we use lazy initialization
    std::vector<decltype(std::ranges::subrange(details::IterW<T>{}, details::IterW<T>{}))> ranges_;
    decltype(ranges_ | std::views::join | std::views::common)  vranges_;
    bool isRangesInitialized_;
    std::size_t count_;
    constexpr void initRange();

public:
    using range_type = decltype(vranges_);
    constexpr SieveResults(std::vector<T>&& prefix, std::vector<details::Bitmap>&& bitmaps);
    /// Returns a range suitable for iterating over the prime numbers resulting from the sieve
    constexpr auto range();
    /// Implicit cast to a range
    constexpr operator range_type ();
    /// Returns the number of prime numbers found by the sieve.
    constexpr std::size_t count();

    friend constexpr auto begin(SieveResults & rng) {
	rng.initRange();
	return rng.vranges_.begin();
    }
    friend constexpr auto end(SieveResults & rng) {
	rng.initRange();
	return rng.vranges_.end();
    }

    static_assert(std::is_same_v<range_type, decltype(vranges_)>);
};

template <typename T>
constexpr SieveResults<T>::SieveResults(std::vector<T>&& prefix, std::vector<details::Bitmap> && bitmaps)
    : prefix_(std::move(prefix))
    , bmps_(std::move(bitmaps))
    , ranges_()
    , vranges_(ranges_ | std::views::join | std::views::common)
    , isRangesInitialized_(false)
    , count_(0)
{}

template <typename T>
constexpr void SieveResults<T>::initRange()
{
    if(isRangesInitialized_) {
	return;
    }
    ranges_.reserve(bmps_.size() + prefix_.empty() ? 0 : 1);
    if(!prefix_.empty()) {
	ranges_.push_back(std::ranges::subrange(details::IterW<T>{prefix_.begin()},
			details::IterW<T>{prefix_.end()}));
    }
    std::for_each(std::begin(bmps_), std::end(bmps_), [this](auto & bmp) {
        ranges_.push_back(std::ranges::subrange(details::IterW<T>{details::PrimesIterator<T>{&bmp}}, 
			   details::IterW<T>{details::PrimesIterator<T>{&bmp, true}}));
      });
    vranges_ = ranges_ | std::views::join | std::views::common;
    isRangesInitialized_ = true;
}

template <typename T>
constexpr auto SieveResults<T>::range()
{
    initRange();
    return vranges_;
}

template <typename T>
constexpr SieveResults<T>::operator SieveResults<T>::range_type()
{
    return range();
}

template <typename T>
constexpr std::size_t SieveResults<T>::count()
{
    if(count_) {
	return count_;
    }
    count_ = std::distance(std::begin(prefix_), std::end(prefix_)) +
	    std::accumulate(std::begin(bmps_), std::end(bmps_), std::size_t{},
		[](auto x, auto const & bmp) {
		    return x + bmp.popcount();
		});
    return count_;
}


namespace details {

template <typename T>
inline constexpr auto u16primes = []() {
    auto sv = [] {
        Bitmap bmp;
	sieve_data sievdat{.bitmap_ = &bmp};
        return inner_sieve<uint16_t>(
		    u8primes<uint16_t>, uint16_t(0), uint16_t(65535),
		    collectSieveResults<uint16_t>,
		    sievdat);
    };
    std::array<T, sv().size()> u16primesArr;
    std::ranges::copy(sv(), std::begin(u16primesArr));
    return u16primesArr;
  }();


template <typename U>
struct segment
{
    U low_;
    U high_;
    Bitmap * bitmap_;
    bucket<U> * bucket_;
};

template <typename T, typename I, typename Fct>
constexpr auto
sieve(I k0, I k1, Fct ff)
{
    static_assert(is_one_of_v<I, int8_t, uint8_t, int16_t, uint16_t,
            int32_t, uint32_t, int64_t, uint64_t, decltype(0ull)>);

    if constexpr(std::numeric_limits<I>::is_signed) {
        k0 = (std::max)(I{0}, k0);
        k1 = (std::max)(I{0}, k1);
    }
    using U = std::make_unsigned_t<I>;
    const U n0 = U(k0), n1 = U(k1);
    
    LFP_LOG("sieving range [" << n0 << ", " << n1 << "[");

    constexpr U maxn = std::numeric_limits<U>::max();
    constexpr U segmentSize = []() {
        if constexpr (is_one_of_v<U, uint8_t, uint16_t>) {
            return maxn;
        } else { 
            return U{32*1024*1024};
        } }();
    std::vector<T> prefix;
    prefix.reserve(3);
    auto const estimatedNumSegments = (n1 - n0) / segmentSize + 1;
    std::vector<Bitmap> bitmaps;
    bitmaps.reserve(estimatedNumSegments);
    std::vector<bucket<U>> buckets;
    buckets.reserve(estimatedNumSegments);
    std::vector<segment<U>> segments;
    segments.reserve(estimatedNumSegments);
    std::vector<std::uint32_t, allocator<std::uint32_t, 64>> scratchOffsets;
    scratchOffsets.reserve(262144 * 10);
    for(auto a0 = n0, a1 = std::min(n1, (maxn - segmentSize < n0) ? maxn : U(n0 + segmentSize));
        a0 < n1;
        a0 = (maxn - segmentSize < a0) ? maxn : a0 + segmentSize,
        a1 = std::min(n1, maxn - segmentSize < a0 ? maxn : U(a0 + segmentSize))) {
	bitmaps.push_back(Bitmap{});
	buckets.push_back(bucket<U>{scratchOffsets, initial_bucket_capacity});
        segments.emplace_back(segment<U>{
	    .low_ = a0,
	    .high_ = a1,
	    .bitmap_ = &bitmaps.back(),
            .bucket_ = &buckets.back()});
    }
    auto const numSegments = segments.size();

    const U basePrimesRangeSize = [n1](){
        if constexpr (is_one_of_v<U, uint8_t, uint16_t, uint32_t>) {
            return (U{1} << (std::numeric_limits<U>::digits / 2)) - 1;
        } else {
            // Established through tests
            if(n1 >= U{55}<<54) {
                return U{16*1024*1024};
            }
            return (std::min)(U{16*1024*1024},
                        U{1} << ((std::bit_width(n1) + 1)/2));
        } }();
    constexpr auto maxm = (U{1} << (std::numeric_limits<U>::digits / 2)) - 1;

    for(U m0 = 0, m1 = basePrimesRangeSize;
        (m0 < (U{1} << (std::numeric_limits<U>::digits / 2)) - 1) &&  (m0 * m0 <= n1);
        m0 = m1, 
        m1 = (maxm - m1 >= basePrimesRangeSize) ? m1 + basePrimesRangeSize : maxm) {
        constexpr auto basePrimes = []() {
                if constexpr (std::is_same_v<U, uint8_t> || std::is_same_v<U, uint16_t>) {
                    return u8primes<U>;
                } else {
                    return u16primes<uint16_t>;
                }
              }();
        Bitmap basePrimesBmp;
            details::inner_sieve<U>(basePrimes, m0, m1,
                [](auto, auto, details::Bitmap const *){ }, sieve_data{
                  .bitmap_ = &basePrimesBmp,
                          .have_to_initialize_bitmap_ = true
              });
    
        auto pSquared = U{};
        details::PrimesIterator<U> itP{&basePrimesBmp}, itPe{&basePrimesBmp, true};
        for(auto p : std::ranges::subrange(itP, itPe) 
		     | std::views::drop_while([](auto q){ return q < bucket_sieve_threshold; })) {
	    pSquared = p * p;

	    auto highest = segments.back().high_;
	    for(std::size_t i{}; i < numSegments;) {
		uint8_t pmod30{}, kpmod30{};
		auto kp = find_first_multiple_ae_b(p, segments[i].low_, highest, pSquared, pmod30, kpmod30);
		if(!kp) {
		    break;
		}
		LFP_ASSERT(kp < highest);
		i = (kp - n0) / segmentSize;
		LFP_ASSERT(i < numSegments);
		auto & segment = segments[i];
		auto kpmod30Idx = cp30res_to_idx(kpmod30);
		auto indexOfMultipleInBitmap = index_of(kp, kpmod30Idx, compute_gte_coprime<decltype(n0)>(segment.low_));
		segment.bucket_->add(p, indexOfMultipleInBitmap, (cp30res_to_idx(pmod30) << 3) | kpmod30Idx);
		if(kp + 2 * p >= highest) { // @todo: check for overflow!!!
		    break;
		}
		++i;
	    }
        }
        if(pSquared >= n1) {
            break;
        }
    }

    for(auto & currentSegment : segments) {
        details::inner_sieve<T>(u16primes<U>, currentSegment.low_, currentSegment.high_,
            [&](auto it, auto ite, details::Bitmap const*){
                if(it != ite) {
                    prefix = std::vector<T>{it, ite};
                }
                return 0;
            },  sieve_data{ .bitmap_ = currentSegment.bitmap_,
                            .have_to_initialize_bitmap_ = true,
			    .have_to_ignore_bucketable_primes_ = true });
        currentSegment.bitmap_->apply_and_clear(*currentSegment.bucket_);
    }

    return ff(prefix, bitmaps);
}

} // namespace details


inline unsigned int
Threads::defaultCount()
{
    static auto v = []() {
	auto c = std::thread::hardware_concurrency();
	return c ? c : 1;
      }();
    return v;
}

inline Threads::Threads()
    : count_(Threads::defaultCount())
{}

inline Threads::Threads(unsigned int numThreads)
    : count_(numThreads ? numThreads : 1)
{}

inline unsigned int Threads::count() const
{
    return count_;
}

template <typename T, typename U>
constexpr SieveResults<T>
sieve(U n0, U n1)
{
    return details::sieve<T>(n0, n1,
	       [](auto & prefix, std::vector<details::Bitmap> & bitmaps) {
                   return SieveResults<T>{std::move(prefix), std::move(bitmaps)};
               });
}

template <typename T, typename U>
constexpr std::vector<T>
sieve_to_vector(U n0, U n1)
{
    auto res = sieve<T>(n0, n1);
    auto rng = res.range();
    return std::vector<T>(rng.begin(), rng.end());
}


template <typename T, typename U>
SieveResults<T>
sieve(U n0, U n1, Threads const & threads)
{
    if(threads.count() == 1 || (n1 <= n0) || (n1 - n0) <= threads.count()) {
	return sieve<T>(n0, n1);
    }
    auto numThreads = threads.count();
    std::vector<std::future<std::vector<details::Bitmap>>> results;
    std::vector<T> prefix;
    for(auto k = n0, dk = (n1 - n0) / numThreads, ek = (n1 - n0) % numThreads; k < n1; ek = ek ? ek - 1 : 0) {
	auto kmax = k + dk + (ek ? 1 : 0);
	results.emplace_back(std::async(std::launch::async,
		[&prefix](U v0, U v1){
		   return details::sieve<T>(v0, v1,
			[&prefix](auto & pref, std::vector<details::Bitmap> & bmps) {
			    if(!pref.empty()) {
			        prefix = std::move(pref);
			    }
			    return std::move(bmps);
			});
		}, k, kmax));
	k = kmax;
    }
    std::vector<details::Bitmap> bmps =
    std::accumulate(std::begin(results), std::end(results), std::vector<details::Bitmap>{},
        [](auto x, auto & y) {
	    for(auto & b : y.get()) {
	        x.emplace_back(std::move(b));
	    }
	    return std::move(x);
	});
    return SieveResults<T>{std::move(prefix), std::move(bmps)};
}


template <typename U>
constexpr std::size_t count_primes(U n0, U n1)
{
    std::size_t count = 0;
    constexpr auto rangeSize = 24*1024*1024;
    constexpr auto maxn = std::numeric_limits<U>::max();
    for(auto a0 = n0, a1 = std::min(n1, (maxn - rangeSize < n0) ? maxn : n0 + rangeSize);
	a0 < n1;
        a0 = (maxn - rangeSize < a0) ? maxn : a0 + rangeSize, 
	  a1 = std::min(n1, maxn - rangeSize < a0 ? maxn : a0 + rangeSize)) {
	count += sieve<U>(a0, a1).count(); 
    }
    return count; 
}

template <typename U>
std::size_t count_primes(U n0, U n1, Threads const & threads)
{
    if(n0 >= n1) {
	return 0;
    }
    auto numThreads = threads.count();
    if(n1 - n0 < numThreads) {
        return count_primes(n0, n1);
    }
    std::vector<std::future<std::size_t>> results;
    for(auto k = n0, dk = (n1 - n0) / numThreads, ek = (n1 - n0) % numThreads; k < n1; ek = ek ? ek - 1 : 0) {
	auto kmax = k + dk + (ek ? 1 : 0);
	results.emplace_back(std::async(std::launch::async, static_cast<std::size_t(*)(U,U)>(count_primes<U>), k, kmax));
	k = kmax;
    }
    return std::accumulate(std::begin(results), std::end(results),
		    std::size_t{}, [](auto x, auto & y) { return x + y.get(); });
}

} // namespace lfp


