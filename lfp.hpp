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
#include <variant>


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


namespace details {

template <typename T>
inline constexpr auto u8primes = std::to_array<T>({
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251
});


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


using use_dictionary_t = bool;

template <typename U, std::size_t P, use_dictionary_t UseDictionary = P < 32, std::size_t Alignment = 64>
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
    static constexpr U word_at(U * data, std::size_t idx);

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
    return offs_[(c % 30) * 4 / 15];
}


template <typename U, std::size_t P, use_dictionary_t UseDictionary, std::size_t Alignment>
constexpr U
bitmask_impl<U, P, UseDictionary, Alignment>::word_at(U * data, std::size_t idx) {
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
        offsets[(i % 30) * 4 / 15] = j;
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
    /// Returns the number of bitmask objects in this pack
    static constexpr auto size();
    /// Returns a tuple containing all bitmask objects in this pack
    constexpr auto const & get_all() const;
    /// Gets the bitmask object at index I
    template <std::size_t I>
    constexpr auto const & get() const;
    /// Returns the largest prime in the SmallPrimes
    static constexpr std::uint8_t max_prime();
    /// Returns the combination of the masks at the given offsets
    template <std::size_t N>
    constexpr U combined_masks(std::size_t const (&offsets) [N]) const;

private:
    using primes_sequence = std::integer_sequence<std::size_t, SmallPrimes...>;

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
};


template <typename U, uint8_t... SmallPrimes>
constexpr auto bitmask_pack<U, SmallPrimes...>::size()
{
    return sizeof...(SmallPrimes);
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

template <typename U, typename V>
constexpr auto
find_first_multiple_above(U p, V n0)
{
    const auto p2 = p * p;
    constexpr auto c_max = std::numeric_limits<decltype(p2)>::max();
    U dp2;
    auto c = (p2 >= n0) ? p2 : (((dp2 = (n0 - p2 + 2 * p - 1)/(2 * p)*(2 * p)), c_max - p2 < dp2) ? 0 : p2 + dp2);
    if(!c) {
        return c;
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
    return c;
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
    constexpr void reset(std::size_t index);
    template <std::size_t Prime>
    constexpr void apply(bitmask_impl<uint64_t, Prime> const & mask);
    template <uint8_t... Primes>
    constexpr void apply(bitmask_pack<uint64_t, Primes...> const & maskPack);
    constexpr uint64_t popcount() const;
    void check() const;
    constexpr uint8_t at(std::size_t index) const;
    template <typename Func>
    constexpr void foreach_setbit(Func ff) const;

private:
    template <typename Int> static constexpr std::size_t indexInResidues(Int);
    
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

    std::vector<uint64_t> vec_;
    std::size_t size_;
    uint64_t n0_;
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
{}

constexpr
Bitmap::Bitmap(uint64_t n0, std::size_t size)
  : vec_((size + NumLim::digits - 1)/NumLim::digits,
            ~decltype(vec_)::value_type{})
  , size_(size)
  , n0_(n0 + d_[n0 % 30])
{
    if(size % NumLim::digits) {
        vec_.back() &= ~ElemType{} << (NumLim::digits - size % NumLim::digits);
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
        vec_.back() &= ~ElemType{} << (NumLim::digits - size % NumLim::digits);
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
    auto c = find_first_multiple_above(p, n0_);
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
    mappData.current_bitmap_index_ = 0;
    return mappData;
}


template <std::size_t Prime>
constexpr
void Bitmap::apply(bitmask_impl<uint64_t, Prime> const & bmk)
{
    auto mappData = compute_mask_application_data(bmk);
    apply(bmk, mappData, size());
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
	(offsets[I] = (offsets[I] >= maskPack.template get<I>().size()) ? offsets[I] - maskPack.template get<I>().size() : offsets[I])),...);
      };
    auto incOffsets = [&](){ incOffsetsImpl(std::make_index_sequence<std::size(offsets)>{}); };
    for(std::size_t i = mappData[0].current_bitmap_index_ / digits_; i < vec_.size(); ++i) {
	auto mask = maskPack.combined_masks(offsets);
	auto offs0 = offsets[0];
	incOffsets();
	vec_[i] &= mask;
    }
}


inline constexpr uint8_t wheel[8][8] = {
    {6,4,2,4,2,4,6,2},
    {4,2,4,2,4,6,2,6},
    {2,4,2,4,6,2,6,4},
    {4,2,4,6,2,6,4,2},
    {2,4,6,2,6,4,2,4},
    {4,6,2,6,4,2,4,2},
    {6,2,6,4,2,4,2,4},
    {2,6,4,2,4,2,4,6}
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


// Returns the largest ineteger coprime to 30 and less than @param n1
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

template <typename T>
inline constexpr std::array<T,3> primesBelowSix = {2, 3, 5};

template <typename T, typename SP, typename U, typename Func>
constexpr auto 
inner_sieve(SP const & smallPrimes, U n0, U n1, Func ff, Bitmap & bmp, bool initBmp = true)
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
    auto ne = compute_lt_coprime<std::size_t>(n1);
    if(n0 > ne) {
        return ff(it0, it0, nullptr);
    }
    if(initBmp) {
        bmp.assign(n0, (ne - n0)/30 * 8 + ((ne % 30) >= (n0 % 30) ? (ne%30)*4/15 - (n0%30)*4/15 : 8 - (n0%30)*4/15 + (ne%30)*4/15) + 1);
    }

    // Primes below a certain threshold are dealt with by applying precomputed masks to the bitmap
    constexpr unsigned int lastSmallPrime = 31;
    constexpr unsigned int smallPrimesThreshold = lastSmallPrime + 1;
    bitmask_pack<std::remove_cvref_t<decltype(bmp)>::value_type,
	         7, 11, 13, 17, 19, 23, 29, lastSmallPrime> bitmasks;
    bmp.apply(bitmasks);

    for(auto p : smallPrimes | std::views::drop_while([smallPrimesThreshold](auto p) { return p < smallPrimesThreshold; })) {
	auto p2 = U{p} * p;
	if(p2 > ne) {
	    break;
	}
	auto c = find_first_multiple_above(decltype(p2){p}, n0);
	if(!c) {
	    continue;
	}
        
	constexpr auto c_max = std::numeric_limits<decltype(p2)>::max();
	auto count = 0;
	int32_t firstIndex = -1;
	std::array<int,7> offsets{};
        auto offsIdx = 0;
	auto prevDelta = 0;
	for(auto j = whoffs[(p%30)*4/15][(c%30)*4/15]; c <= ne;
	    c = ((c_max - c < wheel[(p%30)*4/15][j]*p) ? ne + 1 : c + wheel[(p%30)*4/15][j]*p), j = (j+1)%8) {
	    auto currIdx = bmp.indexOf(c);
	    if(offsIdx == 0) {
		firstIndex = currIdx;
	    }

	    if(offsIdx > 0 && offsIdx <= offsets.size()) {
		offsets[offsIdx - 1] = currIdx - firstIndex;
	    }
            ++offsIdx;
	    if(offsIdx > offsets.size()) {
		break;
	    }
	}
	if((firstIndex < 0) || (firstIndex >= bmp.size())) {
	    continue;
	}
	std::size_t i = firstIndex;
	if(offsets[6]) {
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
	const auto i0 = i;
	for(auto j = 0; i < bmp.size(); i = i0 + offsets[j], ++j) {
	    bmp.reset(i);
            if((j == offsets.size()) || !offsets[j]) {
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
        return inner_sieve<uint16_t>(
		    u8primes<uint16_t>, uint16_t(0), uint16_t(65535),
		    collectSieveResults<uint16_t>,
		    bmp);
    };
    std::array<T, sv().size()> u16primesArr;
    std::ranges::copy(sv(), std::begin(u16primesArr));
    return u16primesArr;
  }();


template <typename T, typename... Rest>
inline constexpr bool is_one_of_v = (std::is_same_v<T, Rest> || ...);


template <typename T, typename I, typename Fct>
constexpr auto
sieve(I k0, I k1, Fct ff)
{
    static_assert(is_one_of_v<I, int8_t, uint8_t, int16_t, uint16_t,
		    int32_t, uint32_t, int64_t, uint64_t>);
    if constexpr(std::numeric_limits<I>::is_signed) {
	k0 = (std::max)(I{0}, k0);
	k1 = (std::max)(I{0}, k1);
    }
    using U = std::make_unsigned_t<I>;
    const U n0 = U(k0), n1 = U(k1);

    constexpr U maxn = std::numeric_limits<U>::max();
    constexpr U innerRangeSize = []() {
		if constexpr (is_one_of_v<U, uint8_t, uint16_t>) {
		    return maxn;
		} else { 
		    return U{48*1024*1024};
		} }();
    std::vector<details::Bitmap> bitmaps;
    bitmaps.reserve((n1 - n0)/innerRangeSize + 1);
    std::vector<T> prefix;
    prefix.reserve(3);
 

    for(auto a0 = n0, a1 = std::min(n1, (maxn - innerRangeSize < n0) ? maxn : U(n0 + innerRangeSize));
        a0 < n1;
        a0 = (maxn - innerRangeSize < a0) ? maxn : a0 + innerRangeSize,
	a1 = std::min(n1, maxn - innerRangeSize < a0 ? maxn : U(a0 + innerRangeSize))) {

        const U rangeSize = [n1](){
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
	 bitmaps.emplace_back(details::Bitmap{});
	 auto & currSegBmp = bitmaps.back();

         for(U m0 = 0, m1 = rangeSize;
             (m0 < (U{1} << (std::numeric_limits<U>::digits / 2)) - 1) &&  (m0 * m0 <= a1);
	     m0 = m1, 
	       m1 = (maxm - m1 >= rangeSize) ? m1 + rangeSize : maxm) {
	     details::Bitmap primesBmp;
	     constexpr auto basePrimes = []() {
		   if constexpr (std::is_same_v<U, uint8_t> || std::is_same_v<U, uint16_t>) {
		       return u8primes<U>;
		   } else {
		       return u16primes<uint16_t>;
		   }
		}();
	     details::inner_sieve<U>(basePrimes, m0, m1,
	        [](auto, auto, details::Bitmap const *){ }, primesBmp);
	     details::PrimesIterator<U> itP{&primesBmp}, itPe{&primesBmp, true};
	     auto basePrimesRange = std::ranges::subrange(itP, itPe);
	     details::inner_sieve<T>(basePrimesRange, a0, a1,
	             [&](auto it, auto ite, details::Bitmap const*){
		         if(it != ite) {
			     prefix = std::vector<T>{it, ite};
			 }
		         return 0;
		     },  currSegBmp, m0 == 0);
        }
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



