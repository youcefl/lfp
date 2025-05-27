/*
 * MIT License
 * Copyright (c) 2025 Youcef Lemsafer
 * See LICENSE file for more details.
 * Creation date: February 2025.
 */
#include "lfp.hpp"

namespace lfp {

auto string_to_uint(char const * str)
{
#if LFP_HAS_UINT128
    return lfp::uint128_from_string(str);
#else
    std::istringstream istr{str};
    uint64_t res;
    istr >> res;
    return res;
#endif
}

void displayUsage()
{
    std::cerr << "Usage:\n\tlfp [(-t|--threads) num_threads] [-p|--primes] n0 n1" << std::endl;
}

} // namespace lfp


int main(int argc, char** argv)
{
#if LFP_HAS_UINT128
    using lfp::operator<<;
#endif
    if((argc < 3) || (argc > 6)) {
	lfp::displayUsage();
	return 1;
    }
    int32_t numThreads{};
    bool haveToOutputPrimes = false;
    int optionIdx = 1;
    if(argc >= 4) {
	if((argv[optionIdx] == std::string{"-t"}) || (argv[optionIdx] == std::string{"--threads"})) {
	    std::istringstream istr{argv[optionIdx + 1]};
	    istr >> numThreads;
	    optionIdx += 2;
	}
	if((argv[optionIdx] == std::string{"-p"}) || (argv[optionIdx] == std::string{"--primes"})) {
            haveToOutputPrimes = true;
	    optionIdx += 1;
        }
    }

    auto n0 = lfp::string_to_uint(argv[optionIdx]);
    auto n1 = lfp::string_to_uint(argv[optionIdx + 1]);
    auto nmax = std::max(n0, n1);
    using sieve_res_t = std::variant<lfp::sieve_results<uint32_t, uint32_t>
	                            , lfp::sieve_results<uint64_t, uint64_t>
#if LFP_HAS_UINT128
				    , lfp::sieve_results<lfp::uint128_t, lfp::uint128_t>
#endif
                                    >;
    auto sieve = [haveToOutputPrimes]<typename UInt>(UInt u0, UInt u1, lfp::threads & threadz){
        if(haveToOutputPrimes) {
            auto results = lfp::sieve<UInt>(u0, u1, threadz);
	    auto primesCount = results.count();
	    return std::make_pair(primesCount, std::optional<sieve_res_t>{std::move(results)});
	} else {
	    return std::make_pair(lfp::count_primes(u0, u1, threadz), std::optional<sieve_res_t>{});
	}
      };


    auto const startt = std::chrono::steady_clock::now();
    lfp::threads threadz{numThreads < 0 ? 1u : (unsigned int)numThreads};
    auto results = (nmax <= std::numeric_limits<std::uint32_t>::max())
	           ? sieve(uint32_t(n0), uint32_t(n1), threadz)
#if LFP_HAS_UINT128
		   : (nmax <= std::numeric_limits<std::uint64_t>::max())
		     ? sieve(uint64_t(n0), uint64_t(n1), threadz)
#endif
		       : sieve(n0, n1, threadz)
		       ;
    auto const endt = std::chrono::steady_clock::now();

    std::cout << "The number of prime numbers in range [" << n0 << ", " << n1 << "[ is "
	   << results.first << "." << std::endl;
    std::cout << "Took " << std::chrono::duration<double>(endt - startt) << std::endl;

    if(haveToOutputPrimes) {
	std::cout << "Primes:" << std::endl;
	std::visit([](auto && sieveRes){
	    for(auto p : sieveRes) {
	        std::cout << p << '\n';
            }
	  }, results.second.value());
    }

    return 0;
}

