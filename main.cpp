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
    std::cerr << "Usage:\n\tlfp [-t num_threads] n0 n1" << std::endl;
}

} // namespace lfp


int main(int argc, char** argv)
{
#if LFP_HAS_UINT128
    using lfp::operator<<;
#endif
    if((argc != 3) && (argc != 5)) {
	lfp::displayUsage();
	return 1;
    }
    int n0idx = 1, n1idx = 2;
    int32_t numThreads{};
    if(argc == 5) {
	if(argv[1] != std::string{"-t"}) {
            lfp::displayUsage();
	    return 1;
	}
	std::istringstream istr{argv[2]};
	istr >> numThreads;
	n0idx += 2;
	n1idx += 2;
    }

    auto n0 = lfp::string_to_uint(argv[n0idx]);
    auto n1 = lfp::string_to_uint(argv[n1idx]);
    auto nmax = std::max(n0, n1);

    auto const startt = std::chrono::steady_clock::now();
    lfp::threads threadz{numThreads < 0 ? 1u : (unsigned int)numThreads};
    auto numPrimes = (nmax <= std::numeric_limits<std::uint32_t>::max()) 
	               ? lfp::count_primes(uint32_t(n0), uint32_t(n1), threadz)
#if LFP_HAS_UINT128
		       : (nmax <= std::numeric_limits<std::uint64_t>::max()) 
		       ? lfp::count_primes(uint64_t(n0), uint64_t(n1), threadz)
#endif
		       : lfp::count_primes(n0, n1, threadz)
		       ;
    auto const endt = std::chrono::steady_clock::now();

    std::cout << "The number of prime numbers in range [" << n0 << ", " << n1 << "[ is "
	   << numPrimes << "." << std::endl;
    std::cout << "Took " << std::chrono::duration<double>(endt - startt) << std::endl;


    return 0;
}

