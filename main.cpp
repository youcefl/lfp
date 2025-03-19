/*
 * MIT License
 * Copyright (c) 2025 Youcef Lemsafer
 * See LICENSE file for more details.
 * Creation date: February 2025.
 */
#include "lfp.hpp"

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
    uint64_t n0, n1;
    std::istringstream istr0{argv[n0idx]};
    istr0 >> n0;
    std::istringstream istr1{argv[n1idx]};
    istr1 >> n1;

    auto const startt = std::chrono::steady_clock::now();
    auto numPrimes = lfp::threaded_count_primes(numThreads, n0, n1);
    auto const endt = std::chrono::steady_clock::now();

    std::cout << "The number of prime numbers in range [" << n0 << ", " << n1 << "[ is "
	   << numPrimes << "." << std::endl;
    std::cout << "Took " << std::chrono::duration<double>(endt - startt) << std::endl;
    return 0;
}

