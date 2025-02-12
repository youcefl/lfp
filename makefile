# MIT License
# Copyright (c) Youcef Lemsafer
# See LICENSE file for more details.
# Creation date: december 2024.

CXX ?= g++
CXFLAGS ?= -std=c++20 -O3 -march=native

lfp: lfp.cpp
	$(CXX) $(CXFLAGS) -o lfp lfp.cpp

clean:
	rm -f lfp

