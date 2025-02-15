# MIT License
# Copyright (c) Youcef Lemsafer
# See LICENSE file for more details.
# Creation date: december 2024.

CXX ?= g++
CXFLAGS ?= -std=c++20 -O3 -march=native
TEST_LIBS ?= -lCatch2Main -lCatch2
TEST_LIB_INSTALL ?= ./Catch2-install
TEST_LIB_INCL = $(TEST_LIB_INSTALL)/include
TEST_LIB_INCL_OPT ?= -I$(TEST_LIB_INCL)
TEST_LIB_LIB = $(TEST_LIB_INSTALL)/lib
TEST_LIB_LIB_OPT ?= -L$(TEST_LIB_LIB)

OBJS=static_tests.o

all: $(OBJS) lfp tests

lfp: main.cpp lfp.hpp
	$(CXX) $(CXFLAGS) -o lfp main.cpp

static_tests.o: static_tests.cpp lfp.hpp
	$(CXX) $(CXFLAGS) -c static_tests.cpp

tests: tests.cpp lfp.hpp
	$(CXX) $(CXFLAGS) $(TEST_LIB_INCL_OPT) -o tests tests.cpp $(TEST_LIB_LIB_OPT)  $(TEST_LIBS)

clean:
	rm -f lfp tests
	rm -f $(OBJS)

