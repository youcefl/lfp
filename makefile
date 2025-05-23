# MIT License
# Copyright (c) Youcef Lemsafer
# See LICENSE file for more details.
# Creation date: december 2024.

CXX ?= g++
CXFLAGS ?= -std=c++20 -O3 -march=native -DNDEBUG $(ADDITIONAL_CXFLAGS)
STATIC_TESTS_CXFLAGS ?=
# By default we activate assertions during testing
DYNAMIC_TESTS_FLAGS ?= -UNDEBUG
SYNTAX_ONLY_FLAG=-fsyntax-only
TEST_LIBS ?= -lCatch2Main -lCatch2
TEST_LIB_INSTALL ?= ./Catch2-install
TEST_LIB_INCL = $(TEST_LIB_INSTALL)/include
TEST_LIB_INCL_OPT ?= -I$(TEST_LIB_INCL)
TEST_LIB_LIB = $(TEST_LIB_INSTALL)/lib
TEST_LIB_LIB_OPT ?= -L$(TEST_LIB_LIB)


lfp: main.cpp lfp.hpp
	$(CXX) $(CXFLAGS) -o $@ main.cpp

static_tests: static_tests_1 static_tests_2 static_tests_internals
	

static_tests_%: static_tests_%.cpp lfp.hpp
	$(CXX) $(CXFLAGS) $(SYNTAX_ONLY_FLAG) $(STATIC_TESTS_CXFLAGS) $<

tests: tests.cpp lfp.hpp lfp_tests.hpp
	$(CXX) $(CXFLAGS) $(DYNAMIC_TESTS_FLAGS) $(TEST_LIB_INCL_OPT) -o $@ $< $(TEST_LIB_LIB_OPT)  $(TEST_LIBS)

tests_internals: tests_internals.cpp lfp.hpp lfp_tests.hpp
	$(CXX) $(CXFLAGS) $(DYNAMIC_TESTS_FLAGS) $(TEST_LIB_INCL_OPT) -o $@ $< $(TEST_LIB_LIB_OPT)  $(TEST_LIBS)

all: static_tests lfp tests tests_internals

clean:
	rm -f lfp tests

