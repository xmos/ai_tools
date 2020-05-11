

PLATFORM_NAME = x86

PLATFORM_FLAGS_DEFAULT := 
PLATFORM_INCLUDES :=

ifeq ($(OS),Windows_NT)
  ifeq ($(findstring windows32,$(shell uname -s)),windows32)
    PLATFORM_EXE_SUFFIX = .a
  else
    PLATFORM_EXE_SUFFIX = .exe
  endif
else
  PLATFORM_EXE_SUFFIX = 
endif


PLATFORM_FLAGS := $(PLATFORM_FLAGS_DEFAULT)

CC := gcc
XCC := gcc
CXX := g++

CC_FLAGS  := -g -O3
XCC_FLAGS := -g -O3
CXX_FLAGS := -g -O3 -std=c++11

LD_FLAGS  := -L/usr/local/lib -lm