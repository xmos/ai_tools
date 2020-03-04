

PLATFORM_NAME = x86

PLATFORM_FLAGS_DEFAULT := 
PLATFORM_INCLUDES :=

ifeq ($(OS),Windows_NT)
  PLATFORM_EXE_SUFFIX = .exe
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

LD_FLAGS  := -L/usr/local/lib -lstdc++ -lm