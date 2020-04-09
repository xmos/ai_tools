// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_PAR_H_
#define XCORE_PAR_H_
#include <cassert>
#include <cstdint>
#include <iostream>

namespace xcore {

constexpr size_t maxthreads = 5;

typedef struct ParRegion {
  int32_t top;
  int32_t left;
  int32_t rows;
  int32_t cols;
} ParRegion;

class ParRegionArray {
 public:
  ParRegionArray() : size(0) {}
  int size;
  ParRegion regions[maxthreads];
  const ParRegion& operator[](int i) {
    assert(i < size);
    return regions[i];
  }
  void append(const ParRegion& region) {
    assert(size < maxthreads);
    regions[size] = std::move(region);
    size++;
  }
  void clear() { size = 0; }
};

}  // namespace xcore

#endif  // XCORE_PAR_H_