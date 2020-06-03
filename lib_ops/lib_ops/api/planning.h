// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_OPERATORS_PLANNING_H_
#define XCORE_OPERATORS_PLANNING_H_
#include <cassert>
#include <cstdint>
#include <iostream>

namespace xcore {

constexpr size_t max_threads = 5;

typedef struct RowColRegion {
  int32_t top;
  int32_t left;
  int32_t rows;
  int32_t cols;
} RowColRegion;

class RowColRegionArray {
 public:
  RowColRegionArray() : size(0) {}
  int size;
  RowColRegion regions[max_threads];
  const RowColRegion& operator[](int i) {
    assert(i < size);
    return regions[i];
  }
  void append(const RowColRegion& region) {
    assert(size < max_threads);
    regions[size] = std::move(region);
    size++;
  }
  void clear() { size = 0; }
};

enum ExecutionPlanType { ROWCOL = 1, CHANGRP = 2, CHANGRP_ROWCOL = 3 };

typedef struct ExecutionPlan {
  ExecutionPlanType type;
  int32_t threads;
  RowColRegionArray regions;
} ExecutionPlan;

}  // namespace xcore

#endif  // XCORE_OPERATORS_PLANNING_H_