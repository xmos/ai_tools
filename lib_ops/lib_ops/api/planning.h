// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_OPERATORS_PLANNING_H_
#define XCORE_OPERATORS_PLANNING_H_
#include <cassert>
#include <cstdint>
#include <iostream>

namespace xcore {

constexpr size_t max_regions = 5;

typedef struct RowColRegion {
  int32_t top;
  int32_t left;
  int32_t rows;
  int32_t cols;
} RowColRegion;

class RowColRegionArray {
 public:
  RowColRegionArray() : size_(0) {}
  const RowColRegion &operator[](int i);
  void Append(const RowColRegion &region);

  int32_t GetSize() { return size_; }
  // void Clear() { size_ = 0; }

  RowColRegion regions[max_regions];

 private:
  int32_t size_;
};

typedef struct ChannelGroup {
  int32_t index;
  int32_t start;
  int32_t size;
} ChannelGroup;

class ChannelGroupArray {
 public:
  ChannelGroupArray() : n_chans_(0) {}
  const ChannelGroup &operator[](int i);
  void SetNumChannels(int32_t chans) { n_chans_ = chans; }
  int32_t GetSize();

 private:
  int32_t n_chans_;
  ChannelGroup chan_group_;
};

class ExecutionPlan {
 public:
  ExecutionPlan() : n_threads_(0) {}
  ~ExecutionPlan() {}

  void SetNumThreads(int32_t n_threads) { n_threads_ = n_threads; }
  int32_t GetNumThreads() { return n_threads_; }

  int32_t GetNumJobs();

  RowColRegionArray regions;
  ChannelGroupArray changrps;

 private:
  int32_t n_threads_;
};

}  // namespace xcore

#endif  // XCORE_OPERATORS_PLANNING_H_