// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_OPERATORS_PLANNING_H_
#define XCORE_OPERATORS_PLANNING_H_
#include <cassert>
#include <cstdint>
#include <iostream>

namespace xcore {

typedef struct RowColRegion {
  int32_t top;
  int32_t left;
  int32_t rows;
  int32_t cols;
} RowColRegion;

class RowColRegionArray {
 public:
  RowColRegionArray();
  void Init(size_t size);
  const RowColRegion &operator[](int i);
  void Append(const RowColRegion &region);

  size_t GetSize();

 private:
  int32_t next_;
  int32_t size_;
  RowColRegion *regions_;
};

typedef struct ChannelGroup {
  int32_t index;
  int32_t start;
  int32_t size;
} ChannelGroup;

class ChannelGroupArray {
 public:
  ChannelGroupArray();
  void Init(size_t size);
  const ChannelGroup &operator[](int i);
  void Append(const ChannelGroup &changrp);
  size_t GetSize();

 private:
  int32_t next_;
  int32_t size_;
  ChannelGroup *chan_groups_;
};

class ExecutionPlan {
 public:
  ExecutionPlan() : n_threads_(0) {}
  ~ExecutionPlan() {}

  void SetNumThreads(int32_t n_threads) { n_threads_ = n_threads; }
  int32_t GetNumThreads() { return n_threads_; }

  RowColRegionArray regions;
  ChannelGroupArray changrps;

 private:
  int32_t n_threads_;
};

}  // namespace xcore

#endif  // XCORE_OPERATORS_PLANNING_H_