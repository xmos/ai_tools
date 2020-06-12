// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_OPERATORS_PLANNING_H_
#define XCORE_OPERATORS_PLANNING_H_
#include <cassert>
#include <cstdint>
#include <iostream>

//#include "nn_op_utils.h"

#define CHANNEL_GROUP_LENGTH 16
#define CHANNEL_GROUP_LENGTH_LOG2 4
#define BSO_CHANNEL_GROUP_LENGTH (7 * 16)
#define BSO_CHANNEL_GROUP_BYTES (BSO_CHANNEL_GROUP_LENGTH * 2)

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
  RowColRegionArray() : size_(0) {}
  RowColRegion regions[max_threads];
  const RowColRegion& operator[](int i) {
    assert(i < size_);
    return regions[i];
  }
  void Append(const RowColRegion& region) {
    assert(size_ < max_threads);
    regions[size_] = std::move(region);
    size_++;
  }
  int32_t GetSize() { return size_; }
  void Clear() { size_ = 0; }

 private:
  int32_t size_;
};

typedef struct ChannelGroup {
  int32_t start;
  ;
  int32_t size;
} ChannelGroup;

class ChannelGroupArray {
 public:
  ChannelGroupArray() : n_chans_(0) {}
  const ChannelGroup& operator[](int i) {
    assert(i < GetSize());
    chan_group_.start = i * CHANNEL_GROUP_LENGTH;
    if ((chan_group_.start + CHANNEL_GROUP_LENGTH) <= n_chans_)
      chan_group_.size = CHANNEL_GROUP_LENGTH;
    else
      chan_group_.size = n_chans_ - chan_group_.start;
    return chan_group_;
  }

  void SetNumChannels(int32_t chans) { n_chans_ = chans; }
  int32_t GetSize() {
    return (n_chans_ + CHANNEL_GROUP_LENGTH - 1) >> CHANNEL_GROUP_LENGTH_LOG2;
  }

 private:
  int32_t n_chans_;
  ChannelGroup chan_group_;
};

enum ExecutionPlanType {
  NONE = 0,
  ROWCOL = 1,
  CHANGRP = 2,
  CHANGRP_ROWCOL = 3
};

class ExecutionPlan {
 public:
  ExecutionPlan() : type_(NONE), n_threads_(0), n_channels_(0) {}
  ~ExecutionPlan() {}

  void SetType(ExecutionPlanType type) { type_ = type; }

  void SetNumChannels(int32_t n_channels) {
    chan_groups.SetNumChannels(n_channels);
  }
  // int32_t GetNumChannels() { return n_channels_; }

  void SetNumThreads(int32_t n_threads) { n_threads_ = n_threads; }
  int32_t GetNumThreads() { return n_threads_; }

  int32_t GetNumJobs() { return chan_groups.GetSize() * regions.GetSize(); }

  RowColRegionArray regions;
  ChannelGroupArray chan_groups;

 private:
  ExecutionPlanType type_;
  int32_t n_threads_;
  int32_t n_channels_;
};

}  // namespace xcore

#endif  // XCORE_OPERATORS_PLANNING_H_