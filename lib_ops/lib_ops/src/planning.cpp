// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/planning.h"

namespace xcore {

constexpr size_t changrp_len = 16;
constexpr size_t changrp_len_log2 = 4;

//*****************************
//*****************************
//*****************************
// RowColRegionArray
//*****************************
//*****************************
//*****************************

const RowColRegion &RowColRegionArray::operator[](int i) {
  assert(i < size_);
  return regions[i];
}

void RowColRegionArray::Append(const RowColRegion &region) {
  assert(size_ < max_regions);
  regions[size_] = std::move(region);
  size_++;
}

//*****************************
//*****************************
//*****************************
// ChannelGroupArray
//*****************************
//*****************************
//*****************************

const ChannelGroup &ChannelGroupArray::operator[](int i) {
  assert(i < GetSize());
  chan_group_.index = i;
  chan_group_.start = i * changrp_len;
  if ((chan_group_.start + changrp_len) <= n_chans_)
    chan_group_.size = changrp_len;
  else
    chan_group_.size = n_chans_ - chan_group_.start;
  return chan_group_;
}

int32_t ChannelGroupArray::GetSize() {
  return (n_chans_ + changrp_len - 1) >> changrp_len_log2;
}

//*****************************
//*****************************
//*****************************
// ExecutionPlan
//*****************************
//*****************************
//*****************************

int32_t ExecutionPlan::GetNumJobs() {
  return changrps.GetSize() * regions.GetSize();
}

}  // namespace xcore