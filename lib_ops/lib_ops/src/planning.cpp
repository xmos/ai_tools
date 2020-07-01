// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/planning.h"

#include "lib_ops/api/dispatcher.h"

namespace xcore {

//*****************************
//*****************************
//*****************************
// RowColRegionArray
//*****************************
//*****************************
//*****************************

RowColRegionArray::RowColRegionArray()
    : next_(0), size_(0), regions_(nullptr) {}

void RowColRegionArray::Init(size_t size) {
  assert(regions_ == nullptr);
  Dispatcher *dispatcher = GetDispatcher();

  size_ = size;
  regions_ = reinterpret_cast<RowColRegion *>(
      GetDispatcher()->AllocatePersistantBuffer(sizeof(RowColRegion) * size_));
}

const RowColRegion &RowColRegionArray::operator[](int i) {
  assert(i < size_);
  return regions_[i];
}

void RowColRegionArray::Append(const RowColRegion &region) {
  assert(next_ < size_);
  regions_[next_] = std::move(region);
  next_++;
}

size_t RowColRegionArray::GetSize() { return next_; }

//*****************************
//*****************************
//*****************************
// ChannelGroupArray
//*****************************
//*****************************
//*****************************

ChannelGroupArray::ChannelGroupArray()
    : next_(0), size_(0), chan_groups_(nullptr) {}

void ChannelGroupArray::Init(size_t size) {
  assert(chan_groups_ == nullptr);
  Dispatcher *dispatcher = GetDispatcher();

  size_ = size;
  chan_groups_ = reinterpret_cast<ChannelGroup *>(
      GetDispatcher()->AllocatePersistantBuffer(sizeof(ChannelGroup) * size_));
}

const ChannelGroup &ChannelGroupArray::operator[](int i) {
  assert(i < GetSize());

  return chan_groups_[i];
}

void ChannelGroupArray::Append(const ChannelGroup &changrp) {
  assert(next_ < size_);
  chan_groups_[next_] = std::move(changrp);
  next_++;
}

size_t ChannelGroupArray::GetSize() { return next_; }

}  // namespace xcore