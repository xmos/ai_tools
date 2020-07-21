// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/allocator.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>

#include "lib_ops/src/xcore_reporter.h"

void MemoryAllocator::SetHeap(void *buffer, size_t size) {
  assert(buffer);
  assert(size > 0);

  buffer_ = buffer;
  buffer_size_ = size;

  ResetHeap();
}

void MemoryAllocator::ResetHeap() {
  alloc_tail_ = std::align(WORD_ALIGNMENT, 1, buffer_, buffer_size_);
  scratch_head_ = alloc_tail_;
}

size_t MemoryAllocator::GetSize() { return buffer_size_; }

size_t MemoryAllocator::GetAllocatedSize() {
  return ((uintptr_t)alloc_tail_ - (uintptr_t)buffer_);
}

size_t MemoryAllocator::GetFreeSize() {
  return buffer_size_ - GetAllocatedSize();
}

void *MemoryAllocator::AllocatePersistantBuffer(size_t size, size_t alignment) {
  assert(size > 0);

  void *ptr = AllocateBuffer(size, alignment);

  if (ptr) scratch_head_ = alloc_tail_;
  return ptr;
}

void *MemoryAllocator::AllocateScratchBuffer(size_t size, size_t alignment) {
  assert(size > 0);

  void *ptr = AllocateBuffer(size, alignment);
  return ptr;
}

void MemoryAllocator::ResetScratch() { alloc_tail_ = scratch_head_; }

void *MemoryAllocator::GetScratchBuffer() { return scratch_head_; }

void *MemoryAllocator::AllocateBuffer(size_t size, size_t alignment) {
  alloc_tail_ = std::align(alignment, size, alloc_tail_, buffer_size_);

  if (GetFreeSize() >= size) {
    void *ptr = alloc_tail_;
    alloc_tail_ = (void *)((uintptr_t)alloc_tail_ + size);
    return ptr;
  }
  // Allocator is out of memory for this allocation
  // TF_LITE_REPORT_ERROR(error_reporter_,
  //                      "Failed to allocate memory, %d bytes required",
  //                      size - GetFreeSize());
  return nullptr;
}
