// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/allocator.h"

#include <algorithm>
#include <cassert>
#include <memory>

#include "lib_ops/api/tracing.h"

#define ALIGNMENT (4)

void MemoryAllocator::SetHeap(void *buffer, size_t size) {
  assert(buffer);
  assert(size > 0);

  buffer_ = buffer;
  buffer_size_ = size;

  ResetHeap();
}

void MemoryAllocator::ResetHeap() {
  alloc_tail_ = std::align(ALIGNMENT, 1, buffer_, buffer_size_);
  scratch_head_ = alloc_tail_;
  max_allocated_ = 0;
}

void MemoryAllocator::ResetScratch() { alloc_tail_ = scratch_head_; }

size_t MemoryAllocator::GetSize() { return buffer_size_; }

size_t MemoryAllocator::GetAllocatedSize() {
  return ((uintptr_t)alloc_tail_ - (uintptr_t)buffer_);
}

size_t MemoryAllocator::GetMaxAllocatedSize() { return max_allocated_; }

size_t MemoryAllocator::GetFreeSize() {
  return buffer_size_ - GetAllocatedSize();
}

void *MemoryAllocator::AllocateBuffer(size_t size) {
  alloc_tail_ = std::align(ALIGNMENT, size, alloc_tail_, buffer_size_);

  if (GetFreeSize() >= size) {
    void *ptr = alloc_tail_;
    alloc_tail_ = (void *)((uintptr_t)alloc_tail_ + size);
    max_allocated_ = std::max(max_allocated_, GetAllocatedSize());
    return ptr;
  }
  // Allocator is out of memory for this allocation
  TRACE_ERROR("Failed to allocate memory, %d bytes required\n",
              size - GetFreeSize());
  return nullptr;
}

void *MemoryAllocator::AllocatePersistantBuffer(size_t size) {
  assert(size > 0);

  void *ptr = AllocateBuffer(size);

  if (ptr) scratch_head_ = alloc_tail_;
  return ptr;
}

void *MemoryAllocator::AllocateScratchBuffer(size_t size) {
  assert(size > 0);

  void *ptr = AllocateBuffer(size);
  return ptr;
}
