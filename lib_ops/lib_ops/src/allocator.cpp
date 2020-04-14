// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include <cassert>
#include <cstdlib>
#include <iostream>

#include "lib_ops/api/allocator.h"

namespace xcore {

constexpr int32_t align = 4;

inline uintptr_t align_forward(uintptr_t ptr) {
  uintptr_t p, a, modulo;

  p = ptr;
  a = (uintptr_t)align;
  modulo = p & (a - 1);

  if (modulo != 0) {
    // If 'p' address is not aligned, push the address to the
    // next value which is aligned
    p += a - modulo;
  }
  return p;
}

LinearAllocator::LinearAllocator(void* buffer, size_t size) {
  SetBuffer(buffer, size);
}

void LinearAllocator::SetBuffer(void* buffer, size_t size) {
  assert(buffer);
  assert(size > 0);

  buffer_ = align_forward(reinterpret_cast<uintptr_t>(buffer));
  buffer_size_ = size;
  allocated_size_ = 0;
  last_allocation_offset_ = size;
}

void* LinearAllocator::Allocate(size_t size) {
  uintptr_t curr_ptr = buffer_ + allocated_size_;
  uintptr_t offset = align_forward(curr_ptr) - buffer_;

  // Check to see if the backing memory has space left
  if (offset + size <= buffer_size_) {
    last_allocation_offset_ = offset;
    allocated_size_ = offset + size;

    return reinterpret_cast<void*>(buffer_ + offset);
  }

  // Allocator is out of memory for this allocation
  return nullptr;
}

void* LinearAllocator::Reallocate(void* ptr, size_t size) {
  if (ptr == nullptr)
    return Allocate(size);  // Unallocated, no need to Reallocate

  uintptr_t last_allocation = buffer_ + last_allocation_offset_;
  uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(ptr);
  if (last_allocation == raw_ptr) {
    allocated_size_ = raw_ptr - buffer_;
    return Allocate(size);
  } else {
    // Reallocating an arbitrary allocation is not supported
    return nullptr;
  }
}

void LinearAllocator::Reset() {
  allocated_size_ = 0;
  last_allocation_offset_ = buffer_size_;
}

}  // namespace xcore
