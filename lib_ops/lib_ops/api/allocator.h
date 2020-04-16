// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_OPERATOR_ALLOCATOR_H_
#define XCORE_OPERATOR_ALLOCATOR_H_
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace xcore {

class LinearAllocator {
 public:
  LinearAllocator() {}
  LinearAllocator(void* buffer, size_t size);
  ~LinearAllocator() {}

  void SetBuffer(void* buffer, size_t size);
  void* Allocate(size_t size);
  void* Reallocate(void* ptr, size_t size);
  void Reset();

  size_t GetAllocatedSize() { return allocated_size_; }
  size_t GetFreeSize() { return buffer_size_ - allocated_size_; }

 private:
  // Prevent copies because it might cause errors
  LinearAllocator& operator=(const LinearAllocator&);

  uintptr_t buffer_;
  size_t buffer_size_;
  size_t allocated_size_;
  size_t last_allocation_offset_;  // For realloc support only
};

}  // namespace xcore

#endif  // XCORE_OPERATOR_ALLOCATOR_H_
