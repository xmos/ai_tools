// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_OPERATORS_ALLOCATOR_H_
#define XCORE_OPERATORS_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>

class MemoryAllocator {
 public:
  /** Construct Allocator.
   *
   * All pointers returned by the heap will be word-aligned.
   * Some heap memory may be lost due to this alignment.
   *
   *
   * \param buffer  Pointer to beginning of heap data.
   * \param size    Size of buffer (in bytes)
   */
  MemoryAllocator()
      : buffer_(nullptr),
        buffer_size_(0),
        alloc_tail_(nullptr),
        scratch_head_(nullptr),
        max_allocated_(0) {}
  ~MemoryAllocator() {}

  void SetHeap(void *buffer, size_t size);

  /** Get the size (in bytes) of the heap.
   */
  size_t GetSize();

  /** Get the size (in bytes) of memory allocated from the heap.
   */
  size_t GetAllocatedSize();

  /** Get the maximum size (in bytes) of memory allocated from the heap.
   */
  size_t GetMaxAllocatedSize();

  /** Get the size (in bytes) of available memory in the heap.
   */
  size_t GetFreeSize();

  /** Reset the allocator so all memory can be re-used.
   */
  void ResetHeap();

  /** Reset the allocator so the scratch memory can be re-used.
   */
  void ResetScratch();

  /** Allocate memory that is intended to persist for the lifetime of the
   * allocator. \param size    Size of allocation (in bytes)
   */
  void *AllocatePersistantBuffer(size_t size);

  /** Allocate scratch memory that will only be used temporarilly.
   * \param size    Size of allocation (in bytes)
   */
  void *AllocateScratchBuffer(size_t size);

 private:
  void *AllocateBuffer(size_t size);

  void *buffer_;
  size_t buffer_size_;
  void *alloc_tail_;
  void *scratch_head_;
  size_t max_allocated_;
};

#endif  // XCORE_OPERATORS_ALLOCATOR_H_
