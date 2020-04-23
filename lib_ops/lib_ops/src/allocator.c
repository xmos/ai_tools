// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/allocator.h"

#include <assert.h>
#include <stdlib.h>

#define ALIGNMENT 4

static uintptr_t kBuffer;
static size_t kBufferSize;
static size_t kAllocatedSize;
static size_t kLastAllocationOffset;  // For LIFO realloc and free support only

static uintptr_t align_forward(uintptr_t ptr) {
  uintptr_t aligned_result = ((ptr + (ALIGNMENT - 1)) / ALIGNMENT) * ALIGNMENT;
  return aligned_result;
}

void xcSetHeap(void *buffer, size_t size) {
  assert(buffer);
  assert(size > 0);

  kBuffer = align_forward((uintptr_t)(buffer));
  kBufferSize = size;
  kAllocatedSize = 0;
  kLastAllocationOffset = size;
}

void xcResetHeap() {
  kAllocatedSize = 0;
  kLastAllocationOffset = kBufferSize;
}

size_t xcGetHeapSize() { return kBufferSize; }
size_t xcGetHeapAllocatedSize() { return kAllocatedSize; }
size_t xcGetHeapFreeSize() { return kBufferSize - kAllocatedSize; }

void *xcMalloc(size_t size) {
  uintptr_t curr_ptr = kBuffer + kAllocatedSize;
  uintptr_t offset = align_forward(curr_ptr) - kBuffer;

  // Check to see if the backing memory has space left
  if (offset + size <= kBufferSize) {
    kLastAllocationOffset = offset;
    kAllocatedSize = offset + size;

    return (void *)(kBuffer + offset);
  }

  // Allocator is out of memory for this allocation
  return NULL;
}

void *xcRealloc(void *ptr, size_t size) {
  if (ptr == NULL) return xcMalloc(size);  // Unallocated, no need to Reallocate

  uintptr_t last_allocation = kBuffer + kLastAllocationOffset;
  uintptr_t raw_ptr = (uintptr_t)(ptr);
  if (last_allocation == raw_ptr) {
    kAllocatedSize = raw_ptr - kBuffer;
    return xcMalloc(size);
  }
  // Reallocating an arbitrary allocation is not supported
  return NULL;
}

void xcFree(void *ptr) {
  uintptr_t last_allocation = kBuffer + kLastAllocationOffset;
  uintptr_t raw_ptr = (uintptr_t)(ptr);
  if (last_allocation == raw_ptr) {
    // Only the last allocation can be freed (FIFO)
    kAllocatedSize = raw_ptr - kBuffer;
  }
}
