// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/allocator.h"

#include <assert.h>
#include <stdlib.h>

#include "lib_ops/api/tracing.h"

#define ALIGNMENT (4)
#define ALIGNMENT_MASK (0x0003)

static uintptr_t kBuffer;
static size_t kBufferSize;
static size_t kNextAllocationOffset;
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
  kNextAllocationOffset = 0;
  kLastAllocationOffset = size;
}

void xcResetHeap() {
  kNextAllocationOffset = 0;
  kLastAllocationOffset = kBufferSize;
}

size_t xcGetHeapSize() { return kBufferSize; }
size_t xcGetHeapAllocatedSize() { return kNextAllocationOffset; }
size_t xcGetHeapFreeSize() { return kBufferSize - kNextAllocationOffset; }

void *xcMalloc(size_t size) {
  assert(size > 0);

  if (size & ALIGNMENT_MASK) {
    // forward align
    size += (ALIGNMENT - (size & ALIGNMENT_MASK));
  }

  if ((kNextAllocationOffset + size) <= kBufferSize) {
    uintptr_t ptr = kBuffer + kNextAllocationOffset;
    kLastAllocationOffset = kNextAllocationOffset;
    kNextAllocationOffset += size;
    return (void *)(ptr);
  }

  // Allocator is out of memory for this allocation
  TRACE_ERROR("Failed to allocate memory, %d bytes required\n",
              (kNextAllocationOffset + size) - kBufferSize);
  return NULL;
}

void *xcRealloc(void *ptr, size_t size) {
  if (ptr == NULL) return xcMalloc(size);  // Unallocated, no need to Reallocate

  uintptr_t last_allocation = kBuffer + kLastAllocationOffset;
  if (last_allocation == (uintptr_t)(ptr)) {
    kNextAllocationOffset = kLastAllocationOffset;
    return xcMalloc(size);
  }
  // Reallocating an arbitrary allocation is not supported
  TRACE_ERROR("Reallocating an arbitrary allocation is not supported\n");
  return NULL;
}

void xcFree(void *ptr) {
  uintptr_t last_allocation = kBuffer + kLastAllocationOffset;
  if (last_allocation == (uintptr_t)(ptr)) {
    // Only the last allocation can be freed (FIFO)
    kNextAllocationOffset = kLastAllocationOffset;

    return;
  }
  TRACE_ERROR("Freeing an arbitrary allocation is not supported\n");
}
