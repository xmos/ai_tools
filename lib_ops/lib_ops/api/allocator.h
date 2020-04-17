// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_ALLOCATOR_H_
#define XCORE_ALLOCATOR_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Specify memory to use for dynamic allocations.
 *
 * All pointers returned by the heap word be word-aligned.
 * Some heap memory may be lost due to this alignment.
 *
 *
 * \param buffer  Pointer to beginning of heap data.
 * \param size    Size of buffer (in bytes)
 */
void xcSetHeap(void *buffer, size_t size);

/** Get the size (in bytes) of the heap.
 */
size_t xcGetHeapSize();

/** Get the size (in bytes) of memory allocated from the heap.
 */
size_t xcGetHeapAllocatedSize();

/** Get the size (in bytes) of available memory in the heap.
 */
size_t xcGetHeapFreeSize();

/** Reset the heap so the memory can be re-used.
 */
void xcResetHeap();

/** Allocate memory.
 * \param size    Size of allocation (in bytes)
 */
void *xcMalloc(size_t size);

/** Reallocate memory
 *
 * This heap implementation only allow the last allocation to be
 * reallocated (LIFO). Attempting to reallocate a block that is not the
 * last allocation will return a null pointer.
 *
 *
 * \param ptr     Pointer to previously allocated data.
 * \param size    Size of new allocation (in bytes)
 */
void *xcRealloc(void *ptr, size_t size);

/** Free an allocation
 *
 * This heap implementation only allow the last allocation to be
 * freed (LIFO). Attempting to free a block that is not the
 * last allocation results in a no-op.
 *
 *
 * \param ptr     Pointer to previously allocated data.
 */
void xcFree(void *ptr);

#ifdef __cplusplus
}
#endif
#endif  // XCORE_ALLOCATOR_H_
