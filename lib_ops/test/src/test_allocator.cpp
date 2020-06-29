// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include <iostream>

#include "lib_ops/api/allocator.h"
#include "unity.h"
#include "unity_fixture.h"

TEST_GROUP(allocator);

TEST_SETUP(allocator) {}

TEST_TEAR_DOWN(allocator) {}

TEST(allocator, test_allocate) {
  size_t buffer_size = 100;
  void *buffer[buffer_size];
  size_t data_size = 20;
  void *data;
  size_t allocated_size = 0;
  size_t free_size = 0;
  MemoryAllocator allocator;

  allocator.SetHeap(buffer, buffer_size);
  data = allocator.AllocatePersistantBuffer(data_size);
  TEST_ASSERT_NOT_NULL(data);

  allocated_size = allocator.GetAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, data_size);

  free_size = allocator.GetFreeSize();
  TEST_ASSERT_EQUAL_INT(free_size, buffer_size - data_size);
}

TEST(allocator, test_failed_allocate) {
  size_t buffer_size = 100;  // must be multiple of 4
  void *buffer[buffer_size];
  size_t data_size = 20;  // must be multiple of 4
  void *data;
  size_t valid_allocations = buffer_size / data_size;
  MemoryAllocator allocator;

  allocator.SetHeap(buffer, buffer_size);

  for (int i = 0; i < valid_allocations; i++) {
    data = allocator.AllocatePersistantBuffer(data_size);
    TEST_ASSERT_NOT_NULL(data);
  }

  data = allocator.AllocatePersistantBuffer(data_size);
  TEST_ASSERT_NULL(data);
}

TEST(allocator, test_scratch) {
  size_t buffer_size = 100;
  void *buffer[buffer_size];
  size_t persistant_data_size = 20;  // must be multiple of 4
  void *persistant_data;
  size_t scratch_data_size = 12;  // must be multiple of 4
  void *scratch_data;
  size_t allocated_size = 0;
  MemoryAllocator allocator;

  allocator.SetHeap(buffer, buffer_size);

  persistant_data = allocator.AllocatePersistantBuffer(persistant_data_size);
  allocated_size = allocator.GetAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, persistant_data_size);

  scratch_data = allocator.AllocateScratchBuffer(scratch_data_size);
  TEST_ASSERT_NOT_NULL(scratch_data);
  allocated_size = allocator.GetAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size,
                        persistant_data_size + scratch_data_size);

  scratch_data = allocator.AllocateScratchBuffer(scratch_data_size);
  TEST_ASSERT_NOT_NULL(scratch_data);
  allocated_size = allocator.GetAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size,
                        persistant_data_size + scratch_data_size * 2);
}

TEST(allocator, test_scratch_reset) {
  size_t buffer_size = 100;
  void *buffer[buffer_size];
  size_t persistant_data_size = 20;  // must be multiple of 4
  void *persistant_data;
  size_t scratch_data_size = 12;  // must be multiple of 4
  void *scratch_data;
  MemoryAllocator allocator;

  allocator.SetHeap(buffer, buffer_size);

  persistant_data = allocator.AllocatePersistantBuffer(persistant_data_size);
  scratch_data = allocator.AllocateScratchBuffer(scratch_data_size);

  allocator.ResetScratch();

  // after resetting the scratch memory, the next persistant allocation should
  // be at the previous scratch allocation
  persistant_data = allocator.AllocatePersistantBuffer(persistant_data_size);
  TEST_ASSERT_POINTERS_EQUAL(persistant_data, scratch_data);
}

TEST(allocator, test_reset) {
  size_t buffer_size = 100;
  void *buffer[buffer_size];
  void *data;
  size_t allocated_size = 0;
  MemoryAllocator allocator;

  allocator.SetHeap(buffer, buffer_size);

  data = allocator.AllocatePersistantBuffer(buffer_size);

  allocator.ResetHeap();
  allocated_size = allocator.GetAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, 0);

  data = allocator.AllocatePersistantBuffer(buffer_size);
  TEST_ASSERT_NOT_NULL(data);
}

TEST(allocator, test_default_alignment) {
  size_t buffer_size = 100;
  void *buffer[buffer_size];
  size_t data_size = 21;          // must NOT be multiple of 4
  size_t aligned_data_size = 24;  // must be multiple of 4
  void *data1;
  void *data2;
  size_t allocated_size_data1 = 0;
  size_t allocated_size_data2 = 0;
  MemoryAllocator allocator;

  allocator.SetHeap(buffer, buffer_size);

  data1 = allocator.AllocatePersistantBuffer(data_size);
  allocated_size_data1 = allocator.GetAllocatedSize();
  TEST_ASSERT_EQUAL_INT(data_size, allocated_size_data1);
  data2 = allocator.AllocatePersistantBuffer(data_size);
  allocated_size_data2 = allocator.GetAllocatedSize();
  TEST_ASSERT_EQUAL_INT((aligned_data_size + data_size), allocated_size_data2);
}

TEST(allocator, test_custom_alignment) {
  size_t alignment = 32;
  size_t buffer_size = 100;
  void *buffer[buffer_size];

  size_t data_size = alignment + 1;  // must NOT be multiple of alignment
  void *data1;
  void *data2;
  MemoryAllocator allocator;

  allocator.SetHeap(buffer, buffer_size);

  data1 = allocator.AllocatePersistantBuffer(data_size, alignment);
  TEST_ASSERT_EQUAL_INT(0, (long)data1 % alignment);
  data2 = allocator.AllocatePersistantBuffer(data_size, alignment);
  TEST_ASSERT_EQUAL_INT(0, (long)data2 % alignment);
}

TEST(allocator, test_max_allocated) {
  size_t buffer_size = 100;
  void *buffer[buffer_size];
  size_t persistant_data_size = 20;  // must be multiple of 4
  void *persistant_data;
  size_t scratch_data_size = 12;  // must be multiple of 4
  void *scratch_data;
  size_t allocated_size = 0;
  size_t max_allocated_size = 0;
  MemoryAllocator allocator;

  allocator.SetHeap(buffer, buffer_size);

  persistant_data = allocator.AllocatePersistantBuffer(persistant_data_size);
  scratch_data = allocator.AllocateScratchBuffer(scratch_data_size);
  scratch_data = allocator.AllocateScratchBuffer(scratch_data_size);

  allocated_size = allocator.GetAllocatedSize();
  max_allocated_size = allocator.GetMaxAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, max_allocated_size);

  allocator.ResetScratch();
  TEST_ASSERT_EQUAL_INT(max_allocated_size, allocator.GetMaxAllocatedSize());

  scratch_data = allocator.AllocateScratchBuffer(scratch_data_size);
  TEST_ASSERT_EQUAL_INT(max_allocated_size, allocator.GetMaxAllocatedSize());
}

TEST_GROUP_RUNNER(allocator) {
  RUN_TEST_CASE(allocator, test_allocate);
  RUN_TEST_CASE(allocator, test_failed_allocate);
  RUN_TEST_CASE(allocator, test_scratch);
  RUN_TEST_CASE(allocator, test_scratch_reset);
  RUN_TEST_CASE(allocator, test_reset);
  RUN_TEST_CASE(allocator, test_default_alignment);
  RUN_TEST_CASE(allocator, test_custom_alignment);
  RUN_TEST_CASE(allocator, test_max_allocated);
}
