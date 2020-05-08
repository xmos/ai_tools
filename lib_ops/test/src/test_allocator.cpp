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

  xcSetHeap(buffer, buffer_size);
  data = xcMalloc(data_size);
  TEST_ASSERT_NOT_NULL(data);

  allocated_size = xcGetHeapAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, data_size);

  free_size = xcGetHeapFreeSize();
  TEST_ASSERT_EQUAL_INT(free_size, buffer_size - data_size);
}

TEST(allocator, test_failed_allocate) {
  size_t buffer_size = 100;  // must be multiple of 4
  void *buffer[buffer_size];
  size_t data_size = 20;  // must be multiple of 4
  void *data;
  size_t valid_allocations = buffer_size / data_size;

  xcSetHeap(buffer, buffer_size);

  for (int i = 0; i < valid_allocations; i++) {
    data = xcMalloc(data_size);
    TEST_ASSERT_NOT_NULL(data);
  }

  data = xcMalloc(data_size);
  TEST_ASSERT_NULL(data);
}

TEST(allocator, test_reallocate) {
  size_t buffer_size = 100;
  void *buffer[buffer_size];
  size_t original_data_size = 20;
  void *original_data;
  size_t reallocate_data_size = 40;
  void *reallocate_data;
  size_t allocated_size = 0;

  xcSetHeap(buffer, buffer_size);

  original_data = xcMalloc(original_data_size);
  TEST_ASSERT_NOT_NULL(original_data);
  allocated_size = xcGetHeapAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, original_data_size);

  reallocate_data = xcRealloc(original_data, reallocate_data_size);
  TEST_ASSERT_NOT_NULL(reallocate_data);
  allocated_size = xcGetHeapAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, reallocate_data_size);

  TEST_ASSERT_POINTERS_EQUAL(reallocate_data, original_data);
}

TEST(allocator, test_free) {
  size_t buffer_size = 100;
  void *buffer[buffer_size];
  size_t first_data_size = 20;
  void *first_data;
  size_t second_data_size = 40;
  void *second_data;
  void *realloc_data;
  size_t allocated_size = 0;

  xcSetHeap(buffer, buffer_size);

  // Allocate some memory
  first_data = xcMalloc(first_data_size);
  TEST_ASSERT_NOT_NULL(first_data);
  allocated_size = xcGetHeapAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, first_data_size);

  // Allocate some other memory
  second_data = xcMalloc(second_data_size);
  TEST_ASSERT_NOT_NULL(second_data);
  allocated_size = xcGetHeapAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, first_data_size + second_data_size);

  // Free the second allocation and check that the memory is available
  xcFree(second_data);
  allocated_size = xcGetHeapAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, first_data_size);

  // Try to free the first allocation which is not allowed
  xcFree(first_data);
  TEST_ASSERT_NOT_NULL(first_data);
  TEST_ASSERT_EQUAL_INT(allocated_size, first_data_size);

  // Try to realloc the first allocation which is not allowed
  realloc_data = xcRealloc(first_data, second_data_size);
  TEST_ASSERT_NULL(realloc_data);
  TEST_ASSERT_EQUAL_INT(allocated_size, first_data_size);

  // Allocate some other memory again
  second_data = xcMalloc(second_data_size);
  TEST_ASSERT_NOT_NULL(second_data);
  allocated_size = xcGetHeapAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, first_data_size + second_data_size);
}

TEST(allocator, test_reset) {
  size_t buffer_size = 100;
  void *buffer[buffer_size];
  void *data;
  size_t allocated_size = 0;

  xcSetHeap(buffer, buffer_size);

  data = xcMalloc(buffer_size);
  TEST_ASSERT_NOT_NULL(data);

  xcResetHeap();
  allocated_size = xcGetHeapAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, 0);

  data = xcMalloc(buffer_size);
  TEST_ASSERT_NOT_NULL(data);
}

TEST(allocator, test_align) {
  size_t buffer_size = 100;
  void *buffer[buffer_size];
  size_t data_size = 21;          // must NOT be multiple of 4
  size_t aligned_data_size = 24;  // must be multiple of 4
  void *data1;
  void *data2;
  size_t allocated_size_data1 = 0;
  size_t allocated_size_data2 = 0;

  xcSetHeap(buffer, buffer_size);

  data1 = xcMalloc(data_size);
  TEST_ASSERT_NOT_NULL(data1);
  allocated_size_data1 = xcGetHeapAllocatedSize();
  TEST_ASSERT_EQUAL_INT(aligned_data_size, allocated_size_data1);
  data2 = xcMalloc(data_size);
  TEST_ASSERT_NOT_NULL(data2);
  allocated_size_data2 = xcGetHeapAllocatedSize();
  TEST_ASSERT_EQUAL_INT((aligned_data_size * 2), allocated_size_data2);
}

TEST_GROUP_RUNNER(allocator) {
  RUN_TEST_CASE(allocator, test_allocate);
  RUN_TEST_CASE(allocator, test_failed_allocate);
  RUN_TEST_CASE(allocator, test_reallocate);
  RUN_TEST_CASE(allocator, test_free);
  RUN_TEST_CASE(allocator, test_reset);
  RUN_TEST_CASE(allocator, test_align);
}
