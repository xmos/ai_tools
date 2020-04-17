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
  void* buffer[buffer_size];
  size_t data_size = 20;
  void* data;
  size_t allocated_size = 0;
  size_t free_size = 0;

  xcore::LinearAllocator allocator;
  allocator.SetBuffer(buffer, buffer_size);
  data = allocator.Allocate(data_size);
  TEST_ASSERT_NOT_NULL(data);

  allocated_size = allocator.GetAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, data_size);

  free_size = allocator.GetFreeSize();
  TEST_ASSERT_EQUAL_INT(free_size, buffer_size - data_size);
}

TEST(allocator, test_failed_allocate) {
  size_t buffer_size = 100;  // must be multiple of 4
  void* buffer[buffer_size];
  size_t data_size = 20;  // must be multiple of 4
  void* data;
  size_t valid_allocations = buffer_size / data_size;

  xcore::LinearAllocator allocator(buffer, buffer_size);

  for (int i = 0; i < valid_allocations; i++) {
    data = allocator.Allocate(data_size);
    TEST_ASSERT_NOT_NULL(data);
  }

  data = allocator.Allocate(data_size);
  TEST_ASSERT_NULL(data);
}

TEST(allocator, test_reallocate) {
  size_t buffer_size = 100;
  void* buffer[buffer_size];
  size_t original_data_size = 20;
  void* original_data;
  size_t reallocate_data_size = 40;
  void* reallocate_data;
  size_t allocated_size = 0;

  xcore::LinearAllocator allocator(buffer, buffer_size);

  original_data = allocator.Allocate(original_data_size);
  TEST_ASSERT_NOT_NULL(original_data);
  allocated_size = allocator.GetAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, original_data_size);

  reallocate_data = allocator.Reallocate(original_data, reallocate_data_size);
  TEST_ASSERT_NOT_NULL(reallocate_data);
  allocated_size = allocator.GetAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, reallocate_data_size);

  TEST_ASSERT_POINTERS_EQUAL(reallocate_data, original_data);
}

TEST(allocator, test_reset) {
  size_t buffer_size = 100;
  void* buffer[buffer_size];
  void* data;
  size_t allocated_size = 0;

  xcore::LinearAllocator allocator(buffer, buffer_size);

  data = allocator.Allocate(buffer_size);
  TEST_ASSERT_NOT_NULL(data);

  allocator.Reset();
  allocated_size = allocator.GetAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size, 0);

  data = allocator.Allocate(buffer_size);
  TEST_ASSERT_NOT_NULL(data);
}

TEST(allocator, test_align) {
  size_t buffer_size = 100;
  void* buffer[buffer_size];
  size_t data_size = 21;  // must NOT be multiple of 4
  void* data1;
  void* data2;
  size_t allocated_size_data1 = 0;
  size_t allocated_size_data2 = 0;

  xcore::LinearAllocator allocator(buffer, buffer_size);

  data1 = allocator.Allocate(data_size);
  TEST_ASSERT_NOT_NULL(data1);
  allocated_size_data1 = allocator.GetAllocatedSize();
  TEST_ASSERT_EQUAL_INT(allocated_size_data1, data_size);
  data2 = allocator.Allocate(data_size);
  TEST_ASSERT_NOT_NULL(data2);
  allocated_size_data2 = allocator.GetAllocatedSize();
  TEST_ASSERT_GREATER_THAN_INT(data_size,
                               (allocated_size_data2 - allocated_size_data1));

  int mod4_diff = ((char*)data2 - (char*)data1) % 4;
  TEST_ASSERT_EQUAL_INT(mod4_diff, 0);
}

TEST_GROUP_RUNNER(allocator) {
  RUN_TEST_CASE(allocator, test_allocate);
  RUN_TEST_CASE(allocator, test_failed_allocate);
  RUN_TEST_CASE(allocator, test_reallocate);
  RUN_TEST_CASE(allocator, test_reset);
  RUN_TEST_CASE(allocator, test_align);
}
