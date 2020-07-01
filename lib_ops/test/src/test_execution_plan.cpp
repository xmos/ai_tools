// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/dispatcher.h"
#include "lib_ops/api/planning.h"
#include "unity.h"
#include "unity_fixture.h"

TEST_GROUP(execution_plan);

TEST_SETUP(execution_plan) {}

TEST_TEAR_DOWN(execution_plan) {}

TEST(execution_plan, test_rowcol_region_array) {
  size_t buffer_size = 1000;
  void* buffer[buffer_size];

  size_t num_regions = 5;
  int32_t top = 0;
  int32_t left = 1;
  int32_t rows = 2;
  int32_t cols = 3;
  xcore::RowColRegionArray regions;

  xcore::Dispatcher dispatcher(buffer, buffer_size, true);
  InitializeXCore(&dispatcher);

  regions.Init(num_regions);
  TEST_ASSERT_EQUAL_INT(regions.GetSize(), 0);

  for (int i = 0; i < num_regions; i++) {
    xcore::RowColRegion region = {top, left, rows, cols};
    regions.Append(region);
    TEST_ASSERT_EQUAL_INT(regions.GetSize(), i + 1);
  }

  TEST_ASSERT_EQUAL_INT(regions.GetSize(), num_regions);

  for (int i = 0; i < num_regions; i++) {
    const xcore::RowColRegion& region = regions[i];
    TEST_ASSERT_EQUAL_INT(region.top, top);
    TEST_ASSERT_EQUAL_INT(region.left, left);
    TEST_ASSERT_EQUAL_INT(region.rows, rows);
    TEST_ASSERT_EQUAL_INT(region.cols, cols);
  }
}

TEST(execution_plan, test_channel_group_array) {
  size_t buffer_size = 1000;
  void* buffer[buffer_size];

  size_t num_changrps = 50;
  int32_t size = 16;
  xcore::ChannelGroupArray changrps;

  xcore::Dispatcher dispatcher(buffer, buffer_size, true);
  InitializeXCore(&dispatcher);

  changrps.Init(num_changrps);
  TEST_ASSERT_EQUAL_INT(changrps.GetSize(), 0);

  for (int i = 0; i < num_changrps; i++) {
    xcore::ChannelGroup changrp = {i, i * size - 1, size};
    changrps.Append(changrp);
    TEST_ASSERT_EQUAL_INT(changrps.GetSize(), i + 1);
  }

  TEST_ASSERT_EQUAL_INT(changrps.GetSize(), num_changrps);

  for (int i = 0; i < num_changrps; i++) {
    const xcore::ChannelGroup& changrp = changrps[i];
    TEST_ASSERT_EQUAL_INT(changrp.index, i);
    TEST_ASSERT_EQUAL_INT(changrp.start, i * size - 1);
    TEST_ASSERT_EQUAL_INT(changrp.size, size);
  }
}

TEST_GROUP_RUNNER(execution_plan) {
  RUN_TEST_CASE(execution_plan, test_rowcol_region_array);
  RUN_TEST_CASE(execution_plan, test_channel_group_array);
}
