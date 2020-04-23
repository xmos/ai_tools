// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/par.h"
#include "unity.h"
#include "unity_fixture.h"

TEST_GROUP(par_region);

TEST_SETUP(par_region) {}

TEST_TEAR_DOWN(par_region) {}

TEST(par_region, test_par_region_array) {
  size_t num_regions = 5;
  int32_t top = 0;
  int32_t left = 1;
  int32_t rows = 2;
  int32_t cols = 3;
  xcore::ParRegionArray regions;

  TEST_ASSERT_EQUAL_INT(regions.size, 0);

  for (int i = 0; i < num_regions; i++) {
    xcore::ParRegion region = {top, left, rows, cols};
    regions.append(region);
    TEST_ASSERT_EQUAL_INT(regions.size, i + 1);
  }

  TEST_ASSERT_EQUAL_INT(regions.size, num_regions);

  for (int i = 0; i < num_regions; i++) {
    const xcore::ParRegion& region = regions[i];
    TEST_ASSERT_EQUAL_INT(region.top, top);
    TEST_ASSERT_EQUAL_INT(region.left, left);
    TEST_ASSERT_EQUAL_INT(region.rows, rows);
    TEST_ASSERT_EQUAL_INT(region.cols, cols);
  }
}

TEST_GROUP_RUNNER(par_region) {
  RUN_TEST_CASE(par_region, test_par_region_array);
}
