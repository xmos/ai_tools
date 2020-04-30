// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/benchmarking.h"
#include "unity.h"
#include "unity_fixture.h"

TEST_GROUP(stopwatch);

TEST_SETUP(stopwatch) {}

TEST_TEAR_DOWN(stopwatch) {}

TEST(stopwatch, test_stopwatch) {
  xcore::Stopwatch sw;

  sw.Start();
  sw.Stop();

  TEST_ASSERT_GREATER_THAN_INT(0, sw.GetEllapsedNanoseconds());
  TEST_ASSERT_GREATER_THAN_INT(sw.GetEllapsedMicroseconds(),
                               sw.GetEllapsedNanoseconds());
}

TEST(stopwatch, test_macros) {
  TIMER_START();
  TIMER_STOP("test_macros");
}

TEST_GROUP_RUNNER(stopwatch) {
  RUN_TEST_CASE(stopwatch, test_stopwatch);
  RUN_TEST_CASE(stopwatch, test_macros);
}
