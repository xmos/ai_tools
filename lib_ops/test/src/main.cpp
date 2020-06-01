// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include <iostream>

#include "unity.h"
#include "unity_fixture.h"

static void RunTests(void) {
  RUN_TEST_GROUP(allocator);
  RUN_TEST_GROUP(dispatcher);
  RUN_TEST_GROUP(execution_plan);
  RUN_TEST_GROUP(stopwatch);
}

int main(int argc, const char* argv[]) {
  UnityGetCommandLineOptions(argc, argv);
  UnityBegin(argv[0]);
  RunTests();
  UnityEnd();

  return (int)Unity.TestFailures;
}
