// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include <iostream>

#include "unity.h"

void test_one();
void test_two();

int main(int argc, char *argv[]) {
  UNITY_BEGIN();
  RUN_TEST(test_one);
  RUN_TEST(test_two);
  return UNITY_END();
}