// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include <iostream>

#include "unity.h"
#include "lib_ops/api/allocator.h"


void setUp(void) {}
void tearDown(void) {}

void test_one()
{
    TEST_ASSERT_EQUAL_INT(2, 2);
}

void test_two()
{
    TEST_ASSERT_EQUAL_INT(2, 1);
}