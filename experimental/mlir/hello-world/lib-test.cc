// Copyright 2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include "lib-test.h"
#include <cstdio>
#include <math.h>

#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"

void print_some_stuff()
{
    printf("Some stuff %0.8f \n", M_PI);
}
