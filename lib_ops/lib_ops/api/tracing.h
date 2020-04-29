// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include <stdio.h>

#ifndef LIB_OPS_TRACING_H_
#define LIB_OPS_TRACING_H_

#ifdef ENABLE_TRACING

#define TRACE_ERROR(...) \
  printf("ERROR: ");     \
  printf(__VA_ARGS__)

#define TRACE_INFO(...) \
  printf("INFO: ");     \
  printf(__VA_ARGS__)

#else

#define TRACE_ERROR(...)
#define TRACE_INFO(...)

#endif  // ENABLE_TRACING

#endif  // LIB_OPS_TRACING_H_