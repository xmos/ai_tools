// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include <stdio.h>

#ifndef XCORE_LOGGING_H_
#define XCORE_LOGGING_H_

#ifdef XCORE_ERROR
#define xcError(...) printf(__VA_ARGS__)
#else
#define xcError(...)
#endif  // XCORE_ERROR

#ifdef XCORE_TRACE
#define xcTrace(...) printf(__VA_ARGS__)
#else
#define xcTrace(...)
#endif  // XCORE_TRACE

#endif  // XCORE_LOGGING_H_