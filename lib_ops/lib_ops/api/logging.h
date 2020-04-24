// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include <stdio.h>

#ifndef LIB_OPS_LOGGING_H_
#define LIB_OPS_LOGGING_H_

#ifdef ENABLE_ERROR_LOGGING
#define LOG_ERROR(...) printf(__VA_ARGS__)
#else
#define LOG_ERROR(...)
#endif  // ENABLE_ERROR_LOGGING

#ifdef ENABLE_TRACE_LOGGING
#define LOG_TRACE(...) printf(__VA_ARGS__)
#else
#define LOG_TRACE(...)
#endif  // ENABLE_TRACE_LOGGING

#endif  // LIB_OPS_LOGGING_H_