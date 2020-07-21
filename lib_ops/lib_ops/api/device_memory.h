// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_OPERATORS_DEVICE_MEMORY_H_
#define XCORE_OPERATORS_DEVICE_MEMORY_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef XCORE
#include <xcore/thread.h>

void swmem_setup();
void swmem_handler(void *ignored);
void swmem_teardown();

#endif  // XCORE

void memload(void **dest, void *src, size_t size);

#ifdef __cplusplus
}
#endif

#endif  // XCORE_OPERATORS_DEVICE_MEMORY_H_