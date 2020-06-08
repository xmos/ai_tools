// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_OPERATORS_DEVICE_MEMORY_H_
#define XCORE_OPERATORS_DEVICE_MEMORY_H_

#include <cstddef>
#include <cstdint>

#ifdef XCORE

extern "C" {

void swmem_setup(void);

void swmem_teardown(void);

void swmem_handler(void *ignored);

void flash_read(void *dest, size_t size);

}  // extern "C"

#endif  // XCORE

void memload(void **dest, void *src, size_t size);

#endif  // XCORE_OPERATORS_DEVICE_MEMORY_H_
