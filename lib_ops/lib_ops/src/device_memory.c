// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/device_memory.h"

#include <stddef.h>
#include <string.h>

#ifdef XCORE

#include <xcore/port.h>
#include <xcore/swmem_fill.h>
#include <xmos_flash.h>

#define SWMEM_BASE (0x40000000)

#define IS_RAM(a) (((uintptr_t)a >= 0x80000) && ((uintptr_t)a <= 0x100000))
#define IS_EXTMEM(a) \
  (((uintptr_t)a >= 0x10000000) && (((uintptr_t)a <= 0x20000000)))
#define IS_SWMEM(a) \
  (((uintptr_t)a >= SWMEM_BASE) && (((uintptr_t)a <= 0x80000000)))

flash_ports_t flash_ports_0 = {PORT_SQI_CS, PORT_SQI_SCLK, PORT_SQI_SIO,
                               XS1_CLKBLK_5};

flash_clock_config_t flash_clock_config = {
    1, 8, 8, 1, 0,
};

flash_qe_config_t flash_qe_config_0 = {flash_qe_location_status_reg_0,
                                       flash_qe_bit_6};

flash_handle_t flash_handle;
swmem_fill_t swmem_fill_handle;

// We must initialise this to a value such that it is not memset to zero during
// C runtime startup
#define SWMEM_ADDRESS_UNINITIALISED 0xffffffff
volatile unsigned int __swmem_address = SWMEM_ADDRESS_UNINITIALISED;

static unsigned int nibble_swap_word(unsigned int x) {
  return ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
}

void swmem_fill(swmem_fill_t handle, fill_slot_t address) {
  swmem_fill_buffer_t buf;
  unsigned int *buf_ptr = (unsigned int *)buf;

  flash_read_quad(&flash_handle,
                  (address - (void *)XS1_SWMEM_BASE + __swmem_address) >> 2,
                  buf_ptr, SWMEM_FILL_SIZE_WORDS);
  for (unsigned int i = 0; i < SWMEM_FILL_SIZE_WORDS; i++) {
    buf_ptr[i] = nibble_swap_word(buf_ptr[i]);
  }

  swmem_fill_populate_from_buffer(handle, address, buf);
}

void swmem_setup() {
  flash_connect(&flash_handle, &flash_ports_0, flash_clock_config,
                flash_qe_config_0);

  if (__swmem_address == SWMEM_ADDRESS_UNINITIALISED) {
    __swmem_address = 0;
  }

  swmem_fill_handle = swmem_fill_get();
}

void swmem_teardown() {
  swmem_fill_free(swmem_fill_handle);
  flash_disconnect(&flash_handle);
}

void swmem_handler(void *ignored) {
  fill_slot_t address = 0;
  while (1) {
    address = swmem_fill_in_address(swmem_fill_handle);
    swmem_fill(swmem_fill_handle, address);
    swmem_fill_populate_word_done(swmem_fill_handle, address);
  }
}

void memload(void **dest, void *src, size_t size) {
  if (IS_RAM(src)) {
    *dest = src;
  } else if (IS_SWMEM(src)) {
    flash_read_quad(&flash_handle, ((uintptr_t)src - SWMEM_BASE) >> 2,
                    (unsigned int *)*dest, size);
  } else if (IS_EXTMEM(src)) {
    memcpy(*dest, src, size);
  }
}

#else  // not XCORE

void memload(void **dest, void *src, size_t size) { *dest = src; }

#endif
