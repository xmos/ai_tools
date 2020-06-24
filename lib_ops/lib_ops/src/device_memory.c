// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/device_memory.h"

#include <stddef.h>
#include <string.h>

#ifdef XCORE

#define IS_EXTMEM(a) \
  (((uintptr_t)a >= 0x10000000) && (((uintptr_t)a <= 0x20000000)))
#define IS_SWMEM(a) \
  (((uintptr_t)a >= 0x40000000) && (((uintptr_t)a <= 0x80000000)))

#include <xcore/port.h>
#include <xcore/swmem_fill.h>
#include <xmos_flash.h>

flash_ports_t flash_ports_0 = {PORT_SQI_CS, PORT_SQI_SCLK, PORT_SQI_SIO,
                               XS1_CLKBLK_5};

flash_clock_config_t flash_clock_config = {
    1, 8, 8, 1, 0,
};

flash_qe_config_t flash_qe_config_0 = {flash_qe_location_status_reg_0,
                                       flash_qe_bit_6};

flash_handle_t flash_handle;
swmem_fill_t swmem_fill_handle;

void swmem_fill(swmem_fill_t handle, fill_slot_t address) {
  flash_read_quad(&flash_handle, (address - (void *)XS1_SWMEM_BASE) >> 2,
                  address, SWMEM_FILL_SIZE_WORDS);
}

void swmem_setup() {
  printf("swmem_setup start\n");
  flash_connect(&flash_handle, &flash_ports_0, flash_clock_config,
                flash_qe_config_0);
  printf("swmem_setup mid\n");

  swmem_fill_handle = swmem_fill_get();
  printf("swmem_setup end\n");
}

void swmem_teardown() {
  swmem_fill_free(swmem_fill_handle);
  flash_disconnect(&flash_handle);
}

void swmem_handler(void *ignored) {
  printf("swmem_handler start\n");
  fill_slot_t address = 0;
  while (1) {
    printf("swmem_fill_handle = %ld \n", (long)swmem_fill_handle);
    address = swmem_fill_in_address(swmem_fill_handle);
    printf("swmem_handler address=0x%08x\n", address);
    swmem_fill(swmem_fill_handle, address);
    swmem_fill_populate_word_done(swmem_fill_handle, address);
  }
}

void memload(void **dest, void *src, size_t size) {
  if (IS_SWMEM(src)) {
    flash_read_quad(&flash_handle, ((uintptr_t)src - XS1_SWMEM_BASE) >> 2,
                    (unsigned int *)*dest, size);
  } else if (IS_EXTMEM(src)) {
    memcpy(*dest, src, size);
  }
}

#endif  // XCORE
