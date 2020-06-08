// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/device_memory.h"

#ifdef XCORE

extern "C" {

#define IS_RAM(a) (((uintptr_t)a >= 0x80000) && ((uintptr_t)a <= 0x100000)))
#define IS_EXTMEM(a) \
  (((uintptr_t)a >= 0x10000000) && (((uintptr_t)a <= 0x20000000)))
#define IS_SWMEM(a) \
  (((uintptr_t)a >= 0x40000000) && (((uintptr_t)a <= 0x80000000)))

#include <xcore/port.h>
#include <xmos_flash.h>

flash_ports_t flash_ports_0 = {PORT_SQI_CS, PORT_SQI_SCLK, PORT_SQI_SIO,
                               XS1_CLKBLK_5};

flash_clock_config_t flash_clock_config = {
    1, 8, 8, 1, 0,
};

flash_qe_config_t flash_qe_config_0 = {flash_qe_location_status_reg_0,
                                       flash_qe_bit_6};

flash_handle_t flash_handle;

void flash_read(void *dest, void *src, size_t size) {
  flash_read_quad(&flash_handle, (src - (void *)XS1_SWMEM_BASE) >> 2,
                  (unsigned int *)dest, size);
}

void swmem_fill(swmem_fill_t handle, fill_slot_t address) {
  swmem_fill_buffer_t buf;
  flash_read_quad(&flash_handle, (address - (void *)XS1_SWMEM_BASE) >> 2,
                  (unsigned int *)buf, SWMEM_FILL_SIZE_WORDS);
  swmem_fill_populate_from_buffer(handle, address, buf);
}

swmem_fill_t swmem_setup() {
  flash_connect(&flash_handle, &flash_ports_0, flash_clock_config,
                flash_qe_config_0);
  return swmem_fill_get();
}

void swmem_teardown(swmem_fill_t fill_handle) {
  swmem_fill_free(fill_handle);
  flash_disconnect(&flash_handle);
}

void swmem_handler(swmem_fill_t fill_handle) {
  fill_slot_t address = 0;
  while (1) {
    address = swmem_fill_in_address(fill_handle);
    swmem_fill(fill_handle, address);
    swmem_fill_populate_word_done(fill_handle, address);
  }
}

void memload(void **dest, void *src, size_t size) {
  if (IS_RAM(src)) {
    *dest = src;
  } else if (IS_SWMEM(src)) {
    flash_read(*dest, src, size);
  } else if (IS_DDR(src)) {
    memcpy(*dest, src, size);
  }
}

}  // extern "C"

#else  // not XCORE

void memload(void **dest, void *src, size_t size) { *dest = src; }

#endif  // XCORE
