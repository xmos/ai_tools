#include "flash_server.h"
#include "stdio.h"
#include <platform.h>
#include <quadflash.h>
#include <stdint.h>

#define NUMBER_OF_MODELS 1
#define NFLASH_SPECS 1

fl_QuadDeviceSpec flash_spec[NFLASH_SPECS] = {
    FL_QUADDEVICE_DEFAULT //FL_QUADDEVICE_MACRONIX_MX25R6435FM2IH0
};

on tile[0]: fl_QSPIPorts qspi = {
    PORT_SQI_CS,
    PORT_SQI_SCLK,
    PORT_SQI_SIO,
    XS1_CLKBLK_2
};

extern void model_init(chanend f);
extern void inference();

int main(void) {
  chan c_flash[1];

  par {
    on tile[0] : {
      flash_t headers[NUMBER_OF_MODELS];
      flash_server(c_flash, headers, NUMBER_OF_MODELS, qspi, flash_spec, 1);
    }

    on tile[1] : {
      unsafe {
        c_flash[0] <: FLASH_SERVER_INIT;
        model_init(c_flash[0]);

        inference();

        c_flash[0] <: FLASH_SERVER_QUIT;
      }
    }
  }
  return 0;
}
