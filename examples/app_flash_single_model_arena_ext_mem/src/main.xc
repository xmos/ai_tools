#include <platform.h>
#include <stdio.h>
#include "flash_server.h"

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

extern void inferencer(chanend x);

int main(void) {
    chan x[1];

    par {
        on tile[0]: {
            flash_t headers[1];
            flash_server(x, headers, 1, qspi, flash_spec, 1);            
        }

        on tile[1]: {
            unsafe {
                inferencer(x[0]);
            }
        }
    }
    return 0;
}
