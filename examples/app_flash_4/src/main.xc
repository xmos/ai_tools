#include <platform.h>
#include "flash_server.h"

#define NFLASH_SPECS 1
fl_QuadDeviceSpec flash_spec[NFLASH_SPECS] = {
    FL_QUADDEVICE_MACRONIX_MX25R6435FM2IH0
};

fl_QSPIPorts qspi = {
    PORT_SQI_CS,
    PORT_SQI_SCLK,
    PORT_SQI_SIO,
    XS1_CLKBLK_2
};

extern void network_1(chanend channel_to_network_2);
extern void network_2(chanend channel_to_network_1);

int main(void) {
    chan c_flash[2], c_1_to_2;
    par {
        on tile[0] : {
            flash_t headers[NNETWORKS];
            chanend c_flash[NNETWORKS];

            flash_server(c_flash, headers, NNETWORKS,
                         &qspi, &flash_spec, NFLASH_SPEC);
        }
        on tile[0] : {
            network_1(c_flash[0], c_1_to_2);
        }
        on tile[1] : {
            network_2(c_flash[1], c_1_to_2);
        }
    }
    
}
