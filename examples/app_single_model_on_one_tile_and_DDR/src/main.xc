#include <stdint.h>
#include <stdio.h>
#include <platform.h>
#include "tile_ram_server.h"
// #include "flash_server.h"

// #define NFLASH_SPECS 1

// fl_QuadDeviceSpec flash_spec[NFLASH_SPECS] = {
//     FL_QUADDEVICE_DEFAULT //FL_QUADDEVICE_MACRONIX_MX25R6435FM2IH0
// };

// on tile[0]: fl_QSPIPorts qspi = {
//     PORT_SQI_CS,
//     PORT_SQI_SCLK,
//     PORT_SQI_SIO,
//     XS1_CLKBLK_2
// };

#define NUMBER_OF_MODELS 1
extern void model_init(unsigned);
extern void inference();

// Hack for xc to be happy with EXTMEM
// Have to specify the extern array size
extern const int8_t tile_server_weights[3587012];
int8_t* weights = (int8_t*)&tile_server_weights;

int main(void) {
    //chan c_flash_or_tile[NUMBER_OF_MODELS];

    par {
        on tile[0]: {
            //flash_t headers[NUMBER_OF_MODELS];
            //tile_ram_server(c_flash_or_tile, headers, NUMBER_OF_MODELS, weights);
            // flash_server(c_flash_or_tile, headers, NUMBER_OF_MODELS, qspi, flash_spec, 1);
        }

        on tile[1]: {
            unsafe {
            //c_flash_or_tile[0] <: FLASH_SERVER_INIT;

            model_init((unsigned)(weights + 20));

            inference();

            //c_flash_or_tile[0] <: FLASH_SERVER_QUIT;
            }
        }
    }
    return 0;
}