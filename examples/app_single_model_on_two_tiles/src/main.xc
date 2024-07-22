#include <stdint.h>
#include <stdio.h>
#include <platform.h>
#include "tile_ram_server.h"

#define NUMBER_OF_MODELS 1
#include "model_weights.h"

extern void model_init(chanend f);
extern void inference();

int main(void) {
    chan c_flash_or_tile[NUMBER_OF_MODELS];

    par {
        on tile[0]: {
            flash_t headers[NUMBER_OF_MODELS];
            tile_ram_server(c_flash_or_tile, headers, NUMBER_OF_MODELS, tile_server_weights);
        }

        on tile[1]: {
            unsafe {
            c_flash_or_tile[0] <: FLASH_SERVER_INIT;

            model_init(c_flash_or_tile[0]);

            inference();

            c_flash_or_tile[0] <: FLASH_SERVER_QUIT;
            }
        }
    }
    return 0;
}