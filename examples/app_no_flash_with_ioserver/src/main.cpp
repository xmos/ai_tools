/**
 * @file
 * @brief Main functions suitable for use with XCORE-400.
 * @note Operation of this example on XCORE-400 depends on the availability of
 *       a version of lib_xud suitable for XCORE-400.
 */

#include <netmain.h>
#include <xcore/chanend.h>

#include "ioserver.h"
#include "model.tflite.h"

#define NUMBER_OF_MODELS 1

int main_tile_0(chanend_t io_channel) {
    model_init(NULL);
    model_ioserver(io_channel);
    return 0;
}

int main_tile_1(chanend_t io_channel) {
    chanend_t io_channels[NUMBER_OF_MODELS] = {io_channel};
    ioserver(io_channels, NUMBER_OF_MODELS);
    return 0;
}

DECLARE_CHAN(io_channel)

NETWORK_MAIN(
    TILE_MAIN(main_tile_0, 0, (CHAN(io_channel))),
    TILE_MAIN(main_tile_1, 1, (CHAN(io_channel)))
)