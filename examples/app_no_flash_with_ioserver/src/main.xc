#include <platform.h>
#include "ioserver.h"

#define NUMBER_OF_MODELS 1

extern void inferencer(chanend io_channel);

int main(void) {
    chan io_channel[NUMBER_OF_MODELS];
    par {
        on tile[0]: {
            unsafe {
                inferencer(io_channel[0]);
            }
        }
        on tile[1]: {
            // ioserver uses three threads
            ioserver(io_channel, NUMBER_OF_MODELS);
        }
    }
    return 0;
}
