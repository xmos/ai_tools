#include <platform.h>
#include "ioserver.h"

#define NUMBER_OF_MODELS 1

extern void run(chanend io_channel);

int main(void) {
    chan io_channel[NUMBER_OF_MODELS];
    par {
        on tile[0]: {
            unsafe {
                run(io_channel[0]);
            }
        }
        on tile[1]: {
            // ioserver uses three threads
            ioserver(io_channel, NUMBER_OF_MODELS);
        }
    }
    return 0;
}
