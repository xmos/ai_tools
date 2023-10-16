#include "model.tflite.h"

void run(unsigned io_channel) {
    model_init(NULL);
    model_ioserver(io_channel);
}

extern "C" {
    void inferencer(unsigned io_channel) {
        run(io_channel);
    }
}
