#include <stdint.h>
#include <stdio.h>
#include "model.tflite.h"

extern "C" {
void run(unsigned io_channel) {
  model_init(NULL);
  model_ioserver(io_channel);
}
}
