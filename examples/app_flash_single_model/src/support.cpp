#include <stdio.h>
#include <string.h>
#include <xcore/channel.h>
#include "flash_server.h"

#include "model.tflite.h"
#include "image.h"
#include "image2.h"

void run(unsigned x) {
    model_init((void *)x);
    int8_t *inputs = (int8_t *)model_input_ptr(0);
    int8_t *outputs = (int8_t *)model_output_ptr(0);
    memcpy(inputs, image, sizeof(image));     // copy data to inputs
    model_invoke();
    printf("%s (%d%%)\n", outputs[0] > outputs[1] ? "No human" : "Human", (outputs[1]+128)*100/255);
    memcpy(inputs, image2, sizeof(image2));   // copy data to inputs
    model_invoke();
    printf("%s (%d%%)\n", outputs[0] > outputs[1] ? "No human" : "Human", (outputs[1]+128)*100/255);
    chan_out_word(x, FLASH_SERVER_QUIT);
}

extern "C" {
    void inferencer(unsigned x) {
        run(x);
    }
}
