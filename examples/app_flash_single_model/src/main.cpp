#include <xcore/chanend.h>
#include <xcore/channel.h>
#include <xcore/parallel.h>
#include <xcore/port.h>
#include <stdio.h>
#include <string.h>
#include "flash_server.h"
#include "model.tflite.h"

#include "image.h"
#include "image2.h"

#define NFLASH_SPECS 1

fl_QuadDeviceSpec flash_spec[NFLASH_SPECS] = {
    FL_QUADDEVICE_DEFAULT //FL_QUADDEVICE_MACRONIX_MX25R6435FM2IH0
};

fl_QSPIPorts qspi = {
    PORT_SQI_CS,
    PORT_SQI_SCLK,
    PORT_SQI_SIO,
    XS1_CLKBLK_2
};

DECLARE_JOB(flash_server, (chanend_t *, flash_t*, int,
                           fl_QSPIPorts*, fl_QuadDeviceSpec*, int));
DECLARE_JOB(nn_thread,  (chanend_t));

void nn_thread(chanend_t x) {
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

#if 1
#define NNETWORKS 1

int main(void) {
    flash_t headers[NNETWORKS];
    chanend_t flash_server_chanends[NNETWORKS];
    channel_t c_flash     = chan_alloc();
    flash_server_chanends[0] = c_flash.end_a;

    PAR_JOBS(
        PJOB(flash_server, (flash_server_chanends, headers, NNETWORKS,
                            &qspi, &flash_spec[0], NFLASH_SPECS)),
        PJOB(nn_thread,  (c_flash.end_b))
        );
}
#endif
