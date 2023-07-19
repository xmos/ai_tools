#include <xcore/chanend.h>
#include <xcore/channel.h>
#include <xcore/parallel.h>
#include <xcore/port.h>
#include <stdio.h>
#include <string.h>
#include "flash_server.h"
#include "model1.tflite.h"
#include "model2.tflite.h"

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
DECLARE_JOB(nn_thread,  (chanend_t, chanend_t));

void nn_thread(chanend_t flash1, chanend_t flash2) {
    model1_init((void *)flash1);
    int8_t *inputs1 = (int8_t *)model1_input_ptr(0);
    int8_t *outputs1 = (int8_t *)model1_output_ptr(0);
    model2_init((void *)flash2);
    int8_t *inputs2 = (int8_t *)model2_input_ptr(0);
    int8_t *outputs2 = (int8_t *)model2_output_ptr(0);
    memcpy(inputs1, image, sizeof(image));     // copy data to inputs
    model1_invoke();
    printf("%s (%d%%)\n", outputs1[0] > outputs1[1] ? "No human" : "Human", (outputs1[1]+128)*100/255);

    memcpy(inputs2, image2, sizeof(image2));   // copy data to inputs
    model2_invoke();
    printf("%s (%d%%)\n", outputs2[0] > outputs2[1] ? "No human" : "Human", (outputs2[1]+128)*100/255);
    chan_out_word(flash1, FLASH_SERVER_QUIT);
}

#define NNETWORKS 2

int main(void) {
    flash_t headers[NNETWORKS];
    chanend_t flash_server_chanends[NNETWORKS];
    channel_t c_flash1     = chan_alloc();
    channel_t c_flash2     = chan_alloc();
    flash_server_chanends[0] = c_flash1.end_a;
    flash_server_chanends[1] = c_flash2.end_a;
    PAR_JOBS(
        PJOB(flash_server, (flash_server_chanends, headers, NNETWORKS,
                            &qspi, &flash_spec[0], NFLASH_SPECS)),
        PJOB(nn_thread,  (c_flash1.end_b, c_flash2.end_b))
        );
}
