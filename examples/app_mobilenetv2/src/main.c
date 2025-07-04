// Copyright 2023 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <xcore/channel.h>
#include <xcore/parallel.h>

#include "flash_server.h"
#include "stdio.h"
#include <platform.h>
#include <quadflash.h>
#include <stdint.h>


DECLARE_JOB(f_server, (chanend_t*, flash_t*, int,
                       fl_QSPIPorts *, fl_QuadDeviceSpec*,
                       int) );

DECLARE_JOB(f_client, (chanend_t));


flash_t headers[2];

#define NFLASH_SPECS 1

// #define FL_QUADDEVICE_DEFAULT {0,0}
// #define PORT_SQI_CS XS1_PORT_1B
// #define PORT_SQI_SCLK XS1_PORT_1C
// #define PORT_SQI_SIO XS1_PORT_4B

fl_QuadDeviceSpec flash_spec[NFLASH_SPECS] = {
    FL_QUADDEVICE_DEFAULT //FL_QUADDEVICE_MACRONIX_MX25R6435FM2IH0
};

fl_QSPIPorts qspi = {
    PORT_SQI_CS,
    PORT_SQI_SCLK,
    PORT_SQI_SIO,
    XS1_CLKBLK_2
};


extern void init(void * c_flash);
extern void inference();

void f_server(chanend_t c_flash[], flash_t headers[], int n_flash,
              fl_QSPIPorts *qspi, fl_QuadDeviceSpec flash_spec[],
              int n_flash_spec) {
    flash_server(c_flash, headers, n_flash, qspi, flash_spec, n_flash_spec);
}

void f_client(chanend_t c_flash) {
    init((void*)c_flash);
    inference();
    chan_out_word(c_flash, FLASH_SERVER_QUIT);
}

int main() {
    channel_t a = chan_alloc();
    chanend_t fs[1] = {a.end_a};
    
    PAR_JOBS(
        PJOB(f_server, (fs,headers,1,&qspi,flash_spec,NFLASH_SPECS)),
        PJOB(f_client, (a.end_b)));

    return 0;
}
