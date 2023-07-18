#include <xcore/chanend.h>
#include <xcore/channel.h>
#include <xcore/parallel.h>
#include <xcore/port.h>
#include <stdio.h>
#include <string.h>
#include "flash_server.h"
#include "model.tflite.h"
#include "QuadSpecMacros.h"

#include "image.h"
#include "image2.h"

#if 0
#define FL_QUADDEVICE_MACRONIX_MX25R6435FM2IH0 \
{ \
    16,                     /* MX25R6435FM2IH0 */ \
    256,                    /* page size */ \
    32768,                  /* num pages */ \
    3,                      /* address size */ \
    3,                      /* log2 clock divider */ \
    0x9F,                   /* QSPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0xC22817,               /* device id */ \
    0x20,                   /* QSPI_SE */ \
    4096,                   /* Sector erase is always 4KB */ \
    0x06,                   /* QSPI_WREN */ \
    0x04,                   /* QSPI_WRDI */ \
    PROT_TYPE_NONE,         /* no protection */ \
    {{0,0},{0x00,0x00}},    /* QSPI_SP, QSPI_SU */ \
    0x02,                   /* QSPI_PP */ \
    0xEB,                   /* QSPI_READ_FAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* mad sectors */ \
    {4096,{0,{0}}},         /* regular sector sizes */ \
    0x05,                   /* QSPI_RDSR */ \
    0x01,                   /* QSPI_WRSR */ \
    0x01,                   /* QSPI_WIP_BIT_MASK */ \
}

#define FL_QUADDEVICE_MACRONIX_MX25R3235FM1IH0 \
{ \
    15,                     /* MX25R3235FM1IH0 */ \
    256,                    /* page size */ \
    32768,                  /* num pages */ \
    3,                      /* address size */ \
    3,                      /* log2 clock divider */ \
    0x9F,                   /* QSPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0xC22816,               /* device id */ \
    0x20,                   /* QSPI_SE */ \
    4096,                   /* Sector erase is always 4KB */ \
    0x06,                   /* QSPI_WREN */ \
    0x04,                   /* QSPI_WRDI */ \
    PROT_TYPE_NONE,         /* no protection */ \
    {{0,0},{0x00,0x00}},    /* QSPI_SP, QSPI_SU */ \
    0x02,                   /* QSPI_PP */ \
    0xEB,                   /* QSPI_READ_FAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* mad sectors */ \
    {4096,{0,{0}}},         /* regular sector sizes */ \
    0x05,                   /* QSPI_RDSR */ \
    0x01,                   /* QSPI_WRSR */ \
    0x01,                   /* QSPI_WIP_BIT_MASK */ \
}

#define FL_QUADDEVICE_AT25FF321A \
{ \
  0,        /* UNKNOWN */ \
  256,        /* page size */ \
  16384,        /* num pages */ \
  3,        /* address size */ \
  3,        /* log2 clock divider */ \
  0x9F,       /* QSPI_RDID */ \
  0,        /* id dummy bytes */ \
  3,        /* id size in bytes */ \
  0x1F4708,       /* device id */ \
  0x20,       /* QSPI_SE */ \
  4096,       /* Sector erase is always 4KB */ \
  0x06,       /* QSPI_WREN */ \
  0x04,       /* QSPI_WRDI */ \
  PROT_TYPE_SR,     /* Protection via SR */ \
  {{0x3C,0x00},{0,0}},  /* QSPI_SP, QSPI_SU */ \
  0x02,       /* QSPI_PP */ \
  0xEB,       /* QSPI_READ_FAST */ \
  1,        /* 1 read dummy byte */ \
  SECTOR_LAYOUT_REGULAR,  /* mad sectors */ \
  {4096,{0,{0}}},     /* regular sector sizes */ \
  0x05,       /* QSPI_RDSR */ \
  0x01,       /* QSPI_WRSR */ \
  0x01,       /* QSPI_WIP_BIT_MASK */ \
}

#define NFLASH_SPECS 3
fl_QuadDeviceSpec flash_spec[NFLASH_SPECS] = {
    FL_QUADDEVICE_AT25FF321A,
    FL_QUADDEVICE_MACRONIX_MX25R6435FM2IH0,
    FL_QUADDEVICE_MACRONIX_MX25R3235FM1IH0
};

//fl_QuadDeviceSpec flash_spec[NFLASH_SPECS] = {
//    FL_QUADDEVICE_DEFAULT //FL_QUADDEVICE_MACRONIX_MX25R6435FM2IH0
//};

fl_QSPIPorts qspi = {
    PORT_SQI_CS,
    PORT_SQI_SCLK,
    PORT_SQI_SIO,
    XS1_CLKBLK_2
};

#endif

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
    port_enable(   qspi.qspiCS);
    port_set_clock(qspi.qspiCS, 6);
    port_out(      qspi.qspiCS, 1);
                   
    port_enable(   qspi.qspiSCLK);
    port_set_clock(qspi.qspiSCLK, 6);
    port_out(      qspi.qspiSCLK, 0);
                   
    port_start_buffered(qspi.qspiSIO, 32);
    port_set_clock(     qspi.qspiSIO, 6);

    clock_enable(qspi.qspiClkblk);

    PAR_JOBS(
        PJOB(flash_server, (flash_server_chanends, headers, NNETWORKS,
                            &qspi, &flash_spec[0], NFLASH_SPECS)),
        PJOB(nn_thread,  (c_flash.end_b))
        );
}
#endif
