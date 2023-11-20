#ifndef USER_MAIN_H
#define USER_MAIN_H

#ifdef __XC__

#include "i2c.h"
#include <print.h>
#include <xs1.h>
#include <platform.h>

extern unsafe client interface i2c_master_if i_i2c_client;
extern void interface_saver(client interface i2c_master_if i);
extern void ctrlPort();

/* I2C interface ports */
extern port p_scl;
extern port p_sda;

extern void dsp_main(();

#define USER_MAIN_DECLARATIONS \
    interface i2c_master_if i2c[1]; \
    chan c_hid_tuning_from_host, c_hid_tuning_to_host;

#define USER_MAIN_CORES \
    on tile[0]: {                                                       \
        ctrlPort();                                                     \
        i2c_master(i2c, 1, p_scl, p_sda, 100);                          \
    }                                                                   \
    on tile[1]: {      /* create DSP process for AWE processing */      \
        unsafe                                                          \
        {                                                               \
            i_i2c_client = i2c[0];      /* TODO: delete, audiohw on tile 1 */              \
        }                                                               \
        dsp_main();         \
    }
#endif                  /* End of DSP process */

#endif
