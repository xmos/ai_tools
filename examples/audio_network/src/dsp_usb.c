#include <xcore/channel.h>
#include <xcore/port.h>
#include "awe_xcore.h"

static chanend_t g_c_to_dspc;

void UserBufferManagementInit()
{
}

void UserBufferManagement(unsigned sampsFromUsbToAudio[], unsigned sampsFromAudioToUsb[])
{
    dsp_offload_data_to_nn(g_c_to_dspc, sampsFromUsbToAudio, sampsFromAudioToUsb);
}


void dsp_main(chanend_t c_tuning_from_host, chanend_t c_tuning_to_host) {
    channel_t c_data = chan_alloc();
    g_c_to_dspc = c_data.end_a;

    dsp_main(c_tuning_from_host, c_tuning_to_host, c_data.end_b);
}
