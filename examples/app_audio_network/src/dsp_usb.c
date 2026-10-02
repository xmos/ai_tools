#include <xcore/channel.h>
#include <xcore/port.h>
#include <xcore/chanend.h>
#include <xcore/parallel.h>

static chanend_t g_c_to_dspc;

void UserBufferManagementInit()
{
}

extern void nn_offload_data_to_dsp_engine(chanend_t c_to_dspc, unsigned sampstoNN[], unsigned fromNN[]);

void UserBufferManagement(unsigned sampsFromUsbToAudio[], unsigned sampsFromAudioToUsb[])
{
    nn_offload_data_to_dsp_engine(g_c_to_dspc, sampsFromUsbToAudio, sampsFromAudioToUsb);
}

DECLARE_JOB(nn_dsp_thread, (uint32_t, chanend_t, chanend_t));
DECLARE_JOB(nn_data_transport_thread, (chanend_t, chanend_t));

void dsp_main(chanend_t c_button_state) {
    channel_t c_data = chan_alloc();
    channel_t t;
    t = chan_alloc();
    g_c_to_dspc = c_data.end_a;

    PAR_JOBS(
        PJOB(nn_dsp_thread, (0, t.end_a, c_button_state)),
        PJOB(nn_data_transport_thread, (c_data.end_b, t.end_b))
        );
}
