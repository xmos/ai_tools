#include <xcore/channel.h>
#include <xcore/select.h>
#include <stdio.h>
#include "assert.h"
#include "nn_xcore_internal.h"

void nn_offload_data_to_dsp_engine(chanend_t c_to_dspc, unsigned sampstoNN[], unsigned fromNN[])
{
    // TODO: reroll these loops
    chanend_out_word(c_to_dspc, sampstoNN[0]);
    chanend_out_word(c_to_dspc, sampstoNN[1]);
    chanend_out_word(c_to_dspc, sampstoNN[2]);
    chanend_out_word(c_to_dspc, sampstoNN[3]);
    chanend_out_word(c_to_dspc, sampstoNN[4]);
    chanend_out_word(c_to_dspc, sampstoNN[5]);
    chanend_out_word(c_to_dspc, sampstoNN[6]);
    chanend_out_word(c_to_dspc, sampstoNN[7]);
    chanend_out_word(c_to_dspc, fromNN[0]);
    chanend_out_word(c_to_dspc, fromNN[1]);
    chanend_out_word(c_to_dspc, fromNN[2]);
    chanend_out_word(c_to_dspc, fromNN[3]);
    chanend_out_word(c_to_dspc, fromNN[4]);
    chanend_out_word(c_to_dspc, fromNN[5]);
    chanend_out_word(c_to_dspc, fromNN[6]);
    chanend_out_word(c_to_dspc, fromNN[7]);
    chanend_out_end_token(c_to_dspc);
    sampstoNN[0] = chanend_in_word(c_to_dspc);
    sampstoNN[1] = chanend_in_word(c_to_dspc);
    sampstoNN[2] = chanend_in_word(c_to_dspc);
    sampstoNN[3] = chanend_in_word(c_to_dspc);
    sampstoNN[4] = chanend_in_word(c_to_dspc);
    sampstoNN[5] = chanend_in_word(c_to_dspc);
    sampstoNN[6] = chanend_in_word(c_to_dspc);
    sampstoNN[7] = chanend_in_word(c_to_dspc);
    fromNN[0] = chanend_in_word(c_to_dspc);
    fromNN[1] = chanend_in_word(c_to_dspc);
    fromNN[2] = chanend_in_word(c_to_dspc);
    fromNN[3] = chanend_in_word(c_to_dspc);
    fromNN[4] = chanend_in_word(c_to_dspc);
    fromNN[5] = chanend_in_word(c_to_dspc);
    fromNN[6] = chanend_in_word(c_to_dspc);
    fromNN[7] = chanend_in_word(c_to_dspc);
    chanend_check_end_token(c_to_dspc);
}

#include "dsp.h"

int32_t data_processed       [UPSAMPLED_WINDOW_ADVANCE][NN_OUTPUT_CHANNELS],
int32_t data_to_be_processed [UPSAMPLED_WINDOW_ADVANCE][NN_INPUT_CHANNELS],

/** Function that transfers data to and from the NN subsystem. This
 * provides one fundamental block worth of audio frames, and reads one
 * fundamental block.
 *
 * \param  data_from_nn 32-bit sample values that have been processed
 * \param  data_to_nn   32-bit sample values ready for audio processing
 */
static void data_to_and_from_nn(
    int32_t data_from_nn[UPSAMPLED_WINDOW_ADVANCE][NN_OUTPUT_CHANNELS],
    int32_t data_to_nn  [UPSAMPLED_WINDOW_ADVANCE][NN_INPUT_CHANNELS],
    chanend_t c_dsp_threads[]) {
    memcpy(data_to_be_processed, data_to_nn, 4 * UPSAMPLED_WINDOW_ADVANCE * NN_INPUT_CHANNELS);
    memcpy(data_from_nn, data_processed, 4 * UPSAMPLED_WINDOW_ADVANCE * NN_OUTPUT_CHANNELS);
    chanend_out_end_token(c_dsp_threads[0]);
}

/** Function that blocks data up for the NN stack, and when ready starts the NN
 * threads.
 */
void nn_data_transport_thread(chanend_t c_data, chanend_t c_children[]) {
    int32_t input_data[UPSAMPLED_WINDOW_ADVANCE][NN_INPUT_CHANNELS];
    int32_t output_data[UPSAMPLED_WINDOW_ADVANCE][NN_OUTPUT_CHANNELS];
    int frame = 0;
    while(1) {
        for(int i = 0; i < AUDIO_INPUT_CHANNELS; i++) {
            int x = chanend_in_word(c_data);
            if (i < NN_INPUT_CHANNELS) {
                input_data[frame][i] = x;
            }
        }
        chanend_check_end_token(c_data);
        int nn_channel = 0;
        for(int i = 0; i < AUDIO_OUTPUT_CHANNELS; i++) {
            int x = output_data[frame][nn_channel];
            chanend_out_word(c_data, x);
            nn_channel ++;
            if (nn_channel == NN_OUTPUT_CHANNELS) {
                nn_channel = 0;
            }
        }
        for(int i = 0; i < NN_OUTPUT_CHANNELS; i++) {
            output_data[frame][i] = 0;
        }
        chanend_out_end_token(c_data);
        frame++;
        if (frame == UPSAMPLED_WINDOW_ADVANCE) { // TODO: this may not fit between samples
            frame = 0;
            data_to_and_from_nn(output_data, input_data, c_children);
        }
    }
}

void nn_dsp_thread(uint32_t thread_number,
                    chanend_t c_parent) {
    while(1) {
        chanend_check_end_token(c_parent);
        dsp_preprocess((int *)data_to_be_processed);
        dsp_postprocess((int *)data_processed);
    }
}
