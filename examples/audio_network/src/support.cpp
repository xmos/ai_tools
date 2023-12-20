#include <string.h>
#include <print.h>

#include "model_audio.tflite.h"

#define MEL_BINS 64
#define MEL_Q_VALUE       30
#define MEL_ONE_VALUE     (1 << MEL_Q_VALUE)

extern "C" {

    
int8_t state[64];
    int cnt = 0;
    int skip = 10000;
    int inputs_saved[16][64];
    int outputs_saved[16][64];
    
void nn_predict_masks(int *masks, int *mels) {
    model_init(NULL);
    int8_t *inputs = (int8_t *)model_input_ptr(0);
    int8_t *state_inputs = (int8_t *)model_input_ptr(1);
    int8_t *state_outputs = (int8_t *)model_output_ptr(0);
    int8_t *outputs = (int8_t *)model_output_ptr(1);
    memcpy(state_inputs, state, 64);
    for(int i = 0; i < MEL_BINS; i++) {
        int x = mels[i];
        if (x < -128) x = -128;
        if (x >  127) x = 127;
        inputs[i] = x;
    }
    if (skip == 0) {
        for(int i = 0; i < MEL_BINS; i++) {
            inputs_saved[cnt][i] = inputs[i];
        }
    }
    model_invoke();
    for(int i = 0; i < MEL_BINS; i++) {
        masks[i] = (outputs[i]+128) << (MEL_Q_VALUE-8);
    }
    if (skip == 0) {
        for(int i = 0; i < MEL_BINS; i++) {
            outputs_saved[cnt][i] = outputs[i];
        }
        cnt++;
        if (cnt == 16) {
            for(int i = 0; i < MEL_BINS; i++) {
                for(int c = 0; c < cnt; c++) {
                    printint(inputs_saved[c][i]);
                    printchar(' ');
                }
                printchar('\n');
            }
            printstr("outputs\n");
            for(int i = 0; i < MEL_BINS; i++) {
                for(int c = 0; c < cnt; c++) {
                    printint(outputs_saved[c][i]);
                    printchar(' ');
                }
                printchar('\n');
            }
            _exit(0);
        }
    } else {
        skip--;
    }
    memcpy(state, state_outputs, 64);
}

};
