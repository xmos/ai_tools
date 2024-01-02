#include <string.h>
#include <print.h>

#include "model_audio.tflite.h"

#define MEL_BINS 64
#define MEL_Q_VALUE       30
#define MEL_ONE_VALUE     (1 << MEL_Q_VALUE)

extern "C" {

    int32_t output_lookup[256] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33,
        34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 52,
        53, 54, 55, 56, 58, 59, 60, 61, 63, 64, 65, 66, 68, 69, 70, 71,
        73, 74, 75, 77, 78, 79, 81, 82, 83, 84, 86, 87, 88, 90, 91, 93,
        94, 95, 97, 98, 99,101,102,104,105,106,108,109,111,112,114,115,
        116,118,119,121,122,124,125,127,128,130,131,133,134,136,137,139,
        141,142,144,145,147,148,150,151,153,155,156,158,160,161,163,164,
        166,168,169,171,173,174,176,178,179,181,183,185,186,188,190,192,
        193,195,197,199,200,202,204,206,208,209,211,213,215,217,219,220,
        222,224,226,228,230,232,234,236,237,239,241,243,245,247,249,251,
        253,255,257,259,261,263,265,267,269,271,273,275,278,280,282,284,
        286,288,290,292,294,297,299,301,303,305,308,310,312,314,316,319,
        321,323,325,328,330,332,335,337,339,342,344,346,349,351,353,356,
        358,361,363,365,368,370,373,375,378,380,383,385,388,390,393,395,
        398,400,403,405,408,411,413,416,418,421,424,426,429,432,434,437,
    };
    
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
#if 0
    if (skip == 0) {
        for(int i = 0; i < MEL_BINS; i++) {
            inputs_saved[cnt][i] = inputs[i];
        }
    }
#endif
    model_invoke();
    for(int i = 0; i < MEL_BINS; i++) {
        masks[i] = output_lookup[outputs[i]+128] << (MEL_Q_VALUE-8);
    }
#if 0
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
#endif
    memcpy(state, state_outputs, 64);
}

};
