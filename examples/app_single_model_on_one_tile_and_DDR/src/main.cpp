// Copyright 2023 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <platform.h>
#include <xcore/hwtimer.h>
#include <xcore/assert.h>
#include <print.h>

#include "image.h"
#include "labels.h"
#include "model.tflite.h"

extern const int8_t weights[];

static float dequantize_output(int n) {
  return (n - model_output_zeropoint(0)) * model_output_scale(0);
}

static void print_top3_classes(int idx[3], int8_t top[3]) {
    printf("Top 3 classes:\n");
    for (unsigned i = 0; i < 3; i++) {
        float prob = dequantize_output(top[i]);
        printf("Top %d, class:%u, prob:%.2f, label:'%s'\n", i, idx[i], prob, imagenet_classes[idx[i]]);
    }
}

static void find_top3_classes(
    int8_t *out, 
    size_t out_size, 
    int idx[3],
    int8_t top[3]
) {
    for (size_t i = 0; i < out_size; i++) {
        if (out[i] > top[0]) {
            top[2] = top[1];idx[2] = idx[1];
            top[1] = top[0];idx[1] = idx[0];
            top[0] = out[i];idx[0] = i;
        } else if (out[i] > top[1]) {
            top[2] = top[1];idx[2] = idx[1];
            top[1] = out[i];idx[1] = i;
        } else if (out[i] > top[2]) {
            top[2] = out[i];idx[2] = i;
        }
    }
}

void run() {
    // model Init
    printf("Init Mobilenet DDR model\n");
    model_init((void*)&weights);

    // sizes and I/O
    const size_t input_size = model_input_size(0);
    const size_t output_size = model_output_size(0);
    int8_t* inputs = (int8_t*)model_input_ptr(0);
    int8_t* outputs = (int8_t*)model_output_ptr(0);
    printf("Input size = %zu\n", input_size);
    printf("Output size = %zu\n", output_size);

    // set input
    for (unsigned i = 0; i < input_size; i++) {
        inputs[i] = lion[i] - 128;
    }

    // invoke
    model_invoke();

    // Find top three classes
    int idx[3] = {-1,-1,-1};
    int8_t top[3] = {INT8_MIN, INT8_MIN, INT8_MIN};
    find_top3_classes(outputs, output_size, idx, top);
    
    // Print output
    print_top3_classes(idx, top);
}

int main() {
    run();
    return 0;
}
