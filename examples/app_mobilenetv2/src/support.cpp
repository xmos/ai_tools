#include <platform.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <xcore/channel.h>
#include "model.tflite.h"
#include <lion.h>

// The sample input image is initialized at the beginning of the tensor arena.
// Before we run inference, the input image is copied to the input tensor
// location in the tensor arena.
// With this optimization, we don't need an extra array to store the input
// image. Sample input image is of a LION and of size 160x160x3 = 76800 bytes
uint8_t tensor_arena[LARGEST_TENSOR_ARENA_SIZE] __attribute__((aligned(8))) =
    LION_IMAGE;
#define LION_CLASS 291

void init(unsigned flash_data) { model_init((void *)flash_data); }

void run() {
  int8_t *p = model_input(0)->data.int8;
  // Copy the input image into input tensor location
  // The input image values are between 0 and 255
  // Adjust the input image values to be between -128 and 127
  for (int i = 0; i < model_input_size(0); ++i) {
    p[i] = tensor_arena[i] - 128;
  }

  model_invoke();

  int maxIndex = -1;
  int max = -128;
  int8_t *out = model_output(0)->data.int8;
  for (int i = 0; i < model_output_size(0); ++i) {
    if (out[i] > max) {
      max = out[i];
      maxIndex = i;
    }
  }
  if (maxIndex == LION_CLASS) {
    printf("\nCorrect - Inferred class is LION!\n");
  } else {
    printf("\nIncorrect class!\n");
  }
}

extern "C" {
void model_init(unsigned flash_data) { init(flash_data); }

void inference() { run(); }
}
