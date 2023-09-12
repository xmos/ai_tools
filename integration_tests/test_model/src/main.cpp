#include <stdint.h>
#include <stdio.h>
#include "model.tflite.h"

int main() {
  if (model_init(NULL)) {
    printf("Error!\n");
  }

  for (int n = 0; n < model_inputs(); ++n) {
    char filename[8];
    sprintf(filename, "in%d", n);
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
      printf("Failed to open file.\n");
      return 1;
    }
    int8_t *in = model_input(n)->data.int8;
    for (int i = 0; i < model_input_size(n); ++i) {
      int byte = fgetc(fp);
      if (byte == EOF) {
        printf("Input too small for model. EOF occured.\n");
        break;
      }
      printf("%d,", byte);
      in[i] = (int8_t)byte;
    }
    printf("\n");
  }

  model_invoke();

  for (int n = 0; n < model_outputs(); ++n) {
    char filename[8];
    sprintf(filename, "out%d", n);
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
      printf("Failed to open file.\n");
      return 1;
    }
    int8_t *out = model_output(n)->data.int8;
    for (int i = 0; i < model_output_size(n); ++i) {
      printf("%d,", (int)out[i]);
    }
    printf("\n");
    fwrite(out, sizeof(int8_t), model_output_size(n) * sizeof(int8_t), fp);
  }
  return 0;
}
