
// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "inference_engine.h"

#define MAX_MODEL_CONTENT_SIZE 100000
static int model_content_bytes = 0;
unsigned char model_content[MAX_MODEL_CONTENT_SIZE];

#define TENSOR_ARENA_SIZE 100000
unsigned char tensor_arena[TENSOR_ARENA_SIZE];

static int input_bytes = 0;
static int input_size;
static unsigned char *input_buffer;

static int output_size;
static unsigned char *output_buffer;

enum ReceiveState { Model, Input };
static enum ReceiveState state;

#ifdef XCORE

#include "xscope.h"

void app_main() {}

void send_output() {
  xscope_bytes(OUTPUT_TENSOR, output_size,
               (const unsigned char *)output_buffer);
}

void app_data(void *data, size_t size) {
  if (strncmp(data, "START_MODEL", 11) == 0) {
    model_content_bytes = 0;
    state = Model;
    return;
  } else if (strncmp(data, "END_MODEL", 9) == 0) {
    // Note, initialize will log error and exit if initialize fails
    initialize(model_content, tensor_arena, TENSOR_ARENA_SIZE, &input_buffer,
               &input_size, &output_buffer, &output_size);
    input_bytes = 0;
    state = Input;
    return;
  }

  switch (state) {
    case Model:
      // printf("Model size=%d\n", size);
      memcpy(model_content + model_content_bytes, data, size);
      model_content_bytes += size;
      if (model_content_bytes > MAX_MODEL_CONTENT_SIZE) {
        // Return error if too big
        printf("Model exceeds maximum size of %d bytes\n",
               MAX_MODEL_CONTENT_SIZE);
        exit(1);
      }
      break;
    case Input:
      memcpy(input_buffer + input_bytes, data, size);
      input_bytes += size;
      if (input_bytes == input_size) {
        invoke();
        send_output();
        input_bytes = 0;
      }
      break;
  }
}

#else  // must be x86

static int load_model(const char *filename, char **buffer, size_t *size) {
  FILE *fd = fopen(filename, "rb");
  fseek(fd, 0, SEEK_END);
  size_t fsize = ftell(fd);

  *buffer = (char *)malloc(fsize);

  fseek(fd, 0, SEEK_SET);
  fread(*buffer, 1, fsize, fd);
  fclose(fd);

  *size = fsize;

  return 1;
}

static int load_input(const char *filename, char *input, size_t esize) {
  FILE *fd = fopen(filename, "rb");
  fseek(fd, 0, SEEK_END);
  size_t fsize = ftell(fd);

  if (fsize != esize) {
    printf("Incorrect input file size. Expected %d bytes.\n", esize);
    return 0;
  }

  fseek(fd, 0, SEEK_SET);
  fread(input, 1, fsize, fd);
  fclose(fd);

  return 1;
}

static int save_output(const char *filename, const char *output, size_t osize) {
  FILE *fd = fopen(filename, "wb");
  fwrite(output, sizeof(int8_t), osize, fd);
  fclose(fd);

  return 1;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("Three arguments expected: mode.tflite input-file output-file\n");
    return -1;
  }

  char *model_filename = argv[1];
  char *input_filename = argv[2];
  char *output_filename = argv[3];

  // load model
  char *model_content = NULL;
  size_t model_size;
  if (!load_model(model_filename, &model_content, &model_size)) {
    printf("error loading model filename=%s\n", model_filename);
    return -1;
  }

  // setup runtime
  initialize(model_content, tensor_arena, TENSOR_ARENA_SIZE, &input_buffer,
             &input_size, &output_buffer, &output_size);

  // Load input tensor
  if (!load_input(input_filename, input_buffer, input_size)) {
    printf("error loading input filename=%s\n", input_filename);
    return -1;
  }

  // Run inference, and report any error
  invoke();

  // save output
  if (!save_output(output_filename, output_buffer, output_size)) {
    printf("error saving output filename=%s\n", output_filename);
    return -1;
  }
  return 0;
}

#endif