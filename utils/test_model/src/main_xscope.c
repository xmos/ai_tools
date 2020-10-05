
// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference_engine.h"
#include "xscope.h"

// USE RAM
// #define MAX_MODEL_CONTENT_SIZE 50000
// unsigned char model_content[MAX_MODEL_CONTENT_SIZE];
// static size_t model_received_bytes = 0;
// #define TENSOR_ARENA_SIZE 125000
// unsigned char tensor_arena[TENSOR_ARENA_SIZE];

// USE DDR
#define MAX_MODEL_CONTENT_SIZE 500000
__attribute__((section(
    ".ExtMem_data"))) unsigned char model_content[MAX_MODEL_CONTENT_SIZE];
static size_t model_received_bytes = 0;
#define TENSOR_ARENA_SIZE 200000
unsigned char tensor_arena[TENSOR_ARENA_SIZE];

static int tensor_index = -1;
static size_t tensor_size = 0;
static size_t tensor_received_bytes = 0;

static size_t input_size;
static unsigned char *input_buffer;

static size_t output_size;
static unsigned char *output_buffer;

enum AppState { Model, SetTensor, Invoke, GetTensor };
static enum AppState state;

void xscope_main() {}

void send_tensor(void *buffer, size_t size) {
  xscope_bytes(GET_TENSOR, size, (const unsigned char *)buffer);
}

void xscope_data(void *data, size_t size) {
  void *tensor_buffer;

  // Handle state protocol messages
  if (strncmp(data, "PING_RECV", 9) == 0) {
    printf("Received PING_RECV\n");
    xscope_int(PING_AWK, 0);
    return;
  } else if (strncmp(data, "START_MODEL", 11) == 0) {
    printf("Received START_MODEL\n");
    state = Model;
    model_received_bytes = 0;
    return;
  } else if (strncmp(data, "END_MODEL", 9) == 0) {
    printf("Received END_MODEL\n");
    // Note, initialize will log error and exit if initialize fails
    initialize(model_content, tensor_arena, TENSOR_ARENA_SIZE, &input_buffer,
               &input_size, &output_buffer, &output_size);
    printf("Inference engine initialized\n");
    return;
  } else if (strncmp(data, "SET_TENSOR", 9) == 0) {
    printf("Received SET_TENSOR\n");
    state = SetTensor;
    tensor_received_bytes = 0;
    sscanf(data, "SET_TENSOR %d %d", &tensor_index, &tensor_size);
    printf("SET_TENSOR index=%d   size=%d\n", tensor_index, tensor_size);
    return;
  } else if (strncmp(data, "GET_TENSOR", 9) == 0) {
    printf("Received GET_TENSOR\n");
    state = GetTensor;
    sscanf(data, "GET_TENSOR %d", &tensor_index);
    get_tensor_bytes(tensor_index, &tensor_buffer, &tensor_size);
    printf("GET_TENSOR index=%d  size=%d\n", tensor_index, tensor_size);
    send_tensor(tensor_buffer, tensor_size);
    return;
  } else if (strncmp(data, "CALL_INVOKE", 11) == 0) {
    printf("Received INVOKE\n");
    state = Invoke;
    invoke();
    printf("Invoke compete\n");
    return;
  }

  // Handle data payload messages
  switch (state) {
    case Model:
      // printf("%d    %d\n", model_received_bytes, size);
      memcpy(model_content + model_received_bytes, data, size);
      model_received_bytes += size;
      // printf("    %d    %d\n", model_received_bytes, MAX_MODEL_CONTENT_SIZE);

      if (model_received_bytes > MAX_MODEL_CONTENT_SIZE) {
        // Return error if too big
        printf("Model exceeds maximum size of %d bytes\n",
               MAX_MODEL_CONTENT_SIZE);
        exit(1);
      }
      xscope_int(RECV_AWK, 0);
      break;
    case SetTensor:
      get_tensor_bytes(tensor_index, &tensor_buffer, &tensor_size);
      memcpy(tensor_buffer + tensor_received_bytes, data, size);
      tensor_received_bytes += size;
      if (tensor_received_bytes > tensor_size) {
        // Return error if too big
        printf("Tensor exceeds size of %d bytes\n", tensor_size);
        exit(1);
      }
      xscope_int(RECV_AWK, 0);
      break;
    case Invoke:
    case GetTensor:
      break;
  }
}
