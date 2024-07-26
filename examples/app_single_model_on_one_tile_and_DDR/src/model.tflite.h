// This file is generated. Do not edit.
// Generated on: 24.07.2024 21:50:48

#ifndef model_GEN_H
#define model_GEN_H

#include "tensorflow/lite/c/common.h"

#ifdef SHARED_TENSOR_ARENA
  #ifndef LARGEST_TENSOR_ARENA_SIZE
    #define LARGEST_TENSOR_ARENA_SIZE 258472
  #elif LARGEST_TENSOR_ARENA_SIZE < 258472
    #define LARGEST_TENSOR_ARENA_SIZE 258472
  #endif
#endif

// Sets up the model with init and prepare steps.
TfLiteStatus model_init(void *flash_data);
// Returns the input tensor with the given index.
TfLiteTensor *model_input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *model_output(int index);
// Runs inference for the model.
TfLiteStatus model_invoke();
// Resets variable tensors in the model.
// This should be called after invoking a model with stateful ops such as LSTM.
TfLiteStatus model_reset();

// Returns the number of input tensors.
inline size_t model_inputs() {
  return 1;
}
// Returns the number of output tensors.
inline size_t model_outputs() {
  return 1;
}

inline void *model_input_ptr(int index) {
  return model_input(index)->data.data;
}
inline size_t model_input_size(int index) {
  return model_input(index)->bytes;
}
inline int model_input_dims_len(int index) {
  return model_input(index)->dims->data[0];
}
inline int *model_input_dims(int index) {
  return &model_input(index)->dims->data[1];
}

inline void *model_output_ptr(int index) {
  return model_output(index)->data.data;
}
inline size_t model_output_size(int index) {
  return model_output(index)->bytes;
}
inline int model_output_dims_len(int index) {
  return model_output(index)->dims->data[0];
}
inline int *model_output_dims(int index) {
  return &model_output(index)->dims->data[1];
}
// Only returns valid value if input is quantized
inline int32_t model_input_zeropoint(int index) {
  return model_input(index)->params.zero_point;
}
// Only returns valid value if input is quantized
inline float model_input_scale(int index) {
  return model_input(index)->params.scale;
}
// Only returns valid value if output is quantized
inline int32_t model_output_zeropoint(int index) {
  return model_output(index)->params.zero_point;
}
// Only returns valid value if output is quantized
inline float model_output_scale(int index) {
  return model_output(index)->params.scale;
}

// Sets up the model part of ioserver to communicate 
// with this model from host.
// Requires that ioserver() has been setup and running.
// This is an infinite loop and does not exit.
TfLiteStatus model_ioserver(unsigned io_channel);

#endif
