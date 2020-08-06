
// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "cifar10_model.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_device_memory.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_interpreter.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_profiler.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_reporter.h"
#include "tensorflow/lite/version.h"

tflite::ErrorReporter *reporter = nullptr;
tflite::Profiler *profiler = nullptr;
const tflite::Model *model = nullptr;
tflite::micro::xcore::XCoreInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
constexpr int kTensorArenaSize = 300000;
uint8_t tensor_arena[kTensorArenaSize];

__attribute__((aligned(8))) static char swmem_handler_stack[1024];

static int argmax(const int8_t *A, const int N) {
  assert(N > 0);

  int m = 0;

  for (int i = 1; i < N; i++) {
    if (A[i] > A[m]) {
      m = i;
    }
  }

  return m;
}

static int load_test_input(const char *filename, char *input, size_t esize) {
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

static void setup_tflite() {
  // Set up logging
  static tflite::micro::xcore::XCoreReporter xcore_reporter;
  reporter = &xcore_reporter;
  // Set up profiling.
  static tflite::micro::xcore::XCoreProfiler xcore_profiler(reporter);
  profiler = &xcore_profiler;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(cifar10_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  static tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddSoftmax();
  resolver.AddPad();
  resolver.AddCustom("XC_maxpool2d",
                     tflite::ops::micro::xcore::Register_MaxPool2D());
  resolver.AddCustom("XC_fc_deepin_anyout",
                     tflite::ops::micro::xcore::Register_FullyConnected_16());
  resolver.AddCustom("XC_conv2d_shallowin",
                     tflite::ops::micro::xcore::Register_Conv2D_Shallow());
  resolver.AddCustom("XC_conv2d_deep",
                     tflite::ops::micro::xcore::Register_Conv2D_Deep());
  resolver.AddCustom("XC_requantize_16_to_8",
                     tflite::ops::micro::xcore::Register_Requantize_16_to_8());

  // Build an interpreter to run the model with
  static tflite::micro::xcore::XCoreInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, reporter, true,
      profiler);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
}

int main(int argc, char *argv[]) {
#if defined(USE_SWMEM) || defined(USE_EXTMEM)
  // start SW_Mem handler
  swmem_setup();
  size_t stack_words;
  GET_STACKWORDS(stack_words, swmem_handler);
  run_async(swmem_handler, NULL,
            stack_base(swmem_handler_stack, stack_words + 2));
#endif

  setup_tflite();

  if (argc > 1) {
    printf("Input filename = %s\n", argv[1]);
    // Load input tensor
    if (!load_test_input(argv[1], input->data.raw, input->bytes)) return -1;
  } else {
    printf("No input file\n");
    memset(input->data.raw, 0, input->bytes);
  }

  // Run inference, and report any error
  printf("Running inference...\n");
  TfLiteStatus invoke_status = interpreter->Invoke();

  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(reporter, "Invoke failed on input filename=%s\n",
                         argv[1]);
    return -1;
  }

  char classification[12] = {0};
  int m = argmax(output->data.int8, 10);

  switch (m) {
    case 0:
      snprintf(classification, 9, "Airplane");
      break;
    case 1:
      snprintf(classification, 11, "Automobile");
      break;
    case 2:
      snprintf(classification, 5, "Bird");
      break;
    case 3:
      snprintf(classification, 4, "Cat");
      break;
    case 4:
      snprintf(classification, 5, "Deer");
      break;
    case 5:
      snprintf(classification, 4, "Dog");
      break;
    case 6:
      snprintf(classification, 5, "Frog");
      break;
    case 7:
      snprintf(classification, 6, "Horse");
      break;
    case 8:
      snprintf(classification, 5, "Ship");
      break;
    case 9:
      snprintf(classification, 6, "Truck");
      break;
    default:
      break;
  }
  printf("Classification = %s\n", classification);

  return 0;
}
