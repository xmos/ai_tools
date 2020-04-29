
// Copyright (c) 2019, XMOS Ltd, All rights reserved

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "lib_ops/api/lib_ops.h"
#include "lib_ops/api/stopwatch.h"
#include "mobilenet_ops_resolver.h"
#include "mobilenet_v1.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/version.h"

tflite::ErrorReporter *error_reporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
constexpr int kTensorArenaSize = 148700;
uint8_t tensor_arena[kTensorArenaSize];

xcore::Dispatcher *dispatcher = nullptr;
constexpr int num_threads = 5;
constexpr int kXCOREArenaSize = 8000;
uint8_t xcore_arena[kXCOREArenaSize];

static int load_input(const char *filename, char *input, size_t esize) {
  FILE *fd = fopen(filename, "rb");
  fseek(fd, 0, SEEK_END);
  size_t fsize = ftell(fd);

  if (fsize != esize) {
    printf("Incorrect input file size. Expected %d bytes.\n", esize);
    return 0;
  }

  fseek(fd, 0, SEEK_SET);
  fread(input, 1, esize, fd);
  fclose(fd);

  return 1;
}

static void print_output(const char *output, size_t osize) {
  for (int i = 0; i < osize; i++) {
    printf("i=%u   output=%d\n", i, output[i]);
  }
}

static void setup_tflite() {
  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(mobilenet_v1_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Setup xCORE dispatcher (BEFORE calling AllocateTensors)
  static xcore::Dispatcher static_dispatcher(xcore_arena, kXCOREArenaSize,
                                             num_threads);
  xcore::XCoreStatus xcore_status = xcore::InitializeXCore(&static_dispatcher);
  if (xcore_status != xcore::kXCoreOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "InitializeXCore() failed");
    return;
  }
  dispatcher = &static_dispatcher;

  // This pulls in all the operation implementations we need.
  static tflite::ops::micro::xcore::MobileNetOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_tensors_status = interpreter->AllocateTensors();
  if (allocate_tensors_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
}

int main(int argc, char *argv[]) {
  // setup runtime
  setup_tflite();

  if (argc > 1) {
    printf("input filename=%s\n", argv[1]);
    // Load input tensor
    if (!load_input(argv[1], input->data.raw, input->bytes)) return -1;
  } else {
    printf("no input file\n");
    memset(input->data.raw, 0, input->bytes);
  }

  // Run inference, and report any error
  printf("Running inference...\n");
  xcore::Stopwatch sw;
  sw.Start();
  TfLiteStatus invoke_status = interpreter->Invoke();
  sw.Stop();
  printf("Inference duration %u (us)\n", sw.GetEllapsedMicroseconds());

  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
    return -1;
  }

  print_output(output->data.raw, output->bytes);
  return 0;
}
