
// Copyright (c) 2019, XMOS Ltd, All rights reserved

#include "ai_engine.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "lib_ops/api/benchmarking.h"
#include "lib_ops/api/lib_ops.h"
#include "mobilenet_ops_resolver.h"
#include "mobilenet_v1.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/version.h"

tflite::ErrorReporter *error_reporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
constexpr int kTensorArenaSize = 150108;
uint8_t tensor_arena[kTensorArenaSize];

xcore::Dispatcher *dispatcher = nullptr;
constexpr int num_threads = 5;
constexpr int kXCOREHeapSize = 30024;
uint8_t xcore_heap[kXCOREHeapSize];

void ai_invoke() {
  // Run inference, and report any error
  printf("Running inference...\n");
  xcore::Stopwatch sw;
  sw.Start();
  TfLiteStatus invoke_status = interpreter->Invoke();
  sw.Stop();

  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
  }

  printf("Inference duration %u (us)\n", sw.GetEllapsedMicroseconds());
}

void ai_initialize(unsigned char **input, unsigned *input_size,
                   unsigned char **output, unsigned *output_size) {
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
  static xcore::Dispatcher static_dispatcher(xcore_heap, kXCOREHeapSize,
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
  *input = (unsigned char *)(interpreter->input(0)->data.raw);
  *input_size = interpreter->input(0)->bytes;
  *output = (unsigned char *)(interpreter->output(0)->data.raw);
  *output_size = interpreter->output(0)->bytes;
}
