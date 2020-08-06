
// Copyright (c) 2019, XMOS Ltd, All rights reserved

#include "inference_engine.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "mobilenet_ops_resolver.h"
#include "mobilenet_v1.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_device_memory.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_interpreter.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_profiler.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_reporter.h"
#include "tensorflow/lite/version.h"

tflite::ErrorReporter *reporter = nullptr;
tflite::Profiler *profiler = nullptr;
const tflite::Model *model = nullptr;
tflite::micro::xcore::XCoreInterpreter *interpreter = nullptr;
constexpr int kTensorArenaSize = 220000 + 140000;
uint8_t tensor_arena[kTensorArenaSize];

void invoke() {
  // Run inference, and report any error
  printf("Running inference...\n");
  TfLiteStatus invoke_status = interpreter->Invoke();

  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(reporter, "Invoke failed\n");
  }
}

void initialize(unsigned char **input, unsigned *input_size,
                unsigned char **output, unsigned *output_size) {
  // Set up logging
  static tflite::micro::xcore::XCoreReporter xcore_reporter;
  reporter = &xcore_reporter;
  // Set up profiling.
  static tflite::micro::xcore::XCoreProfiler xcore_profiler(reporter);
  profiler = &xcore_profiler;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(mobilenet_v1_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  static tflite::ops::micro::xcore::MobileNetOpsResolver resolver;

  // Build an interpreter to run the model with
  static tflite::micro::xcore::XCoreInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, reporter, true,
      profiler);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_tensors_status = interpreter->AllocateTensors();
  if (allocate_tensors_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  *input = (unsigned char *)(interpreter->input(0)->data.raw);
  *input_size = interpreter->input(0)->bytes;
  *output = (unsigned char *)(interpreter->output(0)->data.raw);
  *output_size = interpreter->output(0)->bytes;
}
