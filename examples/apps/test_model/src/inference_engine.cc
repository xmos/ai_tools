// Copyright (c) 2019, XMOS Ltd, All rights reserved

#include "inference_engine.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "operators/device_memory.h"
#include "operators/xcore_interpreter.h"
#include "operators/xcore_profiler.h"
#include "operators/xcore_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"
#include "tensorflow/lite/version.h"

tflite::ErrorReporter *reporter = nullptr;
tflite::Profiler *profiler = nullptr;
const tflite::Model *model = nullptr;
xcore::XCoreInterpreter *interpreter = nullptr;

// static buffer for XCoreInterpreter class allowcation
unsigned char interpreter_buffer[sizeof(xcore::XCoreInterpreter)];

void invoke() {
  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();

  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(reporter, "Invoke failed\n");
  }
}

void initialize(unsigned char *model_content, unsigned char *tensor_arena,
                size_t tensor_arena_size, unsigned char **input,
                unsigned *input_size, unsigned char **output,
                unsigned *output_size) {
  // Set up logging
  static xcore::XCoreReporter xcore_reporter;
  if (reporter == nullptr) {
    reporter = &xcore_reporter;
  }
  // Set up profiling.
  static xcore::XCoreProfiler xcore_profiler(reporter);
  if (profiler == nullptr) {
    profiler = &xcore_profiler;
  }

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model_content);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Create all ops resolver and add xCORE custom operators
  tflite::AllOpsResolver resolver;
  resolver.AddCustom("XC_maxpool2d",
                     tflite::ops::micro::xcore::Register_MaxPool2D());
  resolver.AddCustom("XC_avgpool2d",
                     tflite::ops::micro::xcore::Register_AvgPool2D());
  resolver.AddCustom("XC_avgpool2d_global",
                     tflite::ops::micro::xcore::Register_AvgPool2D_Global());
  resolver.AddCustom("XC_fc_deepin_anyout",
                     tflite::ops::micro::xcore::Register_FullyConnected_16());
  resolver.AddCustom("XC_conv2d_shallowin",
                     tflite::ops::micro::xcore::Register_Conv2D_Shallow());
  resolver.AddCustom("XC_conv2d_deep",
                     tflite::ops::micro::xcore::Register_Conv2D_Deep());
  resolver.AddCustom("XC_conv2d_1x1",
                     tflite::ops::micro::xcore::Register_Conv2D_1x1());
  resolver.AddCustom("XC_conv2d_depthwise",
                     tflite::ops::micro::xcore::Register_Conv2D_Depthwise());
  resolver.AddCustom("XC_requantize_16_to_8",
                     tflite::ops::micro::xcore::Register_Requantize_16_to_8());
  resolver.AddCustom("XC_lookup_8",
                     tflite::ops::micro::xcore::Register_Lookup_8());

  // Build an interpreter to run the model with
  if (interpreter) {
    // We already have an interpreter so we need to explicitly call the
    // destructor here but NOT delete the object
    interpreter->~XCoreInterpreter();
  }
  interpreter = new (interpreter_buffer)
      xcore::XCoreInterpreter(model, resolver, tensor_arena, tensor_arena_size,
                              reporter, true, profiler);

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
