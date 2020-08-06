
// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "tensorflow/lite/micro/kernels/xcore/xcore_interpreter.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_profiler.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_reporter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/version.h"

tflite::ErrorReporter *reporter = nullptr;
tflite::Profiler *profiler = nullptr;
const tflite::Model *model = nullptr;
tflite::micro::xcore::XCoreInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
constexpr int kTensorArenaSize =
    300000;  // Hopefully this is big enough for all tests
uint8_t tensor_arena[kTensorArenaSize];

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

static void setup_tflite(const char *model_buffer) {
  // Set up logging
  static tflite::micro::xcore::XCoreReporter xcore_reporter;
  reporter = &xcore_reporter;
  // Set up profiling.
  static tflite::micro::xcore::XCoreProfiler xcore_profiler(reporter);
  profiler = &xcore_profiler;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model_buffer);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  static tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddSoftmax();
  resolver.AddPad();
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
  input = interpreter->input(0);
  output = interpreter->output(0);
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
  char *model_buffer = nullptr;
  size_t model_size;
  if (!load_model(model_filename, &model_buffer, &model_size)) {
    printf("error loading model filename=%s\n", model_filename);
    return -1;
  }

  // setup runtime
  setup_tflite(model_buffer);

  // Load input tensor
  if (!load_input(input_filename, input->data.raw, input->bytes)) {
    printf("error loading input filename=%s\n", input_filename);
    return -1;
  }

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(reporter, "Invoke failed\n");
    return -1;
  }

  // save output
  if (!save_output(output_filename, output->data.raw, output->bytes)) {
    printf("error saving output filename=%s\n", output_filename);
    return -1;
  }
  return 0;
}
