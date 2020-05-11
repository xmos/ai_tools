
// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "lib_ops/api/benchmarking.h"
#include "lib_ops/api/lib_ops.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/version.h"
#include "xcore_model.h"

#define TEST_INPUT_SIZE = 32 * 32 * 4

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 65000;
uint8_t tensor_arena[kTensorArenaSize];

xcore::Dispatcher* dispatcher = nullptr;
constexpr int num_threads = 5;
constexpr int kXCOREArenaSize = 5000;
uint8_t xcore_arena[kXCOREArenaSize];

static int argmax(const int8_t* A, const int N) {
  assert(N > 0);

  int m = 0;

  for (int i = 1; i < N; i++) {
    if (A[i] > A[m]) {
      m = i;
    }
  }

  return m;
}

static int load_test_input(const char* filename, char* input, size_t esize) {
  FILE* fd = fopen(filename, "rb");
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
  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(xcore_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  static tflite::ops::micro::xcore::XcoreOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Setup xCORE dispatcher (BEFORE calling AllocateTensors)
  static xcore::Dispatcher static_dispatcher(xcore_arena, kXCOREArenaSize,
                                             num_threads);
  xcore::XCoreStatus xcore_status = xcore::InitializeXCore(&static_dispatcher);
  if (xcore_status != xcore::kXCoreOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "InitializeXCore() failed");
    return;
  }
  dispatcher = &static_dispatcher;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
}

int main(int argc, char* argv[]) {
  setup_tflite();

  if (argc > 1) {
    printf("input filename=%s\n", argv[1]);
    // Load input tensor
    if (!load_test_input(argv[1], input->data.raw, input->bytes)) return -1;
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
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on input filename=%s\n",
                         argv[1]);
    return -1;
  }

  char classification[12] = {0};
  int m = argmax(output->data.int8, 10);

  switch (m) {
    case 0:
      snprintf(classification, 9, "airplane");
      break;
    case 1:
      snprintf(classification, 11, "automobile");
      break;
    case 2:
      snprintf(classification, 5, "bird");
      break;
    case 3:
      snprintf(classification, 4, "cat");
      break;
    case 4:
      snprintf(classification, 5, "deer");
      break;
    case 5:
      snprintf(classification, 4, "dog");
      break;
    case 6:
      snprintf(classification, 5, "frog");
      break;
    case 7:
      snprintf(classification, 6, "horse");
      break;
    case 8:
      snprintf(classification, 5, "ship");
      break;
    case 9:
      snprintf(classification, 6, "truck");
      break;
    default:
      break;
  }
  printf("classification=%s\n", classification);

  return 0;
}
