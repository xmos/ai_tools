// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

#include "lib_ops/api/lib_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/version.h"

namespace xcore {

typedef void (*invoke_callback_t)(int);

//****************************
//****************************
//****************************
// InterpreterErrorReporter
//****************************
//****************************
//****************************
class InterpreterErrorReporter : public tflite::ErrorReporter {
 public:
  ~InterpreterErrorReporter() {}

  int Report(const char* format, ...);
  int Report(const char* format, va_list args);
  std::string GetError();

 private:
  std::stringstream log_stream_;
};

//****************************
//****************************
//****************************
// XCOREInterpreter
//****************************
//****************************
//****************************
class XCOREInterpreter {
 public:
  XCOREInterpreter();
  ~XCOREInterpreter();

  XCoreStatus Initialize(const char* model_buffer, size_t model_buffer_size,
                         size_t tensor_arena_size, size_t xcore_arena_size);

  XCoreStatus AllocateTensors();

  size_t tensors_size() const { return interpreter_->tensors_size(); }
  size_t inputs_size() const { return interpreter_->inputs_size(); }
  size_t input_tensor_index(size_t input_index);
  size_t outputs_size() const { return interpreter_->outputs_size(); }
  size_t output_tensor_index(size_t output_index);

  XCoreStatus Invoke(invoke_callback_t preinvoke_callback = nullptr,
                     invoke_callback_t postinvoke_callback = nullptr);

  XCoreStatus SetTensor(size_t tensor_index, const void* value, const int size,
                        const int* shape, const int type);

  XCoreStatus GetTensor(size_t tensor_index, void* value, const int size,
                        const int* shape, const int type);

  XCoreStatus GetTensorDetailsBufferSizes(size_t tensor_index, size_t* dims,
                                          size_t* scales, size_t* zero_points);

  XCoreStatus GetTensorDetails(size_t tensor_index, char* name, int name_len,
                               int* shape, int* type, float* scale,
                               int32_t* zero_point);

  XCoreStatus GetOperatorDetailsBufferSizes(size_t operator_index,
                                            size_t* inputs, size_t* outputs);

  XCoreStatus GetOperatorDetails(size_t operator_index, char* name,
                                 int name_len, int* version, int* inputs,
                                 int* outputs);
  size_t GetError(char* msg);

 private:
  InterpreterErrorReporter* error_reporter_;
  tflite::ops::micro::AllOpsResolver* resolver_;
  tflite::MicroInterpreter* interpreter_;
  const tflite::Model* model_;
  char* model_buffer_;
  uint8_t* tensor_arena_;
  size_t tensor_arena_size_;
  xcore::Dispatcher* dispatcher_;
  uint8_t* xcore_arena_;
  size_t xcore_arena_size_;
};

}  // namespace xcore
