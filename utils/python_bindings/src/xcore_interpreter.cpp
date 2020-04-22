// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
#include "xcore_interpreter.h"

#include <iostream>
#include <vector>

namespace xcore {

constexpr int max_log_len = 256;
constexpr int num_threads = 5;

typedef TfLiteStatus (*invoke_function_t)(TfLiteContext*, TfLiteNode*);

//****************************
//****************************
//****************************
// Callback classes
//****************************
//****************************
//****************************
class CallbackContext {
 public:
  CallbackContext()
      : current_operator(0),
        preinvoke_callback(nullptr),
        postinvoke_callback(nullptr) {}
  void Reset() {
    current_operator = 0;
    preinvoke_callback = nullptr;
    postinvoke_callback = nullptr;
    invoke_functions.clear();
  }
  int current_operator;
  invoke_callback_t preinvoke_callback;
  invoke_callback_t postinvoke_callback;
  std::vector<invoke_function_t> invoke_functions;
};
static CallbackContext gCallbackContext;

TfLiteStatus CallbackInvoke(TfLiteContext* context, TfLiteNode* node) {
  int current_operator = gCallbackContext.current_operator;

  invoke_function_t invoke =
      gCallbackContext.invoke_functions[current_operator];

  if (gCallbackContext.preinvoke_callback)
    gCallbackContext.preinvoke_callback(current_operator);
  TfLiteStatus status = invoke(context, node);
  if (gCallbackContext.postinvoke_callback)
    gCallbackContext.postinvoke_callback(current_operator);
  gCallbackContext.current_operator++;

  return status;
}

//****************************
//****************************
//****************************
// InterpreterErrorReporter
//****************************
//****************************
//****************************
int InterpreterErrorReporter::Report(const char* format, ...) {
  va_list args;
  va_start(args, format);
  int code = Report(format, args);
  va_end(args);
  return code;
}

int InterpreterErrorReporter::Report(const char* format, va_list args) {
  char log_buffer[max_log_len];
  std::vsnprintf(log_buffer, max_log_len, format, args);
  log_stream_ << log_buffer;
  return 0;
}

std::string InterpreterErrorReporter::GetError() {
  std::string error = log_stream_.str();
  log_stream_.clear();
  return error;
}

//****************************
//****************************
//****************************
// XCOREInterpreter
//****************************
//****************************
//****************************
XCOREInterpreter::XCOREInterpreter() {
  error_reporter_ = nullptr;
  resolver_ = nullptr;
  interpreter_ = nullptr;
  model_ = nullptr;
  model_buffer_ = nullptr;
  tensor_arena_ = nullptr;
  tensor_arena_size_ = 0;
  dispatcher_ = nullptr;
  xcore_arena_ = nullptr;
  xcore_arena_size_ = 0;
}

XCOREInterpreter::~XCOREInterpreter() {
  if (interpreter_)
    delete interpreter_;  // NOTE: interpreter_ must be deleted before
                          // resolver, error_reporter_, tensor_arena_
                          // xcore_arena_ and model_buffer_
  if (resolver_) delete resolver_;
  if (error_reporter_) delete error_reporter_;
  if (tensor_arena_) delete[] tensor_arena_;
  if (dispatcher_) delete dispatcher_;
  if (xcore_arena_) delete[] xcore_arena_;
  if (model_buffer_) delete[] model_buffer_;
}

XCoreStatus XCOREInterpreter::Initialize(const char* model_buffer,
                                         size_t model_buffer_size,
                                         size_t tensor_arena_size,
                                         size_t xcore_arena_size) {
  // We need to keep a copy of the model content
  model_buffer_ = new char[model_buffer_size];
  memcpy(model_buffer_, model_buffer, model_buffer_size);

  // Create error reporter
  error_reporter_ = new InterpreterErrorReporter();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model_ = tflite::GetModel(model_buffer_);
  if (model_->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter_->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model_->version(), TFLITE_SCHEMA_VERSION);
    return kXCoreError;
  }

  xcore_arena_size_ = xcore_arena_size;
  xcore_arena_ = new uint8_t[xcore_arena_size_];
  memset(xcore_arena_, 0, xcore_arena_size_);
  // Setup xCORE dispatcher
  dispatcher_ =
      new xcore::Dispatcher(xcore_arena_, xcore_arena_size_, num_threads);
  xcore::XCoreStatus xcore_status = xcore::InitializeXCore(dispatcher_);
  if (xcore_status != xcore::kXCoreOk) {
    error_reporter_->Report("InitializeXCore() failed");
    return kXCoreError;
  }

  // Create all ops resolver and add xCORE custom operators
  resolver_ = new tflite::ops::micro::AllOpsResolver();
  tflite::ops::micro::xcore::add_custom_ops(
      reinterpret_cast<tflite::MicroMutableOpResolver*>(resolver_));

  tensor_arena_size_ = tensor_arena_size;
  tensor_arena_ = new uint8_t[tensor_arena_size_];
  memset(tensor_arena_, 0, tensor_arena_size_);

  // Build an interpreter to run the model with.
  interpreter_ = new tflite::MicroInterpreter(
      model_, *resolver_, tensor_arena_, tensor_arena_size_, error_reporter_);

  return kXCoreOk;
}

XCoreStatus XCOREInterpreter::AllocateTensors() {
  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_tensors_status = interpreter_->AllocateTensors();
  if (allocate_tensors_status != kTfLiteOk) {
    return kXCoreError;
  }

  return kXCoreOk;
}

size_t XCOREInterpreter::input_tensor_index(size_t input_index) {
  const TfLiteTensor* input_tensor = interpreter_->input(input_index);
  for (size_t i = 0; i < tensors_size(); i++) {
    const TfLiteTensor* tensor = interpreter_->tensor(i);
    if (tensor == input_tensor) return i;
  }
  return -1;
}

size_t XCOREInterpreter::output_tensor_index(size_t output_index) {
  const TfLiteTensor* output_tensor = interpreter_->output(output_index);
  for (size_t i = 0; i < tensors_size(); i++) {
    const TfLiteTensor* tensor = interpreter_->tensor(i);
    if (tensor == output_tensor) return i;
  }
  return -1;
}

XCoreStatus XCOREInterpreter::Invoke(invoke_callback_t preinvoke_callback,
                                     invoke_callback_t postinvoke_callback) {
  if (preinvoke_callback || postinvoke_callback) {
    gCallbackContext.preinvoke_callback = preinvoke_callback;
    gCallbackContext.postinvoke_callback = postinvoke_callback;
    // Save the registered invoke functions
    for (size_t node_index = 0; node_index < interpreter_->operators_size();
         node_index++) {
      tflite::NodeAndRegistration node_and_reg =
          interpreter_->node_and_registration(static_cast<int>(node_index));

      gCallbackContext.invoke_functions.push_back(
          node_and_reg.registration->invoke);
    }
    // Set the invoke function to the CallbackInvoke
    for (size_t node_index = 0; node_index < interpreter_->operators_size();
         node_index++) {
      tflite::NodeAndRegistration node_and_reg =
          interpreter_->node_and_registration(static_cast<int>(node_index));
      const TfLiteRegistration* reg = node_and_reg.registration;

      // Disregard const qualifier to workaround existing API.
      (const_cast<TfLiteRegistration*>(reg))->invoke = CallbackInvoke;
    }
  }

  TfLiteStatus invoke_status = interpreter_->Invoke();

  // Set back the original invoke function
  if (preinvoke_callback || postinvoke_callback) {
    // Set the invoke function to the CallbackInvoke
    for (size_t node_index = 0; node_index < interpreter_->operators_size();
         node_index++) {
      tflite::NodeAndRegistration node_and_reg =
          interpreter_->node_and_registration(static_cast<int>(node_index));
      const TfLiteRegistration* reg = node_and_reg.registration;

      // Disregard const qualifier to workaround existing API.
      (const_cast<TfLiteRegistration*>(reg))->invoke =
          gCallbackContext.invoke_functions[node_index];
    }
    gCallbackContext.Reset();
  }
  if (invoke_status != kTfLiteOk) {
    return kXCoreError;
  }

  return kXCoreOk;
}

XCoreStatus XCOREInterpreter::SetTensor(size_t tensor_index, const void* value,
                                        const int size, const int* shape,
                                        const int type) {
  TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
  if (tensor == nullptr) {
    return kXCoreError;
  }

  if (tensor->dims->size != size) {
    error_reporter_->Report("tensor dims size %d != %d", tensor->dims->size,
                            size);
    return kXCoreError;
  }

  for (int i = 0; i < size; i++) {
    if (tensor->dims->data[i] != shape[i]) {
      error_reporter_->Report("tensor dim %d != %d", tensor->dims->data[i],
                              shape[i]);
      return kXCoreError;
    }
  }

  if (tensor->type != type) {
    error_reporter_->Report("tensor type %d != %d", tensor->type, type);
    return kXCoreError;
  }

  std::memcpy(tensor->data.raw, value, tensor->bytes);
  return kXCoreOk;
}

XCoreStatus XCOREInterpreter::GetTensor(size_t tensor_index, void* value,
                                        const int size, const int* shape,
                                        const int type) {
  TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
  if (tensor == nullptr) {
    return kXCoreError;
  }

  if (tensor->dims->size != size) {
    error_reporter_->Report("tensor dims size %d != %d", tensor->dims->size,
                            size);
    return kXCoreError;
  }

  for (int i = 0; i < size; i++) {
    if (tensor->dims->data[i] != shape[i]) {
      error_reporter_->Report("tensor dim %d != %d", tensor->dims->data[i],
                              shape[i]);
      return kXCoreError;
    }
  }

  if (tensor->type != type) {
    error_reporter_->Report("tensor type %d != %d", tensor->type, type);
    return kXCoreError;
  }

  std::memcpy(value, tensor->data.raw, tensor->bytes);
  return kXCoreOk;
}

XCoreStatus XCOREInterpreter::GetTensorDetailsBufferSizes(size_t tensor_index,
                                                          size_t* dims,
                                                          size_t* scales,
                                                          size_t* zero_points) {
  TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
  if (tensor == nullptr) {
    return kXCoreError;
  }
  *dims = tensor->dims->size;
  TfLiteAffineQuantization* quantization_params =
      static_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
  if (quantization_params) {
    *scales = quantization_params->scale->size;
    *zero_points = quantization_params->zero_point->size;
  } else {
    *scales = 1;
    *zero_points = 1;
  }
  return kXCoreOk;
}

XCoreStatus XCOREInterpreter::GetTensorDetails(size_t tensor_index, char* name,
                                               int name_len, int* shape,
                                               int* type, float* scale,
                                               int32_t* zero_point) {
  TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
  if (tensor == nullptr) {
    return kXCoreError;
  }
  std::strncpy(name, tensor->name, name_len);

  for (int i = 0; i < tensor->dims->size; i++) {
    shape[i] = tensor->dims->data[i];
  }
  *type = tensor->type;
  TfLiteAffineQuantization* quantization_params =
      static_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
  if (quantization_params) {
    for (int i = 0; i < quantization_params->scale->size; i++) {
      scale[i] = quantization_params->scale->data[i];
    }
    for (int i = 0; i < quantization_params->zero_point->size; i++) {
      zero_point[i] = quantization_params->zero_point->data[i];
    }
  } else {
    *scale = 0.0;
    *zero_point = 0;
  }
  return kXCoreOk;
}

XCoreStatus XCOREInterpreter::GetOperatorDetailsBufferSizes(
    size_t operator_index, size_t* inputs, size_t* outputs) {
  if (operator_index >= interpreter_->operators_size()) {
    error_reporter_->Report("Invalid operator index %d", operator_index);
    return kXCoreError;
  }

  tflite::NodeAndRegistration node_and_reg =
      interpreter_->node_and_registration(static_cast<int>(operator_index));
  const TfLiteNode& node = node_and_reg.node;
  *inputs = node.inputs->size;
  *outputs = node.outputs->size;

  return kXCoreOk;
}

XCoreStatus XCOREInterpreter::GetOperatorDetails(size_t operator_index,
                                                 char* name, int name_len,
                                                 int* version, int* inputs,
                                                 int* outputs) {
  if (operator_index >= interpreter_->operators_size()) {
    error_reporter_->Report("Invalid operator index %d", operator_index);
    return kXCoreError;
  }

  tflite::NodeAndRegistration node_and_reg =
      interpreter_->node_and_registration(static_cast<int>(operator_index));
  const TfLiteNode& node = node_and_reg.node;
  const TfLiteRegistration* reg = node_and_reg.registration;

  if (reg->custom_name != nullptr) {
    std::strncpy(name, reg->custom_name, name_len);
  } else {
    std::strncpy(name, tflite::EnumNamesBuiltinOperator()[reg->builtin_code],
                 name_len);
  }
  *version = reg->version;
  for (int i = 0; i < node.inputs->size; i++) {
    inputs[i] = node.inputs->data[i];
  }
  for (int i = 0; i < node.outputs->size; i++) {
    outputs[i] = node.outputs->data[i];
  }

  return kXCoreOk;
}

size_t XCOREInterpreter::GetError(char* msg) {
  const std::string& error_msg = error_reporter_->GetError();
  std::strncpy(msg, error_msg.c_str(), error_msg.length());
  return error_msg.length();
}

}  // namespace xcore

//*****************************************
//*****************************************
//*****************************************
// C-API callable from Python
//*****************************************
//*****************************************
//*****************************************

extern "C" {
xcore::XCOREInterpreter* new_interpreter() {
  return new xcore::XCOREInterpreter();
}

void delete_interpreter(xcore::XCOREInterpreter* interpreter) {
  delete interpreter;
}

int initialize(xcore::XCOREInterpreter* interpreter, const char* model_content,
               size_t model_content_size, size_t tensor_arena_size,
               size_t xcore_arena_size) {
  return interpreter->Initialize(model_content, model_content_size,
                                 tensor_arena_size, xcore_arena_size);
}

int allocate_tensors(xcore::XCOREInterpreter* interpreter) {
  return interpreter->AllocateTensors();
}

int tensors_size(xcore::XCOREInterpreter* interpreter) {
  return interpreter->tensors_size();
}

size_t inputs_size(xcore::XCOREInterpreter* interpreter) {
  return interpreter->inputs_size();
}

size_t outputs_size(xcore::XCOREInterpreter* interpreter) {
  return interpreter->outputs_size();
}

int set_tensor(xcore::XCOREInterpreter* interpreter, size_t tensor_index,
               const void* value, const int size, const int* shape,
               const int type) {
  return interpreter->SetTensor(tensor_index, value, size, shape, type);
}

int get_tensor(xcore::XCOREInterpreter* interpreter, size_t tensor_index,
               void* value, const int size, const int* shape, const int type) {
  return interpreter->GetTensor(tensor_index, value, size, shape, type);
}

size_t get_tensor_details_buffer_sizes(xcore::XCOREInterpreter* interpreter,
                                       size_t tensor_index, size_t* dims,
                                       size_t* scales, size_t* zero_points) {
  return interpreter->GetTensorDetailsBufferSizes(tensor_index, dims, scales,
                                                  zero_points);
}

int get_tensor_details(xcore::XCOREInterpreter* interpreter,
                       size_t tensor_index, char* name, int name_len,
                       int* shape, int* type, float* scale, int* zero_point) {
  return interpreter->GetTensorDetails(tensor_index, name, name_len, shape,
                                       type, scale, zero_point);
}

size_t get_operator_details_buffer_sizes(xcore::XCOREInterpreter* interpreter,
                                         size_t operator_index, size_t* inputs,
                                         size_t* outputs) {
  return interpreter->GetOperatorDetailsBufferSizes(operator_index, inputs,
                                                    outputs);
}

int get_operator_details(xcore::XCOREInterpreter* interpreter,
                         size_t operator_index, char* name, int name_len,
                         int* version, int* inputs, int* outputs) {
  return interpreter->GetOperatorDetails(operator_index, name, name_len,
                                         version, inputs, outputs);
}

size_t input_tensor_index(xcore::XCOREInterpreter* interpreter,
                          size_t input_index) {
  return interpreter->input_tensor_index(input_index);
}

size_t output_tensor_index(xcore::XCOREInterpreter* interpreter,
                           size_t output_index) {
  return interpreter->output_tensor_index(output_index);
}

int invoke(xcore::XCOREInterpreter* interpreter,
           xcore::invoke_callback_t preinvoke_callback = nullptr,
           xcore::invoke_callback_t postinvoke_callback = nullptr) {
  return interpreter->Invoke(preinvoke_callback, postinvoke_callback);
}

size_t get_error(xcore::XCOREInterpreter* interpreter, char* msg) {
  return interpreter->GetError(msg);
}

}  // extern "C"
