// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
#include <iostream>

#include "tensorflow/lite/micro/kernels/xcore/xcore_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/version.h"

#include "lib_ops/api/lib_ops.h"

namespace xcore {

class XCOREInterpreter {
 public:
  XCOREInterpreter() {}
  ~XCOREInterpreter() {
    if (interpreter_)
      delete interpreter_;  // NOTE: interpreter_ must be deleted before
                            // resolver, micro_error_reporter_, tensor_arena_
                            // amd model_buffer_
    if (resolver_) delete resolver_;
    if (micro_error_reporter_) delete micro_error_reporter_;
    if (tensor_arena_) delete tensor_arena_;
    if (model_buffer_) delete model_buffer_;
  }

  XCoreStatus Initialize(const char* model_buffer, size_t model_buffer_size, size_t arena_size) {
    // We need to keep a copy of the model content
    model_buffer_ = new char[model_buffer_size];
    memcpy(model_buffer_, model_buffer, model_buffer_size);

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model_ = tflite::GetModel(model_buffer_);
    if (model_->version() != TFLITE_SCHEMA_VERSION) {
      error_msg_.clear();
      error_msg_ << "Model provided is schema version " << model_->version()
                 << " not equal to supported version " << TFLITE_SCHEMA_VERSION;
      return kXCoreError;
    }

    micro_error_reporter_ = new tflite::MicroErrorReporter();
    resolver_ = new tflite::ops::micro::AllOpsResolver();
    
    // Adds xCORE custom operators
    tflite::ops::micro::xcore::add_custom_ops(
        reinterpret_cast<tflite::MicroMutableOpResolver*>(resolver_));

    // Build an interpreter to run the model with.
    tensor_arena_ = new uint8_t[arena_size];
    interpreter_ = new tflite::MicroInterpreter(
        model_, *resolver_, tensor_arena_, arena_size, micro_error_reporter_);

    return kXCoreOk;
  }

  XCoreStatus AllocateTensors() {
    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_tensors_status = interpreter_->AllocateTensors();
    if (allocate_tensors_status != kTfLiteOk) {
      error_msg_.clear();
      error_msg_ << "AllocateTensors failed";
      return kXCoreError;
    }

    // Allocate xCORE OperatorDispatcher
    //   NOTE: must be called after AllocateTensors
    xcore::XCoreStatus allocate_xcore_status =
        xcore::AllocateOperatorDispatcher();
    if (allocate_xcore_status != xcore::kXCoreOk) {
      error_msg_.clear();
      error_msg_ << "AllocateOperatorDispatcher failed";
      return kXCoreError;
    }

    return kXCoreOk;
  }

  XCoreStatus Invoke() {
    TfLiteStatus invoke_status = interpreter_->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_msg_.clear();
      return kXCoreError;
    }

    return kXCoreOk;
  }

  XCoreStatus SetTensor(int tensor_index, const void* value, const int size,
                        const int* shape, const int type) {
    TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
    if (tensor == nullptr) {
      error_msg_.clear();
      error_msg_ << "tensor_index " << tensor_index << " not found";
      return kXCoreError;
    }

    if (tensor->dims->size != size) {
      error_msg_.clear();
      error_msg_ << "tensor dims size " << tensor->dims->size << " != " << size;
      return kXCoreError;
    }

    for (int i = 0; i < size; i++) {
      if (tensor->dims->data[i] != shape[i]) {
        error_msg_.clear();
        error_msg_ << "tensor dim " << tensor->dims->data[i]
                   << " != " << shape[i];
        return kXCoreError;
      }
    }

    if (tensor->type != type) {
      error_msg_.clear();
      error_msg_ << "tensor type " << tensor->type << " != " << type;
      return kXCoreError;
    }

    std::memcpy(tensor->data.raw, value, tensor->bytes);
    return kXCoreOk;
  }

  XCoreStatus GetTensor(int tensor_index, void* value, const int size,
                        const int* shape, const int type) {
    TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
    if (tensor == nullptr) {
      error_msg_.clear();
      error_msg_ << "tensor_index " << tensor_index << " not found";
      return kXCoreError;
    }

    if (tensor->dims->size != size) {
      error_msg_.clear();
      error_msg_ << "tensor dims size " << tensor->dims->size << " != " << size;
      return kXCoreError;
    }

    for (int i = 0; i < size; i++) {
      if (tensor->dims->data[i] != shape[i]) {
        error_msg_.clear();
        error_msg_ << "tensor dim " << tensor->dims->data[i]
                   << " != " << shape[i];
        return kXCoreError;
      }
    }

    if (tensor->type != type) {
      error_msg_.clear();
      error_msg_ << "tensor type " << tensor->type << " != " << type;
      return kXCoreError;
    }

    std::memcpy(value, tensor->data.raw, tensor->bytes);
    return kXCoreOk;
  }

  size_t GetError(char* msg) {
    std::strncpy(msg, error_msg_.str().c_str(), error_msg_.str().length());
    return error_msg_.str().length();
  }

 private:
  tflite::ops::micro::AllOpsResolver* resolver_ = nullptr;
  tflite::MicroErrorReporter* micro_error_reporter_ = nullptr;
  const tflite::Model* model_ = nullptr;
  tflite::MicroInterpreter* interpreter_ = nullptr;
  char* model_buffer_ = nullptr;
  uint8_t* tensor_arena_ = nullptr;
  std::stringstream error_msg_;
};

}  // namespace xcore

extern "C" {

xcore::XCOREInterpreter* new_interpreter() {
  return new xcore::XCOREInterpreter();
}

void delete_interpreter(xcore::XCOREInterpreter* interpreter) {
  delete interpreter;
}

int initialize(xcore::XCOREInterpreter* interpreter, const char* model_content,
               size_t model_content_size, size_t arena_size) {
  return interpreter->Initialize(model_content, model_content_size, arena_size);
}

int allocate_tensors(xcore::XCOREInterpreter* interpreter) {
  return interpreter->AllocateTensors();
}

int set_tensor(xcore::XCOREInterpreter* interpreter, int tensor_index,
               const void* value, const int size, const int* shape,
               const int type) {
  return interpreter->SetTensor(tensor_index, value, size, shape, type);
}

int get_tensor(xcore::XCOREInterpreter* interpreter, int tensor_index,
               void* value, const int size, const int* shape, const int type) {
  return interpreter->GetTensor(tensor_index, value, size, shape, type);
}

int invoke(xcore::XCOREInterpreter* interpreter) {
  return interpreter->Invoke();
}

int get_error(xcore::XCOREInterpreter* interpreter, char* msg) {
  return interpreter->GetError(msg);
}

}  // extern "C"
