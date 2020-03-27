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
    if (tensor_arena_) delete tensor_arena_;
    if (interpreter_) delete interpreter_;
  }

  XCoreStatus Initialize(const char* model_buffer, size_t arena_size) {
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model_ = tflite::GetModel(model_buffer);
    if (model_->version() != TFLITE_SCHEMA_VERSION) {
      return kXCoreError;

      //   std::ostringstream msg;
      //   msg << "Model provided is schema version " << model_->version()
      //       << " not equal to supported version " << TFLITE_SCHEMA_VERSION;
      //   throw std::runtime_error(msg.str());
    }

    // This adds the custom operation implementations
    tflite::ops::micro::xcore::add_custom_ops(
        reinterpret_cast<tflite::MicroMutableOpResolver*>(&resolver_));

    // Build an interpreter to run the model with.
    tensor_arena_ = new uint8_t[arena_size];
    interpreter_ = new tflite::MicroInterpreter(
        model_, resolver_, tensor_arena_, arena_size, &micro_error_reporter_);
    return kXCoreOk;
  }

  XCoreStatus AllocateTensors() {
    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_tensors_status = interpreter_->AllocateTensors();
    if (allocate_tensors_status != kTfLiteOk) {
      return kXCoreError;
    }

    // Allocate xCORE KernelDispatcher after AllocateTensors
    xcore::XCoreStatus allocate_xcore_status =
        xcore::AllocateOperatorDispatcher();
    if (allocate_xcore_status != xcore::kXCoreOk) {
      return kXCoreError;
    }

    return kXCoreOk;
  }

  XCoreStatus Invoke() {
    TfLiteStatus invoke_status = interpreter_->Invoke();
    if (invoke_status != kTfLiteOk) {
      return kXCoreError;
    }

    return kXCoreOk;
  }

  XCoreStatus SetTensor(int tensor_index, const void* value) {
    TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
    if (tensor == nullptr) {
      return kXCoreError;
    }

    std::memcpy(tensor->data.raw, value, tensor->bytes);
    return kXCoreOk;
  }

  XCoreStatus GetTensor(int tensor_index, void* value) {
    TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
    if (tensor == nullptr) {
      return kXCoreError;
    }

    std::memcpy(value, tensor->data.raw, tensor->bytes);
    return kXCoreOk;
  }

 private:
  tflite::ops::micro::AllOpsResolver resolver_;
  tflite::MicroErrorReporter micro_error_reporter_;
  const tflite::Model* model_ = nullptr;
  uint8_t* tensor_arena_ = nullptr;
  tflite::MicroInterpreter* interpreter_ = nullptr;
};

}  // namespace xcore

extern "C" {

xcore::XCOREInterpreter* new_interpreter() {
  return new xcore::XCOREInterpreter();
}

int initialize(xcore::XCOREInterpreter* interpreter, const char* model_content,
               size_t arena_size) {
  return interpreter->Initialize(model_content, arena_size);
}

void delete_interpreter(xcore::XCOREInterpreter* interpreter) {
  delete interpreter;
}

int allocate_tensors(xcore::XCOREInterpreter* interpreter) {
  return interpreter->AllocateTensors();
}

int set_tensor(xcore::XCOREInterpreter* interpreter, int tensor_index, const void* value) {
  return interpreter->SetTensor(tensor_index, value);
}

int get_tensor(xcore::XCOREInterpreter* interpreter, int tensor_index, void* value) {
  return interpreter->GetTensor(tensor_index, value);
}

int invoke(xcore::XCOREInterpreter* interpreter) {
  return interpreter->Invoke();
}

}  // extern "C"
