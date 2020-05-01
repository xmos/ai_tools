// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
#include "arena_size.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "xcore_interpreter.h"

namespace xcore {

//*****************************************
//*****************************************
//*****************************************
// verify_micro_interpreter
//   returns true if the tensor_arena_size
//   is adequate
//*****************************************
//*****************************************
//*****************************************

static bool verify_micro_interpreter(const char* model_content,
                                     size_t model_buffer_size,
                                     size_t tensor_arena_size) {
  xcore::XCOREInterpreter interpreter = xcore::XCOREInterpreter();
  interpreter.Initialize(model_content, model_buffer_size, tensor_arena_size,
                         64000);
  if (interpreter.AllocateTensors() == xcore::kXCoreOk) {
    if (interpreter.Invoke() != xcore::kXCoreOk) {
      return false;
    }
  } else {
    return false;
  }

  return true;
}

size_t search_tensor_arena_size(const char* model_content,
                                size_t model_content_size, size_t min_size,
                                size_t max_size) {
  size_t align_to = 4;
  size_t curr_size;
  size_t return_size = max_size;
  bool interpreter_ok;

  while ((max_size - min_size) >= 32) {
    curr_size = (min_size + max_size) / 2;

    if (min_size == curr_size || max_size == curr_size) {
      break;
    }

    interpreter_ok =
        verify_micro_interpreter(model_content, model_content_size, curr_size);
    if (interpreter_ok) {
      return_size = curr_size;
      max_size = curr_size;
    } else {
      min_size = curr_size;
    }
  }
  return return_size + align_to - (return_size % align_to);
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

size_t get_tensor_arena_size(const char* model_content,
                             size_t model_content_size, size_t max_size) {
  return xcore::search_tensor_arena_size(model_content, model_content_size,
                                         1024, max_size);
}

}  // extern "C"
