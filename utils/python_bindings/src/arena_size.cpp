// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
#include "arena_size.h"

#include "lib_ops/api/allocator.h"
#include "lib_ops/api/dispatcher.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
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
constexpr int max_xcore_heap_size = 512000;
constexpr int xcore_heap_size_adjustment = 2048;

static bool verify_micro_interpreter(const char *model_content,
                                     size_t model_buffer_size,
                                     size_t tensor_arena_size) {
  xcore::XCOREInterpreter interpreter = xcore::XCOREInterpreter();
  interpreter.Initialize(model_content, model_buffer_size, tensor_arena_size,
                         max_xcore_heap_size);
  if (interpreter.AllocateTensors() == xcore::kXCoreOk) {
    if (interpreter.Invoke() != xcore::kXCoreOk) {
      return false;
    }
  } else {
    return false;
  }

  return true;
}

void search_arena_sizes(const char *model_content, size_t model_content_size,
                        size_t min_arena_size, size_t max_arena_size,
                        size_t *arena_size, size_t *heap_size) {
  size_t align_to = 12;
  size_t curr_arena_size;
  size_t return_size = max_arena_size;
  bool interpreter_ok;

  while ((max_arena_size - min_arena_size) >= 32) {
    curr_arena_size = (min_arena_size + max_arena_size) / 2;

    if (min_arena_size == curr_arena_size ||
        max_arena_size == curr_arena_size) {
      break;
    }

    interpreter_ok = verify_micro_interpreter(model_content, model_content_size,
                                              curr_arena_size);
    if (interpreter_ok) {
      return_size = curr_arena_size;
      max_arena_size = curr_arena_size;
    } else {
      min_arena_size = curr_arena_size;
    }
  }
  *arena_size = return_size + align_to - (return_size % align_to);

  Dispatcher *dispatcher = GetDispatcher();
  *heap_size = dispatcher->GetAllocatedSize() + xcore_heap_size_adjustment;
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

void get_arena_sizes(const char *model_content, size_t model_content_size,
                     size_t max_size, size_t *arena_size, size_t *heap_size) {
  xcore::search_arena_sizes(model_content, model_content_size, 1024, max_size,
                            arena_size, heap_size);
}

}  // extern "C"
