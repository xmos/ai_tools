// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/arg_min_max.h"

#include "lib_ops/api/tracing.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace arg_min_max {

XCoreStatus ArgMax16::Eval(const int16_t* A, int32_t* C, const int32_t length) {
  TRACE_INFO("ArgMax16 Eval id=%p\n", this);

  argmax_16(A, C, length);

  return kXCoreOk;
}

}  // namespace arg_min_max
}  // namespace xcore