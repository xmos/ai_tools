// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/arg_min_max.h"

#include "lib_ops/api/benchmarking.h"
#include "lib_ops/api/tracing.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace arg_min_max {

XCoreStatus ArgMax16::Eval(int32_t* Y, const int16_t* X, const int32_t length) {
  TRACE_INFO("ArgMax16 Eval id=%p\n", this);
  TIMER_START();

  argmax_16(Y, X, length);

  TIMER_STOP("ArgMax16 id=%p", this);
  return kXCoreOk;
}

}  // namespace arg_min_max
}  // namespace xcore