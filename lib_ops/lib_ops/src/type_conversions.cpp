// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/type_conversions.h"

#include "lib_ops/api/logging.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace type_conversions {

XCoreStatus Requantize_16_to_8::Eval(int8_t* Y, const int16_t* X,
                                     const int32_t length) {
  LOG_TRACE("Requantize_16_to_8 Eval id=%p\n", this);
  requantize_16_to_8(Y, X, length);

  return kXCoreOk;
}

}  // namespace type_conversions
}  // namespace xcore
