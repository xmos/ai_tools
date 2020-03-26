// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/type_conversions.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace type_conversions {

XCoreStatus Requantize_16_to_8::Eval(int8_t* Y, const int16_t* X,
                                     const int32_t length) {
  requantize_16_to_8(Y, X, length);

  return kXCoreOk;
}

}  // namespace type_conversions
}  // namespace xcore
