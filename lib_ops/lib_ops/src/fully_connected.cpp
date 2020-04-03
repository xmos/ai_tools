// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/fully_connected.h"

namespace xcore {
namespace fully_connected {

XCoreStatus FullyConnected_16::Init(int32_t C_in, int32_t C_out) {
  fully_connected_init(&plan_, C_in, C_out);
  return kXCoreOk;
}

XCoreStatus FullyConnected_16::Eval(int16_t* Y, const int8_t* W,
                                    const int8_t* X, const int16_t* BSS) {
  fully_connected_16(Y, W, X, (data16_t*)BSS, &plan_);
  return kXCoreOk;
}

}  // namespace fully_connected
}  // namespace xcore
