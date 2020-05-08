// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/fully_connected.h"

#include "lib_ops/api/logging.h"

namespace xcore {
namespace fully_connected {

XCoreStatus FullyConnected_16::Init(int32_t C_in, int32_t C_out) {
  LOG_TRACE("FullyConnected_16 Init id=%p C_in=%ld C_out=%ld\n", this, C_in,
            C_out);

  fully_connected_init(&plan_, C_in, C_out);
  return kXCoreOk;
}

XCoreStatus FullyConnected_16::Eval(int16_t* Y, const int8_t* W,
                                    const int8_t* X, const int16_t* BSO) {
  fully_connected_16(Y, W, X, (nn_bso_block_t*)BSO, &plan_);
  return kXCoreOk;
}

}  // namespace fully_connected
}  // namespace xcore
