// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "operators/activations.h"

#include "operators/dispatcher.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace activations {

TfLiteStatus Lookup8::Eval(uint8_t* Y, const uint8_t* X, const uint8_t* lut,
                           const int32_t length) {
  lookup8(Y, X, lut, length);

  return kTfLiteOk;
}

}  // namespace activations
}  // namespace xcore
