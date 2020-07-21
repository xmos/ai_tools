// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_ACTIVATION_OPERATORS_H_
#define XCORE_ACTIVATION_OPERATORS_H_

#include <cstdint>

#include "tensorflow/lite/c/common.h"

namespace xcore {
namespace activations {

class Lookup8 {
 public:
  Lookup8() {}
  ~Lookup8() {}

  TfLiteStatus Eval(uint8_t* Y, const uint8_t* X, const uint8_t* lut,
                    const int32_t length);
};

}  // namespace activations
}  // namespace xcore

#endif  // XCORE_ACTIVATIONS_OPERATORS_H_