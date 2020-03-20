// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_TYPE_CONVERSION_OPERATORS_H_
#define XCORE_TYPE_CONVERSION_OPERATORS_H_

#include <cstdint>
#include "lib_ops/api/lib_ops.h"

extern "C" {
    #include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace type_conversions { 

class Requantize_16_to_8 {
    public:
        Requantize_16_to_8() {}
        ~Requantize_16_to_8() {}

        XCoreStatus Eval(int8_t* Y, const int16_t* X, const int32_t length);
};

} // namespace type_conversions
} // namespace xcore

#endif  // XCORE_TYPE_CONVERSION_OPERATORS_H_