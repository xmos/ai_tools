// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_ARG_MIN_MAX_OPERATORS_H_
#define XCORE_ARG_MIN_MAX_OPERATORS_H_

#include <cstdint>
#include "lib_ops/api/lib_ops.h"

extern "C" {
    #include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace arg_min_max { 

class ArgMax16 {
    public:
        ArgMax16() {}
        ~ArgMax16() {}

        XCoreStatus Eval(const int16_t* A, int32_t* C, const int32_t length);
};

} // namespace arg_min_max
} // namespace xcore

#endif  // XCORE_ARG_MIN_MAX_OPERATORS_H_