// Copyright (c) 2020, XMOS Ltd, All rights reserved

#ifndef XCORE_FULLY_CONNECTED_OPERATOR_HPP_
#define XCORE_FULLY_CONNECTED_OPERATOR_HPP_

#include <cstdint>
#include "lib_ops/api/lib_ops.h"

extern "C" {
    #include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace fully_connected { 

class FullyConnected_16 {
    public:
        FullyConnected_16() {}
        ~FullyConnected_16() {}

        XCoreStatus Init(int32_t C_in, int32_t C_out);
        XCoreStatus Eval(int16_t* Y, const int8_t* W, const int8_t* X, const int16_t* BSS);

    private:
        nn_fully_connected_plan_t plan_;
};

} // namespace fully_connected
} // namespace xcore

#endif  // XCORE_FULLY_CONNECTED_OPERATOR_HPP_