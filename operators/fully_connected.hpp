// Copyright (c) 2020, XMOS Ltd, All rights reserved

#ifndef XCORE_FULLY_CONNECTED_OPERATOR_HPP_
#define XCORE_FULLY_CONNECTED_OPERATOR_HPP_

#include "operators.hpp"

namespace xcore {
namespace fully_connected { 

class FullyConnected_16 {
    public:
        FullyConnected_16() {}
        ~FullyConnected_16() {}

        void Init(int32_t C_in, int32_t C_out) {
            fully_connected_init(&plan_, C_in, C_out);
        }

        void Eval(int16_t* Y, const int8_t* W, const int8_t* X, const int16_t* BSS) {
            fully_connected_16(Y, W, X, (data16_t*) BSS, &plan_);
        }

    private:
        nn_fully_connected_plan_t plan_;
};

} // namespace fully_connected
} // namespace xcore

#endif  // XCORE_FULLY_CONNECTED_OPERATOR_HPP_