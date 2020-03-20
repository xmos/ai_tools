// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_CONV2D_OPERATOR_H_
#define XCORE_CONV2D_OPERATOR_H_

#include <vector>
#include <cstdint>
#include "lib_ops/api/lib_ops.h"

extern "C" {
    #include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace conv { 

struct Conv2DOptions {
    padding_mode_t padding_mode;
    int32_t C_in;
    int32_t C_out;
    int32_t K_h;
    int32_t K_w;
    int32_t stride_h;
    int32_t stride_w;
};

struct Conv2DDIDOThreadData {
    int8_t* Y;
    const nn_conv2d_dido_params_t* params;
    const int8_t* X;
    const int8_t* K;
    const int16_t* SS;
};

class Conv2D_DIDO {
    public:
        Conv2D_DIDO() {}
        ~Conv2D_DIDO() {}

        XCoreStatus Init(int32_t X_h, int32_t X_w, int32_t zero_point, 
                  int32_t rows, int32_t cols, const int8_t* K, const int16_t* bias);
        XCoreStatus Eval(int8_t* Y, const int8_t* X, const int8_t* K, const int16_t* SS);

        Conv2DOptions options;
        ParPlan par;
    private:
        std::vector<nn_conv2d_dido_params_t> params_;
};

} // namespace conv
} // namespace xcore

#endif  // XCORE_CONV2D_OPERATOR_H_