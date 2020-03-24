// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_CONV2D_OPERATOR_H_
#define XCORE_CONV2D_OPERATOR_H_

#include <cstdint>
#include <vector>
#include "lib_ops/api/lib_ops.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace conv {

union padding_t {
  padding_mode_t mode;
  struct padding_data_t {
    int8_t top;
    int8_t left;
    int8_t zero_point;
  } data;
};

struct Conv2DUnpaddedShape {
  int32_t C_in;
  int32_t C_out;
};

struct Conv2DOptions {
  padding_t padding;
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

  XCoreStatus Init(int32_t X_h, int32_t X_w, int32_t C_in, int32_t Y_h,
                   int32_t Y_w, int32_t C_out, int32_t zero_point,
                   const int8_t* K, const int16_t* bias);
  XCoreStatus Eval(int8_t* Y, const int8_t* X, const int8_t* K,
                   const int16_t* SS);

  Conv2DOptions options;
  ParPlan par;

 private:
  std::vector<nn_conv2d_dido_params_t> params_;
};

class Conv2D_SIDO {
 public:
  Conv2D_SIDO() {}
  ~Conv2D_SIDO() {}

  XCoreStatus Init(int32_t X_h, int32_t X_w, int32_t C_in, int32_t Y_h,
                   int32_t Y_w, int32_t zero_point, const int8_t* K,
                   const int16_t* bias);
  XCoreStatus Eval(int8_t* Y, const int8_t* X, const int8_t* K,
                   const int16_t* SS);

  Conv2DOptions options;
  Conv2DUnpaddedShape unpadded_shape;

 private:
  nn_conv2d_sido_params_t params_;
};

class Conv2D_1x1 {
 public:
  Conv2D_1x1() {}
  ~Conv2D_1x1() {}

  XCoreStatus Init(int32_t X_h, int32_t X_w, int32_t C_in, int32_t Y_h,
                   int32_t Y_w, int32_t C_out, int32_t start_row,
                   int32_t start_col, int32_t out_pixels);
  XCoreStatus Eval(int8_t* Y, const int8_t* X, const int8_t* K,
                   const int16_t* BSS);

  Conv2DOptions options;

 private:
  nn_conv2d_1x1_plan_t plan_;
};

class Conv2D_Depthwise {
 public:
  Conv2D_Depthwise() {}
  ~Conv2D_Depthwise() {}

  XCoreStatus Init(int32_t X_h, int32_t X_w, int32_t C_in, int32_t Y_h,
                   int32_t Y_w, int32_t C_out);
  XCoreStatus Eval(int8_t* Y, const int8_t* X, const int8_t* K,
                   const int16_t* BSS);

  Conv2DOptions options;

 private:
  nn_conv2d_depthwise_plan_t plan_;
  nn_conv2d_depthwise_job_t job_;
};

}  // namespace conv
}  // namespace xcore

#endif  // XCORE_CONV2D_OPERATOR_H_