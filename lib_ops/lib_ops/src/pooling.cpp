// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/pooling.h"

extern "C" {
#include "lib_nn/api/nn_types.h"
}

namespace xcore {
namespace pooling {

//**************************************
//**************************************
//**************************************
// MaxPool
//**************************************
//**************************************
//**************************************
XCoreStatus MaxPool::Init(int32_t X_h, int32_t X_w, int32_t C_in, int32_t Y_h,
                          int32_t Y_w, int32_t C_out) {
  nn_image_params_t params_in;
  params_in.height = X_h;
  params_in.width = X_w;
  params_in.channels = C_in;

  nn_image_params_t params_out;
  params_out.height = Y_h;
  params_out.width = Y_w;
  params_out.channels = C_out;

  nn_window_op_config_t config;
  nn_window_op_config_simple(&config, &params_in, &params_out, options.pool_h,
                             options.pool_w, options.stride_h,
                             options.stride_w);

  maxpool2d_init(&plan_, &params_in, &params_out, &config);

  return kXCoreOk;
}

XCoreStatus MaxPool::Eval(int8_t* Y, const int8_t* X) {
  maxpool2d(Y, X, &plan_);
  return kXCoreOk;
}

//**************************************
//**************************************
//**************************************
// AvgPool
//**************************************
//**************************************
//**************************************
XCoreStatus AvgPool::Init(int32_t X_h, int32_t X_w, int32_t C_in, int32_t Y_h,
                          int32_t Y_w, int32_t C_out) {
  nn_image_params_t params_in;
  params_in.height = X_h;
  params_in.width = X_w;
  params_in.channels = C_in;

  nn_image_params_t params_out;
  params_out.height = Y_h;
  params_out.width = Y_w;
  params_out.channels = C_out;

  nn_window_op_config_t config;
  nn_window_op_config_simple(&config, &params_in, &params_out, options.pool_h,
                             options.pool_w, options.stride_h,
                             options.stride_w);

  avgpool2d_init(&plan_, &params_in, &params_out, &config);

  return kXCoreOk;
}

XCoreStatus AvgPool::Eval(int8_t* Y, const int8_t* X) {
  avgpool2d(Y, X, &plan_);
  return kXCoreOk;
}

//**************************************
//**************************************
//**************************************
// AvgPool_Global
//**************************************
//**************************************
//**************************************
XCoreStatus AvgPool_Global::Init(int32_t bias, int32_t shift, int32_t scale) {
  bias_ = bias;
  shift_ = shift;
  scale_ = scale;

  return kXCoreOk;
}

XCoreStatus AvgPool_Global::Eval(int8_t* Y, const int8_t* X, int32_t X_h,
                                 int32_t X_w, uint32_t C_in) {
  avgpool2d_global(Y, X, X_h, X_w, C_in, bias_, shift_, scale_);

  return kXCoreOk;
}

}  // namespace pooling
}  // namespace xcore