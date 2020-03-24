// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/conv2d.h"

namespace xcore {
namespace conv {

extern "C" {
ATTRIBUTE_KERNEL_FUNCTION void conv2d_dido_thread_worker(void* context) {
  Conv2DDIDOThreadData* data = (Conv2DDIDOThreadData*)context;
  conv2d_deepin_deepout(data->Y, data->params, data->X, data->K, data->SS);
}
}

//**************************************
//**************************************
//**************************************
// Conv2D_DIDO
//**************************************
//**************************************
//**************************************
XCoreStatus Conv2D_DIDO::Init(int32_t X_h, int32_t X_w, int32_t C_in,
                              int32_t Y_h, int32_t Y_w, int32_t C_out,
                              int32_t zero_point, const int8_t* K,
                              const int16_t* bias) {
  nn_conv2d_init_params_t init_params;
  nn_conv2d_region_params_t region_params;

  init_params.X_height = X_h;
  init_params.X_width = X_w;
  init_params.K_h = options.K_h;
  init_params.K_w = options.K_w;
  init_params.C_in = C_in;
  init_params.C_out = C_out;
  init_params.pad_mode = options.padding.mode;
  init_params.zero_point = zero_point;

  if (par.size() == 0) {
    // there is no par plan so process entire input
    par.emplace_back(0, 0, Y_h, Y_w);
  }

  for (const auto& region : par) {
    nn_conv2d_dido_params_t params;

    region_params.top = region.top;
    region_params.left = region.left;
    region_params.rows = region.rows;
    region_params.cols = region.cols;

    conv2d_deepin_deepout_init(&params, &init_params, &region_params, K,
                               (data16_t*)bias);
    params_.emplace_back(std::move(params));
  }

  // reserve threads and stack memory
  KernelDispatcher& dispatcher = GetKernelDispatcher();
  size_t stack_words=0;
  GET_STACKWORDS(stack_words, conv2d_dido_thread_worker);
  dispatcher.Reserve(params_.size(), stack_words);

  return kXCoreOk;
}

XCoreStatus Conv2D_DIDO::Eval(int8_t* Y, const int8_t* X, const int8_t* K,
                              const int16_t* SS) {
  KernelDispatcher& dispatcher = GetKernelDispatcher();

  size_t stack_words;
  GET_STACKWORDS(stack_words, conv2d_dido_thread_worker);

  for (auto it = params_.cbegin(); it < params_.cend(); ++it) {
    Conv2DDIDOThreadData* data = new Conv2DDIDOThreadData();

    data->Y = Y;
    data->params = &(*it);
    data->X = X;
    data->K = K;
    data->SS = SS;
    dispatcher.Add(conv2d_dido_thread_worker, (void*)data, stack_words);
  }

  dispatcher.Start();
  dispatcher.Wait();

  return kXCoreOk;
}

//**************************************
//**************************************
//**************************************
// Conv2D_SIDO
//**************************************
//**************************************
//**************************************
XCoreStatus Conv2D_SIDO::Init(int32_t X_h, int32_t X_w, int32_t C_in,
                              int32_t Y_h, int32_t Y_w, int32_t zero_point,
                              const int8_t* K, const int16_t* bias) {
  nn_conv2d_init_params_t init_params;
  nn_conv2d_region_params_t region_params;

  init_params.X_height = X_h;
  init_params.X_width = X_w;
  init_params.K_h = options.K_h;
  init_params.K_w = options.K_w;
  init_params.C_in = C_in;
  init_params.C_out = unpadded_shape.C_out;
  init_params.pad_mode = options.padding.mode;
  init_params.zero_point = zero_point;

  region_params.top = 0;
  region_params.left = 0;
  region_params.rows = Y_h;
  region_params.cols = Y_w;

  conv2d_shallowin_deepout_init(&params_, &init_params, &region_params, K,
                                (data16_t*)bias);

  return kXCoreOk;
}

XCoreStatus Conv2D_SIDO::Eval(int8_t* Y, const int8_t* X, const int8_t* K,
                              const int16_t* SS) {
  conv2d_shallowin_deepout(Y, &params_, X, K, SS);
  return kXCoreOk;
}

//**************************************
//**************************************
//**************************************
// Conv2D_1x1
//**************************************
//**************************************
//**************************************
XCoreStatus Conv2D_1x1::Init(int32_t X_h, int32_t X_w, int32_t C_in,
                             int32_t Y_h, int32_t Y_w, int32_t C_out,
                             int32_t start_row, int32_t start_col,
                             int32_t out_pixels) {
  nn_image_params_t params_in;
  params_in.height = X_h;
  params_in.width = X_w;
  params_in.channels = C_in;

  nn_image_params_t params_out;
  params_out.height = Y_h;
  params_out.width = Y_w;
  params_out.channels = C_out;

  conv2d_1x1_init(&plan_, &params_in, &params_out, start_row, start_col,
                  out_pixels);

  return kXCoreOk;
}

XCoreStatus Conv2D_1x1::Eval(int8_t* Y, const int8_t* X, const int8_t* K,
                             const int16_t* BSS) {
  conv2d_1x1(Y, X, K, (data16_t*)BSS, &plan_);
  return kXCoreOk;
}

//**************************************
//**************************************
//**************************************
// Conv2D_depthwise
//**************************************
//**************************************
//**************************************
XCoreStatus Conv2D_Depthwise::Init(int32_t X_h, int32_t X_w, int32_t C_in,
                                   int32_t Y_h, int32_t Y_w, int32_t C_out) {
  nn_image_params_t params_in;
  params_in.height = X_h;
  params_in.width = X_w;
  params_in.channels = C_in;

  nn_image_params_t params_out;
  params_out.height = Y_h;
  params_out.width = Y_w;
  params_out.channels = C_out;

  int8_t window_start_row = -options.padding.data.top;
  int8_t window_start_col = -options.padding.data.left;
  int8_t zero_point = options.padding.data.zero_point;

  conv2d_depthwise_init(&plan_, &job_, &params_in, &params_out,
                        nullptr,  // job_params
                        window_start_row, window_start_col, options.K_h,
                        options.K_w, options.stride_h, options.stride_w,
                        zero_point,
                        1  // job_count
  );

  return kXCoreOk;
}

XCoreStatus Conv2D_Depthwise::Eval(int8_t* Y, const int8_t* X, const int8_t* K,
                                   const int16_t* BSS) {
  conv2d_depthwise(Y, X, K, (nn_bss_block_t*)BSS, &plan_, &job_);
  return kXCoreOk;
}

}  // namespace conv
}  // namespace xcore
