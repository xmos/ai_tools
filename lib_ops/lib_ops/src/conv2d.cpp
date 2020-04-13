// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/conv2d.h"
#include <iostream>

namespace xcore {
namespace conv {

//**************************************
//**************************************
//**************************************
// Conv2D_Deep
//**************************************
//**************************************
//**************************************
struct Conv2DDeepThreadData {
  nn_image_t* Y;
  const nn_image_t* X;
  const nn_tensor_t* K;
  const nn_bss_block_t* BSS;
  const nn_conv2d_deep_plan_t* plan;
  const nn_conv2d_deep_job_t* job;
};

extern "C" {
ATTRIBUTE_KERNEL_FUNCTION void conv2d_deep_thread_worker(void* context) {
  Conv2DDeepThreadData* data = (Conv2DDeepThreadData*)context;
  conv2d_deep(data->Y, data->X, data->K, data->BSS, data->plan, data->job);
}
}

Conv2D_Deep::Conv2D_Deep(const Conv2DParams& params,
                         const ParRegionArray& par_regions)
    : params(params), par_regions(par_regions), jobs_(nullptr) {
  OperatorDispatcher& dispatcher = GetOperatorDispatcher();

  jobs_ = reinterpret_cast<nn_conv2d_deep_job_t*>(
      dispatcher.AllocatePersistentBuffer(sizeof(nn_conv2d_deep_job_t) *
                                          par_regions.size));
}

XCoreStatus Conv2D_Deep::Init(int32_t X_h, int32_t X_w, int32_t C_in,
                              int32_t Y_h, int32_t Y_w, int32_t C_out) {
  nn_image_params_t in_params;
  in_params.height = X_h;
  in_params.width = X_w;
  in_params.channels = C_in;

  nn_image_params_t out_params;
  out_params.height = Y_h;
  out_params.width = Y_w;
  out_params.channels = C_out;

  nn_conv2d_window_params_t window_params;
  window_params.shape.height = params.K_h;
  window_params.shape.width = params.K_w;
  window_params.start.row = -params.pad.top;
  window_params.start.column = -params.pad.left;
  window_params.stride.vertical = params.stride_h;
  window_params.stride.horizontal = params.stride_w;

  if (par_regions.size == 0) {
    // there is no par plan so process entire input
    par_regions.append({0, 0, Y_h, Y_w});
  }

  nn_conv2d_job_params_t job_params[par_regions.size];

  for (int i = 0; i < par_regions.size; i++) {
    const ParRegion& region = par_regions[i];
    job_params[i].start.rows = region.top;
    job_params[i].start.cols = region.left;
    job_params[i].start.channels = 0;
    job_params[i].size.rows = region.rows;
    job_params[i].size.cols = region.cols;
    job_params[i].size.channels = C_out;
  }

  conv2d_deep_init(&plan_, jobs_, &in_params, &out_params, &job_params[0],
                   &window_params, params.pad.zero_point,
                   par_regions.size  // job_count
  );

  // reserve threads and stack memory
  OperatorDispatcher& dispatcher = GetOperatorDispatcher();
  size_t stack_words = 0;
  GET_STACKWORDS(stack_words, conv2d_deep_thread_worker);
  dispatcher.AllocateStackBuffer(par_regions.size, stack_words);

  return kXCoreOk;
}

XCoreStatus Conv2D_Deep::Eval(int8_t* Y, const int8_t* X, const int8_t* K,
                              const int16_t* BSS) {
  OperatorDispatcher& dispatcher = GetOperatorDispatcher();

  size_t stack_words;
  GET_STACKWORDS(stack_words, conv2d_deep_thread_worker);

  Conv2DDeepThreadData thread_data[par_regions.size];

  for (int i = 0; i < par_regions.size; i++) {
    thread_data[i].Y = (nn_image_t*)Y;
    thread_data[i].X = (const nn_image_t*)X;
    thread_data[i].K = (const nn_tensor_t*)K;
    thread_data[i].BSS = (const nn_bss_block_t*)BSS;
    thread_data[i].plan = &plan_;
    thread_data[i].job = &jobs_[i];
    dispatcher.Add(conv2d_deep_thread_worker,
                   reinterpret_cast<void*>(&thread_data[i]), stack_words);
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
  init_params.K_h = unpadded_shape.K_h;
  init_params.K_w = unpadded_shape.K_w;
  init_params.C_in = C_in;
  init_params.C_out = unpadded_shape.C_out;
  init_params.pad_mode = padding_mode_;
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
  nn_image_params_t in_params;
  in_params.height = X_h;
  in_params.width = X_w;
  in_params.channels = C_in;

  nn_image_params_t out_params;
  out_params.height = Y_h;
  out_params.width = Y_w;
  out_params.channels = C_out;

  conv2d_1x1_init(&plan_, &in_params, &out_params, start_row, start_col,
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
  nn_image_params_t in_params;
  in_params.height = X_h;
  in_params.width = X_w;
  in_params.channels = C_in;

  nn_image_params_t out_params;
  out_params.height = Y_h;
  out_params.width = Y_w;
  out_params.channels = C_out;

  conv2d_depthwise_init(&plan_, &job_, &in_params, &out_params,
                        nullptr,           // job_params
                        -params.pad.top,   // window_start_row
                        -params.pad.left,  // window_start_col
                        params.K_h, params.K_w, params.stride_h,
                        params.stride_w, params.pad.zero_point,
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
