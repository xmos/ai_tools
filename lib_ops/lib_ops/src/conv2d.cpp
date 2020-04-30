// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/conv2d.h"

#include <iostream>

#include "lib_ops/api/allocator.h"
#include "lib_ops/api/benchmarking.h"
#include "lib_ops/api/tracing.h"

namespace xcore {
namespace conv {

struct Conv2DThreadData {
  nn_image_t *Y;
  const nn_image_t *X;
  const nn_tensor_t *K;
  const nn_bss_block_t *BSS;
};

//**************************************
//**************************************
//**************************************
// Conv2D_Deep
//**************************************
//**************************************
//**************************************
struct Conv2DDeepThreadData {
  Conv2DThreadData data;
  const nn_conv2d_deep_plan_t *plan;
  const nn_conv2d_deep_job_t *job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void conv2d_deep_thread_worker(void *context) {
  Conv2DDeepThreadData *td = (Conv2DDeepThreadData *)context;
  conv2d_deep(td->data.Y, td->data.X, td->data.K, td->data.BSS, td->plan,
              td->job);
}
}

Conv2D_Deep::Conv2D_Deep(const Conv2DParams &params,
                         const ParRegionArray &par_regions)
    : params(params), par_regions(par_regions), jobs_(nullptr) {}

XCoreStatus Conv2D_Deep::Init(int32_t X_h, int32_t X_w, int32_t C_in,
                              int32_t Y_h, int32_t Y_w, int32_t C_out) {
  TRACE_INFO(
      "Conv2D_Deep Init id=%p X_h=%ld X_w=%ld C_in=%ld Y_h=%ld Y_w=%ld "
      "C_out=%ld\n",
      this, X_h, X_w, C_in, Y_h, Y_w, C_out);

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

  jobs_ = reinterpret_cast<nn_conv2d_deep_job_t *>(
      xcMalloc(sizeof(nn_conv2d_deep_job_t) * par_regions.size));

  nn_conv2d_job_params_t job_params[par_regions.size];

  for (int i = 0; i < par_regions.size; i++) {
    const ParRegion &region = par_regions[i];
    TRACE_INFO(
        "Conv2D_Deep Init id=%p region top=%ld left=%ld rows=%ld cols=%ld\n",
        this, region.top, region.left, region.rows, region.cols);
    job_params[i].start.rows = region.top;
    job_params[i].start.cols = region.left;
    job_params[i].start.channels = 0;
    job_params[i].size.rows = region.rows;
    job_params[i].size.cols = region.cols;
    job_params[i].size.channels = C_out;
  }

  conv2d_deep_init(&plan_, jobs_, &in_params, &out_params, &job_params[0],
                   &window_params, params.pad.zero_point, par_regions.size);

  return kXCoreOk;
}

XCoreStatus Conv2D_Deep::Eval(int8_t *Y, const int8_t *X, const int8_t *K,
                              const int16_t *BSS) {
  TRACE_INFO("Conv2D_Deep Eval id=%p\n", this);
  TIMER_START();

  Dispatcher *dispatcher = GetDispatcher();

  size_t stack_words;
  GET_STACKWORDS(stack_words, conv2d_deep_thread_worker);

  Conv2DDeepThreadData deep_thread_data[par_regions.size];

  for (int i = 0; i < par_regions.size; i++) {
    deep_thread_data[i].data.Y = (nn_image_t *)Y;
    deep_thread_data[i].data.X = (const nn_image_t *)X;
    deep_thread_data[i].data.K = (const nn_tensor_t *)K;
    deep_thread_data[i].data.BSS = (const nn_bss_block_t *)BSS;
    deep_thread_data[i].plan = &plan_;
    deep_thread_data[i].job = &jobs_[i];
    dispatcher->AddThread(conv2d_deep_thread_worker,
                          reinterpret_cast<void *>(&deep_thread_data[i]),
                          stack_words);
  }

  dispatcher->Join();

  TIMER_STOP("Conv2D_Deep id=%p", this);
  return kXCoreOk;
}

//**************************************
//**************************************
//**************************************
// Conv2D_Shallow
//**************************************
//**************************************
//**************************************
struct Conv2DShallowThreadData {
  Conv2DThreadData data;
  const nn_conv2d_shallowin_plan_t *plan;
  const nn_conv2d_shallowin_job_t *job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void conv2d_shallow_thread_worker(void *context) {
  Conv2DShallowThreadData *td = (Conv2DShallowThreadData *)context;
  conv2d_shallowin(td->data.Y, td->data.X, td->data.K, td->data.BSS, td->plan,
                   td->job);
}
}

Conv2D_Shallow::Conv2D_Shallow(const Conv2DParams &params,
                               const ParRegionArray &par_regions)
    : params(params), par_regions(par_regions), jobs_(nullptr) {}

XCoreStatus Conv2D_Shallow::Init(int32_t X_h, int32_t X_w, int32_t C_in,
                                 int32_t Y_h, int32_t Y_w, int32_t C_out) {
  TRACE_INFO(
      "Conv2D_Shallow Init id=%p X_h=%ld X_w=%ld C_in=%ld Y_h=%ld Y_w=%ld "
      "C_out=%ld\n",
      this, X_h, X_w, C_in, Y_h, Y_w, C_out);

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

  jobs_ = reinterpret_cast<nn_conv2d_shallowin_job_t *>(
      xcMalloc(sizeof(nn_conv2d_shallowin_job_t) * par_regions.size));

  nn_conv2d_job_params_t job_params[par_regions.size];

  for (int i = 0; i < par_regions.size; i++) {
    const ParRegion &region = par_regions[i];
    TRACE_INFO(
        "Conv2D_Shallow Init id=%p region top=%ld left=%ld rows=%ld cols=%ld\n",
        this, region.top, region.left, region.rows, region.cols);
    job_params[i].start.rows = region.top;
    job_params[i].start.cols = region.left;
    job_params[i].start.channels = 0;
    job_params[i].size.rows = region.rows;
    job_params[i].size.cols = region.cols;
    job_params[i].size.channels = C_out;
  }

  conv2d_shallowin_init(&plan_, jobs_, &in_params, &out_params, &job_params[0],
                        &window_params, params.pad.zero_point,
                        par_regions.size);

  return kXCoreOk;
}

XCoreStatus Conv2D_Shallow::Eval(int8_t *Y, const int8_t *X, const int8_t *K,
                                 const int16_t *BSS) {
  TRACE_INFO("Conv2D_Shallow Eval id=%p\n", this);
  TIMER_START();

  Dispatcher *dispatcher = GetDispatcher();

  size_t stack_words;
  GET_STACKWORDS(stack_words, conv2d_shallow_thread_worker);

  Conv2DShallowThreadData shallow_thread_data[par_regions.size];

  for (int i = 0; i < par_regions.size; i++) {
    shallow_thread_data[i].data.Y = (nn_image_t *)Y;
    shallow_thread_data[i].data.X = (const nn_image_t *)X;
    shallow_thread_data[i].data.K = (const nn_tensor_t *)K;
    shallow_thread_data[i].data.BSS = (const nn_bss_block_t *)BSS;
    shallow_thread_data[i].plan = &plan_;
    shallow_thread_data[i].job = &jobs_[i];
    dispatcher->AddThread(conv2d_shallow_thread_worker,
                          reinterpret_cast<void *>(&shallow_thread_data[i]),
                          stack_words);
  }

  dispatcher->Join();

  TIMER_STOP("Conv2D_Shallow id=%p", this);
  return kXCoreOk;
}

//**************************************
//**************************************
//**************************************
// Conv2D_1x1
//**************************************
//**************************************
//**************************************
struct Conv2D1x1ThreadData {
  Conv2DThreadData data;
  const nn_conv2d_1x1_plan_t *plan;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void conv2d_1x1_thread_worker(void *context) {
  Conv2D1x1ThreadData *td = (Conv2D1x1ThreadData *)context;
  conv2d_1x1(td->data.Y, td->data.X, td->data.K, (data16_t *)td->data.BSS,
             td->plan);
}
}

Conv2D_1x1::Conv2D_1x1(const Conv2DParams &params,
                       const ParRegionArray &par_regions)
    : params(params), par_regions(par_regions), plans_(nullptr) {}

XCoreStatus Conv2D_1x1::Init(int32_t X_h, int32_t X_w, int32_t C_in,
                             int32_t Y_h, int32_t Y_w, int32_t C_out) {
  TRACE_INFO(
      "Conv2D_1x1 Init id=%p X_h=%ld X_w=%ld C_in=%ld Y_h=%ld Y_w=%ld "
      "C_out=%ld\n",
      this, X_h, X_w, C_in, Y_h, Y_w, C_out);

  nn_image_params_t in_params;
  in_params.height = X_h;
  in_params.width = X_w;
  in_params.channels = C_in;

  nn_image_params_t out_params;
  out_params.height = Y_h;
  out_params.width = Y_w;
  out_params.channels = C_out;

  if (par_regions.size == 0) {
    // there is no par plan so process entire input
    par_regions.append({0, 0, Y_h, Y_w});
  }

  plans_ = reinterpret_cast<nn_conv2d_1x1_plan_t *>(
      xcMalloc(sizeof(nn_conv2d_1x1_plan_t) * par_regions.size));

  // nn_conv2d_1x1_plan_t job_params[par_regions.size];

  for (int i = 0; i < par_regions.size; i++) {
    const ParRegion &region = par_regions[i];
    TRACE_INFO(
        "Conv2D_1x1 Init id=%p region top=%ld left=%ld rows=%ld cols=%ld\n",
        this, region.top, region.left, region.rows, region.cols);

    // job_params[i].start.rows = region.top;
    // job_params[i].start.cols = region.left;
    // job_params[i].start.channels = 0;
    // job_params[i].size.rows = region.rows;
    // job_params[i].size.cols = region.cols;
    // job_params[i].size.channels = C_out;
    conv2d_1x1_init(&plans_[i], &in_params, &out_params, region.top,
                    region.left, region.rows * region.cols);
  }
  return kXCoreOk;
}

XCoreStatus Conv2D_1x1::Eval(int8_t *Y, const int8_t *X, const int8_t *K,
                             const int16_t *BSS) {
  TRACE_INFO("Conv2D_1x1 Eval id=%p\n", this);
  TIMER_START();

  Dispatcher *dispatcher = GetDispatcher();

  size_t stack_words;
  GET_STACKWORDS(stack_words, conv2d_1x1_thread_worker);

  Conv2D1x1ThreadData thread_data[par_regions.size];

  for (int i = 0; i < par_regions.size; i++) {
    thread_data[i].data.Y = (nn_image_t *)Y;
    thread_data[i].data.X = (const nn_image_t *)X;
    thread_data[i].data.K = (const nn_tensor_t *)K;
    thread_data[i].data.BSS = (const nn_bss_block_t *)BSS;
    thread_data[i].plan = &plans_[i];
    dispatcher->AddThread(conv2d_1x1_thread_worker,
                          reinterpret_cast<void *>(&thread_data[i]),
                          stack_words);
  }

  dispatcher->Join();

  TIMER_STOP("Conv2D_1x1 id=%p", this);
  return kXCoreOk;
}

//**************************************
//**************************************
//**************************************
// Conv2D_depthwise
//**************************************
//**************************************
//**************************************
struct Conv2DDepthwiseThreadData {
  Conv2DThreadData data;
  const nn_conv2d_depthwise_plan_t *plan;
  const nn_conv2d_depthwise_job_t *job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void conv2d_depthwise_thread_worker(void *context) {
  Conv2DDepthwiseThreadData *td = (Conv2DDepthwiseThreadData *)context;
  conv2d_depthwise(td->data.Y, td->data.X, td->data.K,
                   (nn_bss_block_t *)td->data.BSS, td->plan, td->job);
}
}

Conv2D_Depthwise::Conv2D_Depthwise(const Conv2DParams &params,
                                   const ParRegionArray &par_regions)
    : params(params), par_regions(par_regions), jobs_(nullptr) {}

XCoreStatus Conv2D_Depthwise::Init(int32_t X_h, int32_t X_w, int32_t C_in,
                                   int32_t Y_h, int32_t Y_w, int32_t C_out) {
  TRACE_INFO(
      "Conv2D_Depthwise Init id=%p X_h=%ld X_w=%ld C_in=%ld Y_h=%ld Y_w=%ld "
      "C_out=%ld\n",
      this, X_h, X_w, C_in, Y_h, Y_w, C_out);

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

  jobs_ = reinterpret_cast<nn_conv2d_depthwise_job_t *>(
      xcMalloc(sizeof(nn_conv2d_depthwise_job_t) * par_regions.size));

  nn_conv2d_job_params_t job_params[par_regions.size];

  for (int i = 0; i < par_regions.size; i++) {
    const ParRegion &region = par_regions[i];
    TRACE_INFO(
        "Conv2D_Depthwise Init id=%p region top=%ld left=%ld rows=%ld "
        "cols=%ld\n",
        this, region.top, region.left, region.rows, region.cols);

    job_params[i].start.rows = region.top;
    job_params[i].start.cols = region.left;
    job_params[i].start.channels = 0;
    job_params[i].size.rows = region.rows;
    job_params[i].size.cols = region.cols;
    job_params[i].size.channels = C_out;
  }

  conv2d_depthwise_init(&plan_, jobs_, &in_params, &out_params,
                        &job_params[0],    // job_params
                        -params.pad.top,   // window_start_row
                        -params.pad.left,  // window_start_col
                        params.K_h, params.K_w, params.stride_h,
                        params.stride_w, params.pad.zero_point,
                        par_regions.size  // job_count
  );

  return kXCoreOk;
}

XCoreStatus Conv2D_Depthwise::Eval(int8_t *Y, const int8_t *X, const int8_t *K,
                                   const int16_t *BSS) {
  TRACE_INFO("Conv2D_Depthwise Eval id=%p\n", this);
  TIMER_START();

  Dispatcher *dispatcher = GetDispatcher();

  size_t stack_words;
  GET_STACKWORDS(stack_words, conv2d_depthwise_thread_worker);

  Conv2DDepthwiseThreadData thread_data[par_regions.size];

  for (int i = 0; i < par_regions.size; i++) {
    thread_data[i].data.Y = (nn_image_t *)Y;
    thread_data[i].data.X = (const nn_image_t *)X;
    thread_data[i].data.K = (const nn_tensor_t *)K;
    thread_data[i].data.BSS = (const nn_bss_block_t *)BSS;
    thread_data[i].plan = &plan_;
    thread_data[i].job = &jobs_[i];
    dispatcher->AddThread(conv2d_depthwise_thread_worker,
                          reinterpret_cast<void *>(&thread_data[i]),
                          stack_words);
  }

  dispatcher->Join();

  TIMER_STOP("Conv2D_Depthwise id=%p", this);
  return kXCoreOk;
}

}  // namespace conv
}  // namespace xcore
