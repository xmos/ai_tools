// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/conv2d.h"

#include <iostream>

#include "lib_ops/api/benchmarking.h"
#include "lib_ops/api/dispatcher.h"
#include "lib_ops/api/tracing.h"

namespace xcore {
namespace conv {

struct Conv2DThreadData {
  nn_image_t *Y;
  const nn_image_t *X;
  const nn_tensor_t *K;
  const nn_bso_block_t *BSO;
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
  conv2d_deep(td->data.Y, td->data.X, td->data.K, td->data.BSO, td->plan,
              td->job);
}
}

Conv2D_Deep::Conv2D_Deep(const Conv2DParams &params,
                         const ExecutionPlan &execution_plan)
    : params(params), execution_plan(execution_plan), jobs_(nullptr) {}

XCoreStatus Conv2D_Deep::Init(int32_t X_h, int32_t X_w, int32_t C_in,
                              int32_t Y_h, int32_t Y_w, int32_t C_out) {
  TRACE_INFO(
      "Conv2D_Deep Init id=%p X_h=%ld X_w=%ld C_in=%ld Y_h=%ld Y_w=%ld "
      "C_out=%ld\n",
      this, X_h, X_w, C_in, Y_h, Y_w, C_out);

  // compute size (in bytes) of 1 output channel's weights
  weights_preload_size_ = params.K_h * params.K_w * C_in;

  // setup kernel parameters
  nn_image_params_t in_params = {(uint32_t)X_h, (uint32_t)X_w, (uint32_t)C_in};
  nn_image_params_t out_params = {(uint32_t)Y_h, (uint32_t)Y_w,
                                  (uint32_t)C_out};
  nn_conv2d_window_params_t window_params = {
      (uint32_t)params.K_h, (uint32_t)params.K_w, -params.pad.top,
      -params.pad.left,     params.stride_h,      params.stride_w};

  // allocate the jobs
  int32_t n_jobs =
      execution_plan.changrps.GetSize() * execution_plan.regions.GetSize();
  jobs_ = reinterpret_cast<nn_conv2d_deep_job_t *>(
      GetDispatcher()->AllocatePersistantBuffer(sizeof(nn_conv2d_deep_job_t) *
                                                n_jobs));

  // set job parameters
  nn_conv2d_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];
    for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = execution_plan.regions[i_rg];
      TRACE_INFO(
          "Conv2D_Deep Init id=%p, chan group start=%ld size=%ld, region "
          "top=%ld left=%ld rows=%ld "
          "cols=%ld\n",
          this, changrp.start, changrp.size, region.top, region.left,
          region.rows, region.cols);

      job_params[i_cg * execution_plan.regions.GetSize() + i_rg] = {
          region.top,  region.left, changrp.start,
          region.rows, region.cols, changrp.size};
    }
  }

  // initialize the kernel
  conv2d_deep_init(&plan_, jobs_, &in_params, &out_params, &job_params[0],
                   &window_params, params.pad.zero_point, n_jobs);

  return kXCoreOk;
}

XCoreStatus Conv2D_Deep::Eval(int8_t *Y, const int8_t *X, const int8_t *K,
                              const int16_t *BSO) {
  TRACE_INFO("Conv2D_Deep Eval id=%p\n", this);
  TIMER_START();

  // initialize the dispatcher
  Dispatcher *dispatcher = GetDispatcher();
  size_t stack_words;
  GET_STACKWORDS(stack_words, conv2d_deep_thread_worker);
  dispatcher->InitializeTasks(conv2d_deep_thread_worker, stack_words);

  // create thread data and tasks
  Conv2DDeepThreadData thread_data[execution_plan.GetNumThreads()];
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];

    // preload the weights and biases
    dispatcher->PreloadWeights(&tK, K, weights_preload_size_, changrp);
    dispatcher->PreloadBiases(&tBSO, BSO, changrp);

    for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
      int32_t i_job = i_cg * execution_plan.regions.GetSize() + i_rg;
      thread_data[i_rg].data.Y = (nn_image_t *)Y;
      thread_data[i_rg].data.X = (const nn_image_t *)X;
      thread_data[i_rg].data.K = (const nn_tensor_t *)tK;
      thread_data[i_rg].data.BSO = (const nn_bso_block_t *)tBSO;
      thread_data[i_rg].plan = &plan_;
      jobs_[i_job].stride.start.K = 0;
      jobs_[i_job].stride.start.BSO = 0;
      thread_data[i_rg].job = &jobs_[i_job];
      dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_rg]));
    }
    // start and wait for tasks to complete
    dispatcher->JoinTasks();
  }

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
  const nn_conv2d_shallowin_plan_t *plan;
  nn_conv2d_shallowin_job_t *job;
  Conv2DThreadData data;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void conv2d_shallow_thread_worker(void *context) {
  Conv2DShallowThreadData *td = (Conv2DShallowThreadData *)context;
  conv2d_shallowin(td->data.Y, td->data.X, td->data.K, td->data.BSO, td->plan,
                   td->job);
}
}

Conv2D_Shallow::Conv2D_Shallow(const Conv2DParams &params,
                               const ExecutionPlan &execution_plan)
    : params(params), execution_plan(execution_plan), jobs_(nullptr) {}

XCoreStatus Conv2D_Shallow::Init(int32_t X_h, int32_t X_w, int32_t C_in,
                                 int32_t Y_h, int32_t Y_w, int32_t C_out,
                                 int32_t K_w_padded) {
  TRACE_INFO(
      "Conv2D_Shallow Init id=%p X_h=%ld X_w=%ld C_in=%ld Y_h=%ld Y_w=%ld "
      "C_out=%ld, K_w_padded=%ld\n",
      this, X_h, X_w, C_in, Y_h, Y_w, C_out, K_w_padded);

  // compute size (in bytes) of 1 output channel's weights
  weights_preload_size_ = params.K_h * K_w_padded * C_in;

  // setup kernel parameters
  nn_image_params_t in_params = {(uint32_t)X_h, (uint32_t)X_w, (uint32_t)C_in};
  nn_image_params_t out_params = {(uint32_t)Y_h, (uint32_t)Y_w,
                                  (uint32_t)C_out};
  nn_conv2d_window_params_t window_params = {
      (uint32_t)params.K_h, (uint32_t)params.K_w, -params.pad.top,
      -params.pad.left,     params.stride_h,      params.stride_w};

  // allocate the jobs
  int32_t n_jobs =
      execution_plan.changrps.GetSize() * execution_plan.regions.GetSize();
  jobs_ = reinterpret_cast<nn_conv2d_shallowin_job_t *>(
      GetDispatcher()->AllocatePersistantBuffer(
          sizeof(nn_conv2d_shallowin_job_t) * n_jobs));

  // set job parameters
  nn_conv2d_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];
    for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = execution_plan.regions[i_rg];
      TRACE_INFO(
          "Conv2D_Shallow Init id=%p, chan group start=%ld size=%ld, region "
          "top=%ld left=%ld rows=%ld "
          "cols=%ld\n",
          this, changrp.start, changrp.size, region.top, region.left,
          region.rows, region.cols);

      job_params[i_cg * execution_plan.regions.GetSize() + i_rg] = {
          region.top,  region.left, changrp.start,
          region.rows, region.cols, changrp.size};
    }
  }

  // initialize the kernel
  conv2d_shallowin_init(&plan_, jobs_, &in_params, &out_params, &job_params[0],
                        &window_params, params.pad.zero_point, n_jobs);

  return kXCoreOk;
}  // namespace conv

XCoreStatus Conv2D_Shallow::Eval(int8_t *Y, const int8_t *X, const int8_t *K,
                                 const int16_t *BSO) {
  TRACE_INFO("Conv2D_Shallow Eval id=%p\n", this);
  TIMER_START();

  // initialize the dispatcher
  Dispatcher *dispatcher = GetDispatcher();
  size_t stack_words;
  GET_STACKWORDS(stack_words, conv2d_shallow_thread_worker);
  dispatcher->InitializeTasks(conv2d_shallow_thread_worker, stack_words);

  // create thread data and tasks
  Conv2DShallowThreadData thread_data[execution_plan.GetNumThreads()];
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];

    // preload the weights and biases
    dispatcher->PreloadWeights(&tK, K, weights_preload_size_, changrp);
    dispatcher->PreloadBiases(&tBSO, BSO, changrp);

    for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
      int32_t i_job = i_cg * execution_plan.regions.GetSize() + i_rg;
      thread_data[i_rg].data.Y = (nn_image_t *)Y;
      thread_data[i_rg].data.X = (const nn_image_t *)X;
      thread_data[i_rg].data.K = (const nn_tensor_t *)tK;
      thread_data[i_rg].data.BSO = (const nn_bso_block_t *)tBSO;
      thread_data[i_rg].plan = &plan_;
      jobs_[i_job].stride.start.K = 0;
      jobs_[i_job].stride.start.BSO = 0;
      thread_data[i_rg].job = &jobs_[i_job];
      dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_rg]));
    }
    // start and wait for tasks to complete
    dispatcher->JoinTasks();
  }

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
  nn_conv2d_1x1_job_t *job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void conv2d_1x1_thread_worker(void *context) {
  Conv2D1x1ThreadData *td = (Conv2D1x1ThreadData *)context;
  conv2d_1x1(td->data.Y, td->data.X, td->data.K, td->data.BSO, td->plan,
             td->job);
}
}

Conv2D_1x1::Conv2D_1x1(const Conv2DParams &params,
                       const ExecutionPlan &execution_plan)
    : params(params), execution_plan(execution_plan), jobs_(nullptr) {}

XCoreStatus Conv2D_1x1::Init(int32_t X_h, int32_t X_w, int32_t C_in,
                             int32_t Y_h, int32_t Y_w, int32_t C_out) {
  TRACE_INFO(
      "Conv2D_1x1 Init id=%p X_h=%ld X_w=%ld C_in=%ld Y_h=%ld Y_w=%ld "
      "C_out=%ld\n",
      this, X_h, X_w, C_in, Y_h, Y_w, C_out);

  // compute size (in bytes) of 1 output channel's weights
  weights_preload_size_ = C_in;

  // setup kernel parameters
  nn_image_params_t in_params = {(uint32_t)X_h, (uint32_t)X_w, (uint32_t)C_in};
  nn_image_params_t out_params = {(uint32_t)Y_h, (uint32_t)Y_w,
                                  (uint32_t)C_out};

  // allocate the jobs
  int32_t n_jobs =
      execution_plan.changrps.GetSize() * execution_plan.regions.GetSize();
  jobs_ = reinterpret_cast<nn_conv2d_1x1_job_t *>(
      GetDispatcher()->AllocatePersistantBuffer(sizeof(nn_conv2d_1x1_job_t) *
                                                n_jobs));

  // set job parameters
  nn_conv2d_1x1_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];
    for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = execution_plan.regions[i_rg];
      TRACE_INFO(
          "Conv2D_1x1 Init id=%p, chan group start=%ld size=%ld, region "
          "top=%ld left=%ld rows=%ld "
          "cols=%ld\n",
          this, changrp.start, changrp.size, region.top, region.left,
          region.rows, region.cols);

      job_params[i_cg * execution_plan.regions.GetSize() + i_rg] = {
          region.top, region.left, changrp.start,
          (uint32_t)(region.rows * region.cols), (uint32_t)changrp.size};
    }
  }

  // initialize the kernel
  conv2d_1x1_init(&plan_, jobs_, &in_params, &out_params, &job_params[0],
                  n_jobs);

  return kXCoreOk;
}

XCoreStatus Conv2D_1x1::Eval(int8_t *Y, const int8_t *X, const int8_t *K,
                             const int16_t *BSO) {
  TRACE_INFO("Conv2D_1x1 Eval id=%p\n", this);
  TIMER_START();

  // initialize the dispatcher
  Dispatcher *dispatcher = GetDispatcher();
  size_t stack_words;
  GET_STACKWORDS(stack_words, conv2d_1x1_thread_worker);
  dispatcher->InitializeTasks(conv2d_1x1_thread_worker, stack_words);

  // create thread data and tasks
  Conv2D1x1ThreadData thread_data[execution_plan.GetNumThreads()];
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];

    // preload the weights and biases
    dispatcher->PreloadWeights(&tK, K, weights_preload_size_, changrp);
    dispatcher->PreloadBiases(&tBSO, BSO, changrp);

    for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
      int32_t i_job = i_cg * execution_plan.regions.GetSize() + i_rg;
      thread_data[i_rg].data.Y = (nn_image_t *)Y;
      thread_data[i_rg].data.X = (const nn_image_t *)X;
      thread_data[i_rg].data.K = (const nn_tensor_t *)tK;
      thread_data[i_rg].data.BSO = (const nn_bso_block_t *)tBSO;
      thread_data[i_rg].plan = &plan_;
      jobs_[i_job].start.K = 0;
      jobs_[i_job].start.BSO = 0;
      thread_data[i_rg].job = &jobs_[i_job];
      dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_rg]));
    }
    // start and wait for tasks to complete
    dispatcher->JoinTasks();
  }

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
  conv2d_depthwise(td->data.Y, td->data.X, td->data.K, td->data.BSO, td->plan,
                   td->job);
}
}

Conv2D_Depthwise::Conv2D_Depthwise(const Conv2DParams &params,
                                   const ExecutionPlan &execution_plan)
    : params(params), execution_plan(execution_plan), jobs_(nullptr) {}

XCoreStatus Conv2D_Depthwise::Init(int32_t X_h, int32_t X_w, int32_t C_in,
                                   int32_t Y_h, int32_t Y_w, int32_t C_out) {
  TRACE_INFO(
      "Conv2D_Depthwise Init id=%p X_h=%ld X_w=%ld C_in=%ld Y_h=%ld Y_w=%ld "
      "C_out=%ld\n",
      this, X_h, X_w, C_in, Y_h, Y_w, C_out);

  // compute size (in bytes) of 1 output channel's weights
  weights_preload_size_ = params.K_h * params.K_w * C_in;

  // setup kernel parameters
  nn_image_params_t in_params = {(uint32_t)X_h, (uint32_t)X_w, (uint32_t)C_in};
  nn_image_params_t out_params = {(uint32_t)Y_h, (uint32_t)Y_w,
                                  (uint32_t)C_out};
  nn_conv2d_window_params_t window_params = {
      (uint32_t)params.K_h, (uint32_t)params.K_w, -params.pad.top,
      -params.pad.left,     params.stride_h,      params.stride_w};

  // allocate the jobs
  int32_t n_jobs =
      execution_plan.changrps.GetSize() * execution_plan.regions.GetSize();
  jobs_ = reinterpret_cast<nn_conv2d_depthwise_job_t *>(
      GetDispatcher()->AllocatePersistantBuffer(
          sizeof(nn_conv2d_depthwise_job_t) * n_jobs));

  // set job parameters
  nn_conv2d_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];
    for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = execution_plan.regions[i_rg];
      TRACE_INFO(
          "Conv2D_Depthwise Init id=%p, chan group start=%ld size=%ld, region "
          "top=%ld left=%ld rows=%ld "
          "cols=%ld\n",
          this, changrp.start, changrp.size, region.top, region.left,
          region.rows, region.cols);

      job_params[i_cg * execution_plan.regions.GetSize() + i_rg] = {
          region.top,  region.left, changrp.start,
          region.rows, region.cols, changrp.size};
    }
  }

  // initialize the kernel
  conv2d_depthwise_init(&plan_, jobs_, &in_params, &out_params,
                        &job_params[0],    // job_params
                        -params.pad.top,   // window_start_row
                        -params.pad.left,  // window_start_col
                        params.K_h, params.K_w, params.stride_h,
                        params.stride_w, params.pad.zero_point,
                        n_jobs  // job_count
  );

  return kXCoreOk;
}

XCoreStatus Conv2D_Depthwise::Eval(int8_t *Y, const int8_t *X, const int8_t *K,
                                   const int16_t *BSO) {
  TRACE_INFO("Conv2D_Depthwise Eval id=%p\n", this);
  TIMER_START();

  // initialize the dispatcher
  Dispatcher *dispatcher = GetDispatcher();
  size_t stack_words;
  GET_STACKWORDS(stack_words, conv2d_depthwise_thread_worker);
  dispatcher->InitializeTasks(conv2d_depthwise_thread_worker, stack_words);

  // create thread data and tasks
  Conv2DDepthwiseThreadData thread_data[execution_plan.GetNumThreads()];

  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;

  // preload the weights
  //   NOTE: They all need to be preloaded for each job
  //         This may be changed in the future.
  dispatcher->PreloadBuffer(&tK, K, weights_preload_size_);

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];

    // preload the biases
    dispatcher->PreloadBiases(&tBSO, BSO, changrp);

    for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
      int32_t i_job = i_cg * execution_plan.regions.GetSize() + i_rg;
      thread_data[i_rg].data.Y = (nn_image_t *)Y;
      thread_data[i_rg].data.X = (const nn_image_t *)X;
      thread_data[i_rg].data.K = (const nn_tensor_t *)tK;
      thread_data[i_rg].data.BSO = (const nn_bso_block_t *)tBSO;
      thread_data[i_rg].plan = &plan_;
      jobs_[i_job].stride.start.K = 0;
      jobs_[i_job].stride.start.BSO = 0;
      thread_data[i_rg].job = &jobs_[i_job];
      dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_rg]));
    }
    // start and wait for tasks to complete
    dispatcher->JoinTasks();
  }

  TIMER_STOP("Conv2D_Depthwise id=%p", this);
  return kXCoreOk;
}

}  // namespace conv
}  // namespace xcore
