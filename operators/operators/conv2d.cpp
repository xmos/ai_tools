// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "operators/conv2d.h"

#include <iostream>

#include "operators/dispatcher.h"

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

Conv2D_Deep::Conv2D_Deep()
    : jobs_(nullptr),
      stack_scratch_index_(-1),
      stack_size_(0),
      weights_scratch_index_(-1),
      bias_scratch_index_(-1) {}

void Conv2D_Deep::Init(TfLiteContext *ctx) {
  // allocate the jobs
  ctx->AllocatePersistentBuffer(ctx,
                                sizeof(nn_conv2d_deep_job_t) *
                                    execution_plan.changrps.GetSize() *
                                    execution_plan.regions.GetSize(),
                                reinterpret_cast<void **>(&jobs_));
}

TfLiteStatus Conv2D_Deep::Prepare(TfLiteContext *ctx, const int8_t *K,
                                  const int16_t *BSO, int32_t X_h, int32_t X_w,
                                  int32_t C_in, int32_t Y_h, int32_t Y_w,
                                  int32_t C_out) {
  Dispatcher *dispatcher = GetDispatcher();

  TF_LITE_REPORT_STATUS(
      dispatcher->GetReporter(),
      "Conv2D_Deep Prepare X_h=%d X_w=%d C_in=%d Y_h=%d Y_w=%d "
      "C_out=%d",
      X_h, X_w, C_in, Y_h, Y_w, C_out);

  // setup kernel parameters
  nn_image_params_t in_params = {(uint32_t)X_h, (uint32_t)X_w, (uint32_t)C_in};
  nn_image_params_t out_params = {(uint32_t)Y_h, (uint32_t)Y_w,
                                  (uint32_t)C_out};
  nn_window_params_t conv_window = {
      {(uint32_t)params.K_h, (uint32_t)params.K_w},
      {-params.pad.top, -params.pad.left},
      {params.stride_h, params.stride_w}};

  int32_t n_jobs =
      execution_plan.changrps.GetSize() * execution_plan.regions.GetSize();

  // allocate the stack for thread workers
  GET_STACKSIZE(stack_size_, conv2d_deep_thread_worker);
  TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
      ctx, stack_size_ * execution_plan.GetNumThreads(),
      &stack_scratch_index_));

  // allocate scratch buffers for weights and biases (if necessary)
  if (IS_NOT_RAM(K)) {
    TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
        ctx, execution_plan.GetWeightsScratchSize(), &weights_scratch_index_));
  }
  if (IS_NOT_RAM(BSO)) {
    TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
        ctx, execution_plan.GetBiasScratchSize(), &bias_scratch_index_));
  }

  // set job parameters
  nn_conv2d_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];
    for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = execution_plan.regions[i_rg];
      TF_LITE_REPORT_STATUS(
          dispatcher->GetReporter(),
          "Conv2D_Deep Prepare chan group start=%d size=%d, region "
          "top=%d left=%d rows=%d "
          "cols=%d",
          changrp.start, changrp.size, region.top, region.left, region.rows,
          region.cols);

      job_params[i_cg * execution_plan.regions.GetSize() + i_rg] = {
          {region.top, region.left, changrp.start},
          {region.rows, region.cols, changrp.size}};
    }
  }

  // initialize the kernel
  conv2d_deep_init(&plan_, jobs_, &in_params, &out_params, &job_params[0],
                   &conv_window, params.pad.zero_point, n_jobs);

  return kTfLiteOk;
}

TfLiteStatus Conv2D_Deep::Eval(TfLiteContext *ctx, int8_t *Y, const int8_t *X,
                               const int8_t *K, const int16_t *BSO) {
  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack =
      static_cast<char *>(ctx->GetScratchBuffer(ctx, stack_scratch_index_));
  assert(stack);
  dispatcher->InitializeTasks(conv2d_deep_thread_worker, stack, stack_size_);

  // create thread data and tasks
  Conv2DDeepThreadData thread_data[execution_plan.GetNumThreads()];

  // load weights & bias scratch buffers (if necessary)
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;

  if (weights_scratch_index_ >= 0) {
    tK = static_cast<int8_t *>(
        ctx->GetScratchBuffer(ctx, weights_scratch_index_));
    assert(tK);
  }
  if (bias_scratch_index_ >= 0) {
    tBSO =
        static_cast<int16_t *>(ctx->GetScratchBuffer(ctx, bias_scratch_index_));
    assert(tBSO);
  }

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];

    // fetch the weights and biases
    dispatcher->FetchWeights(&tK, K, execution_plan.GetWeightsScratchSize(),
                             changrp);
    dispatcher->FetchBiases(&tBSO, BSO, execution_plan.GetBiasScratchSize(),
                            changrp);

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

  return kTfLiteOk;
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

Conv2D_Shallow::Conv2D_Shallow()
    : jobs_(nullptr),
      stack_scratch_index_(-1),
      stack_size_(0),
      weights_scratch_index_(-1),
      bias_scratch_index_(-1) {}

void Conv2D_Shallow::Init(TfLiteContext *ctx) {
  // allocate the jobs
  ctx->AllocatePersistentBuffer(ctx,
                                sizeof(nn_conv2d_shallowin_job_t) *
                                    execution_plan.changrps.GetSize() *
                                    execution_plan.regions.GetSize(),
                                reinterpret_cast<void **>(&jobs_));
}

TfLiteStatus Conv2D_Shallow::Prepare(TfLiteContext *ctx, const int8_t *K,
                                     const int16_t *BSO, int32_t X_h,
                                     int32_t X_w, int32_t C_in, int32_t Y_h,
                                     int32_t Y_w, int32_t C_out,
                                     int32_t K_w_padded) {
  Dispatcher *dispatcher = GetDispatcher();

  TF_LITE_REPORT_STATUS(
      dispatcher->GetReporter(),
      "Conv2D_Shallow Prepare X_h=%d X_w=%d C_in=%d Y_h=%d Y_w=%d "
      "C_out=%d, K_w_padded=%d",
      X_h, X_w, C_in, Y_h, Y_w, C_out, K_w_padded);

  // setup kernel parameters
  nn_image_params_t in_params = {(uint32_t)X_h, (uint32_t)X_w, (uint32_t)C_in};
  nn_image_params_t out_params = {(uint32_t)Y_h, (uint32_t)Y_w,
                                  (uint32_t)C_out};
  nn_window_params_t conv_window = {
      {(uint32_t)params.K_h, (uint32_t)params.K_w},
      {-params.pad.top, -params.pad.left},
      {params.stride_h, params.stride_w}};

  int32_t n_jobs =
      execution_plan.changrps.GetSize() * execution_plan.regions.GetSize();

  // allocate the stack for thread workers
  GET_STACKSIZE(stack_size_, conv2d_shallow_thread_worker);
  TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
      ctx, stack_size_ * execution_plan.GetNumThreads(),
      &stack_scratch_index_));

  // allocate scratch buffers for weights and biases (if necessary)
  if (IS_NOT_RAM(K)) {
    TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
        ctx, execution_plan.GetWeightsScratchSize(), &weights_scratch_index_));
  }
  if (IS_NOT_RAM(BSO)) {
    TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
        ctx, execution_plan.GetBiasScratchSize(), &bias_scratch_index_));
  }

  // set job parameters
  nn_conv2d_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];
    for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = execution_plan.regions[i_rg];
      TF_LITE_REPORT_STATUS(
          dispatcher->GetReporter(),
          "Conv2D_Shallow Prepare chan group start=%d size=%d, region "
          "top=%d left=%d rows=%d "
          "cols=%d",
          changrp.start, changrp.size, region.top, region.left, region.rows,
          region.cols);

      job_params[i_cg * execution_plan.regions.GetSize() + i_rg] = {
          {region.top, region.left, changrp.start},
          {region.rows, region.cols, changrp.size}};
    }
  }

  // initialize the kernel
  conv2d_shallowin_init(&plan_, jobs_, &in_params, &out_params, &job_params[0],
                        &conv_window, params.pad.zero_point, n_jobs);

  return kTfLiteOk;
}  // namespace conv

TfLiteStatus Conv2D_Shallow::Eval(TfLiteContext *ctx, int8_t *Y,
                                  const int8_t *X, const int8_t *K,
                                  const int16_t *BSO) {
  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack =
      static_cast<char *>(ctx->GetScratchBuffer(ctx, stack_scratch_index_));
  assert(stack);
  dispatcher->InitializeTasks(conv2d_shallow_thread_worker, stack, stack_size_);

  // create thread data and tasks
  Conv2DShallowThreadData thread_data[execution_plan.GetNumThreads()];

  // load weights & bias scratch buffers (if necessary)
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;

  if (weights_scratch_index_ >= 0) {
    tK = static_cast<int8_t *>(
        ctx->GetScratchBuffer(ctx, weights_scratch_index_));
    assert(tK);
  }
  if (bias_scratch_index_ >= 0) {
    tBSO =
        static_cast<int16_t *>(ctx->GetScratchBuffer(ctx, bias_scratch_index_));
    assert(tBSO);
  }

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];

    // fetch the weights and biases
    dispatcher->FetchWeights(&tK, K, execution_plan.GetWeightsScratchSize(),
                             changrp);
    dispatcher->FetchBiases(&tBSO, BSO, execution_plan.GetBiasScratchSize(),
                            changrp);

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

  return kTfLiteOk;
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

Conv2D_1x1::Conv2D_1x1()
    : jobs_(nullptr),
      stack_scratch_index_(-1),
      stack_size_(0),
      weights_scratch_index_(-1),
      bias_scratch_index_(-1) {}

void Conv2D_1x1::Init(TfLiteContext *ctx) {
  // allocate the jobs
  ctx->AllocatePersistentBuffer(ctx,
                                sizeof(nn_conv2d_1x1_job_t) *
                                    execution_plan.changrps.GetSize() *
                                    execution_plan.regions.GetSize(),
                                reinterpret_cast<void **>(&jobs_));
}

TfLiteStatus Conv2D_1x1::Prepare(TfLiteContext *ctx, const int8_t *K,
                                 const int16_t *BSO, int32_t X_h, int32_t X_w,
                                 int32_t C_in, int32_t Y_h, int32_t Y_w,
                                 int32_t C_out) {
  Dispatcher *dispatcher = GetDispatcher();

  TF_LITE_REPORT_STATUS(dispatcher->GetReporter(),
                        "Conv2D_1x1 Prepare X_h=%d X_w=%d C_in=%d "
                        "Y_h=%d Y_w=%d C_out=%d",
                        X_h, X_w, C_in, Y_h, Y_w, C_out);

  // setup kernel parameters
  nn_image_params_t in_params = {(uint32_t)X_h, (uint32_t)X_w, (uint32_t)C_in};
  nn_image_params_t out_params = {(uint32_t)Y_h, (uint32_t)Y_w,
                                  (uint32_t)C_out};

  int32_t n_jobs =
      execution_plan.changrps.GetSize() * execution_plan.regions.GetSize();

  // allocate the stack for thread workers
  GET_STACKSIZE(stack_size_, conv2d_1x1_thread_worker);
  TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
      ctx, stack_size_ * execution_plan.GetNumThreads(),
      &stack_scratch_index_));

  // allocate scratch buffers for weights and biases (if necessary)
  if (IS_NOT_RAM(K)) {
    TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
        ctx, execution_plan.GetWeightsScratchSize(), &weights_scratch_index_));
  }
  if (IS_NOT_RAM(BSO)) {
    TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
        ctx, execution_plan.GetBiasScratchSize(), &bias_scratch_index_));
  }

  // set job parameters
  nn_conv2d_1x1_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];
    for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = execution_plan.regions[i_rg];
      TF_LITE_REPORT_STATUS(
          dispatcher->GetReporter(),
          "Conv2D_1x1 Prepare chan group start=%d size=%d, region "
          "top=%d left=%d rows=%d "
          "cols=%d",
          changrp.start, changrp.size, region.top, region.left, region.rows,
          region.cols);

      job_params[i_cg * execution_plan.regions.GetSize() + i_rg] = {
          {region.top, region.left, changrp.start},
          {(uint32_t)(region.rows * region.cols), (uint32_t)changrp.size}};
    }
  }

  // initialize the kernel
  conv2d_1x1_init(&plan_, jobs_, &in_params, &out_params, &job_params[0],
                  n_jobs);

  return kTfLiteOk;
}

TfLiteStatus Conv2D_1x1::Eval(TfLiteContext *ctx, int8_t *Y, const int8_t *X,
                              const int8_t *K, const int16_t *BSO) {
  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack =
      static_cast<char *>(ctx->GetScratchBuffer(ctx, stack_scratch_index_));
  assert(stack);
  dispatcher->InitializeTasks(conv2d_1x1_thread_worker, stack, stack_size_);

  // create thread data and tasks
  Conv2D1x1ThreadData thread_data[execution_plan.GetNumThreads()];

  // load weights & bias scratch buffers (if necessary)
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;

  if (weights_scratch_index_ >= 0) {
    tK = static_cast<int8_t *>(
        ctx->GetScratchBuffer(ctx, weights_scratch_index_));
    assert(tK);
  }
  if (bias_scratch_index_ >= 0) {
    tBSO =
        static_cast<int16_t *>(ctx->GetScratchBuffer(ctx, bias_scratch_index_));
    assert(tBSO);
  }

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];

    // fetch the weights and biases
    dispatcher->FetchWeights(&tK, K, execution_plan.GetWeightsScratchSize(),
                             changrp);
    dispatcher->FetchBiases(&tBSO, BSO, execution_plan.GetBiasScratchSize(),
                            changrp);

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

  return kTfLiteOk;
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

Conv2D_Depthwise::Conv2D_Depthwise()
    : jobs_(nullptr),
      stack_scratch_index_(-1),
      stack_size_(0),
      weights_scratch_index_(-1),
      bias_scratch_index_(-1) {}

void Conv2D_Depthwise::Init(TfLiteContext *ctx) {
  // allocate the jobs
  ctx->AllocatePersistentBuffer(ctx,
                                sizeof(nn_conv2d_depthwise_job_t) *
                                    execution_plan.changrps.GetSize() *
                                    execution_plan.regions.GetSize(),
                                reinterpret_cast<void **>(&jobs_));
}

TfLiteStatus Conv2D_Depthwise::Prepare(TfLiteContext *ctx, const int8_t *K,
                                       const int16_t *BSO, int32_t X_h,
                                       int32_t X_w, int32_t C_in, int32_t Y_h,
                                       int32_t Y_w, int32_t C_out) {
  Dispatcher *dispatcher = GetDispatcher();

  TF_LITE_REPORT_STATUS(
      dispatcher->GetReporter(),
      "Conv2D_Depthwise Prepare X_h=%d X_w=%d C_in=%d Y_h=%d Y_w=%d "
      "C_out=%d",
      X_h, X_w, C_in, Y_h, Y_w, C_out);

  // setup kernel parameters
  nn_image_params_t in_params = {(uint32_t)X_h, (uint32_t)X_w, (uint32_t)C_in};
  nn_image_params_t out_params = {(uint32_t)Y_h, (uint32_t)Y_w,
                                  (uint32_t)C_out};
  nn_window_params_t conv_window = {
      {(uint32_t)params.K_h, (uint32_t)params.K_w},
      {-params.pad.top, -params.pad.left},
      {params.stride_h, params.stride_w}};

  int32_t n_jobs =
      execution_plan.changrps.GetSize() * execution_plan.regions.GetSize();

  // allocate the stack for thread workers
  GET_STACKSIZE(stack_size_, conv2d_depthwise_thread_worker);
  TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
      ctx, stack_size_ * execution_plan.regions.GetSize(),
      &stack_scratch_index_));

  // allocate scratch buffers for weights and biases (if necessary)
  if (IS_NOT_RAM(K)) {
    TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
        ctx, execution_plan.GetWeightsScratchSize(), &weights_scratch_index_));
  }
  if (IS_NOT_RAM(BSO)) {
    TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
        ctx, execution_plan.GetBiasScratchSize(), &bias_scratch_index_));
  }

  // set job parameters
  nn_conv2d_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];
    for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = execution_plan.regions[i_rg];
      TF_LITE_REPORT_STATUS(
          dispatcher->GetReporter(),
          "Conv2D_Depthwise Prepare chan group start=%d size=%d, "
          "region "
          "top=%d left=%d rows=%d "
          "cols=%d",
          changrp.start, changrp.size, region.top, region.left, region.rows,
          region.cols);

      job_params[i_cg * execution_plan.regions.GetSize() + i_rg] = {
          {region.top, region.left, changrp.start},
          {region.rows, region.cols, changrp.size}};
    }
  }

  // initialize the kernel
  conv2d_depthwise_init(&plan_, jobs_, &in_params, &out_params, &job_params[0],
                        &conv_window, params.pad.zero_point, n_jobs);

  return kTfLiteOk;
}

TfLiteStatus Conv2D_Depthwise::Eval(TfLiteContext *ctx, int8_t *Y,
                                    const int8_t *X, const int8_t *K,
                                    const int16_t *BSO) {
  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack =
      static_cast<char *>(ctx->GetScratchBuffer(ctx, stack_scratch_index_));
  assert(stack);
  dispatcher->InitializeTasks(conv2d_depthwise_thread_worker, stack,
                              stack_size_);

  // create thread data and tasks
  Conv2DDepthwiseThreadData thread_data[execution_plan.GetNumThreads()];

  // load weights & bias scratch buffers (if necessary)
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;

  if (weights_scratch_index_ >= 0) {
    tK = static_cast<int8_t *>(
        ctx->GetScratchBuffer(ctx, weights_scratch_index_));
    assert(tK);
  }
  if (bias_scratch_index_ >= 0) {
    tBSO =
        static_cast<int16_t *>(ctx->GetScratchBuffer(ctx, bias_scratch_index_));
    assert(tBSO);
  }

  // fetch the weights
  //   NOTE: They all need to be fetched for each job
  //         This may be changed in the future.
  dispatcher->FetchBuffer(&tK, K, execution_plan.GetWeightsScratchSize());

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];

    // fetch the biases
    dispatcher->FetchBiases(&tBSO, BSO, execution_plan.GetBiasScratchSize(),
                            changrp);

    for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
      int32_t i_job = i_cg * execution_plan.regions.GetSize() + i_rg;
      thread_data[i_rg].data.Y = (nn_image_t *)Y;
      thread_data[i_rg].data.X = (const nn_image_t *)X;
      thread_data[i_rg].data.K = (const nn_tensor_t *)tK;
      thread_data[i_rg].data.BSO = (const nn_bso_block_t *)tBSO;
      thread_data[i_rg].plan = &plan_;
      jobs_[i_job].stride.start.BSO = 0;
      thread_data[i_rg].job = &jobs_[i_job];
      dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_rg]));
    }
    // start and wait for tasks to complete
    dispatcher->JoinTasks();
  }

  return kTfLiteOk;
}

}  // namespace conv
}  // namespace xcore
