// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "operators/pooling.h"

#include "operators/dispatcher.h"

extern "C" {
#include "lib_nn/api/nn_types.h"
}

namespace xcore {
namespace pooling {

struct PoolingThreadData {
  int8_t* Y;
  const int8_t* X;
};

//**************************************
//**************************************
//**************************************
// MaxPool
//**************************************
//**************************************
//**************************************
struct MaxPoolThreadData {
  const nn_maxpool2d_plan_t* plan;
  nn_pool2d_job_t* job;
  PoolingThreadData data;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void maxpool_thread_worker(void* context) {
  MaxPoolThreadData* td = (MaxPoolThreadData*)context;
  maxpool2d(td->data.Y, td->data.X, td->plan, td->job);
}
}

MaxPool::MaxPool(const PoolingParams& params,
                 const ExecutionPlan& execution_plan)
    : params(params),
      execution_plan(execution_plan),
      jobs_(nullptr),
      stack_scratch_index_(-1),
      stack_size_(0) {}

TfLiteStatus MaxPool::Prepare(TfLiteContext* ctx, int32_t X_h, int32_t X_w,
                              int32_t C_in, int32_t Y_h, int32_t Y_w,
                              int32_t C_out) {
  Dispatcher* dispatcher = GetDispatcher();

  TF_LITE_REPORT_STATUS(dispatcher->GetReporter(),
                        "MaxPool Prepare X_h=%d X_w=%d C_in=%d Y_h=%d Y_w=%d "
                        "C_out=%d",
                        X_h, X_w, C_in, Y_h, Y_w, C_out);

  nn_image_params_t in_params = {(uint32_t)X_h, (uint32_t)X_w, (uint32_t)C_in};
  nn_image_params_t out_params = {(uint32_t)Y_h, (uint32_t)Y_w,
                                  (uint32_t)C_out};
  nn_window_params_t window_params = {
      {(uint32_t)params.pool_h, (uint32_t)params.pool_w},
      {0, 0},
      {params.stride_h, params.stride_w}};

  // allocate the jobs
  int32_t n_jobs = execution_plan.regions.GetSize();
  TF_LITE_ENSURE_STATUS(ctx->AllocatePersistentBuffer(
      ctx, sizeof(nn_pool2d_job_t) * n_jobs, reinterpret_cast<void**>(&jobs_)));

  // allocate the stack for thread workers
  GET_STACKSIZE(stack_size_, maxpool_thread_worker);
  TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
      ctx, stack_size_ * execution_plan.GetNumThreads(),
      &stack_scratch_index_));

  // set job parameters
  nn_window_op_job_params_t job_params[n_jobs];

  for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
    const RowColRegion& region = execution_plan.regions[i_rg];
    TF_LITE_REPORT_STATUS(
        dispatcher->GetReporter(),
        "MaxPool Prepare region top=%d left=%d rows=%d cols=%d", region.top,
        region.left, region.rows, region.cols);

    job_params[i_rg] = {{region.top, region.left, 0},
                        {region.rows, region.cols, C_out}};
  }

  // initialize the kernel
  maxpool2d_init(&plan_, jobs_, &in_params, &out_params, &window_params,
                 &job_params[0], n_jobs);

  return kTfLiteOk;
}

TfLiteStatus MaxPool::Eval(TfLiteContext* ctx, int8_t* Y, const int8_t* X) {
  Dispatcher* dispatcher = GetDispatcher();

  // initialize the dispatcher
  char* stack =
      static_cast<char*>(ctx->GetScratchBuffer(ctx, stack_scratch_index_));
  assert(stack);
  dispatcher->InitializeTasks(maxpool_thread_worker, stack, stack_size_);

  // create thread data and tasks
  MaxPoolThreadData thread_data[execution_plan.GetNumThreads()];

  for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
    thread_data[i_rg].data.Y = (nn_image_t*)Y;
    thread_data[i_rg].data.X = (const nn_image_t*)X;
    thread_data[i_rg].plan = &plan_;
    thread_data[i_rg].job = &jobs_[i_rg];
    dispatcher->AddTask(reinterpret_cast<void*>(&thread_data[i_rg]));
  }

  // start and wait for tasks to complete
  dispatcher->JoinTasks();

  return kTfLiteOk;
}  // namespace pooling

//**************************************
//**************************************
//**************************************
// AvgPool
//**************************************
//**************************************
//**************************************
struct AvgPoolThreadData {
  const nn_avgpool2d_plan_t* plan;
  nn_pool2d_job_t* job;
  PoolingThreadData data;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void avgpool_thread_worker(void* context) {
  AvgPoolThreadData* td = (AvgPoolThreadData*)context;
  avgpool2d(td->data.Y, td->data.X, td->plan, td->job);
}
}

AvgPool::AvgPool(const PoolingParams& params,
                 const ExecutionPlan& execution_plan)
    : params(params),
      execution_plan(execution_plan),
      jobs_(nullptr),
      stack_scratch_index_(-1),
      stack_size_(0) {}

TfLiteStatus AvgPool::Prepare(TfLiteContext* ctx, int32_t X_h, int32_t X_w,
                              int32_t C_in, int32_t Y_h, int32_t Y_w,
                              int32_t C_out) {
  Dispatcher* dispatcher = GetDispatcher();

  TF_LITE_REPORT_STATUS(dispatcher->GetReporter(),
                        "AvgPool Prepare X_h=%d X_w=%d C_in=%d Y_h=%d Y_w=%d "
                        "C_out=%d",
                        X_h, X_w, C_in, Y_h, Y_w, C_out);

  nn_image_params_t in_params = {(uint32_t)X_h, (uint32_t)X_w, (uint32_t)C_in};
  nn_image_params_t out_params = {(uint32_t)Y_h, (uint32_t)Y_w,
                                  (uint32_t)C_out};

  nn_window_params_t window_params = {
      {(uint32_t)params.pool_h, (uint32_t)params.pool_w},
      {0, 0},
      {params.stride_h, params.stride_w}};

  // allocate the jobs
  int32_t n_jobs = execution_plan.regions.GetSize();
  TF_LITE_ENSURE_STATUS(ctx->AllocatePersistentBuffer(
      ctx, sizeof(nn_pool2d_job_t) * n_jobs, reinterpret_cast<void**>(&jobs_)));

  // allocate the stack for thread workers
  GET_STACKSIZE(stack_size_, avgpool_thread_worker);
  TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
      ctx, stack_size_ * execution_plan.GetNumThreads(),
      &stack_scratch_index_));

  // set job parameters
  nn_window_op_job_params_t job_params[n_jobs];

  for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
    const RowColRegion& region = execution_plan.regions[i_rg];
    TF_LITE_REPORT_STATUS(
        dispatcher->GetReporter(),
        "AvgPool Prepare region top=%d left=%d rows=%d cols=%d", region.top,
        region.left, region.rows, region.cols);

    job_params[i_rg] = {{region.top, region.left, 0},
                        {region.rows, region.cols, C_out}};
  }

  // initialize the kernel
  avgpool2d_init(&plan_, jobs_, &in_params, &out_params, &window_params,
                 &job_params[0], n_jobs);

  return kTfLiteOk;
}

TfLiteStatus AvgPool::Eval(TfLiteContext* ctx, int8_t* Y, const int8_t* X) {
  Dispatcher* dispatcher = GetDispatcher();

  // initialize the dispatcher
  char* stack =
      static_cast<char*>(ctx->GetScratchBuffer(ctx, stack_scratch_index_));
  assert(stack);
  dispatcher->InitializeTasks(avgpool_thread_worker, stack, stack_size_);

  // create thread data and tasks
  AvgPoolThreadData thread_data[execution_plan.regions.GetSize()];

  for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
    thread_data[i_rg].data.Y = (nn_image_t*)Y;
    thread_data[i_rg].data.X = (const nn_image_t*)X;
    thread_data[i_rg].plan = &plan_;
    thread_data[i_rg].job = &jobs_[i_rg];
    dispatcher->AddTask(reinterpret_cast<void*>(&thread_data[i_rg]));
  }

  // start and wait for tasks to complete
  dispatcher->JoinTasks();

  return kTfLiteOk;
}

//**************************************
//**************************************
//**************************************
// AvgPool_Global
//**************************************
//**************************************
//**************************************
struct AvgPoolGlobalThreadData {
  const nn_avgpool2d_global_plan_t* plan;
  nn_avgpool2d_global_job_t* job;
  PoolingThreadData data;
  int32_t bias;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void avgpool_global_thread_worker(void* context) {
  AvgPoolGlobalThreadData* td = (AvgPoolGlobalThreadData*)context;
  avgpool2d_global(td->data.Y, td->data.X, td->bias, td->plan, td->job);
}
}

AvgPool_Global::AvgPool_Global(const ExecutionPlan& execution_plan)
    : execution_plan(execution_plan),
      jobs_(nullptr),
      stack_scratch_index_(-1),
      stack_size_(0) {}

TfLiteStatus AvgPool_Global::Prepare(TfLiteContext* ctx, int32_t X_h,
                                     int32_t X_w, int32_t C_in, int32_t bias,
                                     int32_t shift, int32_t scale) {
  Dispatcher* dispatcher = GetDispatcher();

  TF_LITE_REPORT_STATUS(dispatcher->GetReporter(),
                        "AvgPool_Global Prepare X_h=%d X_w=%d C_in=%d", X_h,
                        X_w, C_in);

  bias_ = bias;

  // setup kernel parameters
  nn_image_params_t in_params = {(uint32_t)X_h, (uint32_t)X_w, (uint32_t)C_in};

  // allocate the jobs
  int32_t n_jobs = execution_plan.changrps.GetSize();
  TF_LITE_ENSURE_STATUS(ctx->AllocatePersistentBuffer(
      ctx, sizeof(nn_avgpool2d_global_job_t) * n_jobs,
      reinterpret_cast<void**>(&jobs_)));

  // allocate the stack for thread workers
  GET_STACKSIZE(stack_size_, avgpool_global_thread_worker);
  TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
      ctx, stack_size_ * execution_plan.GetNumThreads(),
      &stack_scratch_index_));

  // set job parameters
  nn_avgpool2d_global_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup& changrp = execution_plan.changrps[i_cg];
    TF_LITE_REPORT_STATUS(dispatcher->GetReporter(),
                          "AvgPool_Global Prepare chan group start=%d size=%d",
                          changrp.start, changrp.size);

    job_params[i_cg] = {(uint32_t)changrp.start, (channel_count_t)changrp.size};
  }

  // initialize the kernel
  avgpool2d_global_init(&plan_, jobs_, &in_params, &job_params[0], n_jobs);
  // NOTE: Overriding the plan's shift and scale is temporary.
  //       See issue #144
  plan_.shift = shift;
  plan_.scale = scale;

  return kTfLiteOk;
}

TfLiteStatus AvgPool_Global::Eval(TfLiteContext* ctx, int8_t* Y,
                                  const int8_t* X, int32_t X_h, int32_t X_w,
                                  uint32_t C_in) {
  Dispatcher* dispatcher = GetDispatcher();

  // initialize the dispatcher
  char* stack =
      static_cast<char*>(ctx->GetScratchBuffer(ctx, stack_scratch_index_));
  assert(stack);
  dispatcher->InitializeTasks(avgpool_global_thread_worker, stack, stack_size_);

  // create thread data and tasks
  int n_th = execution_plan.GetNumThreads();
  AvgPoolGlobalThreadData thread_data[n_th];

  int i_th = 0;

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    thread_data[i_th].data.Y = Y;
    thread_data[i_th].data.X = X;
    thread_data[i_th].bias = bias_;
    thread_data[i_th].plan = &plan_;
    thread_data[i_th].job = &jobs_[i_cg];

    dispatcher->AddTask(reinterpret_cast<void*>(&thread_data[i_th]));

    i_th++;
    if (i_th == n_th) {
      // start and wait for tasks to complete
      dispatcher->JoinTasks();
      i_th = 0;
    }
  }
  dispatcher->JoinTasks();  // finish up any added tasks

  return kTfLiteOk;
}

}  // namespace pooling
}  // namespace xcore