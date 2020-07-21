// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/pooling.h"

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
    : params(params), execution_plan(execution_plan), jobs_(nullptr) {}

XCoreStatus MaxPool::Prepare(int32_t X_h, int32_t X_w, int32_t C_in,
                             int32_t Y_h, int32_t Y_w, int32_t C_out) {
  TF_LITE_REPORT_STATUS(
      GetDispatcher()->GetReporter(),
      "MaxPool Prepare id=%p X_h=%ld X_w=%ld C_in=%ld Y_h=%ld Y_w=%ld "
      "C_out=%ld\n",
      this, X_h, X_w, C_in, Y_h, Y_w, C_out);

  nn_image_params_t in_params = {(uint32_t)X_h, (uint32_t)X_w, (uint32_t)C_in};
  nn_image_params_t out_params = {(uint32_t)Y_h, (uint32_t)Y_w,
                                  (uint32_t)C_out};
  nn_window_params_t window_params = {
      {(uint32_t)params.pool_h, (uint32_t)params.pool_w},
      {0, 0},
      {params.stride_h, params.stride_w}};

  // allocate the jobs
  int32_t n_jobs = execution_plan.regions.GetSize();
  jobs_ = reinterpret_cast<nn_pool2d_job_t*>(
      GetDispatcher()->AllocatePersistantBuffer(sizeof(nn_pool2d_job_t) *
                                                n_jobs));

  // set job parameters
  nn_window_op_job_params_t job_params[n_jobs];

  for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
    const RowColRegion& region = execution_plan.regions[i_rg];
    TF_LITE_REPORT_STATUS(
        GetDispatcher()->GetReporter(),
        "MaxPool Prepare id=%p, region top=%ld left=%ld rows=%ld cols=%ld\n",
        this, region.top, region.left, region.rows, region.cols);

    job_params[i_rg] = {{region.top, region.left, 0},
                        {region.rows, region.cols, C_out}};
  }

  // initialize the kernel
  maxpool2d_init(&plan_, jobs_, &in_params, &out_params, &window_params,
                 &job_params[0], n_jobs);

  return kXCoreOk;
}

XCoreStatus MaxPool::Eval(int8_t* Y, const int8_t* X) {
  TF_LITE_REPORT_STATUS(GetDispatcher()->GetReporter(), "MaxPool Eval id=%p\n",
                        this);

  // initialize the dispatcher
  Dispatcher* dispatcher = GetDispatcher();
  size_t stack_words;
  GET_STACKWORDS(stack_words, maxpool_thread_worker);
  dispatcher->InitializeTasks(maxpool_thread_worker, stack_words);

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

  return kXCoreOk;
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
    : params(params), execution_plan(execution_plan), jobs_(nullptr) {}

XCoreStatus AvgPool::Prepare(int32_t X_h, int32_t X_w, int32_t C_in,
                             int32_t Y_h, int32_t Y_w, int32_t C_out) {
  TF_LITE_REPORT_STATUS(
      GetDispatcher()->GetReporter(),
      "AvgPool Prepare id=%p X_h=%ld X_w=%ld C_in=%ld Y_h=%ld Y_w=%ld "
      "C_out=%ld\n",
      this, X_h, X_w, C_in, Y_h, Y_w, C_out);

  nn_image_params_t in_params = {(uint32_t)X_h, (uint32_t)X_w, (uint32_t)C_in};
  nn_image_params_t out_params = {(uint32_t)Y_h, (uint32_t)Y_w,
                                  (uint32_t)C_out};

  nn_window_params_t window_params = {
      {(uint32_t)params.pool_h, (uint32_t)params.pool_w},
      {0, 0},
      {params.stride_h, params.stride_w}};

  // allocate the jobs
  int32_t n_jobs = execution_plan.regions.GetSize();
  jobs_ = reinterpret_cast<nn_pool2d_job_t*>(
      GetDispatcher()->AllocatePersistantBuffer(sizeof(nn_pool2d_job_t) *
                                                n_jobs));

  // set job parameters
  nn_window_op_job_params_t job_params[n_jobs];

  for (int i_rg = 0; i_rg < execution_plan.regions.GetSize(); i_rg++) {
    const RowColRegion& region = execution_plan.regions[i_rg];
    TF_LITE_REPORT_STATUS(
        GetDispatcher()->GetReporter(),
        "AvgPool Prepare id=%p, region top=%ld left=%ld rows=%ld cols=%ld\n",
        this, region.top, region.left, region.rows, region.cols);

    job_params[i_rg] = {{region.top, region.left, 0},
                        {region.rows, region.cols, C_out}};
  }

  // initialize the kernel
  avgpool2d_init(&plan_, jobs_, &in_params, &out_params, &window_params,
                 &job_params[0], n_jobs);

  return kXCoreOk;
}

XCoreStatus AvgPool::Eval(int8_t* Y, const int8_t* X) {
  TF_LITE_REPORT_STATUS(GetDispatcher()->GetReporter(), "AvgPool Eval id=%p\n",
                        this);

  // initialize the dispatcher
  Dispatcher* dispatcher = GetDispatcher();
  size_t stack_words;
  GET_STACKWORDS(stack_words, avgpool_thread_worker);
  dispatcher->InitializeTasks(avgpool_thread_worker, stack_words);

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

  return kXCoreOk;
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
    : execution_plan(execution_plan), jobs_(nullptr) {}

XCoreStatus AvgPool_Global::Prepare(int32_t X_h, int32_t X_w, int32_t C_in,
                                    int32_t bias, int32_t shift,
                                    int32_t scale) {
  TF_LITE_REPORT_STATUS(GetDispatcher()->GetReporter(),
                        "AvgPool_Global Prepare id=%p\n", this);
  bias_ = bias;

  // setup kernel parameters
  nn_image_params_t in_params = {(uint32_t)X_h, (uint32_t)X_w, (uint32_t)C_in};

  // allocate the jobs
  int32_t n_jobs = execution_plan.changrps.GetSize();
  jobs_ = reinterpret_cast<nn_avgpool2d_global_job_t*>(
      GetDispatcher()->AllocatePersistantBuffer(
          sizeof(nn_avgpool2d_global_job_t) * n_jobs));

  // set job parameters
  nn_avgpool2d_global_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup& changrp = execution_plan.changrps[i_cg];
    TF_LITE_REPORT_STATUS(
        GetDispatcher()->GetReporter(),
        "AvgPool_Global Prepare id=%p, chan group start=%ld size=%ld\n", this,
        changrp.start, changrp.size);

    job_params[i_cg] = {(uint32_t)changrp.start, (channel_count_t)changrp.size};
  }

  // initialize the kernel
  avgpool2d_global_init(&plan_, jobs_, &in_params, &job_params[0], n_jobs);
  // NOTE: Overriding the plan's shift and scale is temporary.
  //       See issue #144
  plan_.shift = shift;
  plan_.scale = scale;

  return kXCoreOk;
}

XCoreStatus AvgPool_Global::Eval(int8_t* Y, const int8_t* X, int32_t X_h,
                                 int32_t X_w, uint32_t C_in) {
  TF_LITE_REPORT_STATUS(GetDispatcher()->GetReporter(),
                        "AvgPool_Global Eval id=%p\n", this);

  // initialize the dispatcher
  Dispatcher* dispatcher = GetDispatcher();
  size_t stack_words;
  GET_STACKWORDS(stack_words, avgpool_global_thread_worker);
  dispatcher->InitializeTasks(avgpool_global_thread_worker, stack_words);

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

  return kXCoreOk;
}

}  // namespace pooling
}  // namespace xcore