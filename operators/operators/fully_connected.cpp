// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "operators/fully_connected.h"

#include <algorithm>
#include <cstring>

#include "operators/dispatcher.h"

namespace xcore {
namespace fully_connected {

struct FullyConnectedThreadData {
  int16_t *Y;
  const nn_tensor_t *X;
  const nn_tensor_t *W;
  const nn_bso_block_t *BSO;
  nn_fully_connected_plan_t *plan;
  nn_fully_connected_job_t *job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void fully_connected_thread_worker(void *context) {
  FullyConnectedThreadData *td = (FullyConnectedThreadData *)context;
  fully_connected_16(td->Y, td->W, td->X, td->BSO, td->plan, td->job);
}
}

FullyConnected_16::FullyConnected_16(const ExecutionPlan &execution_plan)
    : execution_plan(execution_plan),
      jobs_(nullptr),
      stack_scratch_index_(-1),
      stack_size_(0),
      weights_scratch_index_(-1),
      bias_scratch_index_(-1) {}

TfLiteStatus FullyConnected_16::Prepare(TfLiteContext *ctx, const int8_t *W,
                                        const int16_t *BSO, int32_t C_in,
                                        int32_t C_out) {
  Dispatcher *dispatcher = GetDispatcher();

  TF_LITE_REPORT_STATUS(dispatcher->GetReporter(),
                        "FullyConnected_16 Prepare, C_in=%d C_out=%d", C_in,
                        C_out);

  // allocate the jobs
  int32_t n_jobs = execution_plan.changrps.GetSize();
  TF_LITE_ENSURE_STATUS(ctx->AllocatePersistentBuffer(
      ctx, sizeof(nn_fully_connected_job_t) * n_jobs,
      reinterpret_cast<void **>(&jobs_)));

  // allocate the stack for thread workers
  GET_STACKSIZE(stack_size_, fully_connected_thread_worker);
  TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
      ctx, stack_size_ * execution_plan.GetNumThreads(),
      &stack_scratch_index_));

  // allocate scratch buffers for weights and biases (if necessary)
  if (IS_NOT_RAM(W)) {
    TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
        ctx, execution_plan.GetWeightsScratchSize(), &weights_scratch_index_));
  }
  if (IS_NOT_RAM(BSO)) {
    TF_LITE_ENSURE_STATUS(ctx->RequestScratchBufferInArena(
        ctx, execution_plan.GetBiasScratchSize(), &bias_scratch_index_));
  }

  // set job parameters
  nn_fully_connected_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];
    TF_LITE_REPORT_STATUS(
        dispatcher->GetReporter(),
        "FullyConnected_16 Prepare, chan group start=%d size=%d", changrp.start,
        changrp.size);

    job_params[i_cg] = {(uint32_t)changrp.start, (channel_count_t)changrp.size};
  }

  // initialize the kernel
  fully_connected_init(&plan_, jobs_, C_in, C_out, &job_params[0], n_jobs);

  return kTfLiteOk;
}

TfLiteStatus FullyConnected_16::Eval(TfLiteContext *ctx, int16_t *Y,
                                     const int8_t *X, const int8_t *W,
                                     const int16_t *BSO) {
  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack =
      static_cast<char *>(ctx->GetScratchBuffer(ctx, stack_scratch_index_));
  assert(stack);
  dispatcher->InitializeTasks(fully_connected_thread_worker, stack,
                              stack_size_);

  // create thread data and tasks
  int i_th = 0;
  int n_th = execution_plan.GetNumThreads();
  FullyConnectedThreadData thread_data[n_th];

  // load weights & bias scratch buffers (if necessary)
  size_t weights_fetch_size;
  int8_t *tW[n_th];
  int16_t *tBSO[n_th];
  std::memset(tW, 0, n_th * sizeof(int8_t *));
  std::memset(tBSO, 0, n_th * sizeof(int16_t *));

  if (weights_scratch_index_ >= 0) {
    tW[0] = static_cast<int8_t *>(
        ctx->GetScratchBuffer(ctx, weights_scratch_index_));
    assert(tW);
  }
  if (bias_scratch_index_ >= 0) {
    tBSO[0] =
        static_cast<int16_t *>(ctx->GetScratchBuffer(ctx, bias_scratch_index_));
    assert(tBSO);
  }

  weights_fetch_size =
      std::min((size_t)(changrp_len * execution_plan.GetWeightsScratchSize() /
                        (execution_plan.changrps[n_th - 1].start +
                         execution_plan.changrps[n_th - 1].size)),
               execution_plan.GetWeightsScratchSize());

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];

    // fetch the weights and biases
    dispatcher->FetchWeights(&tW[i_th], W, weights_fetch_size, changrp);

    dispatcher->FetchBiases(&tBSO[i_th], BSO,
                            execution_plan.GetBiasScratchSize(), changrp);

    thread_data[i_th].Y = Y;
    thread_data[i_th].X = X;
    thread_data[i_th].W = tW[i_th];
    thread_data[i_th].BSO = (const nn_bso_block_t *)tBSO[i_th];
    thread_data[i_th].plan = &plan_;
    jobs_[i_cg].stride.start.W = 0;
    jobs_[i_cg].stride.start.BSO = 0;
    thread_data[i_th].job = &jobs_[i_cg];
    dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_th]));

    i_th++;

    if (i_th == n_th) {
      dispatcher->JoinTasks();
      i_th = 0;
    }
  }
  dispatcher->JoinTasks();  // finish up any added tasks

  return kTfLiteOk;
}

}  // namespace fully_connected
}  // namespace xcore
