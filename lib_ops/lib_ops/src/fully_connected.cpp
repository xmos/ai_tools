// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/fully_connected.h"

#include <cstring>

#include "lib_ops/api/benchmarking.h"
#include "lib_ops/api/tracing.h"

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
      W_(nullptr),
      BSO_(nullptr) {}

XCoreStatus FullyConnected_16::Prepare(int32_t C_in, int32_t C_out,
                                       const int8_t *W, const int16_t *BSO) {
  TRACE_INFO("FullyConnected_16 Prepare id=%p C_in=%ld C_out=%ld\n", this, C_in,
             C_out);

  W_ = W;
  BSO_ = BSO;

  // allocate the jobs
  int32_t n_jobs = execution_plan.changrps.GetSize();
  jobs_ = reinterpret_cast<nn_fully_connected_job_t *>(
      GetDispatcher()->AllocatePersistantBuffer(
          sizeof(nn_fully_connected_job_t) * n_jobs));

  // set job parameters
  nn_fully_connected_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];
    TRACE_INFO(
        "FullyConnected_16 Prepare id=%p, chan group start=%ld size=%ld\n",
        this, changrp.start, changrp.size);

    job_params[i_cg] = {(uint32_t)changrp.start, (channel_count_t)changrp.size};
  }

  // initialize the kernel
  fully_connected_init(&plan_, jobs_, C_in, C_out, &job_params[0], n_jobs);

  return kXCoreOk;
}

XCoreStatus FullyConnected_16::Eval(int16_t *Y, const int8_t *X) {
  TRACE_INFO("FullyConnected Eval id=%p\n", this);
  TIMER_START();

  // initialize the dispatcher
  Dispatcher *dispatcher = GetDispatcher();
  size_t stack_words;
  GET_STACKWORDS(stack_words, fully_connected_thread_worker);
  dispatcher->InitializeTasks(fully_connected_thread_worker, stack_words);

  // create thread data and tasks
  int i_th = 0;
  int n_th = execution_plan.GetNumThreads();
  FullyConnectedThreadData thread_data[n_th];
  int8_t *tW[n_th];
  int16_t *tBSO[n_th];

  std::memset(tW, 0, n_th * sizeof(int8_t *));
  std::memset(tBSO, 0, n_th * sizeof(int16_t *));

  for (int i_cg = 0; i_cg < execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = execution_plan.changrps[i_cg];

    // fetch the weights and biases
    dispatcher->FetchWeights(
        &tW[i_th], W_, execution_plan.GetWeightsScratchSize() * n_th, changrp);
    dispatcher->FetchBiases(&tBSO[i_th], BSO_,
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

  TIMER_STOP("FullyConnected id=%p", this);
  return kXCoreOk;
}

}  // namespace fully_connected
}  // namespace xcore
