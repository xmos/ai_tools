// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/type_conversions.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace type_conversions {

struct RequantizeThreadData {
  int8_t* Y;
  const int16_t* X;
  nn_requantize_16_to_8_job_t* job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void requantize_16_to_8_thread_worker(void* context) {
  RequantizeThreadData* td = (RequantizeThreadData*)context;
  requantize_16_to_8(td->Y, td->X, td->job);
}
}

Requantize_16_to_8::Requantize_16_to_8(const ExecutionPlan& execution_plan)
    : execution_plan(execution_plan) {}

XCoreStatus Requantize_16_to_8::Init(int32_t length) {
  TF_LITE_REPORT_STATUS(GetDispatcher()->GetReporter(),
                        "Requantize_16_to_8 Init id=%p length=%ld\n", this,
                        length);

  // allocate the jobs
  int32_t n_jobs = execution_plan.GetNumThreads();
  jobs_ = reinterpret_cast<nn_requantize_16_to_8_job_t*>(
      GetDispatcher()->AllocatePersistantBuffer(
          sizeof(nn_requantize_16_to_8_job_t) * n_jobs));

  // initialize the kernel
  requantize_16_to_8_init(jobs_, length, n_jobs);

  return kXCoreOk;
}

XCoreStatus Requantize_16_to_8::Eval(int8_t* Y, const int16_t* X) {
  TF_LITE_REPORT_STATUS(GetDispatcher()->GetReporter(),
                        "Requantize_16_to_8 Eval id=%p\n", this);

  // initialize the dispatcher
  Dispatcher* dispatcher = GetDispatcher();
  size_t stack_words;
  GET_STACKWORDS(stack_words, requantize_16_to_8_thread_worker);
  dispatcher->InitializeTasks(requantize_16_to_8_thread_worker, stack_words);

  // create thread data and tasks
  RequantizeThreadData thread_data[execution_plan.GetNumThreads()];

  for (int i_job = 0; i_job < execution_plan.GetNumThreads(); i_job++) {
    thread_data[i_job].Y = Y;
    thread_data[i_job].X = X;
    thread_data[i_job].job = &jobs_[i_job];
    dispatcher->AddTask(reinterpret_cast<void*>(&thread_data[i_job]));
  }
  // start and wait for tasks to complete
  dispatcher->JoinTasks();

  return kXCoreOk;
}

}  // namespace type_conversions
}  // namespace xcore
