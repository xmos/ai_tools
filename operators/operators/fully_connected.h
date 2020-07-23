// Copyright (c) 2020, XMOS Ltd, All rights reserved

#ifndef XCORE_FULLY_CONNECTED_OPERATOR_HPP_
#define XCORE_FULLY_CONNECTED_OPERATOR_HPP_

#include <cstdint>

#include "operators/planning.h"
#include "tensorflow/lite/c/common.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace fully_connected {

class FullyConnected_16 {
 public:
  FullyConnected_16(const ExecutionPlan& plan);
  ~FullyConnected_16() {}

  TfLiteStatus Prepare(TfLiteContext* ctx, const int8_t* W, const int16_t* BSO,
                       int32_t C_in, int32_t C_out);
  TfLiteStatus Eval(TfLiteContext* ctx, int16_t* Y, const int8_t* X,
                    const int8_t* W, const int16_t* BSO);

  ExecutionPlan execution_plan;

 private:
  nn_fully_connected_plan_t plan_;
  nn_fully_connected_job_t* jobs_;
  int stack_scratch_index_;
  size_t stack_size_;
  int weights_scratch_index_;
  int bias_scratch_index_;
};

}  // namespace fully_connected
}  // namespace xcore

#endif  // XCORE_FULLY_CONNECTED_OPERATOR_HPP_