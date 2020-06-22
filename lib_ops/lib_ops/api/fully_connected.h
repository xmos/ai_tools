// Copyright (c) 2020, XMOS Ltd, All rights reserved

#ifndef XCORE_FULLY_CONNECTED_OPERATOR_HPP_
#define XCORE_FULLY_CONNECTED_OPERATOR_HPP_

#include <cstdint>

#include "lib_ops/api/lib_ops.h"
#include "lib_ops/api/planning.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace fully_connected {

class FullyConnected_16 {
 public:
  FullyConnected_16(const ExecutionPlan& plan);
  ~FullyConnected_16() {}

  XCoreStatus Init(int32_t C_in, int32_t C_out);
  XCoreStatus Eval(int16_t* Y, const int8_t* W, const int8_t* X,
                   const int16_t* BSO);

  ExecutionPlan execution_plan;

 private:
  size_t weights_preload_size_;
  nn_fully_connected_plan_t plan_;
  nn_fully_connected_job_t* jobs_;
};

}  // namespace fully_connected
}  // namespace xcore

#endif  // XCORE_FULLY_CONNECTED_OPERATOR_HPP_