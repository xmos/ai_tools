// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_TYPE_CONVERSION_OPERATORS_H_
#define XCORE_TYPE_CONVERSION_OPERATORS_H_

#include <cstdint>

#include "lib_ops/api/lib_ops.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace type_conversions {

class Requantize_16_to_8 {
 public:
  Requantize_16_to_8(const ExecutionPlan& plan);
  ~Requantize_16_to_8() {}

  XCoreStatus Init(int32_t length);
  XCoreStatus Eval(int8_t* Y, const int16_t* X);

  ExecutionPlan execution_plan;

 private:
  nn_requantize_16_to_8_job_t* jobs_;
};

}  // namespace type_conversions
}  // namespace xcore

#endif  // XCORE_TYPE_CONVERSION_OPERATORS_H_