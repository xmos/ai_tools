// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_POOLING_OPERATORS_H_
#define XCORE_POOLING_OPERATORS_H_

#include <cstdint>

#include "lib_ops/api/lib_ops.h"
#include "lib_ops/api/planning.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace pooling {

struct PoolingParams {
  int32_t pool_h;
  int32_t pool_w;
  int32_t stride_h;
  int32_t stride_w;
};

class MaxPool {
 public:
  MaxPool(const PoolingParams& params, const ExecutionPlan& execution_plan);
  ~MaxPool() {}

  XCoreStatus Init(int32_t X_h, int32_t X_w, int32_t C_in, int32_t Y_h,
                   int32_t Y_w, int32_t C_out);
  XCoreStatus Eval(int8_t* Y, const int8_t* X);

  PoolingParams params;
  ExecutionPlan execution_plan;

 private:
  nn_maxpool2d_plan_t plan_;
  nn_pool2d_job_t* jobs_;
};

class AvgPool {
 public:
  AvgPool(const PoolingParams& params, const ExecutionPlan& execution_plan);
  ~AvgPool() {}

  XCoreStatus Init(int32_t X_h, int32_t X_w, int32_t C_in, int32_t Y_h,
                   int32_t Y_w, int32_t C_out);
  XCoreStatus Eval(int8_t* Y, const int8_t* X);

  PoolingParams params;
  ExecutionPlan execution_plan;

 private:
  nn_avgpool2d_plan_t plan_;
  nn_pool2d_job_t* jobs_;
};

class AvgPool_Global {
 public:
  AvgPool_Global(const ExecutionPlan& execution_plan);
  ~AvgPool_Global() {}

  XCoreStatus Init(int32_t X_h, int32_t X_w, int32_t C_in, int32_t bias,
                   int32_t shift, int32_t scale);
  XCoreStatus Eval(int8_t* Y, const int8_t* X, int32_t X_h, int32_t X_w,
                   uint32_t C_in);

  ExecutionPlan execution_plan;

 private:
  int32_t bias_;
  nn_avgpool2d_global_plan_t plan_;
  nn_avgpool2d_global_job_t* jobs_;
};

}  // namespace pooling
}  // namespace xcore

#endif  // XCORE_POOLING_OPERATORS_H_