// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_CONV2D_OPERATOR_H_
#define XCORE_CONV2D_OPERATOR_H_

#include <cstdint>
#include <vector>

#include "lib_ops/api/lib_ops.h"
#include "lib_ops/api/planning.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace conv {

struct Conv2DPadding {
  int8_t top;
  int8_t left;
  int8_t zero_point;
  int8_t unused;
};

struct Conv2DParams {
  int32_t K_h;
  int32_t K_w;
  int32_t stride_h;
  int32_t stride_w;
  Conv2DPadding pad;
};

class Conv2D_Deep {
 public:
  Conv2D_Deep(const Conv2DParams& params, const ExecutionPlan& plan);
  ~Conv2D_Deep() {}

  XCoreStatus Prepare(int32_t X_h, int32_t X_w, int32_t C_in, int32_t Y_h,
                      int32_t Y_w, int32_t C_out, const int8_t* K,
                      const int16_t* BSO);
  XCoreStatus Eval(int8_t* Y, const int8_t* X);

  Conv2DParams params;
  ExecutionPlan execution_plan;

 private:
  nn_conv2d_deep_plan_t plan_;
  nn_conv2d_deep_job_t* jobs_;
  const int8_t* K_;     // original kernel tensor
  const int16_t* BSO_;  // original bias tensor
};

class Conv2D_Shallow {
 public:
  Conv2D_Shallow(const Conv2DParams& params, const ExecutionPlan& plan);
  ~Conv2D_Shallow() {}

  XCoreStatus Prepare(int32_t X_h, int32_t X_w, int32_t C_in, int32_t Y_h,
                      int32_t Y_w, int32_t C_out, int32_t K_w_padded,
                      const int8_t* K, const int16_t* BSO);
  XCoreStatus Eval(int8_t* Y, const int8_t* X);

  Conv2DParams params;
  ExecutionPlan execution_plan;

 private:
  nn_conv2d_shallowin_plan_t plan_;
  nn_conv2d_shallowin_job_t* jobs_;
  const int8_t* K_;     // original kernel tensor
  const int16_t* BSO_;  // original bias tensor
};

class Conv2D_1x1 {
 public:
  Conv2D_1x1(const Conv2DParams& params, const ExecutionPlan& plan);
  ~Conv2D_1x1() {}

  XCoreStatus Prepare(int32_t X_h, int32_t X_w, int32_t C_in, int32_t Y_h,
                      int32_t Y_w, int32_t C_out, const int8_t* K,
                      const int16_t* BSO);
  XCoreStatus Eval(int8_t* Y, const int8_t* X);

  Conv2DParams params;
  ExecutionPlan execution_plan;

 private:
  nn_conv2d_1x1_plan_t plan_;
  nn_conv2d_1x1_job_t* jobs_;
  const int8_t* K_;     // original kernel tensor
  const int16_t* BSO_;  // original bias tensor
};

class Conv2D_Depthwise {
 public:
  Conv2D_Depthwise(const Conv2DParams& params, const ExecutionPlan& plan);
  ~Conv2D_Depthwise() {}

  XCoreStatus Prepare(int32_t X_h, int32_t X_w, int32_t C_in, int32_t Y_h,
                      int32_t Y_w, int32_t C_out, const int8_t* K,
                      const int16_t* BSO);
  XCoreStatus Eval(int8_t* Y, const int8_t* X);

  Conv2DParams params;
  ExecutionPlan execution_plan;

 private:
  nn_conv2d_depthwise_plan_t plan_;
  nn_conv2d_depthwise_job_t* jobs_;
  const int8_t* K_;     // original kernel tensor
  const int16_t* BSO_;  // original bias tensor
};

}  // namespace conv
}  // namespace xcore

#endif  // XCORE_CONV2D_OPERATOR_H_