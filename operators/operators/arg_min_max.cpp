// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "operators/arg_min_max.h"

#include "operators/dispatcher.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace xcore {
namespace arg_min_max {

TfLiteStatus ArgMax16::Eval(int32_t* Y, const int16_t* X,
                            const int32_t length) {
  TF_LITE_REPORT_STATUS(GetDispatcher()->GetReporter(), "ArgMax16 Eval id=%p",
                        this);

  argmax_16(Y, X, length);

  return kTfLiteOk;
}

}  // namespace arg_min_max
}  // namespace xcore