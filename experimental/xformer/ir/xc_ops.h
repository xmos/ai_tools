// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_IR_XC_OPS_H
#define XFORMER_IR_XC_OPS_H

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

// clang-format off
#include "ir/xc_dialect.h.inc"
// clang-format on

#define GET_OP_CLASSES
#include "ir/xc_ops.h.inc"

#endif // XFORMER_IR_XC_OPS_H
