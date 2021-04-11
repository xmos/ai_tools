#ifndef XFORMER_IR_XC_OPS_H
#define XFORMER_IR_XC_OPS_H

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

// clang-format off
#include "experimental/mlir/ir/xc_dialect.h.inc"
// clang-format on

#define GET_OP_CLASSES
#include "experimental/mlir/ir/xc_ops.h.inc"

#endif // XFORMER_IR_XC_OPS_H
