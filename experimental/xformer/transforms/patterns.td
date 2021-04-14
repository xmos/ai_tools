// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

// This is the optimization pattern definition file for XCore.
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

include "ir/xc_ops.td"

def : Pat<(TFL_FullyConnectedOp
           : $output $input, $filter, $bias, $fused_activation_function,
             $weights_format, $keep_num_dims),
          (XC_FullyConnectedOp $input, $filter, $bias,
           $fused_activation_function, $weights_format, $keep_num_dims)>;
