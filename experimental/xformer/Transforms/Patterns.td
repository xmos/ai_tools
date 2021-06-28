// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

// This is the optimization pattern definition file for XCore.
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

include "IR/XCoreOps.td"

// Fully connected pattern
def : Pat<(TFL_FullyConnectedOp
           : $output TensorOf<[QI8]>:$input, TensorOf<[QI8]>:$filter,
           TensorOf<[QI32]>:$bias, $fused_activation_function,
             $weights_format, $keep_num_dims),
          (XC_FullyConnectedOp $input, $filter, $bias,
           $fused_activation_function, $weights_format, $keep_num_dims)>;

// Activation lowering patterns
def getLookupTable
    : NativeCodeCall<"getLookupTable($_builder, $0.getDefiningOp())">;

foreach activationOp = [TFL_ReluOp, TFL_Relu6Op, TFL_TanhOp, TFL_LogisticOp] in
  def : Pat<(activationOp
            : $output TensorOf<[QI8]>:$input),
            (XC_Lookup8Op $input, (ConstantOp (getLookupTable $output)))>;
