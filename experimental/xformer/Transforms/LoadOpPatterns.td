// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

// This is the second pattern definition file for XCore.
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

include "IR/XCoreOps.td"

def isFlashImageFileProvided : Constraint<CPred<"isFlashImageFileProvided()">>;

def : Pat<(XC_Conv2DV2Op
           : $output $input, $thread_count, $scratch_bytes,
             (ConstantOp
              : $weights_op $weights),
             (ConstantOp
              : $biases_op $biases),
             (ConstantOp
              : $multipliers_op $multipliers),
             $abstract_kernel_params, $memcpy_fn_params, $aggregate_fn_params,
             $output_transform_fn_params, $conv2d_kernel_type),
          (XC_Conv2DV2Op $input, $thread_count, $scratch_bytes,
           (XC_LoadOp $weights_op), (XC_LoadOp $biases_op),
           (XC_LoadOp $multipliers_op), $abstract_kernel_params,
           $memcpy_fn_params, $aggregate_fn_params, $output_transform_fn_params,
           $conv2d_kernel_type),
          [(isFlashImageFileProvided)]>;
