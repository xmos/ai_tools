// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

// This is the optimization pattern definition file for XCore.
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

include "IR/XCoreOps.td"

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
           $conv2d_kernel_type)>;

// TODO: DPK
// // Fully connected pattern
// def : Pat<(TFL_FullyConnectedOp
//            : $output TensorOf<[QI8]>:$input, TensorOf<[QI8]>:$filter,
//            TensorOf<[QI32]>:$bias, $fused_activation_function,
//              $weights_format, $keep_num_dims),
//           (XC_FullyConnectedOp $input, $filter, $bias,
//            $fused_activation_function, $weights_format, $keep_num_dims)>;

// def : Pat<(TFL_PadOp
//            : $output $input, (TFL_ConstOp
//                               : $padding_op $padding_attr)),
//           (XC_PadOp $input, $padding_op, (getPadValue $input)), [
//             (HasValidPaddingShape $padding_op),
//             (HasNoPaddingForBatchAndChannel $padding_attr),
//             (HasValidBytesPerPixel $input), (HasNoFollowingConv2D $output)
//           ]>;
