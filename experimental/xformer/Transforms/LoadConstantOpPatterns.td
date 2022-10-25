// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

include "IR/XCoreOps.td"

def ShouldBeLoadedExternally
    : Constraint<CPred<"shouldBeLoadedExternally($0)">>;

def : Pat<(XC_Conv2DV2Op
           : $output $input,
             (Arith_ConstantOp
              : $weights_op $weights_attr),
             $mulsbiases_or_thresholds, $conv2d_kernel_type, $memcpy_fn_params,
             $aggregate_fn_params, $output_transform_fn_params, $ot_type,
             $scratch_bytes, $thread_count, $abstract_kernel_params),
          (XC_Conv2DV2Op $input, (XC_LoadConstantOp $weights_op),
           $mulsbiases_or_thresholds, $conv2d_kernel_type, $memcpy_fn_params,
           $aggregate_fn_params, $output_transform_fn_params, $ot_type,
           $scratch_bytes, $thread_count, $abstract_kernel_params),
          [(ShouldBeLoadedExternally $weights_attr)]>;

def : Pat<(XC_Conv2DV2Op
           : $output $input, $weights,
             (Arith_ConstantOp
              : $mulsbiases_or_thresholds_op $mulsbiases_or_thresholds_attr),
             $conv2d_kernel_type, $memcpy_fn_params, $aggregate_fn_params,
             $output_transform_fn_params, $ot_type, $scratch_bytes,
             $thread_count, $abstract_kernel_params),
          (XC_Conv2DV2Op $input, $weights,
           (XC_LoadConstantOp $mulsbiases_or_thresholds_op),
           $conv2d_kernel_type, $memcpy_fn_params, $aggregate_fn_params,
           $output_transform_fn_params, $ot_type, $scratch_bytes, $thread_count,
           $abstract_kernel_params),
          [(ShouldBeLoadedExternally $mulsbiases_or_thresholds_attr)]>;
// This constraint checks if the pattern has already been applied to the const
// op
def IsNotUsedByLoadConstantOp
    : Constraint<CPred<"isNotUsedByLoadConstantOp($0)">>;

foreach constOp = [Arith_ConstantOp, TFL_ConstOp] in {
def:
  Pat<(constOp
       : $op $attr),
      (XC_LoadConstantOp(constOp $attr)),
      [(IsNotUsedByLoadConstantOp $op), (ShouldBeLoadedExternally $attr)]>;
}

def : Pat<(TFL_QConstOp
           : $op $qtype, $attr),
          (XC_LoadConstantOp(TFL_QConstOp $qtype, $attr)),
          [(IsNotUsedByLoadConstantOp $op), (ShouldBeLoadedExternally $attr)]>;
