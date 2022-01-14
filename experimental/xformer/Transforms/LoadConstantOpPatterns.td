// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

// This is the second pattern definition file for XCore.
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

include "IR/XCoreOps.td"

def ShouldBeLoadedExternally
    : Constraint<CPred<"shouldBeLoadedExternally($0)">>;

def : Pat<(XC_Conv2DV2Op
           : $output $input, $thread_count, $scratch_bytes,
             (ConstantOp
              : $weights_op $weights_attr),
             $mulsbiases_or_thresholds, $abstract_kernel_params,
             $memcpy_fn_params, $aggregate_fn_params,
             $output_transform_fn_params, $conv2d_kernel_type),
          (XC_Conv2DV2Op $input, $thread_count, $scratch_bytes,
           (XC_LoadConstantOp $weights_op), $mulsbiases_or_thresholds,
           $abstract_kernel_params, $memcpy_fn_params, $aggregate_fn_params,
           $output_transform_fn_params, $conv2d_kernel_type),
          [(ShouldBeLoadedExternally $weights_attr)]>;

def : Pat<(XC_Conv2DV2Op
           : $output $input, $thread_count, $scratch_bytes, $weights,
             (ConstantOp
              : $mulsbiases_or_thresholds_op $mulsbiases_or_thresholds_attr),
             $abstract_kernel_params, $memcpy_fn_params, $aggregate_fn_params,
             $output_transform_fn_params, $conv2d_kernel_type),
          (XC_Conv2DV2Op $input, $thread_count, $scratch_bytes, $weights,
           (XC_LoadConstantOp $mulsbiases_or_thresholds_op),
           $abstract_kernel_params, $memcpy_fn_params, $aggregate_fn_params,
           $output_transform_fn_params, $conv2d_kernel_type),
          [(ShouldBeLoadedExternally $mulsbiases_or_thresholds_attr)]>;

foreach constOp = [ConstantOp, TFL_ConstOp] in {
def:
  Pat<(TFL_Conv2DOp $input,
       (constOp
        : $op $attr),
       $b, $dh, $dw, $faf, $p, $sh, $sw),
      (TFL_Conv2DOp $input, (XC_LoadConstantOp $op), $b, $dh, $dw, $faf, $p,
       $sh, $sw),
      [(ShouldBeLoadedExternally $attr)]>;

def:
  Pat<(TFL_Conv2DOp $input, $f,
       (constOp
        : $op $attr),
       $dh, $dw, $faf, $p, $sh, $sw),
      (TFL_Conv2DOp $input, $f, (XC_LoadConstantOp $op), $dh, $dw, $faf, $p,
       $sh, $sw),
      [(ShouldBeLoadedExternally $attr)]>;
}

def : Pat<(TFL_Conv2DOp $input,
           (TFL_QConstOp
            : $op $qtype, $attr),
           $b, $dh, $dw, $faf, $p, $sh, $sw),
          (TFL_Conv2DOp $input, (XC_LoadConstantOp $op), $b, $dh, $dw, $faf, $p,
           $sh, $sw),
          [(ShouldBeLoadedExternally $attr)]>;

def : Pat<(TFL_Conv2DOp $input, $f,
           (TFL_QConstOp
            : $op $qtype, $attr),
           $dh, $dw, $faf, $p, $sh, $sw),
          (TFL_Conv2DOp $input, $f, (XC_LoadConstantOp $op), $dh, $dw, $faf, $p,
           $sh, $sw),
          [(ShouldBeLoadedExternally $attr)]>;