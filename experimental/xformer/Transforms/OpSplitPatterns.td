// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

// This is the optimization pattern definition file for XCore.
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"

include "IR/XCoreOps.td"

def getMemcpyFnParam : NativeCodeCall<"getMemcpyFnParam($_builder, $0)">;

def getI32IntegerAttrZero : NativeCodeCall<"getI32IntegerAttrZero($_builder)">;

def getValues : NativeCodeCall<"getValues($0)">;

def HasNoFollowingStridedSlice : Constraint<CPred<"HasNoFollowingStridedSlice($0)">>;
def HasFollowingReshape : Constraint<CPred<"HasFollowingReshape($0)">>;

def createOpSplitBlock : NativeCodeCall<
    "createOpSplitBlock($_builder, $0.getDefiningOp(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10)">;

// // First pass of op split
// // Insert StidedSliceOp
def : Pat<( TFL_Conv2DOp:$conv_out $input, $weights, $muls, $kt, $mp, $aggp, $padding, $stride_h, $stride_w),
            //  (createOpSplitBlock $conv_out,$conv_out, $input, $weights, $muls, $kt, $mp, $aggp, $padding, $stride_h, $stride_w),
             (XC_StridedSliceOp (TFL_Conv2DOp $input, $weights, $muls, $kt, $mp, $aggp, $padding,
              $stride_h, $stride_w), (getI32IntegerAttrZero),
             (getI32IntegerAttrZero), (getMemcpyFnParam $conv_out)), 
            [(HasFollowingReshape $conv_out)]>;

// def : Pat<( TFL_ConcatenationOp:$concat_out $values, $axis, $fused_activation_function ),
//           (XC_StridedSliceOp ( TFL_ConcatenationOp (getValues $values), $axis, $fused_activation_function ), (getI32IntegerAttrZero),
//              (getI32IntegerAttrZero), (getMemcpyFnParam $concat_out)), 
//             [(HasFollowingReshape $concat_out)]>;

// def : Pat<( TFL_ConcatenationOp:$out $values, $axis, $fused_activation_function ),
        //   ( TFL_ConcatenationOp (getValues $values), $axis, $fused_activation_function ),
            // [(HasFollowingReshape $out)]>;
