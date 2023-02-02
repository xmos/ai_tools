// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

// This is the optimization pattern definition file for XCore.
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"

include "IR/XCoreOps.td"

// Activation lowering patterns
def getLookupTable
    : NativeCodeCall<"getLookupTable($_builder, $0.getDefiningOp())">;

foreach activationOp =
    [TFL_ReluOp, TFL_Relu6Op, TFL_TanhOp, TFL_LogisticOp] in {
def:
  Pat<(activationOp
            : $output TensorOf<[QI8]>:$input),
            (XC_LookupOp $input, (Arith_ConstantOp (getLookupTable
            $output)))>;
}

// Pad pattern
// We only support spatial padding
def HasNoPaddingForBatchAndChannel
    : Constraint<CPred<
          "$0.cast<DenseIntElementsAttr>().getNumElements() == 8 && "
          "($0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{0, 0}] == 0 "
          "&&"
          "$0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{0, 1}] == 0 &&"
          "$0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{3, 0}] == 0 &&"
          "$0.cast<DenseIntElementsAttr>().getValues<int32_t>()[{3, 1}] == "
          "0)">>;

// Check that bytes per pixel is a multiple of four
def HasValidBytesPerPixel
    : Constraint<
          CPred<"$0.getType().cast<ShapedType>().getDimSize(3) % 4 == 0">>;

def getPadValue : NativeCodeCall<"getPadValue($_builder, $0)">;

def getPaddingPlan
    : NativeCodeCall<
          "getPaddingPlan($_builder, $0.getDefiningOp<TFL::PadOp>())">;

foreach constOp = [Arith_ConstantOp, TFL_ConstOp] in {
def:
  Pat<(TFL_PadOp
           : $output TensorOf<[QI8]>:$input, (constOp
                              : $padding_op $padding_attr)),
          (XC_PadOp $input, (getPaddingPlan $output), (getPadValue $input)), [
            (HasNoPaddingForBatchAndChannel $padding_attr),
            (HasValidBytesPerPixel $input)
          ]>;
}

// Fuse XC_Conv2D(Reshape()) -> XC_Conv2D()
def : Pat<(XC_Conv2DV2Op
           : $cout(TFL_ReshapeOp
                   : $rout $input, $shape),
             $weights, $muls, $kt, $mp, $aggp, $otp, $ott, $scratch, $tc, $akp),
          (XC_Conv2DV2Op $input, $weights, $muls, $kt, $mp, $aggp, $otp, $ott,
           $scratch, $tc, $akp)>;

// Fuse Reshape(XC_Conv2D()) -> XC_Conv2D()
def : Pat<(TFL_ReshapeOp
           : $rout(XC_Conv2DV2Op $input, $weights, $muls, $kt, $mp, $aggp, $otp,
                   $ott, $scratch, $tc, $akp),
             $shape),
          (XC_Conv2DV2Op $input, $weights, $muls, $kt, $mp, $aggp, $otp, $ott,
           $scratch, $tc, $akp)>;

// Replace LQ_QuantizeOp with XC_bsign_8
def : Pat<(LQ_QuantizeOp $input), (XC_Bsign8Op $input)>;
