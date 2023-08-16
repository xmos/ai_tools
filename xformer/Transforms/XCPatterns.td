// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

// This is the optimization pattern definition file for XCore.
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"

include "IR/XCoreOps.td"
include "Utils/Utils.td"

// Activation lowering patterns
def getLookupTable
    : NativeCodeCall<"getLookupTable($_builder, $0.getDefiningOp())">;

def getThreadCount
    : NativeCodeCall<"getThreadCount($_builder)">;

foreach activationOp =
    [TFL_ReluOp, TFL_Relu6Op, TFL_TanhOp, TFL_LogisticOp, TFL_HardSwishOp] in {
def:
  Pat<(activationOp
            : $output TensorOf<[QI8]>:$input),
            (XC_LookupOp $input, (Arith_ConstantOp (getLookupTable
            $output)), (getThreadCount))>;
}

def getPadValue : NativeCodeCall<"getPadValue($_builder, $0)">;

def getPaddingPlan
    : NativeCodeCall<
          "getPaddingPlan($_builder, $0.getDefiningOp<TFL::PadOp>())">;

def Has3To4Channel
    : Constraint<CPred<"$0.getType().cast<ShapedType>().getDimSize(3) == 3 && "
                       "$1.getType().cast<ShapedType>().getDimSize(3) == 4">>;

foreach constOp = [Arith_ConstantOp, TFL_ConstOp] in {
def:
  Pat<(TFL_PadOp
           : $output TensorOf<[QI8]>:$input, (constOp
                              : $padding_op $padding_attr)),
          (XC_PadOp $input, (getPaddingPlan $output), (getPadValue $input)), [
            (HasOnlySpatialPadding $padding_attr),
            (HasMultipleOfNBytesPerPixel<4> $input)
          ]>;

def:
  Pat<(TFL_PadOp
           : $output TensorOf<[QI8]>:$input, (constOp
                              : $padding_op $padding_attr)),
          (XC_Pad3To4Op $input, (getPadValue $input)), [
            (HasOnlyChannelPadding $padding_attr),
            (Has3To4Channel $input, $output),
          ]>;
}

// Lower special Concatenation op PyTorch remnant to XC_Pad3to4
// If the second input of concatenation is with a constant of all values zero,
// and input has channels 3 and output has channels 4
class HasExactValues<int n>
    : Constraint<CPred<"$0.size() == " #n>, "has exactly " #n #" values">;
def Front : NativeCodeCall<"$0.front()", 1>;
// Checks if second input is integer constant and its splat value is equal to
// zero.
def IsSecondInputSplatAndEqualToZero
    : Constraint<
          CPred<"$0.back().getDefiningOp() && "
                "dyn_cast<TFL::QConstOp>($0.back().getDefiningOp()) && "
                "dyn_cast<TFL::QConstOp>($0.back().getDefiningOp()).getValue()."
                "cast<DenseElementsAttr>().isSplat() && "
                "dyn_cast<TFL::QConstOp>($0.back().getDefiningOp()).getValue()."
                "cast<DenseElementsAttr>().getSplatValue<int8_t>() == 0">>;
def AreChannels3And4
    : Constraint<
          CPred<"$0.front().getType().cast<ShapedType>().getDimSize(3) == 3 && "
                "$1.getType().cast<ShapedType>().getDimSize(3) == 4">>;
def : Pat<(TFL_ConcatenationOp
           : $output $varg, $axis, $faf),
          (XC_Pad3To4Op(Front $varg), ConstantAttr<I32Attr, "0">), [
            (HasExactValues<2> $varg), (IsSecondInputSplatAndEqualToZero $varg),
            (AreChannels3And4 $varg, $output)
          ]>;

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
