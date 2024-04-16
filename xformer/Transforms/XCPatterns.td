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
// Activation lookup i8
def getLookupTableI8
    : NativeCodeCall<"getLookupTableI8($_builder, $0.getDefiningOp())">;

foreach activationOp =
    [TFL_ReluOp, TFL_Relu6Op, TFL_TanhOp, TFL_LogisticOp, TFL_HardSwishOp] in {
def:
  Pat<(activationOp
            : $output TensorOf<[QI8]>:$input),
            (XC_LookupOp $input, (Arith_ConstantOp (getLookupTableI8
            $output)))>;
}

// Activation lookup i16
def getLookupTableI16OrFail
    : NativeCodeCall<"getLookupTableI16($_builder, $0.getDefiningOp()); "
                     "if(blob == nullptr){return failure();}">;

foreach activationOp = [TFL_LogisticOp, TFL_TanhOp, TFL_ReluOp] in {
def:
  Pat<(activationOp
            : $output TensorOf<[QI16]>:$input),
            (XC_LookupOp $input, (Arith_ConstantOp (getLookupTableI16OrFail:$blob
            $output)))>;
}

// Softmax lookup
def getExpLookupF32
    : NativeCodeCall<"getExpLookupF32($_builder, $0.getDefiningOp())">;

// Softmax constraint: batch size must be 1, number of dimensions must be 2
// (batch, dim)
def isSingleSegment
    : Constraint<CPred<"$0.getType().cast<ShapedType>().getRank() == 2">>;

def betaIsOne : Constraint<CPred<"$0.getValue().convertToFloat() == 1.0">>;

// Softmax -> XC_SoftmaxOp
def:
   Pat<(TFL_SoftmaxOp
            : $output TensorOf<[QI8]>:$input, $beta),
           (XC_SoftmaxOp $input, (Arith_ConstantOp (getExpLookupF32
           $output))), [(betaIsOne $beta), (isSingleSegment $input)]>;

// Beta float activation lookup
def getActivationType
    : NativeCodeCall<"getActivationType($_builder, $0.getDefiningOp())">;

foreach activationOp = [TFL_EluOp, TFL_LogisticOp, TFL_TanhOp] in {
def:
  Pat<(activationOp
            : $output TensorOf<[F32]>:$input),
            (XC_Beta_ActivationF32Op $input, (getActivationType $output)), [(isBetaFloatEnabled)]>;
}

// ELU(Dequantize(int16)) -> Dequantize(XC_LookupOp())
def getLookupTableI16WithInputOutputOrFail
    : NativeCodeCall<"getLookupTableI16($_builder, $0.getDefiningOp(), "
                     "$1.getDefiningOp(), $2.getDefiningOp()); if(blob == "
                     "nullptr){return failure();}">;
def:
    Pat<(TFL_EluOp : $output2 (TFL_DequantizeOp : $output1
    TensorOf<[QI16]>:$input)),
            (TFL_DequantizeOp (XC_LookupOp $input, (Arith_ConstantOp
            (getLookupTableI16WithInputOutputOrFail:$blob $output2, $output1, $input)),
            (returnType $input))), []>;

def : Pat<(TFL_ConcatenationOp $input, $axis, $faf),
          (XC_Beta_ConcatF32Op $input), [(isBetaFloatEnabled)]>;

// Unary i16
// If we cannot obtain the blob, we want the pattern to fail.
// The return value of the function is stored in a variable named "blob".
// We use that to check if it is a nullptr.
def getUnaryI16BlobOrFail
    : NativeCodeCall<"getUnaryI16Blob($_builder, $0.getDefiningOp()); if(blob "
                     "== nullptr){return failure();}">;
// TFL Quantize(f32 to i16), Quantize(i16 to i16), Dequantize
def: Pat<(TFL_QuantizeOp
            : $output TensorOf<[F32]>:$input, $qtype),
            (XC_UnaryI16Op $input, (Arith_ConstantOp (getUnaryI16BlobOrFail:$blob
            $output)), XC_UnaryI16_Quantize)>;
def: Pat<(TFL_QuantizeOp
            : $output TensorOf<[QI16]>:$input, $qtype),
            (XC_UnaryI16Op $input, (Arith_ConstantOp (getUnaryI16BlobOrFail:$blob
            $output)), XC_UnaryI16_Requantize)>;
def: Pat<(TFL_DequantizeOp
            : $output TensorOf<[QI16]>:$input),
            (XC_UnaryI16Op $input, (Arith_ConstantOp (getUnaryI16BlobOrFail:$blob
            $output)), XC_UnaryI16_Dequantize)>;

// Binary i16
// If we cannot obtain the blob, we want the pattern to fail.
// The return value of the function is stored in a variable named "blob".
// We use that to check if it is a nullptr.
def getBinaryI16BlobOrFail
    : NativeCodeCall<"getBinaryI16Blob($_builder, $0.getDefiningOp()); if(blob "
                     "== nullptr){return failure();}">;
// TFL Add, Sub, Mul
def: Pat<(TFL_AddOp
            : $output TensorOf<[QI16]>:$input1, TensorOf<[QI16]>:$input2,
            TFL_AF_None), (XC_BinaryI16Op $input1, $input2, (Arith_ConstantOp
            (getBinaryI16BlobOrFail:$blob $output)), XC_BinaryI16_Add)>;

def: Pat<(TFL_SubOp
            : $output TensorOf<[QI16]>:$input1, TensorOf<[QI16]>:$input2,
            TFL_AF_None), (XC_BinaryI16Op $input1, $input2, (Arith_ConstantOp
            (getBinaryI16BlobOrFail:$blob $output)), XC_BinaryI16_Add)>;

def: Pat<(TFL_MulOp
            : $output TensorOf<[QI16]>:$input1, TensorOf<[QI16]>:$input2,
            TFL_AF_None), (XC_BinaryI16Op $input1, $input2, (Arith_ConstantOp
            (getBinaryI16BlobOrFail:$blob $output)), XC_BinaryI16_Mul)>;

// Pad patterns
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
             $weights, $muls, $po, $kt, $mp, $aggp, $otp, $ott, $scratch, $tc,
             $akp),
          (XC_Conv2DV2Op $input, $weights, $muls, $po, $kt, $mp, $aggp, $otp,
           $ott, $scratch, $tc, $akp)>;

// Fuse Reshape(XC_Conv2D()) -> XC_Conv2D()
def : Pat<(TFL_ReshapeOp
           : $rout(XC_Conv2DV2Op $input, $weights, $muls, $po, $kt, $mp, $aggp,
                   $otp, $ott, $scratch, $tc, $akp),
             $shape),
          (XC_Conv2DV2Op $input, $weights, $muls, $po, $kt, $mp, $aggp, $otp,
           $ott, $scratch, $tc, $akp)>;

// Replace LQ_QuantizeOp with XC_bsign_8
def : Pat<(LQ_QuantizeOp $input), (XC_Bsign8Op $input)>;
