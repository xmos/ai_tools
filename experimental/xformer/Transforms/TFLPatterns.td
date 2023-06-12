// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"

include "Utils/Utils.td"

// Casts $1 to a dequantized type and then casts that to a quantized type
// using the quantization parameters from the type in $0
// $0 = tensor<m x n x i8 x !quantization params 1>
// $1 = tensor<x x y x i8 x !quantization params 2>
// We create tensor<x x y x i8 x !quantization params 1>
class UpdateShapeWithAxis<int i>
    : NativeCodeCall<
          "quant::CastQuantizedTypeAttrFromExpressedType($_builder, $0, "
          "quant::QuantizedType::castToExpressedType($1.getType()), " #i #")">;

// Convert Quantize(Reshape()) -> Reshape(Quantize())
// This is to merge Quantize with Conv2D if possible
def : Pat<(TFL_QuantizeOp(TFL_ReshapeOp $input, $shape), $qtype),
          (TFL_ReshapeOp(TFL_QuantizeOp $input,
                         (UpdateShapeWithAxis<-1> $qtype, $input)),
           $shape)>;

// Fuse Quantize(Conv2D()) -> Conv2D()
def : Pat<(TFL_QuantizeOp(TFL_Conv2DOp $input, $f, $b, $dh, $dw, $faf, $p, $sh,
                          $sw),
           $qtype),
          (TFL_Conv2DOp $input, $f, $b, $dh, $dw, $faf, $p, $sh, $sw)>;

def : Pat<(TFL_QuantizeOp(TFL_DepthwiseConv2DOp $input, $f, $b, $dh, $dw, $faf,
                          $p, $sh, $sw, $dm),
           $qtype),
          (TFL_DepthwiseConv2DOp $input, $f, $b, $dh, $dw, $faf, $p, $sh, $sw,
           $dm)>;

// Get padding values and output type as two return values
def GetConv2DPaddingValues
    : NativeCodeCall<"getConv2DPaddingValues<TFL::Conv2DOp>($_builder, "
                     "$0.getDefiningOp<TFL::Conv2DOp>())",
                     2>;

// Get padding values and output type as two return values
def GetDepthwiseConv2DPaddingValues
    : NativeCodeCall<
          "getConv2DPaddingValues<TFL::DepthwiseConv2DOp>($_builder, "
          "$0.getDefiningOp<TFL::DepthwiseConv2DOp>())",
          2>;

// TFL_Conv2D() with SAME padding -> TFL_Conv2D(Pad())
def : Pat<(TFL_Conv2DOp: $output TensorOf<[QI8]>:$input, TensorOf<[QI8]>:$f, AnyTypeOf<[TensorOf<[I32,QI32]>, NoneType]>:$b, $dh, $dw, $faf, TFL_PAD_Same, $sh, $sw),
          (TFL_Conv2DOp (TFL_PadOp $input,
                        (GetConv2DPaddingValues
                         : $ret__0 $output),
                        (returnType $ret__1)), $f, $b, $dh, $dw, $faf, TFL_PAD_Valid, $sh, $sw)>;

// TFL_DepthwiseConv2D() with SAME padding-> TFL_DepthwiseConv2D(Pad())
def : Pat<(TFL_DepthwiseConv2DOp: $output TensorOf<[QI8]>:$input, TensorOf<[QI8]>:$f, TensorOf<[I32,QI32]>:$b, $dh, $dw, $faf, TFL_PAD_Same, $sh, $sw, $dm),
          (TFL_DepthwiseConv2DOp (TFL_PadOp $input,
                        (GetDepthwiseConv2DPaddingValues
                         : $ret__0 $output),
                        (returnType $ret__1)), $f, $b, $dh, $dw, $faf, TFL_PAD_Valid, $sh, $sw, $dm)>;

// PyTorch remnant opt
// Relu(Minimum(x, 127)) -> Relu(x)
// We can ignore minimum of (x,127) as it doesn't do anything
class IsSplatAndEqualTo<int n>
    : Constraint<
          CPred<"dyn_cast<TFL::QConstOp>($0.getDefiningOp()) && "
                "dyn_cast<TFL::QConstOp>($0.getDefiningOp()).getValue()."
                "cast<DenseElementsAttr>().isSplat() && "
                "dyn_cast<TFL::QConstOp>($0.getDefiningOp()).getValue()."
                "cast<DenseElementsAttr>().getSplatValue<int8_t>() == " #n>>;
def : Pat<(TFL_ReluOp(TFL_MinimumOp $lhs, $rhs)), (TFL_ReluOp $lhs),
          [(IsSplatAndEqualTo<127> $rhs)]>;

// Merge Relu with Conv
def : Pat<(TFL_ReluOp(TFL_Conv2DOp $input, $f, $b, $dh, $dw, TFL_AF_None, $p,
                      $sh, $sw)),
          (TFL_Conv2DOp $input, $f, $b, $dh, $dw, TFL_AF_Relu, $p, $sh, $sw)>;

// Get the dimension size as integer attr.
class GetDimAsI32<int n>
    : NativeCodeCall<
          "$_builder.getIntegerAttr($_builder.getIntegerType(32), "
          "$0.getType().cast<RankedTensorType>().getDimSize(" #n #"))">;

def getTypeOf1WithQParamsOf0
    : NativeCodeCall<
          "dyn_cast<mlir::quant::UniformQuantizedType>($0.getType().cast<"
          "ShapedType>().getElementType()).castFromExpressedType(mlir::quant::"
          "UniformQuantizedType::castToExpressedType($1.getType()))">;

def getTypeAttrOf1WithQParamsOf0
    : NativeCodeCall<
          "mlir::TypeAttr::get(dyn_cast<mlir::quant::UniformQuantizedType>($0."
          "getType().cast<ShapedType>().getElementType())."
          "castFromExpressedType(mlir::quant::UniformQuantizedType::"
          "castToExpressedType($1.getType())))">;

def HasSpatialAxisForMean
    : Constraint<CPred<
          "$0.cast<DenseIntElementsAttr>().getNumElements() == 2 && "
          "($0.cast<DenseIntElementsAttr>().getValues<int32_t>()[0] == 1 "
          "&&"
          "$0.cast<DenseIntElementsAttr>().getValues<int32_t>()[1] == 2)">>;

class HasRank<int n>
    : Constraint<CPred<"$0.getType().cast<ShapedType>().getRank() == " #n>>;

def getExpandedShape
    : NativeCodeCall<"RankedTensorType::get({1, 1, "
                     "$0.getType().cast<ShapedType>().getDimSize(0), "
                     "$0.getType().cast<ShapedType>().getDimSize(1)}, "
                     "$0.getType().cast<ShapedType>().getElementType())">;

def getExpandedShapeAttr
    : NativeCodeCall<"DenseIntElementsAttr::get(RankedTensorType::get({2}, "
                     "rewriter.getI32Type()), "
                     "{$0.getType().cast<ShapedType>().getDimSize(0), "
                     "static_cast<int32_t>($0.getType().cast<ShapedType>()."
                     "getDimSize(1))})">;

// If MeanOp with spatial axis and rank 2 output, expand output to rank 4, which
// we later lower to AveragePool2D
def : Pat<(TFL_MeanOp
           : $output $input,
             (TFL_ConstOp
              : $axis_op $axis),
             $kd),
          (TFL_ReshapeOp(TFL_MeanOp $input, $axis_op, $kd,
                         (returnType(getExpandedShape $output))),
           (TFL_ConstOp(getExpandedShapeAttr $output))),
          [(HasSpatialAxisForMean $axis), (HasRank<2> $output)]>;

// Lower MeanOp with spatial axis to AveragePool2D
def : Pat<(TFL_MeanOp
           : $output $input, (TFL_ConstOp $axis), $kd),
          (TFL_QuantizeOp(
               TFL_AveragePool2DOp $input, (GetDimAsI32<1> $input),
               (GetDimAsI32<2> $input), TFL_PAD_Valid,
               ConstantAttr<I32Attr, "1">, ConstantAttr<I32Attr, "1">,
               TFL_AF_None,
               (returnType(getTypeOf1WithQParamsOf0 $input, $output))),
           (getTypeAttrOf1WithQParamsOf0 $output, $output)),
          [(HasSpatialAxisForMean $axis), (HasRank<4> $output)]>;

// def CreateBias
//     : NativeCodeCall<
//           "DenseIntElementsAttr::get(RankedTensorType::get({$0.getType().cast<"
//           "ShapedType>().getDimSize(3)}, rewriter.getI32Type()), "
//           "std::vector<int32_t>($0.getType().cast<ShapedType>().getDimSize(3),
//           " "0))">;

// def HasRank4
//     : Constraint<
//           CPred<"$0.getType().cast<ShapedType>().getRank() == 4">>;

// def : Pat<(TFL_MulOp
//            : $output $in1, (TFL_ReshapeOp $in2, $shape), $faf),
//           (TFL_DepthwiseConv2DOp $in1, $in2, (TFL_ConstOp (CreateBias
//           $output)),
//            ConstantAttr<I32Attr, "1">, ConstantAttr<I32Attr, "1">,
//            TFL_AF_None, TFL_PAD_Valid, ConstantAttr<I32Attr, "1">,
//            ConstantAttr<I32Attr, "1">, ConstantAttr<I32Attr, "1">),[(HasRank4
//            $output)]>;

def Pad3to4OutputType
    : NativeCodeCall<
          "RankedTensorType::get({$0.getType().cast<ShapedType>().getDimSize(0)"
          ", $0.getType().cast<ShapedType>().getDimSize(1), "
          "$0.getType().cast<ShapedType>().getDimSize(2), 4}, "
          "$0.getType().cast<ShapedType>().getElementType())">;

// Pad3to4(Pad3to3) to Pad4to4(Pad3to4)
foreach constOp = [Arith_ConstantOp, TFL_ConstOp] in {
def:
  Pat<(TFL_PadOp(TFL_PadOp $input, $padding3to3),
       (constOp
        : $padding3to4_op $padding3to4_attr)),
      (TFL_PadOp(TFL_PadOp $input, $padding3to4_op,
                 (returnType(Pad3to4OutputType $input))),
       $padding3to3),
      [
        (HasMultipleOfNBytesPerPixel<3> $input),
        (HasOnlyPadding3To4ForChannel $padding3to4_attr),
      ]>;
}

// Get padding values and output type as two return values
def GetBConv2DPaddingValues
    : NativeCodeCall<"getBConv2DPaddingValues($_builder, "
                     "$0.getDefiningOp<lq::Bconv2dOp>())",
                     2>;

// Replace LQ_Bconv2DOp of SAME padding with a pad_value of one with
// LQ_Bconv2DOp(TFL_Pad()) of VALID padding. We cannot do this when the
// pad_value is zero as detailed below.
// Comment copied from
// https://github.com/larq/compute-engine/blob/main/larq_compute_engine/core/bconv2d/zero_padding_correction.h#L6
// "When we compute a convolution that requires "SAME" padding we pad with the
// value zero, meaning bitpacked bits 0, representing the value +1; thus we
// compute 'same one' padding by default. A correction is needed if we want
// 'same zero' padding instead -- we have to add or subtract a value to elements
// at the edge of the output tensor."
def : Pat<(LQ_Bconv2dOp
           : $output $input, $f, $m, $b, $t, $cin, $dh, $dw, $faf,
             ConstantAttr<I32Attr, "1">, TFL_PAD_Same, $sh, $sw),
          (LQ_Bconv2dOp(TFL_PadOp $input,
                        (GetBConv2DPaddingValues
                         : $ret__0 $output),
                        (returnType $ret__1)),
           $f, $m, $b, $t, $cin, $dh, $dw, $faf, ConstantAttr<I32Attr, "0">,
           TFL_PAD_Valid, $sh, $sw)>;
