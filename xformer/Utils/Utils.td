// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

// Utility predicates that are shared by multiple passes.

def isBetaFloatEnabled : Constraint<CPred<"isBetaFloatEnabled()">>;

// Check that bytes per pixel is a multiple of n
class HasMultipleOfNBytesPerPixel<int n>
    : Constraint<CPred<
          "$0.getType().cast<ShapedType>().getDimSize(3) % " #n #" == 0">>;

def HasOnlyChannelPadding
    : Constraint<CPred<"utils::hasOnlyChannelPadding($0.cast<DenseIntElementsAttr>())">>;

def HasOnlySpatialPadding
    : Constraint<CPred<"utils::hasOnlySpatialPadding($0.cast<DenseIntElementsAttr>())">>;

// Casts $1 to a dequantized type and then casts that to a quantized type
// using the quantization parameters from the type in $0
// $0 = tensor<m x n x i8 x !quantization params 1>
// $1 = tensor<x x y x i8 x !quantization params 2>
// We create tensor<x x y x i8 x !quantization params 1>
class UpdateShapeWithAxis<int i>
    : NativeCodeCall<
          "quant::CastQuantizedTypeAttrFromExpressedType($_builder, $0, "
          "quant::QuantizedType::castToExpressedType($1.getType()), " #i #")">;

def IsQuantizedType : Constraint<CPred<"dyn_cast<QuantizedType>($0.getType()."
                                       "cast<ShapedType>().getElementType())">>;

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
                     "$0.getType().cast<ShapedType>().isDynamicDim(0)?1:$0."
                     "getType().cast<ShapedType>().getDimSize(0),"
                     "$0.getType().cast<ShapedType>().getDimSize(1)}, "
                     "$0.getType().cast<ShapedType>().getElementType())">;

def getExpandedShapeAttr
    : NativeCodeCall<
          "DenseIntElementsAttr::get(RankedTensorType::get({2}, "
          "rewriter.getI32Type()), "
          "{static_cast<int32_t>($0.getType().cast<ShapedType>().isDynamicDim("
          "0)?1: $0.getType().cast<ShapedType>().getDimSize(0)), "
          "static_cast<int32_t>($0.getType().cast<ShapedType>()."
          "getDimSize(1))})">;

def PadChannelOutputType
    : NativeCodeCall<
          "RankedTensorType::get({$0.getType().cast<ShapedType>().getDimSize(0)"
          ", $0.getType().cast<ShapedType>().getDimSize(1), "
          "$0.getType().cast<ShapedType>().getDimSize(2), "
          "$1.getType().cast<ShapedType>().getDimSize(3)}, "
          "$0.getType().cast<ShapedType>().getElementType())">;

// Get padding values and output type as two return values
def GetBConv2DPaddingValues
    : NativeCodeCall<"getBConv2DPaddingValues($_builder, "
                     "$0.getDefiningOp<lq::Bconv2dOp>())",
                     2>;
