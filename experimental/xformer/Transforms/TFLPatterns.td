// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "larq_compute_engine/mlir/ir/lce_ops.td"

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
             ConstantAttr<I32Attr, "1">, ConstantAttr<StrAttr, "SAME">, $sh,
             $sw),
          (LQ_Bconv2dOp(TFL_PadOp $input,
                        (GetBConv2DPaddingValues
                         : $ret__0 $output),
                        (returnType $ret__1)),
           $f, $m, $b, $t, $cin, $dh, $dw, $faf, ConstantAttr<I32Attr, "0">,
           ConstantAttr<StrAttr, "VALID">, $sh, $sw)>;
