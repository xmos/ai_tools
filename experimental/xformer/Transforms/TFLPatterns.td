// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

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
