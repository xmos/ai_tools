// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

include "IR/XCoreOps.td"
include "Utils/Utils.td"

def CreateNoneValue : NativeCodeCall<"$_builder.create<TFL::NoValueOp>($0."
                                     "getLoc(), $_builder.getUnitAttr())">;

// Check that input channels equals filter channels
// We don't optimize grouped convolutions yet
def HasEqualChannels
    : Constraint<CPred<"$0.getType().cast<ShapedType>().getDimSize(3) == "
                       "$1.getType().cast<ShapedType>().getDimSize(3)">>;

def IsConstOp
    : Constraint<
          CPred<"dyn_cast_or_null<arith::ConstantOp>($0.getDefiningOp()) || "
                "dyn_cast_or_null<TFL::ConstOp>($0.getDefiningOp()) || "
                "dyn_cast_or_null<TFL::QConstOp>($0.getDefiningOp())">>;

// TFL_Conv2D() -> XC_FakeConv2D()
def :
Pat<(TFL_Conv2DOp: $output TensorOf<[QI8]>:$input, TensorOf<[QI8]>:$f, AnyTypeOf<[TensorOf<[I32,QI32]>, NoneType]>:$b, $dh, $dw, $faf, $wf, $sh, $sw),
          (XC_FakeConv2DOp $input, $f, $b, $dh, $dw, $faf, $wf, (CreateNoneValue $input), $sh, $sw),[
              (HasMultipleOfNBytesPerPixel<4> $input),
              (HasMultipleOfNBytesPerPixel<4> $output),
              (HasEqualChannels $input, $f),
              (IsConstOp $f),
              ]>;

// TFL_DepthwiseConv2D() -> XC_FakeDepthwiseConv2D()
def :
Pat<(TFL_DepthwiseConv2DOp: $output TensorOf<[QI8]>:$input, TensorOf<[QI8]>:$f, TensorOf<[I32,QI32]>:$b, $dh, $dw, $faf, $wf, $sh, $sw, $dm),
          (XC_FakeDepthwiseConv2DOp $input, $f, $b, $dh, $dw, $faf, $wf, (CreateNoneValue $input), $sh, $sw, $dm),[
              (HasMultipleOfNBytesPerPixel<4> $input),
              (HasMultipleOfNBytesPerPixel<4> $output),
              (IsConstOp $f),
              ]>;

// Special case, we only optimize conv with filter width 3, filter height
// 2, and stride height 2
def Hasfw5fh2
    : Constraint<CPred<"$0.getType().cast<ShapedType>().getRank() == 4 && "
                       "$0.getType().cast<ShapedType>().getDimSize(1) == 3 && "
                       "$0.getType().cast<ShapedType>().getDimSize(2) == 2">>;

// F32 TFL_Conv2D() -> XC_Beta_ConvF32()
def :
Pat<(TFL_Conv2DOp: $output TensorOf<[F32]>:$input, TensorOf<[F32]>:$f, TensorOf<[F32]>:$b, $dh, $dw, $faf, $wf, ConstantAttr<I32Attr, "2">, ConstantAttr<I32Attr, "1">),
          (XC_Beta_ConvF32Op $input, $f, $b),
          [(Hasfw5fh2 $f), (isBetaFloatEnabled)]>;

// // F32 TFL_TransposeConv2D() -> XC_Beta_TransposeConvF32()
def :
Pat<(TFL_TransposeConvOp: $output $outshape, TensorOf<[F32]>:$f, TensorOf<[F32]>:$input, TensorOf<[F32]>:$b, $wf, ConstantAttr<I32Attr, "2">, ConstantAttr<I32Attr, "1">, $faf),
          (XC_Beta_TransposeConvF32Op $input, $f, $b),
          [(Hasfw5fh2 $f), (isBetaFloatEnabled)]>;

// // F32 TFL_FullyConnected() -> XC_Beta_FcF32()
def :
Pat<(TFL_FullyConnectedOp: $output TensorOf<[F32]>:$input, TensorOf<[F32]>:$f, $b, $faf, $wf, $knd, $aqi),
          (XC_Beta_FcF32Op $input, $f), [(isBetaFloatEnabled)]>;
