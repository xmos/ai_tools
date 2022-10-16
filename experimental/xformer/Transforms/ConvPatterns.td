// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

include "IR/XCoreOps.td"

def ShouldReduceMemory : Constraint<CPred<"shouldReduceMemory()">>;

def CreateNoneValue : NativeCodeCall<"$_builder.create<TFL::NoValueOp>($0."
                                     "getLoc(), $_builder.getUnitAttr())">;

// Check that depth is a multiple of four
def HasValidDepth
    : Constraint<
          CPred<"$0.getType().cast<ShapedType>().getDimSize(3) % 4 == 0">>;

// Check that input channels equals filter channels
// We don't optimize grouped convolutions yet
def HasEqualChannels
    : Constraint<CPred<"$0.getType().cast<ShapedType>().getDimSize(3) == "
                       "$1.getType().cast<ShapedType>().getDimSize(3)">>;

// TFL_Conv2D() -> XC_FakeConv2D()
def : Pat<(TFL_Conv2DOp: $output TensorOf<[QI8]>:$input, TensorOf<[QI8]>:$f, AnyTypeOf<[TensorOf<[I32,QI32]>, NoneType]>:$b, $dh, $dw, $faf, $p, $sh, $sw),
          (XC_FakeConv2DOp $input, $f, $b, $dh, $dw, $faf, $p, (CreateNoneValue $input), $sh, $sw),[
              (HasValidDepth $input),
              (HasValidDepth $output),
              (HasEqualChannels $input, $f)
              ]>;

// TFL_DepthwiseConv2D() -> XC_FakeDepthwiseConv2D()
def : Pat<(TFL_DepthwiseConv2DOp: $output TensorOf<[QI8]>:$input, TensorOf<[QI8]>:$f, TensorOf<[I32,QI32]>:$b, $dh, $dw, $faf, $p, $sh, $sw, $dm),
          (XC_FakeDepthwiseConv2DOp $input, $f, $b, $dh, $dw, $faf, $p, (CreateNoneValue $input), $sh, $sw, $dm),[
              (HasValidDepth $input),
              (HasValidDepth $output)
              ]>;

// Replace XC_FakeDepthwiseConv2D(Pad()) of VALID padding with
// XC_FakeDepthwiseConv2D() of EXPLICIT padding
def : Pat<(XC_FakeDepthwiseConv2DOp(TFL_PadOp $input, $padding_values), $f, $b,
           $dh, $dw, $faf, TFL_PAD_Valid, $dummy_values, $sh, $sw, $dm),
          (XC_FakeDepthwiseConv2DOp $input, $f, $b, $dh, $dw, $faf,
           ConstantStrAttr<StrAttr, "EXPLICIT">, $padding_values, $sh, $sw,
           $dm),
          [(ShouldReduceMemory)]>;