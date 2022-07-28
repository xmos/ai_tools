// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

include "IR/XCoreOps.td"

// XC_FakeConv2D -> TFL_Conv2D()
def : Pat<(XC_FakeConv2DOp $input, $f, $b, $dh, $dw, $faf, $p, $pv, $sh, $sw),
          (TFL_Conv2DOp $input, $f, $b, $dh, $dw, $faf, $p, $sh, $sw)>;

// XC_FakeDepthwiseConv2D -> TFL_DepthwiseConv2D()
def : Pat<(XC_FakeDepthwiseConv2DOp $input, $f, $b, $dh, $dw, $faf, $p, $pv,
           $sh, $sw, $dm),
          (TFL_DepthwiseConv2DOp $input, $f, $b, $dh, $dw, $faf, $p, $sh, $sw,
           $dm)>;

def getPadOpOutputType
    : NativeCodeCall<
          "getPadOpOutputType($_builder, $0, "
          "std::vector<int32_t>($1.cast<DenseIntElementsAttr>().getValues<"
          "int32_t>().begin(), "
          "$1.cast<DenseIntElementsAttr>().getValues<int32_t>().end()))">;

def : Pat<(XC_FakeDepthwiseConv2DOp $input, $f, $b, $dh, $dw, $faf,
           ConstantStrAttr<StrAttr, "EXPLICIT">,
           (Arith_ConstantOp
            : $pv_op $pv),
           $sh, $sw, $dm),
          (TFL_DepthwiseConv2DOp(TFL_PadOp $input, $pv_op,
                                 (returnType(getPadOpOutputType $input, $pv))),
           $f, $b, $dh, $dw, $faf, TFL_PAD_Valid, $sh, $sw, $dm),
          [], (addBenefit 10)>;