// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

include "IR/XCoreOps.td"

// XC_FakeConv2D -> TFL_Conv2D()
def : Pat<(XC_FakeConv2DOp $input, $f, $b, $po, $dh, $dw, $faf, $p, $pv, $sh,
           $sw, $osbh, $osbw, $osh, $osw, $io),
          (TFL_Conv2DOp $input, $f, $b, $dh, $dw, $faf, $p, $sh, $sw)>;

// XC_FakeDepthwiseConv2D -> TFL_DepthwiseConv2D()
def : Pat<(XC_FakeDepthwiseConv2DOp $input, $f, $b, $dh, $dw, $faf, $p, $pv,
           $sh, $sw, $dm),
          (TFL_DepthwiseConv2DOp $input, $f, $b, $dh, $dw, $faf, $p, $sh, $sw,
           $dm)>;
