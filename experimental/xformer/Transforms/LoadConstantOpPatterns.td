// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

// This is the second pattern definition file for XCore.
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

include "IR/XCoreOps.td"

def ShouldBeLoadedExternally
    : Constraint<CPred<"shouldBeLoadedExternally($0)">>;

def IsUsedByValidOp : Constraint<CPred<"isUsedByValidOp($0)">>;

foreach constOp = [ConstantOp, TFL_ConstOp] in {
def:
  Pat<(constOp
       : $optype $attr),
      (XC_LoadConstantOp $attr),
      [(ShouldBeLoadedExternally $attr), (IsUsedByValidOp $optype)]>;
}

def : Pat<(TFL_QConstOp
           : $optype $qtype, $attr),
          (XC_LoadConstantOp $attr),
          [(ShouldBeLoadedExternally $attr), (IsUsedByValidOp $optype)]>;
