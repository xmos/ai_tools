// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

include "IR/XCoreOps.td"

def ShouldBeLoadedExternally
    : Constraint<CPred<"shouldBeLoadedExternally($0)">>;

// This constraint checks if the pattern has already been applied to the const
// op
def IsNotUsedByLoadConstantOp
    : Constraint<CPred<"isNotUsedByLoadConstantOp($0)">>;

foreach constOp = [Arith_ConstantOp, TFL_ConstOp] in {
def:
  Pat<(constOp
       : $op $attr),
      (XC_LoadConstantOp(constOp $attr)),
      [(IsNotUsedByLoadConstantOp $op), (ShouldBeLoadedExternally $attr)]>;
}

def : Pat<(TFL_QConstOp
           : $op $qtype, $attr),
          (XC_LoadConstantOp(TFL_QConstOp $qtype, $attr)),
          [(IsNotUsedByLoadConstantOp $op), (ShouldBeLoadedExternally $attr)]>;
