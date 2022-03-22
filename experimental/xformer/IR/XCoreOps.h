// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_IR_XCOREOPS_H
#define XFORMER_IR_XCOREOPS_H

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// clang-format off
#include "IR/XCoreDialect.h.inc"
// clang-format on

#include "IR/XCoreEnumOps.h.inc"

#define GET_OP_CLASSES
#include "IR/XCoreOps.h.inc"

// Attribute name to store the required thread count in the module
// The attribute name is required to be prefixed by the dialect
constexpr char xcRequiredThreadCountAttrName[] = "xc.requiredThreadCount";

#endif // XFORMER_IR_XCOREOPS_H
