// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

// Generated dialect defs.
#include "IR/XCoreDialect.cpp.inc"

#define GET_OP_CLASSES
#include "IR/XCoreOps.cpp.inc"

#include "IR/XCoreEnumOps.cpp.inc"

#include "mlir/IR/TypeUtilities.h"

namespace mlir::xcore {

void XCoreDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IR/XCoreOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

OpFoldResult PadOp::fold(FoldAdaptor) {
  if (succeeded(
          verifyCompatibleShapes(getInput().getType(), getOutput().getType())))
    return getInput();

  return {};
}

} // namespace mlir::xcore
