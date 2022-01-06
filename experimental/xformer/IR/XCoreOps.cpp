// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#define GET_OP_CLASSES
#include "IR/XCoreOps.cpp.inc"

#include "IR/XCoreEnumOps.cpp.inc"

namespace mlir {
namespace xcore {

void XCoreDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IR/XCoreOps.cpp.inc"
      >();
}
} // namespace xcore
} // namespace mlir
