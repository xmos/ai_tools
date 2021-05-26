// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#define GET_OP_CLASSES
#include "IR/XCoreOps.cpp.inc"

namespace mlir {
namespace xcore {

std::vector<uint8_t> FullyConnectedOp::buildCustomOptions() { return {}; }

void XCoreDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IR/XCoreOps.cpp.inc"
      >();
}
} // namespace xcore
} // namespace mlir
