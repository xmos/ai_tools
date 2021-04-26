// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "flatbuffers/flexbuffers.h"

#define GET_OP_CLASSES
#include "IR/XCoreOps.cpp.inc"

namespace mlir {
namespace xcore {

std::vector<uint8_t> FullyConnectedOp::buildCustomOptions() {
  // TODO: We are adding "illegal_params" as custom options here as it is
  // expected by xformer 1.0 to run LegalizeXCFullyConnectedPass. With this we
  // can verify that the output of xformer 2.0 can be consumed by xformer 1.0.
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Int("illegal_params", 1); });
  fbb.Finish();
  return fbb.GetBuffer();
}

void XCoreDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IR/XCoreOps.cpp.inc"
      >();
}
} // namespace xcore
} // namespace mlir
