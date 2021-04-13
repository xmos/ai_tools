#include "ir/xc_ops.h"

#define GET_OP_CLASSES
#include "ir/xc_ops.cc.inc"

namespace mlir {
namespace xcore {

std::vector<uint8_t> FullyConnectedOp::buildCustomOptions() {
  return {};
}

void XCoreDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ir/xc_ops.cc.inc"
      >();
}
} // namespace xcore
} // namespace mlir
