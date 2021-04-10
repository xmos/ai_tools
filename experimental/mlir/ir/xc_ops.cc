#include "xc_ops.h"

#include "flatbuffers/flexbuffers.h"
//#include "tensorflow/lite/schema/schema_generated.h"

#define GET_OP_CLASSES
#include "experimental/mlir/ir/xc_ops.cc.inc"

namespace mlir {
namespace xcore {

std::vector<uint8_t> FullyConnectedOp::buildCustomOptions() {
  // return {};
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Int("illegal_params", 1); });
  fbb.Finish();
  return fbb.GetBuffer();
}

void XCoreDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "experimental/mlir/ir/xc_ops.cc.inc"
      >();
}
} // namespace xcore
} // namespace mlir
