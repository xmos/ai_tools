#ifndef XCORE_TRANSFORMS_PASSES_H
#define XCORE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace xcore {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Create a single pipeline that will run all the needed passes in the right
// order.
void buildXCorePassPipeline(OpPassManager &pm);

//===----------------------------------------------------------------------===//
// XCore-specific passes
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> createPrintNestingPass();
std::unique_ptr<OperationPass<FuncOp>> createOptimizeFullyConnectedPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerXCorePassPipeline();

inline void registerAllPasses() {
  registerXCorePassPipeline();

  createOptimizeFullyConnectedPass();
  createPrintNestingPass();
}

} // namespace xcore
} // namespace mlir

#endif // XCORE_TRANSFORMS_PASSES_H