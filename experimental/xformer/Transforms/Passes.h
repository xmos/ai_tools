// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_TRANSFORMS_PASSES_H
#define XFORMER_TRANSFORMS_PASSES_H

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
std::unique_ptr<OperationPass<FuncOp>> createTranslateToCustomOpPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerXCorePassPipeline();

inline void registerAllPasses() {
  registerXCorePassPipeline();

  createPrintNestingPass();
  createOptimizeFullyConnectedPass();
  createTranslateToCustomOpPass();
}

} // namespace xcore
} // namespace mlir

#endif // XFORMER_TRANSFORMS_PASSES_H
