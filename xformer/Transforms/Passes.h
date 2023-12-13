// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#ifndef XFORMER_TRANSFORMS_PASSES_H
#define XFORMER_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeTransposePass();
std::unique_ptr<OperationPass<func::FuncOp>>
createReplaceAvgPoolWithConv2DPass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceFCWithConv2DPass();
std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeConv2DPass();
std::unique_ptr<OperationPass<func::FuncOp>> createOpSplitPass();
std::unique_ptr<OperationPass<func::FuncOp>> createApplyTFLPatternsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceAddPass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceMulPass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceMaxPoolPass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceStridedSlicePass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceConv2DPass();
std::unique_ptr<OperationPass<func::FuncOp>> createApplyXCPatternsPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createApplyLoadConstantOpPatternsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createWriteWeightsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createPlanMemoryPass();
std::unique_ptr<OperationPass<func::FuncOp>> createTranslateToCustomOpPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerXCorePassPipeline();

} // namespace xcore
} // namespace mlir

#endif // XFORMER_TRANSFORMS_PASSES_H
