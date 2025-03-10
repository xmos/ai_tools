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
void buildXCorePreOpSplitPassPipeline(OpPassManager &pm);
void buildXCoreRemainingPassPipeline(OpPassManager &pm);

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
std::unique_ptr<OperationPass<func::FuncOp>>
createVerifySameAllocationTensorsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createRemoveDynamicShapePass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceAddSubPass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceMulPass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceMeanPass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceSumPass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceMaxPoolPass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceStridedSlicePass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceSlicePass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceBroadcastPass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplacePadPass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceConcatPass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceTransposePass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceConv2DPass();
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceTransposeConvPass();
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
