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

std::unique_ptr<OperationPass<FuncOp>> createApplyTFLPatternsPass();
std::unique_ptr<OperationPass<FuncOp>> createInsertStridedSlicePatternsPass();
std::unique_ptr<OperationPass<FuncOp>> createInsertConcatPass();
std::unique_ptr<OperationPass<FuncOp>> createReplaceAvgPoolWithConv2DPass();
std::unique_ptr<OperationPass<FuncOp>> createReplaceFCWithConv2DPass();
std::unique_ptr<OperationPass<FuncOp>> createPad3to4Conv2DPass();
std::unique_ptr<OperationPass<FuncOp>> createReplaceConv2DPass();
std::unique_ptr<OperationPass<FuncOp>> createReplaceStridedSlicePass();
std::unique_ptr<OperationPass<FuncOp>> createReplaceConcatPass();
std::unique_ptr<OperationPass<FuncOp>> createApplyXCPatternsPass();
std::unique_ptr<OperationPass<FuncOp>> createApplyLoadConstantOpPatternsPass();
std::unique_ptr<OperationPass<FuncOp>> createWriteFlashImagePass();
std::unique_ptr<OperationPass<FuncOp>> createTranslateToCustomOpPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerXCorePassPipeline();

} // namespace xcore
} // namespace mlir

#endif // XFORMER_TRANSFORMS_PASSES_H
