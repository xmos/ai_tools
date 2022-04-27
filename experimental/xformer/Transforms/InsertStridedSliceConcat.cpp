// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "lib_nn/api/MemCpyFn.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

namespace {
// Replace TFL StridedSlice with StridedSlice for XCore.
struct InsertStridedSliceConcat
    : public PassWrapper<InsertStridedSliceConcat, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  void runOnFunction() override;
};

struct InsertStridedSliceConcatPattern
    : public OpRewritePattern<TFL::StridedSliceOp> {
  using OpRewritePattern<TFL::StridedSliceOp>::OpRewritePattern;

  
};

void InsertStridedSliceConcat::runOnFunction() {
  auto *ctx = &getContext();
  auto func = getFunction();
  OwningRewritePatternList patterns(ctx);
  patterns.insert<InsertStridedSliceConcatPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the InsertStridedSliceConcat pass.
std::unique_ptr<OperationPass<FuncOp>> createInsertStridedSliceConcatPass() {
  return std::make_unique<InsertStridedSliceConcat>();
}

static PassRegistration<InsertStridedSliceConcat>
    pass("xcore-insert-stridedslice-concat",
         "Insert TFL StridedSlice and TFL Concat.");

} // namespace xcore
} // namespace mlir
