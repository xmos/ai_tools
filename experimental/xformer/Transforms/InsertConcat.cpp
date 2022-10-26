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
// Insert  Concat
struct InsertConcat
    : public PassWrapper<InsertConcat,
                          OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertConcat)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-insert-concat"; }
  StringRef getDescription() const final {
    return "Insert TFL Concat.";
  }
  void runOnOperation() override;
};



struct InsertConcatPattern
    : public OpRewritePattern<DummyStridedSliceOp> {
  using OpRewritePattern<DummyStridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DummyStridedSliceOp stridedSliceOriginal,
                                PatternRewriter &rewriter) const override {

    auto stridedSliceInput = stridedSliceOriginal.input();

	// Extract args from the op
	auto inputType =
	stridedSliceInput.getType().dyn_cast<RankedTensorType>();
	auto inputHeight = inputType.getDimSize(1);
	auto inputWidth = inputType.getDimSize(2);
	auto inputDepth = inputType.getDimSize(3);

	int32_t offset = (inputHeight*inputWidth*inputDepth)/2;

	auto simpleSliceOp0  = rewriter.create<SimpleSliceOp>(
		stridedSliceOriginal.getLoc(), stridedSliceOriginal.getType(),
        stridedSliceInput, rewriter.getI32IntegerAttr(offset)) ; 

	auto simpleSliceOp1  = rewriter.create<SimpleSliceOp>(
		stridedSliceOriginal.getLoc(), stridedSliceOriginal.getType(),
        stridedSliceInput, rewriter.getI32IntegerAttr(offset)) ; 

	auto concatOp  = rewriter.create<ConcatOp>(
		stridedSliceOriginal.getLoc(), stridedSliceOriginal.getType(),
        simpleSliceOp0, simpleSliceOp0, offset );

	rewriter.replaceOp(stridedSliceOriginal, concatOp.output());

    return success();
  }
};

void InsertConcat::runOnOperation() {
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();
  patterns.insert<InsertConcatPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the InsertConcat pass.
std::unique_ptr<OperationPass<func::FuncOp>> createInsertConcatPass() {
  return std::make_unique<InsertConcat>();
}

static PassRegistration<InsertConcat> pass;

} // namespace xcore
} // namespace mlir
