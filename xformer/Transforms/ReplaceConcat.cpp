// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "Utils/Util.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

extern "C" {
#include "lib_nn/api/nn_layers.h"
}

namespace mlir::xcore {

namespace {

// Replace TFL Concatenate with Concat for XCore.
struct ReplaceConcat
    : public PassWrapper<ReplaceConcat, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceConcat)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-concat"; }
  StringRef getDescription() const final {
    return "Replace TFL Concatenate with Concat for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceConcatPattern : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern<TFL::ConcatenationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concatOp,
                                PatternRewriter &rewriter) const override {

    auto values = concatOp.getValues();
    int num_inputs = values.size();
    if (num_inputs > 16)
      return failure();
    ArrayRef<int64_t> inputShapes[16];
    for (int i = 0; i < num_inputs; i++) {
      auto inputType = values[i].getType().cast<RankedTensorType>();
      if (!inputType.hasStaticShape())
        return failure();
      inputShapes[i] = inputType.getShape();
    }
    auto outputType = concatOp.getOutput().getType().cast<RankedTensorType>();

    Type elementType = outputType.getElementType();

    int axis = concatOp.getAxis();
    const int rank = outputType.getRank();
    if (axis < 0)
      axis = rank + axis;
    const size_t dtype_size = utils::getTypeSize(elementType);
    int num_copies = 1;
    for (int i = 0; i < axis; i++) {
      num_copies *= outputType.getShape()[i];
    }

    bool isVpu = true;
    int32_t sizes[16];
    for (int i = 0; i < num_inputs; i++) {
      sizes[i] = dtype_size;
      for (int j = axis; j < rank; j++)
        sizes[i] *= inputShapes[i][j];
      if (sizes[i] % 4 != 0)
        isVpu = false;
    }

    auto binaryObjectConcatOp = rewriter.create<ConcatOp>(
        concatOp.getLoc(), concatOp.getType(), values,
        rewriter.getI32IntegerAttr(num_copies), rewriter.getI32ArrayAttr(sizes),
        rewriter.getI32IntegerAttr(num_inputs), rewriter.getBoolAttr(isVpu));

    rewriter.replaceOp(concatOp, binaryObjectConcatOp.getOutput());

    return success();
  }
};

void ReplaceConcat::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceConcatPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceConcat pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceConcatPass() {
  return std::make_unique<ReplaceConcat>();
}

static PassRegistration<ReplaceConcat> pass;

} // namespace mlir::xcore
