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
    if (values.size() != 2) {
      return failure();
    }

    auto inputType1 = values[0].getType().cast<RankedTensorType>();
    auto inputType2 = values[1].getType().cast<RankedTensorType>();
    auto outputType = concatOp.getOutput().getType().cast<RankedTensorType>();
    if (!inputType1.hasStaticShape() || !inputType2.hasStaticShape()) {
      return failure();
    }

    Type elementType = inputType1.getElementType();

    int axis = concatOp.getAxis();
    const int rank = inputType1.getRank();
    if (axis < 0)
      axis = rank + axis;
    auto in_shp1 = inputType1.getShape();
    auto in_shp2 = inputType2.getShape();
    const size_t dtype_size = utils::getTypeSize(elementType);
    int num_copies = 1;
    for (int i = 0; i < axis; i++) {
      num_copies *= in_shp1[i];
    }
    int size1 = dtype_size;
    int size2 = dtype_size;
    for (int i = axis; i < rank; i++) {
      size1 *= in_shp1[i];
      size2 *= in_shp2[i];
    }
    std::cout << "num_copies: " << num_copies << std::endl;
    std::cout << "size1: " << size1 << std::endl;
    std::cout << "size2: " << size2 << std::endl;
    auto binaryObjectConcatOp = rewriter.create<ConcatOp>(
        concatOp.getLoc(), concatOp.getType(), values[0], values[1],
        rewriter.getI32IntegerAttr(num_copies),
        rewriter.getI32IntegerAttr(size1), rewriter.getI32IntegerAttr(size2));

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
