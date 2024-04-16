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

// Replace TFL Slice with Slice for XCore.
struct ReplaceSlice
    : public PassWrapper<ReplaceSlice, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceSlice)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-slice"; }
  StringRef getDescription() const final {
    return "Replace TFL Slice with Slice for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceSlicePattern : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp sliceOp,
                                PatternRewriter &rewriter) const override {

    // If the input is a constant, LLVM's Canonicalizer will
    // fold the slice into a constant later.
    if (matchPattern(sliceOp.getInput(), m_Constant()) ||
        matchPattern(sliceOp.getInput(), m_Op<TFL::ShapeOp>())) {
      return failure();
    }

    auto inputType = sliceOp.getInput().getType().cast<RankedTensorType>();
    if (!inputType.hasStaticShape())
      return failure();

    Type inputElementType = inputType.getElementType();
    auto outputType = sliceOp.getOutput().getType().cast<RankedTensorType>();

    DenseElementsAttr beginAttr;
    matchPattern(sliceOp.getBegin(), m_Constant(&beginAttr));
    auto beginValues = beginAttr.getValues<int32_t>();

    DenseElementsAttr sizeAttr;
    matchPattern(sliceOp.getSize(), m_Constant(&sizeAttr));
    auto sizeValues = sizeAttr.getValues<int32_t>();

    const int rank = inputType.getRank();

    if (rank != 4)
      return failure();

    if (utils::checkSliceNoOp(beginValues, sizeValues, inputType)) {
      rewriter.replaceOp(sliceOp, sliceOp.getInput());
      return success();
    }

    const size_t dtype_size = utils::getTypeSize(inputElementType);

    int32_t start, offset, size, num_copies;
    const int mulW = inputType.getShape()[3] * dtype_size;

    bool slicingHW = (inputType.getShape()[2] != outputType.getShape()[2]) ||
                     (inputType.getShape()[1] != outputType.getShape()[1]);

    if (slicingHW && (outputType.getShape()[3] != inputType.getShape()[3]))
      return failure();

    if (slicingHW) {
      if (inputType.getShape()[0] != 1 || outputType.getShape()[0] != 1)
        return failure();
      size = outputType.getShape()[2] * mulW;
      offset = inputType.getShape()[2] * mulW;
      start = beginValues[1] * offset + beginValues[2] * mulW;
      num_copies = outputType.getShape()[1];
    } else {
      offset = mulW;
      size = outputType.getShape()[3] * dtype_size;
      start = beginValues[3] * dtype_size;
      num_copies = outputType.getShape()[0] * outputType.getShape()[1] *
                   outputType.getShape()[2];
    }
    bool isVpu = start % 4 == 0 && size % 4 == 0 && offset % 4 == 0;
    auto binaryObjectSliceOp = rewriter.create<SliceOp>(
        sliceOp.getLoc(), sliceOp.getType(), sliceOp.getInput(),
        rewriter.getI32IntegerAttr(start), rewriter.getI32IntegerAttr(offset),
        rewriter.getI32IntegerAttr(size),
        rewriter.getI32IntegerAttr(num_copies), rewriter.getBoolAttr(isVpu));

    rewriter.replaceOp(sliceOp, binaryObjectSliceOp.getOutput());

    return success();
  }
};

void ReplaceSlice::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceSlicePattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceSlice pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceSlicePass() {
  return std::make_unique<ReplaceSlice>();
}

static PassRegistration<ReplaceSlice> pass;

} // namespace mlir::xcore
