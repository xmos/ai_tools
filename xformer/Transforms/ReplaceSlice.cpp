// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

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

    auto inputType = sliceOp.getInput().getType().cast<RankedTensorType>();
    assert(inputType.hasStaticShape() &&
           "SliceOp: Input tensor must have static shape");
    Type inputElementType = inputType.getElementType();
    auto outputType = sliceOp.getOutput().getType().cast<RankedTensorType>();

    DenseElementsAttr beginAttr;
    matchPattern(sliceOp.getBegin(), m_Constant(&beginAttr));
    auto beginValues = beginAttr.getValues<int32_t>();

    DenseElementsAttr sizeAttr;
    matchPattern(sliceOp.getSize(), m_Constant(&sizeAttr));
    auto sizeValues = sizeAttr.getValues<int32_t>();

    int begin[5], end[5], inShape[5], inOffsets[4], outOffsets[4];

    // TFLite supports up to 5 dimensions, if the input is less we pad
    const int rank = inputType.getRank();
    const int numPad = 5 - rank;
    for (int i = 0; i < 5; i++) {
      begin[i] = i > numPad ? beginValues[i - numPad] : 0;
      end[i] = i > numPad ? begin[i] + sizeValues[i - numPad] : 1;
      inShape[i] = i > numPad ? inputType.getShape()[i - numPad] : 1;
    }

    // Merge axes where possible in the end
    while (begin[4] == 0 && end[4] == inShape[4]) {
      int32_t last_begin = begin[3] * inShape[4];
      int32_t last_end = end[3] * inShape[4];
      int32_t last_dim = inShape[3] * inShape[4];
      memmove(begin + 1, begin, 3 * sizeof(int32_t));
      memmove(end + 1, end, 3 * sizeof(int32_t));
      memmove(inShape + 1, inShape, 3 * sizeof(int32_t));
      begin[0] = 0;
      end[0] = 1;
      inShape[0] = 1;
      begin[4] = last_begin;
      end[4] = last_end;
      inShape[4] = last_dim;
    }

    // Treat dtype as an extra axis that we merge with the last axis, to use
    // vpu_memcpy if possible
    auto dtypeSize = inputElementType.getIntOrFloatBitWidth() / 8;
    inShape[4] *= dtypeSize;
    begin[4] *= dtypeSize;
    end[4] *= dtypeSize;

    // Initialise offsets
    inOffsets[0] = inputType.getNumElements() / inShape[0];
    outOffsets[0] = outputType.getNumElements() / (end[0] - begin[0]);
    for (int i = 1; i < 4; i++) {
      inOffsets[i] = inOffsets[i - 1] / inShape[i];
      outOffsets[i] = outOffsets[i - 1] / (end[i] - begin[i]);
    }

    const bool isVpu =
        inShape[4] % 4 == 0 && begin[4] % 4 == 0 && end[4] % 4 == 0;

    auto binaryObjectSliceOp = rewriter.create<SliceOp>(
        sliceOp.getLoc(), sliceOp.getType(), sliceOp.getInput(),
        rewriter.getI32ArrayAttr(begin), rewriter.getI32ArrayAttr(end),
        rewriter.getI32ArrayAttr(inOffsets),
        rewriter.getI32ArrayAttr(outOffsets), rewriter.getBoolAttr(isVpu));

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

} // namespace xcore
} // namespace mlir
