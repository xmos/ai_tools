// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

namespace {
static constexpr char opSplitLabel[] = "opSplitLabel";
// OpSplit
struct OpSplit : public PassWrapper<OpSplit, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpSplit)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-op-split"; }
  StringRef getDescription() const final { return "Op Split."; }
  void runOnOperation() override;
};

struct OpSplitPattern : public OpRewritePattern<TFL::Conv2DOp> {
  using OpRewritePattern<TFL::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::Conv2DOp convOriginal,
                                PatternRewriter &rewriter) const override {
    // Do not split ops already split
    if (convOriginal->hasAttr(opSplitLabel))
      return failure();

    //
    // Check for invalid cases and return
    //
    if (convOriginal.padding() != "VALID")
      return failure();
    auto filterHeight =
        convOriginal.filter().getType().dyn_cast<RankedTensorType>().getDimSize(
            1);
    auto filterWidth =
        convOriginal.filter().getType().dyn_cast<RankedTensorType>().getDimSize(
            2);
    if (filterHeight != filterWidth)
      return failure();
    auto inputElementalType =
        convOriginal.input().getType().cast<ShapedType>().getElementType();
    // Input type must be QI8
    if (!(inputElementalType.isa<quant::QuantizedType>() &&
          inputElementalType.cast<quant::QuantizedType>().isSigned() &&
          inputElementalType.cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }
    auto outputElementalType =
        convOriginal.output().getType().cast<ShapedType>().getElementType();
    // Output type must be QI8
    if (!(outputElementalType.isa<quant::QuantizedType>() &&
          outputElementalType.cast<quant::QuantizedType>().isSigned() &&
          outputElementalType.cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    // Clone the op as we want to replace it with the same conv op but with
    // strided slices and concat inserted after it
    auto convReplacement =
        llvm::cast<TFL::Conv2DOp>(rewriter.clone(*convOriginal));

    // Apply label, so that the same op is not rewritten a second time.
    convReplacement->setAttr(opSplitLabel, rewriter.getUnitAttr());

    // variables that are the same for all strided slices to be created
    int32_t stridesAttr[4] = {1, 1, 1, 1};
    auto stridesConstantOp = rewriter.create<arith::ConstantOp>(
        convReplacement.getLoc(), rewriter.getI32TensorAttr(stridesAttr));
    int32_t begin_mask, end_mask, ellipsis_mask, new_axis_mask,
        shrink_axis_mask;
    begin_mask = end_mask = ellipsis_mask = new_axis_mask = shrink_axis_mask =
        0;

    // Will hold stridec slice op to insert after conv op
    SmallVector<Value> stridedSliceOps;

    auto convOutput = convOriginal.output();
    auto outputType = convOutput.getType().dyn_cast<RankedTensorType>();
    int32_t outputHeight = outputType.getDimSize(1);
    int32_t outputWidth = outputType.getDimSize(2);
    int32_t outputDepth = outputType.getDimSize(3);

    // The number of splits is hardcoded for now,
    // but it should be determined by conv output size
    int numSplits = 4;

    int32_t sliceWidth = outputWidth / numSplits;

    // The remainder will be distrubed between the splits
    // to keep them about the same size
    int32_t sliceWidthRemainder = outputWidth % numSplits;

    // For loop uses end index of previous strided slice created
    // needs to intializec to zero for first slice
    int32_t prevEndIndex = 0;

    // Loops creates strided slices with correct params
    for (size_t i = 0; i < numSplits; i++) {
      // Distibutes remainder between slices
      int32_t currentSliceWidth = sliceWidth;
      if (i < sliceWidthRemainder)
        currentSliceWidth++;

      // Descibes output tensor of strided slice
      // Only currentSliceWidth can be unique to each strided slice
      RankedTensorType stridedSliceOutputType = RankedTensorType::get(
          {1, outputHeight, currentSliceWidth, outputDepth},
          convOutput.getType().cast<ShapedType>().getElementType());

      // Start where the prev slice ended
      int32_t beginAttr[4] = {0, 0, prevEndIndex, 0};
      auto beginConstantOp = rewriter.create<arith::ConstantOp>(
          convReplacement.getLoc(), rewriter.getI32TensorAttr(beginAttr));

      // End is start + slice width
      int32_t endIndex = prevEndIndex + currentSliceWidth;
      // Go to end of tensor for all dims except width
      int32_t endAttr[4] = {1, outputHeight, endIndex, outputDepth};
      auto endConstantOp = rewriter.create<arith::ConstantOp>(
          convReplacement.getLoc(), rewriter.getI32TensorAttr(endAttr));
      prevEndIndex = endIndex;

      auto stridedSliceOp = rewriter.create<TFL::StridedSliceOp>(
          convReplacement.getLoc(), stridedSliceOutputType, convReplacement,
          beginConstantOp, endConstantOp, stridesConstantOp, begin_mask,
          end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);

      // Add label, for safety when raising slice later
      stridedSliceOp->setAttr(opSplitLabel, rewriter.getUnitAttr());

      // Store created strided slice op to use as input to concat
      stridedSliceOps.push_back(stridedSliceOp.getResult());
    }

    // Concat op does not have activation function
    StringRef fused_activation_function = "NONE";

    auto newConcatOp = rewriter.create<TFL::ConcatenationOp>(
        convReplacement.getLoc(), convOutput.getType(), stridedSliceOps, 2,
        fused_activation_function);

    // Replace Conv with
    // Cloned Conv -> Strided Slices -> Concat
    rewriter.replaceOp(convOriginal, newConcatOp.output());

    return success();
  }
};

struct RaiseStridedSlicePattern : public OpRewritePattern<TFL::StridedSliceOp> {
  using OpRewritePattern<TFL::StridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::StridedSliceOp stridedSlice,
                                PatternRewriter &rewriter) const override {
    // Only raise slices that have been inserted with op split pass
    if (!((stridedSlice->hasAttr(opSplitLabel))))
      return failure();

    // Do not raise slices that have already been raised
    static constexpr char raisedStridedSliceLabel[] = "raisedStridedSliceLabel";
    if (stridedSlice->hasAttr(raisedStridedSliceLabel))
      return failure();

    auto convOriginal =
        llvm::cast<TFL::Conv2DOp>(stridedSlice.input().getDefiningOp());
    auto convOriginalInputShape =
        convOriginal.input().getType().cast<RankedTensorType>().getShape();
    auto convOriginalOutputShape =
        convOriginal.output().getType().cast<RankedTensorType>().getShape();
    auto filterWidth =
        convOriginal.filter().getType().dyn_cast<RankedTensorType>().getDimSize(
            2);
    auto strideWidth = convOriginal.stride_w();

    DenseElementsAttr attr;
    if (!matchPattern(stridedSlice.end(), m_Constant(&attr))) {
      return failure();
    }
    auto endIndex = attr.getValues<int32_t>()[2];

    auto newEndIndex = endIndex * strideWidth - strideWidth + filterWidth;

    int32_t endAttr[4] = {1, static_cast<int32_t>(convOriginalInputShape[1]),
                          static_cast<int32_t>(newEndIndex),
                          static_cast<int32_t>(convOriginalInputShape[3])};
    auto endConstantOp = rewriter.create<arith::ConstantOp>(
        stridedSlice.getLoc(), rewriter.getI32TensorAttr(endAttr));

    auto stridedSliceOutputShape =
        stridedSlice.output().getType().cast<RankedTensorType>().getShape();
    auto outputWidth = stridedSliceOutputShape[2];

    int32_t newOutputWidth =
        outputWidth * strideWidth - strideWidth + filterWidth;

    int32_t beginAttr[4] = {
        0, 0, static_cast<int32_t>(newEndIndex - newOutputWidth), 0};
    auto beginConstantOp = rewriter.create<arith::ConstantOp>(
        stridedSlice.getLoc(), rewriter.getI32TensorAttr(beginAttr));

    RankedTensorType newStridedSliceType = RankedTensorType::get(
        {convOriginalInputShape[0], convOriginalInputShape[1], newOutputWidth,
         convOriginalInputShape[3]},
        convOriginal.input().getType().cast<ShapedType>().getElementType());

    auto stridedSliceReplacement = rewriter.create<TFL::StridedSliceOp>(
        stridedSlice.getLoc(), newStridedSliceType, convOriginal.input(),
        beginConstantOp, endConstantOp, stridedSlice.strides(),
        stridedSlice.begin_mask(), stridedSlice.end_mask(),
        stridedSlice.ellipsis_mask(), stridedSlice.new_axis_mask(),
        stridedSlice.shrink_axis_mask());
    stridedSliceReplacement->setAttr(raisedStridedSliceLabel,
                                     rewriter.getUnitAttr());

    RankedTensorType newConvType = RankedTensorType::get(
        {convOriginalOutputShape[0], convOriginalOutputShape[1],
         stridedSliceOutputShape[2], convOriginalOutputShape[3]},
        convOriginal.output().getType().cast<ShapedType>().getElementType());

    auto convReplacement =
        llvm::cast<TFL::Conv2DOp>(rewriter.clone(*convOriginal));
    convReplacement->getResult(0).setType(newConvType);

    convReplacement.setOperand(0, stridedSliceReplacement);

    rewriter.replaceOp(stridedSlice, convReplacement.output());

    return success();
  }
};

void OpSplit::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();

  RewritePatternSet patterns(ctx);
  patterns.insert<OpSplitPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  RewritePatternSet patterns2(ctx);
  patterns2.insert<RaiseStridedSlicePattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns2));
}
} // namespace

// Creates an instance of the OpSplit pass.
std::unique_ptr<OperationPass<func::FuncOp>> createOpSplitPass() {
  return std::make_unique<OpSplit>();
}

static PassRegistration<OpSplit> pass;

} // namespace xcore
} // namespace mlir
