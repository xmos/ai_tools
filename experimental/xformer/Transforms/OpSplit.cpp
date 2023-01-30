// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

#include "tensorflow/core/framework/kernel_shape_util.h"

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

    // Data from conv needed later
    auto convOutput = convOriginal.output();
    auto outputType = convOutput.getType().dyn_cast<RankedTensorType>();
    int32_t outputHeight = outputType.getDimSize(1);
    int32_t outputWidth = outputType.getDimSize(2);
    int32_t outputDepth = outputType.getDimSize(3);
    int32_t outputSize = outputHeight * outputWidth * outputDepth;

    // Number chosen for testing purposes
    // Actul number will depend on application
    int32_t splitTensorSize = 98304;
    // Only op split if output size is too big
    if (outputSize < 2 * splitTensorSize)
      return failure();

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

    // The number of splits is determined by conv output size
    int numSplits = ceil(outputSize / splitTensorSize);

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

    // Create concat op that concats on dim 2, width
    auto concatOp = rewriter.create<TFL::ConcatenationOp>(
        convReplacement.getLoc(), convOutput.getType(), stridedSliceOps, 2,
        fused_activation_function);

    // Replace Conv with
    // Cloned Conv -> Strided Slices -> Concat
    rewriter.replaceOp(convOriginal, concatOp.output());

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

    static constexpr char raisedStridedSliceLabel[] = "raisedStridedSliceLabel";
    // Do not raise slices that have already been raised
    if (stridedSlice->hasAttr(raisedStridedSliceLabel))
      return failure();

    // Get data from conv needed to raise strided slice
    auto convOriginal =
        llvm::cast<TFL::Conv2DOp>(stridedSlice.input().getDefiningOp());

    auto convOriginalInputShape =
        convOriginal.input().getType().cast<RankedTensorType>().getShape();
    auto inputHeight = convOriginalInputShape[1];
    auto inputWidth = convOriginalInputShape[2];

    auto convOriginalOutputShape =
        convOriginal.output().getType().cast<RankedTensorType>().getShape();

    auto filterType =
        convOriginal.filter().getType().dyn_cast<RankedTensorType>();
    auto filterHeight = filterType.getDimSize(1);
    auto filterWidth = filterType.getDimSize(2);

    auto strideWidth = convOriginal.stride_w();

    // get end index of strided slice
    DenseElementsAttr attr;
    if (!matchPattern(stridedSlice.end(), m_Constant(&attr))) {
      return failure();
    }
    auto endIndex = attr.getValues<int32_t>()[2];

    // Get original slice's output width
    auto stridedSliceOutputShape =
        stridedSlice.output().getType().cast<RankedTensorType>().getShape();
    auto outputWidth = stridedSliceOutputShape[2];

    int32_t newEndIndex;
    int32_t newOutputWidth;
    int64_t padTop, padBottom, padLeft, padRight;
    padTop = padBottom = padLeft = padRight = 0;

    int dilation_h_factor = 1;
    int dilation_w_factor = 1;
    int64_t newHeight, newWidth;
    tensorflow::Padding opPadding = convOriginal.padding() == "VALID"
                                        ? tensorflow::Padding::VALID
                                        : tensorflow::Padding::SAME;
    if (tensorflow::GetWindowedOutputSizeVerboseV2(
            inputHeight, filterHeight, dilation_h_factor,
            convOriginal.stride_h(), opPadding, &newHeight, &padTop,
            &padBottom) != tensorflow::Status::OK()) {
      return failure();
    }
    if (tensorflow::GetWindowedOutputSizeVerboseV2(
            inputWidth, filterWidth, dilation_w_factor, strideWidth, opPadding,
            &newWidth, &padLeft, &padRight) != tensorflow::Status::OK()) {
      return failure();
    }

    // Check if padding is same
    if (convOriginal.padding() == "VALID") {
      // Calculate new end index for slice after being raised above conv
      newEndIndex = endIndex * strideWidth - strideWidth + filterWidth;

    } else if (convOriginal.padding() == "SAME") {

      // Check if this is left most split
      if (!matchPattern(stridedSlice.begin(), m_Constant(&attr))) {
        return failure();
      }
      auto beginIndex = attr.getValues<int32_t>()[2];
      if (beginIndex == 0) {
        // Calculate new end index for slice after being raised above conv
        newEndIndex =
            endIndex * strideWidth - strideWidth + filterWidth - padLeft;

        padRight = 0;

      } else if (endIndex == convOriginalOutputShape[2]) { // end
        // Calculate new end index for slice after being raised above conv
        newEndIndex = endIndex * strideWidth - strideWidth + filterWidth -
                      padLeft - padRight;

        padLeft = 0;

      } else { // beginIndex not 0 or end
        // Calculate new end index for slice after being raised above conv
        newEndIndex =
            endIndex * strideWidth - strideWidth + filterWidth - padLeft;

        padLeft = 0;
        padRight = 0;
      }
    }

    // Set end tensor for slice to be above conv with new end index
    int32_t endAttr[4] = {1, static_cast<int32_t>(convOriginalInputShape[1]),
                          static_cast<int32_t>(newEndIndex),
                          static_cast<int32_t>(convOriginalInputShape[3])};
    auto endConstantOp = rewriter.create<arith::ConstantOp>(
        stridedSlice.getLoc(), rewriter.getI32TensorAttr(endAttr));

    // Calculate new output width after raising slice above conv
    newOutputWidth = outputWidth * strideWidth - strideWidth + filterWidth -
                     padLeft - padRight;

    // Set begin tensor to zero for all dims except width
    // set width to new end index - new output width
    int32_t beginAttr[4] = {
        0, 0, static_cast<int32_t>(newEndIndex - newOutputWidth), 0};
    auto beginConstantOp = rewriter.create<arith::ConstantOp>(
        stridedSlice.getLoc(), rewriter.getI32TensorAttr(beginAttr));

    // New strided slice output shape is conv input shape except width
    // The new calculated output width is used for width
    RankedTensorType newStridedSliceType = RankedTensorType::get(
        {convOriginalInputShape[0], convOriginalInputShape[1], newOutputWidth,
         convOriginalInputShape[3]},
        convOriginal.input().getType().cast<ShapedType>().getElementType());

    // Create new strided slice for above conv
    auto stridedSliceReplacement = rewriter.create<TFL::StridedSliceOp>(
        stridedSlice.getLoc(), newStridedSliceType, convOriginal.input(),
        beginConstantOp, endConstantOp, stridedSlice.strides(),
        stridedSlice.begin_mask(), stridedSlice.end_mask(),
        stridedSlice.ellipsis_mask(), stridedSlice.new_axis_mask(),
        stridedSlice.shrink_axis_mask());
    // Label it as raised so it is not raised again
    stridedSliceReplacement->setAttr(raisedStridedSliceLabel,
                                     rewriter.getUnitAttr());

    auto paddedHeight = convOriginalInputShape[1] + padTop + padBottom;
    auto paddedWidth = newOutputWidth + padLeft + padRight;

    std::vector<int32_t> paddingValues{0,
                                       0,
                                       static_cast<int>(padTop),
                                       static_cast<int>(padBottom),
                                       static_cast<int>(padLeft),
                                       static_cast<int>(padRight),
                                       0,
                                       0};

    RankedTensorType paddingsType =
        RankedTensorType::get({4, 2}, rewriter.getI32Type());

    Value paddings = rewriter.create<TFL::ConstOp>(
        stridedSlice.getLoc(),
        DenseIntElementsAttr::get(paddingsType, paddingValues));

    auto paddedResultType = RankedTensorType::get(
        {convOriginalInputShape[0], paddedHeight, paddedWidth,
         convOriginalInputShape[3]},
        convOriginal.input().getType().cast<ShapedType>().getElementType());

    auto padOp =
        rewriter.create<TFL::PadOp>(stridedSlice.getLoc(), paddedResultType,
                                    stridedSliceReplacement, paddings);

    auto convReplacement =
        llvm::cast<TFL::Conv2DOp>(rewriter.clone(*convOriginal));

    // Set new conv output shape with old slice width
    RankedTensorType newConvType = RankedTensorType::get(
        {convOriginalInputShape[0],
         (paddedHeight + strideWidth - filterWidth) / strideWidth,
         (paddedWidth + strideWidth - filterWidth) / strideWidth,
         convOriginalOutputShape[3]},
        convOriginal.output().getType().cast<ShapedType>().getElementType());
    convReplacement->getResult(0).setType(newConvType);

    if (convOriginal.padding() == "VALID") {
      // Connect new conv's input to new strided slice
      convReplacement.setOperand(0, stridedSliceReplacement);

    } else if (convOriginal.padding() == "SAME") {
      // Connect new conv's input to new strided slice
      convReplacement.setOperand(0, padOp);
    }
    convReplacement->setAttr(rewriter.getStringAttr("padding"),
                             rewriter.getStringAttr("VALID"));

    // replace stided slice with new strided slice -> new conv
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
