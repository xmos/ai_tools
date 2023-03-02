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
static constexpr char raisedStridedSliceLabel[] = "raisedStridedSliceLabel";

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

template <typename TargetOp>
struct OpSplitHorizontalPattern : public OpRewritePattern<TargetOp> {
  using OpRewritePattern<TargetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TargetOp targetOp,
                                PatternRewriter &rewriter) const override {
    // Do not split ops already split
    if (targetOp->hasAttr(opSplitLabel))
      return failure();

    //
    // Check for invalid cases and return
    //
    auto outputElementalType = targetOp.output()
                                   .getType()
                                   .template cast<ShapedType>()
                                   .getElementType();
    // Output type must be QI8
    if (!(outputElementalType.template isa<quant::QuantizedType>() &&
          outputElementalType.template cast<quant::QuantizedType>()
              .isSigned() &&
          outputElementalType.template cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    // Data from target op needed later
    auto targetOutput = targetOp.output();
    auto outputType =
        targetOutput.getType().template dyn_cast<RankedTensorType>();
    int32_t outputHeight = outputType.getDimSize(1);
    int32_t outputWidth = outputType.getDimSize(2);
    int32_t outputDepth = outputType.getDimSize(3);
    int32_t outputSize = outputHeight * outputWidth * outputDepth;

    // Number chosen for testing purposes
    // Actual number will depend on application
    int32_t splitTensorSize = 98304;
    // Only op split if output size is too big
    if (outputSize < 2 * splitTensorSize)
      return failure();

    // Clone the op as we want to replace it with the same op type but with
    // strided slices and concat inserted after it
    auto targetReplacement = llvm::cast<TargetOp>(rewriter.clone(*targetOp));

    // Apply label, so that the same op is not rewritten a second time.
    targetReplacement->setAttr(opSplitLabel, rewriter.getUnitAttr());

    // variables that are the same for all strided slices to be created
    int32_t stridesAttr[4] = {1, 1, 1, 1};
    auto stridesConstantOp = rewriter.create<arith::ConstantOp>(
        targetReplacement.getLoc(), rewriter.getI32TensorAttr(stridesAttr));
    int32_t begin_mask, end_mask, ellipsis_mask, new_axis_mask,
        shrink_axis_mask;
    begin_mask = end_mask = ellipsis_mask = new_axis_mask = shrink_axis_mask =
        0;

    // Will hold strided slice op to insert after target op
    SmallVector<Value> stridedSliceOps;

    // The number of splits is determined by target op output size
    int numSplits = ceil(outputSize / splitTensorSize);

    int32_t sliceHeight = outputHeight / numSplits;

    // The remainder will be distributed between the splits
    // to keep them about the same size
    int32_t sliceHeightRemainder = outputHeight % numSplits;

    // For loop uses end index of previous strided slice created
    // needs to initalized to zero for first slice
    int32_t prevEndIndex = 0;

    // Loops creates strided slices with correct params
    for (size_t i = 0; i < numSplits; i++) {
      // Distributes remainder between slices
      int32_t currentSliceHeight = sliceHeight;
      if (i < sliceHeightRemainder)
        currentSliceHeight++;

      // Describes output tensor of strided slice
      // Only currentSliceHeight can be unique to each strided slice
      RankedTensorType stridedSliceOutputType = RankedTensorType::get(
          {1, currentSliceHeight, outputWidth, outputDepth},
          targetOutput.getType().template cast<ShapedType>().getElementType());

      // Start where the prev slice ended
      int32_t beginAttr[4] = {0, prevEndIndex, 0, 0};
      auto beginConstantOp = rewriter.create<arith::ConstantOp>(
          targetReplacement.getLoc(), rewriter.getI32TensorAttr(beginAttr));

      // End is start + slice width
      int32_t endIndex = prevEndIndex + currentSliceHeight;
      // Go to end of tensor for all dims except width
      int32_t endAttr[4] = {1, endIndex, outputWidth, outputDepth};
      auto endConstantOp = rewriter.create<arith::ConstantOp>(
          targetReplacement.getLoc(), rewriter.getI32TensorAttr(endAttr));
      prevEndIndex = endIndex;

      auto stridedSliceOp = rewriter.create<TFL::StridedSliceOp>(
          targetReplacement.getLoc(), stridedSliceOutputType, targetReplacement,
          beginConstantOp, endConstantOp, stridesConstantOp, begin_mask,
          end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);

      // Add label, for safety when raising slice later
      stridedSliceOp->setAttr(opSplitLabel, rewriter.getUnitAttr());

      // Store created strided slice op to use as input to concat
      stridedSliceOps.push_back(stridedSliceOp.getResult());
    }

    // Concat op does not have activation function
    StringRef fused_activation_function = "NONE";

    // Create concat op that concats on dim 1, width
    auto concatOp = rewriter.create<TFL::ConcatenationOp>(
        targetReplacement.getLoc(), targetOutput.getType(), stridedSliceOps, 1,
        fused_activation_function);

    // Replace target op with
    // Cloned target op -> Strided Slices -> Concat
    rewriter.replaceOp(targetOp, concatOp.output());

    return success();
  }
};

struct RaiseStridedSliceHorizontalAddPattern
    : public OpRewritePattern<TFL::StridedSliceOp> {
  using OpRewritePattern<TFL::StridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::StridedSliceOp stridedSlice,
                                PatternRewriter &rewriter) const override {
    // Only raise slices that have been inserted with op split pass
    if (!((stridedSlice->hasAttr(opSplitLabel))))
      return failure();

    // Do not raise slices that have already been raised
    if (stridedSlice->hasAttr(raisedStridedSliceLabel))
      return failure();

    if (!isa<TFL::AddOp>(stridedSlice.input().getDefiningOp())) {
      return failure();
    }

    // Get data from conv needed to raise strided slice
    auto addOriginal =
        llvm::cast<TFL::AddOp>(stridedSlice.input().getDefiningOp());

    auto addOriginalOutput =
        addOriginal.output().getType().template cast<RankedTensorType>();
    auto outputHeight = addOriginalOutput.getDimSize(1);
    auto outputWidth = addOriginalOutput.getDimSize(2);
    auto outputChannels = addOriginalOutput.getDimSize(3);

    // get end index of strided slice
    DenseElementsAttr attr;
    if (!matchPattern(stridedSlice.end(), m_Constant(&attr))) {
      return failure();
    }
    auto endIndex = attr.getValues<int32_t>()[1];
    auto endIndices = attr.getValues<int32_t>();

    // Get original slice's output width
    auto stridedSliceOutput =
        stridedSlice.output().getType().cast<RankedTensorType>();
    auto stridedSliceOutputHeight = stridedSliceOutput.getDimSize(1);
    auto stridedSliceOutputWidth = stridedSliceOutput.getDimSize(2);
    auto stridedSliceOutputChannels = stridedSliceOutput.getDimSize(3);

    // Set end tensor for slice to be above conv with new end index
    int32_t endAttr[4] = {1, static_cast<int32_t>(endIndices[1]),
                          static_cast<int32_t>(endIndices[2]),
                          static_cast<int32_t>(endIndices[3])};
    auto endConstantOp = rewriter.create<arith::ConstantOp>(
        stridedSlice.getLoc(), rewriter.getI32TensorAttr(endAttr));

    // Set begin tensor to zero for all dims except width
    // set width to new end index - new output width
    int32_t beginAttr[4] = {0, static_cast<int32_t>(endIndex - outputHeight), 0,
                            0};
    auto beginConstantOp = rewriter.create<arith::ConstantOp>(
        stridedSlice.getLoc(), rewriter.getI32TensorAttr(beginAttr));

    // Create new strided slice for above conv
    auto stridedSliceLHS =
        llvm::cast<TFL::StridedSliceOp>(rewriter.clone(*stridedSlice));
    stridedSliceLHS.setOperand(0, addOriginal.lhs());
    RankedTensorType stridedSliceLHSType = RankedTensorType::get(
        {1, stridedSliceOutputHeight, stridedSliceOutputWidth,
         stridedSliceOutputChannels},
        addOriginal.lhs()
            .getType()
            .template cast<ShapedType>()
            .getElementType());
    stridedSliceLHS->getResult(0).setType(stridedSliceLHSType);
    // Label it as raised so it is not raised again
    stridedSliceLHS->setAttr(raisedStridedSliceLabel, rewriter.getUnitAttr());

    // Create new strided slice for above conv
    auto stridedSliceRHS =
        llvm::cast<TFL::StridedSliceOp>(rewriter.clone(*stridedSlice));
    stridedSliceRHS.setOperand(0, addOriginal.rhs());
    RankedTensorType stridedSliceRHSType = RankedTensorType::get(
        {1, stridedSliceOutputHeight, stridedSliceOutputWidth,
         stridedSliceOutputChannels},
        addOriginal.rhs()
            .getType()
            .template cast<ShapedType>()
            .getElementType());
    stridedSliceRHS->getResult(0).setType(stridedSliceRHSType);
    // Label it as raised so it is not raised again
    stridedSliceRHS->setAttr(raisedStridedSliceLabel, rewriter.getUnitAttr());

    auto addReplacement = llvm::cast<TFL::AddOp>(rewriter.clone(*addOriginal));
    RankedTensorType addReplacementType = RankedTensorType::get(
        {1, stridedSliceOutputHeight, stridedSliceOutputWidth,
         stridedSliceOutputChannels},
        addOriginal.output()
            .getType()
            .template cast<ShapedType>()
            .getElementType());
    addReplacement->getResult(0).setType(addReplacementType);
    addReplacement.setOperand(0, stridedSliceLHS);
    addReplacement.setOperand(1, stridedSliceRHS);

    // replace strided slice with new strided slice -> new conv
    // or new strided slice -> pad -> new conv
    rewriter.replaceOp(stridedSlice, addReplacement.output());

    return success();
  }
};

template <typename ConvOp>
struct RaiseStridedSliceHorizontalPattern
    : public OpRewritePattern<TFL::StridedSliceOp> {
  using OpRewritePattern<TFL::StridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::StridedSliceOp stridedSlice,
                                PatternRewriter &rewriter) const override {
    // Only raise slices that have been inserted with op split pass
    if (!((stridedSlice->hasAttr(opSplitLabel))))
      return failure();

    // Do not raise slices that have already been raised
    if (stridedSlice->hasAttr(raisedStridedSliceLabel))
      return failure();

    if (!isa<ConvOp>(stridedSlice.input().getDefiningOp())) {
      return failure();
    }

    // Get data from conv needed to raise strided slice
    auto convOriginal =
        llvm::cast<ConvOp>(stridedSlice.input().getDefiningOp());

    auto convOriginalInput =
        convOriginal.input().getType().template cast<RankedTensorType>();
    auto inputHeight = convOriginalInput.getDimSize(1);
    auto inputWidth = convOriginalInput.getDimSize(2);
    auto inputChannels = convOriginalInput.getDimSize(3);

    auto convOriginalOutput =
        convOriginal.output().getType().template cast<RankedTensorType>();
    auto outputHeight = convOriginalOutput.getDimSize(1);
    auto outputChannels = convOriginalOutput.getDimSize(3);

    auto filterType =
        convOriginal.filter().getType().template dyn_cast<RankedTensorType>();
    auto filterHeight = filterType.getDimSize(1);
    auto filterWidth = filterType.getDimSize(2);

    auto strideHeight = convOriginal.stride_h();
    auto strideWidth = convOriginal.stride_w();

    // get end index of strided slice
    DenseElementsAttr attr;
    if (!matchPattern(stridedSlice.end(), m_Constant(&attr))) {
      return failure();
    }
    auto endIndex = attr.getValues<int32_t>()[1];

    // Get original slice's output width
    auto stridedSliceOutput =
        stridedSlice.output().getType().cast<RankedTensorType>();
    auto stridedSliceOutputHeight = stridedSliceOutput.getDimSize(1);

    int32_t newEndIndex;
    int32_t newOutputHeight;
    int64_t padTop, padBottom, padLeft, padRight;
    padTop = padBottom = padLeft = padRight = 0;

    int dilation_h_factor = 1;
    int dilation_w_factor = 1;
    int64_t newHeight, newWidth;
    tensorflow::Padding opPadding = convOriginal.padding() == "VALID"
                                        ? tensorflow::Padding::VALID
                                        : tensorflow::Padding::SAME;
    // Get pad values for conv op
    if (tensorflow::GetWindowedOutputSizeVerboseV2(
            inputHeight, filterHeight, dilation_h_factor, strideHeight,
            opPadding, &newHeight, &padTop,
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
      newEndIndex = endIndex * strideHeight - strideHeight + filterHeight;

    } else if (convOriginal.padding() == "SAME") {

      // Get begin index for slice
      if (!matchPattern(stridedSlice.begin(), m_Constant(&attr))) {
        return failure();
      }
      auto beginIndex = attr.getValues<int32_t>()[1];

      // Check if this is left most split
      if (beginIndex == 0) {
        // Calculate new end index for slice after being raised above conv
        newEndIndex =
            endIndex * strideHeight - strideHeight + filterHeight - padTop;

        // Left split is not padded on right
        padBottom = 0;

      } else if (endIndex == outputHeight) { // end
        // Calculate new end index for slice after being raised above conv
        newEndIndex = endIndex * strideHeight - strideHeight + filterHeight -
                      padTop - padBottom;

        // Right split is not padded on left
        padTop = 0;

      } else { // beginIndex not 0 or end
        // Calculate new end index for slice after being raised above conv
        newEndIndex =
            endIndex * strideHeight - strideHeight + filterHeight - padTop;

        // Center splits are not padded on left or right
        padTop = 0;
        padBottom = 0;
      }
    }

    // Set end tensor for slice to be above conv with new end index
    int32_t endAttr[4] = {1, static_cast<int32_t>(newEndIndex),
                          static_cast<int32_t>(inputWidth),
                          static_cast<int32_t>(inputChannels)};
    auto endConstantOp = rewriter.create<arith::ConstantOp>(
        stridedSlice.getLoc(), rewriter.getI32TensorAttr(endAttr));

    // Calculate new output width after raising slice above conv
    newOutputHeight = stridedSliceOutputHeight * strideHeight - strideHeight +
                      filterHeight - padTop - padBottom;

    // Set begin tensor to zero for all dims except width
    // set width to new end index - new output width
    int32_t beginAttr[4] = {
        0, static_cast<int32_t>(newEndIndex - newOutputHeight), 0, 0};
    auto beginConstantOp = rewriter.create<arith::ConstantOp>(
        stridedSlice.getLoc(), rewriter.getI32TensorAttr(beginAttr));

    // New strided slice output shape is conv input shape except width
    // The new calculated output width is used for width
    RankedTensorType newStridedSliceType =
        RankedTensorType::get({1, newOutputHeight, inputWidth, inputChannels},
                              convOriginal.input()
                                  .getType()
                                  .template cast<ShapedType>()
                                  .getElementType());

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

    // Adjust shape for padding
    // For valid conv the shapes will not change since pad values are zero
    auto paddedHeight = newOutputHeight + padTop + padBottom;
    auto paddedWidth = inputWidth + padLeft + padRight;

    // If padding is same, create pad op to extract padding
    TFL::PadOp padOp;
    if (convOriginal.padding() == "SAME") {
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

      auto paddedResultType =
          RankedTensorType::get({1, paddedHeight, paddedWidth, inputChannels},
                                convOriginal.input()
                                    .getType()
                                    .template cast<ShapedType>()
                                    .getElementType());

      padOp =
          rewriter.create<TFL::PadOp>(stridedSlice.getLoc(), paddedResultType,
                                      stridedSliceReplacement, paddings);
    }

    auto convReplacement = llvm::cast<ConvOp>(rewriter.clone(*convOriginal));

    RankedTensorType newConvType = RankedTensorType::get(
        {1, (paddedHeight + strideHeight - filterHeight) / strideHeight,
         (paddedWidth + strideWidth - filterWidth) / strideWidth,
         outputChannels},
        convOriginal.output()
            .getType()
            .template cast<ShapedType>()
            .getElementType());
    convReplacement->getResult(0).setType(newConvType);

    // if valid padding no need for pad op, connect to strided slice
    // else connect to pad op
    if (convOriginal.padding() == "VALID") {
      // Connect new conv's input to new strided slice
      convReplacement.setOperand(0, stridedSliceReplacement);

    } else if (convOriginal.padding() == "SAME") {
      // Connect new conv's input to pad op
      convReplacement.setOperand(0, padOp);

      // Change padding on cloned conv to valid since
      // padding was extracted to pad op
      convReplacement->setAttr("padding", rewriter.getStringAttr("VALID"));
    }

    // replace strided slice with new strided slice -> new conv
    // or new strided slice -> pad -> new conv
    rewriter.replaceOp(stridedSlice, convReplacement.output());

    return success();
  }
};

void OpSplit::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();

  RewritePatternSet patterns(ctx);

  patterns.insert<OpSplitHorizontalPattern<TFL::Conv2DOp>>(ctx);
  patterns.insert<OpSplitHorizontalPattern<TFL::DepthwiseConv2DOp>>(ctx);
  patterns.insert<OpSplitHorizontalPattern<TFL::AddOp>>(ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  RewritePatternSet patterns2(ctx);

  patterns2.insert<RaiseStridedSliceHorizontalPattern<TFL::Conv2DOp>>(ctx);
  patterns2.insert<RaiseStridedSliceHorizontalPattern<TFL::DepthwiseConv2DOp>>(
      ctx);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns2));

  RewritePatternSet patterns3(ctx);
  patterns3.insert<RaiseStridedSliceHorizontalAddPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns3));
}
} // namespace

// Creates an instance of the OpSplit pass.
std::unique_ptr<OperationPass<func::FuncOp>> createOpSplitPass() {
  return std::make_unique<OpSplit>();
}

static PassRegistration<OpSplit> pass;

} // namespace xcore
} // namespace mlir
