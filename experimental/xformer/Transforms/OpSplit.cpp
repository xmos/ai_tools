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

    // Check for invalid cases and return
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


    int32_t stridesAttr[4] = {1, 1, 1, 1};
    auto stridesConstantOp = rewriter.create<arith::ConstantOp>(
        convReplacement.getLoc(), rewriter.getI32TensorAttr(stridesAttr));

    int32_t begin_mask, end_mask, ellipsis_mask, new_axis_mask,
        shrink_axis_mask;
    begin_mask = end_mask = ellipsis_mask = new_axis_mask = shrink_axis_mask =
        0;

    SmallVector<Value> stridedSliceOps;

    auto convOutput = convOriginal.output();
    auto outputType = convOutput.getType().dyn_cast<RankedTensorType>();
    int32_t outputHeight = outputType.getDimSize(1);
    int32_t outputWidth = outputType.getDimSize(2);
    int32_t outputDepth = outputType.getDimSize(3);

    int numSplits = 4;
    int32_t sliceWidth = outputWidth / numSplits;
    int32_t sliceWidthRemainder = outputWidth % numSplits;
  
    int32_t prevEndIndex = 0;
    for (size_t i = 0; i < numSplits; i++)
    {
      int32_t currentSliceWidth = sliceWidth;
      if (i < sliceWidthRemainder)
        currentSliceWidth++; 
      
      RankedTensorType newOutputType = RankedTensorType::get(
        {1, outputHeight, currentSliceWidth,
         outputDepth},
        convOutput.getType().cast<ShapedType>().getElementType());

      int32_t beginAttr[4] = {0, 0, prevEndIndex, 0};
      auto beginConstantOp = rewriter.create<arith::ConstantOp>(
        convReplacement.getLoc(), rewriter.getI32TensorAttr(beginAttr));

      int32_t endIndex = prevEndIndex + currentSliceWidth;
      int32_t endAttr[4] = {1, outputHeight, endIndex, outputDepth};
      auto endConstantOp = rewriter.create<arith::ConstantOp>(
        convReplacement.getLoc(), rewriter.getI32TensorAttr(endAttr));
      prevEndIndex = endIndex;

      auto stridedSliceOp = rewriter.create<TFL::StridedSliceOp>(
        convReplacement.getLoc(), newOutputType, convReplacement,
        beginConstantOp, endConstantOp, stridesConstantOp, begin_mask,
        end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);

      stridedSliceOp->setAttr(opSplitLabel, rewriter.getUnitAttr());

      stridedSliceOps.push_back(stridedSliceOp.getResult());
    }

    StringRef fused_activation_function = "NONE";

    auto newConcatOp = rewriter.create<TFL::ConcatenationOp>(
        convReplacement.getLoc(), convOutput.getType(), stridedSliceOps, 2,
        fused_activation_function);

    rewriter.replaceOp(convOriginal, newConcatOp.output());

    return success();
  }
};

struct RaiseStridedSlicePattern : public OpRewritePattern<TFL::StridedSliceOp> {
  using OpRewritePattern<TFL::StridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::StridedSliceOp stridedSlice,
                                PatternRewriter &rewriter) const override {
    if (!((stridedSlice->hasAttr(opSplitLabel))))
      return failure();

    static constexpr char raisedStridedSliceLabel[] = "raisedStridedSliceLabel";
    if (stridedSlice->hasAttr(raisedStridedSliceLabel))
      return failure();

    auto stridedSliceOutputShape =
        stridedSlice.output().getType().cast<RankedTensorType>().getShape();

    auto convOriginal =
        llvm::cast<TFL::Conv2DOp>(stridedSlice.input().getDefiningOp());
    auto convOriginalInputShape =
        convOriginal.input().getType().cast<RankedTensorType>().getShape();
    auto convOriginalOutputShape =
        convOriginal.output().getType().cast<RankedTensorType>().getShape();
    auto filterWidth = convOriginal.filter().getType().dyn_cast<RankedTensorType>().getDimSize(
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

    auto outputWidth = stridedSliceOutputShape[2];

    int32_t newOutputWidth = outputWidth * strideWidth - strideWidth + filterWidth;

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

    if (convOriginal.padding() == "VALID"){
      convReplacement.setOperand(0, stridedSliceReplacement);
    }
    else {
      // Find padding values
      auto inputType = convReplacement.input().getType().dyn_cast<RankedTensorType>();
      auto filterType = convReplacement.filter().getType().dyn_cast<RankedTensorType>();
      auto inputHeight = inputType.getDimSize(1);
      auto inputWidth = inputType.getDimSize(2);
      auto filterHeight = filterType.getDimSize(1);
      auto filterWidth = filterType.getDimSize(2);

      int64_t newHeight, newWidth;
      int64_t padTop, padBottom, padLeft, padRight;
      tensorflow::GetWindowedOutputSizeVerboseV2(
          inputHeight, filterHeight, convReplacement.dilation_height_factor(),
          convReplacement.stride_height(), tensorflow::Padding::SAME, &newHeight,
          &padTop, &padBottom)
      tensorflow::GetWindowedOutputSizeVerboseV2(
              inputWidth, filterWidth, convReplacement.dilation_width_factor(),
              convReplacement.stride_width(), tensorflow::Padding::SAME, &newWidth,
              &padLeft, &padRight)

      if (!matchPattern(stridedSlice.begin(), m_Constant(&attr))) {
        return failure();
      }
      auto beginIndex = attr.getValues<int32_t>()[2];
      if (beginIndex != 0) {
        padLeft = 0;
      }

      std::vector<int32_t> paddingValues{0,
                                     0,
                                     static_cast<int>(padTop),
                                     static_cast<int>(padBottom),
                                     static_cast<int>(padLeft),
                                     static_cast<int>(padRight),
                                     0,
                                     0};

      auto stridedSliceReplacementOutputShape =
        stridedSliceReplacement.output().getType().cast<RankedTensorType>().getShape();

      int batch = stridedSliceReplacementOutputShape[0] + paddingValues[0] + paddingValues[1];
      int height = stridedSliceReplacementOutputShape[0] + paddingValues[2] + paddingValues[3];
      int width = stridedSliceReplacementOutputShape[0] + paddingValues[4] + paddingValues[5];
      int depth = stridedSliceReplacementOutputShape[0] + paddingValues[6] + paddingValues[7];

      auto paddedResultType = RankedTensorType::get(
        {batch, height, width, depth},
        stridedSliceReplacement.output().getType().cast<ShapedType>().getElementType());

      RankedTensorType paddingsType =
        RankedTensorType::get({4, 2}, rewriter.getI32Type());
      
      Value paddings = rewriter.create<TFL::ConstOp>(
        stridedSlice.getLoc(),
        DenseIntElementsAttr::get(paddingsType, paddingsValues));
      
      Value padOp = rewriter.create<TFL::PadOp>(
        stridedSlice.getLoc(), paddedResultType, stridedSliceReplacement, paddings);

      convReplacement.setOperand(0, padOp);
      convReplacement.padding() = "VALID";
    }

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
