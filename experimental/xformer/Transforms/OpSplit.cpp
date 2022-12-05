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

    if (convOriginal.stride_h() != 1)
      return failure();

    if (convOriginal.stride_w() != 1)
      return failure();

    auto convOutput = convOriginal.output();

    // Extract args from the op
    auto outputType = convOutput.getType().dyn_cast<RankedTensorType>();
    int32_t outputHeight = outputType.getDimSize(1);
    int32_t outputWidth = outputType.getDimSize(2);
    int32_t outputDepth = outputType.getDimSize(3);

    // Only 2 splits implemented
    int numSplits = 2;

    // outputWidth must be divisible by numSplits
    if (outputWidth % numSplits != 0)
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

    auto outputShape = convOutput.getType().cast<RankedTensorType>().getShape();

    RankedTensorType newOutputType = RankedTensorType::get(
        {outputShape[0], outputShape[1], outputShape[2] / numSplits,
         outputShape[3]},
        convOutput.getType().cast<ShapedType>().getElementType());

    int32_t sliceIndex = outputWidth / numSplits;

    int32_t beginAttr0[4] = {0, 0, 0, 0};
    auto beginConstantOp0 = rewriter.create<arith::ConstantOp>(
        convReplacement.getLoc(), rewriter.getI32TensorAttr(beginAttr0));
    
    int32_t endAttr0[4] = {1, outputHeight, sliceIndex, outputDepth};
    auto endConstantOp0 = rewriter.create<arith::ConstantOp>(
        convReplacement.getLoc(), rewriter.getI32TensorAttr(endAttr0));

    int32_t beginAttr1[4] = {0, 0, sliceIndex, 0};
    auto beginConstantOp1 = rewriter.create<arith::ConstantOp>(
        convReplacement.getLoc(), rewriter.getI32TensorAttr(beginAttr1));

    int32_t endAttr1[4] = {1, outputHeight, outputWidth, outputDepth};
    auto endConstantOp1 = rewriter.create<arith::ConstantOp>(
        convReplacement.getLoc(), rewriter.getI32TensorAttr(endAttr1));

    int32_t stridesAttr[4] = {1, 1, 1, 1};
    auto stridesConstantOp = rewriter.create<arith::ConstantOp>(
        convReplacement.getLoc(), rewriter.getI32TensorAttr(stridesAttr));

    int32_t begin_mask, end_mask, ellipsis_mask, new_axis_mask,
        shrink_axis_mask;
    begin_mask = end_mask = ellipsis_mask = new_axis_mask = shrink_axis_mask =
        0;

    auto stridedSliceOp0 = rewriter.create<TFL::StridedSliceOp>(
        convReplacement.getLoc(), newOutputType, convReplacement,
        beginConstantOp0, endConstantOp0, stridesConstantOp, begin_mask,
        end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);

    stridedSliceOp0->setAttr(opSplitLabel, rewriter.getUnitAttr());

    SmallVector<Value> stridedSliceOps;
    stridedSliceOps.push_back(stridedSliceOp0.getResult());

    auto stridedSliceOp1 = rewriter.create<TFL::StridedSliceOp>(
        convReplacement.getLoc(), newOutputType, convReplacement,
        beginConstantOp1, endConstantOp1, stridesConstantOp, begin_mask,
        end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);

    stridedSliceOp1->setAttr(opSplitLabel, rewriter.getUnitAttr());

    stridedSliceOps.push_back(stridedSliceOp1.getResult());

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

    auto convOriginal =
        llvm::cast<TFL::Conv2DOp>(stridedSlice.input().getDefiningOp());

    auto convOriginalOutputShape =
        convOriginal.output().getType().cast<RankedTensorType>().getShape();
    auto stridedSliceOutputShape =
        stridedSlice.output().getType().cast<RankedTensorType>().getShape();

    auto convOriginalInputShape =
        convOriginal.input().getType().cast<RankedTensorType>().getShape();

    // If Padding = Valid
    // OutputWidth =ceil(  (InputWidth – FilterWidth + 1) /  StrideWidth )

    // floor(OutputWidth) = (InputWidth – FilterWidth + 1) /  StrideWidth
    // OutputWidth is always an integer, no need for floor
    // OutputWidth = (InputWidth – FilterWidth + 1) /  StrideWidth
    // OutputWidth * StrideWidth = InputWidth – FilterWidth + 1
    // OutputWidth * StrideWidth - 1 = InputWidth – FilterWidth
    // OutputWidth * StrideWidth - 1 = InputWidth – FilterWidth
    // OutputWidth * StrideWidth - 1 + FilterWidth = InputWidth

    // InputWidth = OutputWidth * StrideWidth - 1 + FilterWidth

    // If Padding = Same
    // OutputWidth = ceil( InputWidth / StrideWidth)
    // OutputWidth is always an integer, no need for floor
    // OutputWidth = InputWidth / StrideWidth
    // OutputWidth * StrideWidth = InputWidth
    // InputWidth = OutputWidth * StrideWidth

    auto filterWidth =
        convOriginal.filter().getType().dyn_cast<RankedTensorType>().getDimSize(
            2);
    auto strideWidth = convOriginal.stride_w();

    DenseElementsAttr attr;
    if (!matchPattern(stridedSlice.begin(), m_Constant(&attr))) {
      return failure();
    }
    auto beginWidth = attr.getValues<int32_t>()[2];
    if (!matchPattern(stridedSlice.end(), m_Constant(&attr))) {
      return failure();
    }
    auto endWidth = attr.getValues<int32_t>()[2];

    auto outputWidth = endWidth - beginWidth;

    int32_t newOutputWidth;
    if (convOriginal.padding() == "VALID")
      newOutputWidth = outputWidth * strideWidth - 1 + filterWidth;
    else if (convOriginal.padding() == "SAME")
      newOutputWidth = outputWidth * strideWidth;

    RankedTensorType newStridedSliceType = RankedTensorType::get(
        {convOriginalInputShape[0], convOriginalInputShape[1], newOutputWidth,
         convOriginalInputShape[3]},
        convOriginal.input().getType().cast<ShapedType>().getElementType());

    arith::ConstantOp beginConstantOp;
    arith::ConstantOp endConstantOp;

    if (beginWidth == 0) {
      int32_t beginAttr[4] = {0, 0, 0, 0};

      beginConstantOp = rewriter.create<arith::ConstantOp>(
          stridedSlice.getLoc(), rewriter.getI32TensorAttr(beginAttr));

      int32_t newEndWidth;
      if (convOriginal.padding() == "VALID")
        newEndWidth = endWidth * strideWidth - 1 + filterWidth;
      else if (convOriginal.padding() == "SAME")
        newEndWidth = endWidth * strideWidth;

      int32_t endAttr[4] = {1, static_cast<int32_t>(convOriginalInputShape[1]),
                            newEndWidth,
                            static_cast<int32_t>(convOriginalInputShape[3])};
      endConstantOp = rewriter.create<arith::ConstantOp>(
          stridedSlice.getLoc(), rewriter.getI32TensorAttr(endAttr));

    } else {
      int32_t newBeginWidth;
      if (convOriginal.padding() == "VALID")
        newBeginWidth = beginWidth * strideWidth - 1 + filterWidth;
      else if (convOriginal.padding() == "SAME")
        newBeginWidth = beginWidth * strideWidth;

      int32_t beginAttr[4] = {0, 0, newBeginWidth, 0};

      beginConstantOp = rewriter.create<arith::ConstantOp>(
          stridedSlice.getLoc(), rewriter.getI32TensorAttr(beginAttr));

      int32_t endAttr[4] = {1, static_cast<int32_t>(convOriginalInputShape[1]),
                            static_cast<int32_t>(convOriginalInputShape[2]),
                            static_cast<int32_t>(convOriginalInputShape[3])};
      endConstantOp = rewriter.create<arith::ConstantOp>(
          stridedSlice.getLoc(), rewriter.getI32TensorAttr(endAttr));
    }

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
