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
    auto filterHeight =
        convOriginal.filter().getType().dyn_cast<RankedTensorType>().getDimSize(
            1);
    if (filterHeight != 1)
      return failure();

    auto filterWidth =
        convOriginal.filter().getType().dyn_cast<RankedTensorType>().getDimSize(
            2);
    if (filterWidth != 1)
      return failure();

    if (convOriginal.stride_h() != 1)
      return failure();

    if (convOriginal.stride_w() != 1)
      return failure();

    if (convOriginal.padding() != "VALID")
      return failure();

    auto inputWidth =
        convOriginal.input().getType().dyn_cast<RankedTensorType>().getDimSize(
            2);
    // Only handles inputWidth dimensions divisible by 2
    if (inputWidth % 2 != 0)
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

    auto convOutput = convReplacement.output();

    // Extract args from the op
    auto outputType = convOutput.getType().dyn_cast<RankedTensorType>();
    int32_t outputHeight = outputType.getDimSize(1);
    int32_t outputWidth = outputType.getDimSize(2);
    int32_t outputDepth = outputType.getDimSize(3);

    auto outputShape = convOutput.getType().cast<RankedTensorType>().getShape();

    int numSplits = 2;
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

    RankedTensorType newStridedSliceType = RankedTensorType::get(
        {stridedSliceOutputShape[0], stridedSliceOutputShape[1],
         stridedSliceOutputShape[2], convOriginalInputShape[3]},
        convOriginal.input().getType().cast<ShapedType>().getElementType());

    auto stridedSliceReplacement =
        llvm::cast<TFL::StridedSliceOp>(rewriter.clone(*stridedSlice));
    stridedSliceReplacement->setAttr(raisedStridedSliceLabel,
                                     rewriter.getUnitAttr());
    stridedSliceReplacement->getResult(0).setType(newStridedSliceType);
    stridedSliceReplacement.setOperand(0, convOriginal.input());

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
