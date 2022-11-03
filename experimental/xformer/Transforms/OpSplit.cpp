// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "lib_nn/api/MemCpyFn.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

namespace {
// OpSplit
struct OpSplit : public PassWrapper<OpSplit, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpSplit)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-opsplit"; }
  StringRef getDescription() const final { return "OpSplit."; }
  void runOnOperation() override;
};

struct OpSplitPattern : public OpRewritePattern<TFL::Conv2DOp> {
  using OpRewritePattern<TFL::Conv2DOp>::OpRewritePattern;

  // LogicalResult matchAndRewrite(DummyStridedSliceOp stridedSliceOriginal,
  LogicalResult matchAndRewrite(TFL::Conv2DOp convOriginal,
                                PatternRewriter &rewriter) const override {

    static constexpr char kSplitLabel[] = "__split_op__";

    // Do not split ops already split
    if (convOriginal->hasAttr(kSplitLabel))
      return failure();

    auto convReplacement = rewriter.create<TFL::Conv2DOp>(
        convOriginal.getLoc(), convOriginal.getType(), convOriginal.input(),
        convOriginal.filter(), convOriginal.bias(),
        convOriginal.dilation_h_factor(), convOriginal.dilation_w_factor(),
        convOriginal.fused_activation_function(), convOriginal.padding(),
        convOriginal.stride_h(), convOriginal.stride_w());

    // Apply label, so that the same op is not rewritten a second time.
    convReplacement->setAttr(kSplitLabel, rewriter.getUnitAttr());

    auto convOriginalOutput = convOriginal.output();

    // Extract args from the op
    auto outputType = convOriginalOutput.getType().dyn_cast<RankedTensorType>();
    int32_t outputHeight = outputType.getDimSize(1);
    int32_t outputWidth = outputType.getDimSize(2);
    int32_t outputDepth = outputType.getDimSize(3);

    auto outputShape = convOriginalOutput.getType().cast<RankedTensorType>().getShape();
    ArrayRef newOutputShape = {outputShape[0],outputShape[1],outputShape[2]/2,outputShape[3]};

    RankedTensorType newOutputType = RankedTensorType::get(
        newOutputShape,
        convOriginalOutput.getType().cast<ShapedType>().getElementType());

    int32_t sliceIndex = outputWidth / 2;

    int32_t beginAttr0[4] = {0, 0, 0, 0};
    auto beginConstantOp0 = rewriter.create<arith::ConstantOp>(
        convOriginal.getLoc(), rewriter.getI32TensorAttr(beginAttr0));

    int32_t endAttr0[4] = {1, outputHeight, sliceIndex, outputDepth};
    auto endConstantOp0 = rewriter.create<arith::ConstantOp>(
        convOriginal.getLoc(), rewriter.getI32TensorAttr(endAttr0));
    
    int32_t beginAttr1[4] = {0, 0, sliceIndex, 0};
    auto beginConstantOp1 = rewriter.create<arith::ConstantOp>(
        convOriginal.getLoc(), rewriter.getI32TensorAttr(beginAttr1));

    int32_t endAttr1[4] = {1, outputHeight, outputWidth, outputDepth};
    auto endConstantOp1 = rewriter.create<arith::ConstantOp>(
        convOriginal.getLoc(), rewriter.getI32TensorAttr(endAttr1));

    int32_t stridesAttr[4] = {1, 1, 1, 1};
    auto stridesConstantOp = rewriter.create<arith::ConstantOp>(
        convOriginal.getLoc(), rewriter.getI32TensorAttr(stridesAttr));

    int32_t begin_mask, end_mask, ellipsis_mask, new_axis_mask,
        shrink_axis_mask;
    begin_mask = end_mask = ellipsis_mask = new_axis_mask = shrink_axis_mask =
        0;

    auto stridedSliceOp0 = rewriter.create<TFL::StridedSliceOp>(
        convOriginal.getLoc(), newOutputType, convReplacement,
        beginConstantOp0, endConstantOp0, stridesConstantOp, begin_mask,
        end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);

    SmallVector<Value> stridedSliceOps;
    stridedSliceOps.push_back(stridedSliceOp0.getResult());

    auto stridedSliceOp1 = rewriter.create<TFL::StridedSliceOp>(
        convOriginal.getLoc(), newOutputType, convReplacement,
        beginConstantOp1, endConstantOp1, stridesConstantOp, begin_mask,
        end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);

    stridedSliceOps.push_back(stridedSliceOp1.getResult());


    StringRef fused_activation_function = "NONE";

    auto newConcatOp = rewriter.create<TFL::ConcatenationOp>(
        convOriginal.getLoc(), convOriginalOutput.getType(), stridedSliceOps, 2,
        fused_activation_function);

    rewriter.replaceOp(convOriginal, newConcatOp.output());

    return success();
  }
};

void OpSplit::runOnOperation() {
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();
  patterns.insert<OpSplitPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the OpSplit pass.
std::unique_ptr<OperationPass<func::FuncOp>> createOpSplitPass() {
  return std::make_unique<OpSplit>();
}

static PassRegistration<OpSplit> pass;

} // namespace xcore
} // namespace mlir
