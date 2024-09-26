// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Utils/Util.h"

extern "C" {
#include "lib_nn/api/nn_layers.h"
}
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"

namespace mlir::xcore {

namespace {
// Replace TFL Sum with Mean for XCore.
struct ReplaceSum
    : public PassWrapper<ReplaceSum, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceSum)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-sum"; }
  StringRef getDescription() const final {
    return "Replace TFL Sum with mean for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceSumPattern : public OpRewritePattern<TFL::SumOp> {
  using OpRewritePattern<TFL::SumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SumOp sumOp,
                                PatternRewriter &rewriter) const override {

    auto input = sumOp.getInput();
    auto output = sumOp.getOutput();

    DenseElementsAttr axisAttr;
    matchPattern(sumOp.getAxes(), m_Constant(&axisAttr));
    auto axisValues = axisAttr.getValues<int32_t>();
    std::vector<int32_t> axis(axisValues.begin(), axisValues.end());
    int32_t minAxis = *std::min_element(axis.begin(), axis.end());
    int32_t maxAxis = *std::max_element(axis.begin(), axis.end());
    if (maxAxis - minAxis > axis.size() - 1) {
      return failure();
    }

    auto inputType = input.getType().cast<ShapedType>();
    auto outputType = output.getType().cast<ShapedType>();
    if (!utils::isNBitSignedQType<8>(inputType.getElementType()) ||
        !utils::isNBitSignedQType<8>(outputType.getElementType())) {
      return failure();
    }

    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();

    int rank = inputShape.size();

    int beginDims = 1;
    for (int i = 0; i < minAxis; i++) {
      beginDims *= inputShape[i];
    }

    int endDims = 1;
    for (int i = maxAxis + 1; i < rank; i++) {
      endDims *= inputShape[i];
    }

    int sumDims = 1;
    for (int i = minAxis; i <= maxAxis; i++) {
      sumDims *= inputShape[i];
    }

    auto inputQType = utils::getQType(input);
    auto outputQType = utils::getQType(output);

    float inZeroPoint = static_cast<float>(inputQType.getZeroPoint());
    float outZeroPoint = static_cast<float>(outputQType.getZeroPoint());
    float scaleMul = inputQType.getScale() / outputQType.getScale();

    auto beginDimsAttr = rewriter.getI32IntegerAttr(beginDims);
    auto endDimsAttr = rewriter.getI32IntegerAttr(endDims);
    auto meanDimsAttr = rewriter.getI32IntegerAttr(sumDims);
    auto inZeroPointAttr = rewriter.getF32FloatAttr(inZeroPoint);
    auto outZeroPointAttr = rewriter.getF32FloatAttr(outZeroPoint);
    auto scaleMulAttr = rewriter.getF32FloatAttr(scaleMul);

    auto xcSumOp = rewriter.create<MeanOp>(
        sumOp.getLoc(), sumOp.getType(), sumOp.getInput(), beginDimsAttr,
        meanDimsAttr, endDimsAttr, inZeroPointAttr, outZeroPointAttr,
        scaleMulAttr);
    rewriter.replaceOp(sumOp, xcSumOp.getOutput());

    return success();
  }
};

void ReplaceSum::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceSumPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceSum pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceSumPass() {
  return std::make_unique<ReplaceSum>();
}

static PassRegistration<ReplaceSum> pass;

} // namespace mlir::xcore
