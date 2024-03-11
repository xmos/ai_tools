// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1
#include "Utils/Util.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::xcore {

namespace {
// Replace TFL AveragePool2D with TFL DepthwiseConv2D.
struct ReplaceAvgPoolWithConv2D
    : public PassWrapper<ReplaceAvgPoolWithConv2D,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceAvgPoolWithConv2D)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final {
    return "xcore-replace-avgpool-with-conv2d";
  }
  StringRef getDescription() const final {
    return "Replace TFL Avgpool with Conv2D operations.";
  }
  void runOnOperation() override;
};

struct ReplaceAvgPoolWithConv2DPattern
    : public OpRewritePattern<TFL::AveragePool2DOp> {
  using OpRewritePattern<TFL::AveragePool2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::AveragePool2DOp avgPoolOp,
                                PatternRewriter &rewriter) const override {

    auto inputElementalType =
        avgPoolOp.getInput().getType().cast<ShapedType>().getElementType();

    auto outputElementalType =
        avgPoolOp.getOutput().getType().cast<ShapedType>().getElementType();

    // Check for invalid types and return
    // Input type must be QI8
    if (!utils::hasNBitSignedQType(inputElementalType) ||
        !utils::hasNBitSignedQType(outputElementalType)) {
      return failure();
    }

    auto inputType =
        avgPoolOp.getInput().getType().dyn_cast<RankedTensorType>();
    auto inputDepth = inputType.getDimSize(3);

    auto filterHeight = avgPoolOp.getFilterHeight();
    auto filterWidth = avgPoolOp.getFilterWidth();

    float scaleFactor = 1. / (filterHeight * filterWidth);

    int64_t storageTypeMin =
        quant::QuantizedType::getDefaultMinimumForInteger(/*isSigned=*/true, 8);
    int64_t storageTypeMax =
        quant::QuantizedType::getDefaultMaximumForInteger(/*isSigned=*/true, 8);

    /*
    The quantisation stratergy we are using is to set all elements of the filter
    to one and to control the scaling with the QTypes scalar. The zero point
    will be 0. In order to achieve an AvgPool2D we simply sum all elements of
    the receptive field then scale by 1./count(the receptive field).
    */
    UniformQuantizedType int8ElementQtype =
        mlir::quant::UniformQuantizedType::get(
            true, rewriter.getIntegerType(8), rewriter.getF32Type(),
            scaleFactor, 0, storageTypeMin, storageTypeMax);

    auto filterResultType = RankedTensorType::get(
        {1, filterHeight, filterWidth, inputDepth}, int8ElementQtype);

    RankedTensorType filterValueType = RankedTensorType::get(
        {1, filterHeight, filterWidth, inputDepth}, rewriter.getIntegerType(8));

    // These are the actual values that the quantised filter will hold.
    std::vector<int8_t> filterVector(filterHeight * filterWidth * inputDepth,
                                     1);

    Value filter = rewriter.create<TFL::QConstOp>(
        avgPoolOp.getLoc(), mlir::TypeAttr::get(filterResultType),
        DenseElementsAttr::get<int8_t>(filterValueType, filterVector));

    //[asj] This may need to be QI32 but I32 seems to work
    RankedTensorType biasType =
        RankedTensorType::get({inputDepth}, rewriter.getI32Type());
    std::vector<int32_t> biasValues(inputDepth, 0);
    auto bias = rewriter.create<TFL::ConstOp>(
        avgPoolOp->getLoc(), DenseIntElementsAttr::get(biasType, biasValues));

    auto conv2dOp = rewriter.create<TFL::DepthwiseConv2DOp>(
        avgPoolOp.getLoc(), avgPoolOp.getType(), avgPoolOp.getInput(), filter,
        bias, // TODO [asj]how do we drop the bias?
        /*dilation_h_factor=*/1,
        /*dilation_w_factor=*/1,
        /*fused_activation_function=*/avgPoolOp.getFusedActivationFunction(),
        /*padding=*/avgPoolOp.getPadding(),
        /*stride_h=*/avgPoolOp.getStrideH(),
        /*stride_w=*/avgPoolOp.getStrideW(),
        /*depth_multiplier=*/1);

    rewriter.replaceOp(avgPoolOp, conv2dOp.getOutput());

    return success();
  }
};

void ReplaceAvgPoolWithConv2D::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();

  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceAvgPoolWithConv2DPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceAvgPoolWithConv2D pass.
std::unique_ptr<OperationPass<func::FuncOp>>
createReplaceAvgPoolWithConv2DPass() {
  return std::make_unique<ReplaceAvgPoolWithConv2D>();
}

static PassRegistration<ReplaceAvgPoolWithConv2D> pass;

} // namespace mlir::xcore
