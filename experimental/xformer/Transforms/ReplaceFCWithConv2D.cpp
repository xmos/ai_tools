// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

namespace {
// Replace suitable TFL FullyConnected with TFL Conv2D for XCore.
struct ReplaceFCWithConv2D
    : public PassWrapper<ReplaceFCWithConv2D, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceFCWithConv2D)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-fc-with-conv2d"; }
  StringRef getDescription() const final {
    return "Replace suitable TFL FullyConnected with TFL Conv2D";
  }
  void runOnOperation() override;
};

struct ReplaceFCWithConv2DPattern
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fcOp,
                                PatternRewriter &rewriter) const override {
    // Check for invalid types and return
    // Input type must be QI8
    auto fcInputElementType =
        fcOp.getInput().getType().cast<ShapedType>().getElementType();
    if (!(fcInputElementType.isa<quant::QuantizedType>() &&
          fcInputElementType.cast<quant::QuantizedType>().isSigned() &&
          fcInputElementType.cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    // Filter type must be
    auto fcFilterElementType =
        fcOp.getFilter().getType().cast<ShapedType>().getElementType();
    if (!(fcFilterElementType.isa<quant::QuantizedType>() &&
          fcFilterElementType.cast<quant::QuantizedType>().isSigned() &&
          fcFilterElementType.cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    // If bias exists, it must be QI32
    if (!fcOp.getBias().getType().isa<NoneType>()) {
      auto fcBiasElementType =
          fcOp.getBias().getType().cast<ShapedType>().getElementType();

      if (!(fcBiasElementType.isa<quant::QuantizedType>() &&
            fcBiasElementType.cast<quant::QuantizedType>().isSigned() &&
            fcBiasElementType.cast<quant::QuantizedType>()
                    .getStorageTypeIntegralWidth() == 32)) {
        return failure();
      }
    }

    if (fcOp.getWeightsFormat() != "DEFAULT") {
      return failure();
    }

    // Add a ReshapeOp before Conv2D for expanding input to 4 dims
    auto inputType = fcOp.getInput().getType().cast<ShapedType>();
    int spatialInDim = inputType.getRank() - 2;
    int channelInDim = inputType.getRank() - 1;

    std::vector<int64_t> expandedInputShapeVector = {
        1LL, 1LL, inputType.getShape()[spatialInDim],
        inputType.getShape()[channelInDim]};
    auto expandedInputResultType = RankedTensorType::get(
        expandedInputShapeVector, inputType.getElementType());

    std::vector<int32_t> expandedReshapeConstantVector = {
        1, 1,
        static_cast<int>(inputType.isDynamicDim(spatialInDim)
                             ? 1
                             : inputType.getDimSize(spatialInDim)),
        static_cast<int>(inputType.getDimSize(channelInDim))};
    RankedTensorType expandedShapeType =
        RankedTensorType::get({4}, rewriter.getI32Type());
    auto expandedShapeConstantOp = rewriter.create<arith::ConstantOp>(
        fcOp.getLoc(), expandedShapeType,
        DenseIntElementsAttr::get(expandedShapeType,
                                  expandedReshapeConstantVector));
    auto reshapeInputOp = rewriter.create<TFL::ReshapeOp>(
        fcOp.getLoc(), expandedInputResultType, fcOp.getInput(),
        expandedShapeConstantOp);

    // Expand filter to 4 dims
    auto filterQConstOp =
        dyn_cast<TFL::QConstOp>(fcOp.getFilter().getDefiningOp());
    auto filter = filterQConstOp.getValue().cast<DenseElementsAttr>();
    auto filterType = fcOp.getFilter().getType().cast<ShapedType>();
    std::vector<int64_t> expandedFilterVector = {filterType.getShape()[0], 1, 1,
                                                 filterType.getShape()[1]};
    auto expandedFilterType = RankedTensorType::get(
        expandedFilterVector, filterType.getElementType());
    auto expandedFilterQConstOp = rewriter.create<TFL::QConstOp>(
        fcOp.getLoc(), mlir::TypeAttr::get(expandedFilterType), filter);

    // Add a Conv2D to replace the FC
    auto result0Type = fcOp.getResult(0).getType().cast<ShapedType>();
    int spatialOutDim = result0Type.getRank() - 2;
    int channelOutDim = result0Type.getRank() - 1;
    std::vector<int64_t> expandedResultVector = {
        1, 1, result0Type.getShape()[spatialOutDim],
        result0Type.getShape()[channelOutDim]};
    auto expandedResultType = RankedTensorType::get(
        expandedResultVector, result0Type.getElementType());
    auto newConv2DOp = rewriter.create<TFL::Conv2DOp>(
        fcOp.getLoc(), expandedResultType, reshapeInputOp,
        expandedFilterQConstOp, fcOp.getBias(), 1, 1,
        fcOp.getFusedActivationFunction(), "VALID", 1, 1);

    // Add a ReshapeOp after Conv2D for squeezing output back to 2 or 3 dims
    auto newConv2DOutputType =
        newConv2DOp.getOutput().getType().cast<ShapedType>();
    std::vector<int64_t> squeezedOutputShapeVector;
    std::vector<int32_t> squeezedReshapeConstantVector;
    RankedTensorType squeezedShapeType;
    // If three output dimensions for the fully connected
    if (result0Type.getRank() == 3) {
      squeezedOutputShapeVector = {1, newConv2DOutputType.getShape()[2],
                                   newConv2DOutputType.getShape()[3]};
      squeezedReshapeConstantVector = {
          1,
          static_cast<int>(newConv2DOutputType.isDynamicDim(2)
                               ? 1
                               : newConv2DOutputType.getDimSize(2)),
          static_cast<int>(newConv2DOutputType.getDimSize(3))};
      squeezedShapeType = RankedTensorType::get({3}, rewriter.getI32Type());
    } else {
      squeezedOutputShapeVector = {newConv2DOutputType.getShape()[2],
                                   newConv2DOutputType.getShape()[3]};
      squeezedReshapeConstantVector = {
          static_cast<int>(newConv2DOutputType.isDynamicDim(2)
                               ? 1
                               : newConv2DOutputType.getDimSize(2)),
          static_cast<int>(newConv2DOutputType.getDimSize(3))};
      squeezedShapeType = RankedTensorType::get({2}, rewriter.getI32Type());
    }

    auto squeezedOutputResultType = RankedTensorType::get(
        squeezedOutputShapeVector, newConv2DOutputType.getElementType());
    auto squeezedShapeConstantOp = rewriter.create<arith::ConstantOp>(
        fcOp.getLoc(), squeezedShapeType,
        DenseIntElementsAttr::get(squeezedShapeType,
                                  squeezedReshapeConstantVector));
    auto reshapeOutputOp = rewriter.create<TFL::ReshapeOp>(
        fcOp.getLoc(), squeezedOutputResultType, newConv2DOp.getOutput(),
        squeezedShapeConstantOp);

    // Replace the FC with the new ops
    rewriter.replaceOp(fcOp, reshapeOutputOp.getOutput());

    return success();
  }
};

void ReplaceFCWithConv2D::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();

  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceFCWithConv2DPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceFCWithConv2D pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceFCWithConv2DPass() {
  return std::make_unique<ReplaceFCWithConv2D>();
}

static PassRegistration<ReplaceFCWithConv2D> pass;

} // namespace xcore
} // namespace mlir
