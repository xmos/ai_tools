// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "lib_nn/api/Conv2d.hpp"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include <numeric>

namespace mlir {
namespace xcore {

namespace {
// Replace suitable TFL FullyConnected with TFL Conv2D for XCore.
struct ReplaceFCWithConv2D
    : public PassWrapper<ReplaceFCWithConv2D, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  void runOnFunction() override;
};

struct ReplaceFCWithConv2DPattern
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fcOp,
                                PatternRewriter &rewriter) const override {
    // Check for invalid types and return
    // Input type must be QI8
    if (!(fcOp.input()
              .getType()
              .cast<ShapedType>()
              .getElementType()
              .isa<quant::QuantizedType>() &&
          fcOp.input()
              .getType()
              .cast<ShapedType>()
              .getElementType()
              .cast<quant::QuantizedType>()
              .isSigned() &&
          fcOp.input()
                  .getType()
                  .cast<ShapedType>()
                  .getElementType()
                  .cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    // Filter type must be QI8
    if (!(fcOp.filter()
              .getType()
              .cast<ShapedType>()
              .getElementType()
              .isa<quant::QuantizedType>() &&
          fcOp.filter()
              .getType()
              .cast<ShapedType>()
              .getElementType()
              .cast<quant::QuantizedType>()
              .isSigned() &&
          fcOp.filter()
                  .getType()
                  .cast<ShapedType>()
                  .getElementType()
                  .cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    // TODO: What to do if no bias?
    // Bias type must be QI32
    if (!(fcOp.bias()
              .getType()
              .cast<ShapedType>()
              .getElementType()
              .isa<quant::QuantizedType>() &&
          fcOp.bias()
              .getType()
              .cast<ShapedType>()
              .getElementType()
              .cast<quant::QuantizedType>()
              .isSigned() &&
          fcOp.bias()
                  .getType()
                  .cast<ShapedType>()
                  .getElementType()
                  .cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 32)) {
      return failure();
    }

    if (fcOp.weights_format() != "DEFAULT") {
      return failure();
    }

    if (fcOp.keep_num_dims() != false) {
      return failure();
    }

    // Add a ReshapeOp before Conv2D for expanding input to 4 dims
    std::vector<int64_t> expandedShapeVector = {1, 1, 1, 1};
    assert(fcOp.input().getType().cast<ShapedType>().getRank() == 2 &&
           "FullyConnected input should have a rank of 2!");
    expandedShapeVector[0] =
        fcOp.input().getType().cast<ShapedType>().getShape()[0];
    expandedShapeVector[3] =
        fcOp.input().getType().cast<ShapedType>().getShape()[1];
    auto expandedInputResultType = RankedTensorType::get(
        expandedShapeVector,
        fcOp.input().getType().cast<ShapedType>().getElementType());
    auto expandedShapeConstantOp = rewriter.create<ConstantOp>(
        fcOp.getLoc(),
        DenseElementsAttr::get(
            RankedTensorType::get({4}, rewriter.getIntegerType(64)),
            llvm::makeArrayRef(expandedShapeVector)));
    auto reshapeInputOp =
        rewriter.create<TFL::ReshapeOp>(fcOp.getLoc(), expandedInputResultType,
                                        fcOp.input(), expandedShapeConstantOp);

    // Expand filter to 4 dims
    auto filterQConstOp =
        dyn_cast<TFL::QConstOp>(fcOp.filter().getDefiningOp());
    auto filter = filterQConstOp.value().cast<DenseElementsAttr>();
    auto filterShape =
        fcOp.filter().getType().cast<RankedTensorType>().getShape();
    std::vector<int64_t> expandedFilterVector = {1, 1, 1, 1};
    expandedFilterVector[0] = filterShape[0];
    expandedFilterVector[3] = filterShape[1];
    auto expandedFilterType = RankedTensorType::get(
        expandedFilterVector,
        fcOp.filter().getType().cast<ShapedType>().getElementType());
    auto expandedFilterQConstOp = rewriter.create<TFL::QConstOp>(
        fcOp.getLoc(), mlir::TypeAttr::get(expandedFilterType), filter);

    // Add a Conv2D to replace the FC
    std::vector<int64_t> expandedResultVector = {1, 1, 1, 1};
    expandedResultVector[0] =
        fcOp.getResult(0).getType().cast<ShapedType>().getShape()[0];
    expandedResultVector[3] =
        fcOp.getResult(0).getType().cast<ShapedType>().getShape()[1];
    auto expandedResultType = RankedTensorType::get(
        expandedResultVector,
        fcOp.getResult(0).getType().cast<ShapedType>().getElementType());
    auto newConv2DOp = rewriter.create<TFL::Conv2DOp>(
        fcOp.getLoc(), expandedResultType, reshapeInputOp,
        expandedFilterQConstOp, fcOp.bias(), 1, 1,
        fcOp.fused_activation_function(), "VALID", 1, 1);

    // Add a ReshapeOp after Conv2D for squeezing output back to 2 dims
    std::vector<int64_t> squeezedShapeVector = {1, 1};
    squeezedShapeVector[0] =
        newConv2DOp.output().getType().cast<ShapedType>().getShape()[0];
    squeezedShapeVector[1] =
        newConv2DOp.output().getType().cast<ShapedType>().getShape()[3];
    auto squeezedOutputResultType = RankedTensorType::get(
        squeezedShapeVector,
        newConv2DOp.output().getType().cast<ShapedType>().getElementType());
    auto squeezedShapeConstantOp = rewriter.create<ConstantOp>(
        fcOp.getLoc(),
        DenseElementsAttr::get(
            RankedTensorType::get({2}, rewriter.getIntegerType(64)),
            llvm::makeArrayRef(squeezedShapeVector)));
    auto reshapeOutputOp = rewriter.create<TFL::ReshapeOp>(
        fcOp.getLoc(), squeezedOutputResultType, newConv2DOp.output(),
        squeezedShapeConstantOp);

    // Replace the FC with the new ops
    rewriter.replaceOp(fcOp, reshapeOutputOp.output());

    return success();
  }
};

void ReplaceFCWithConv2D::runOnFunction() {
  auto *ctx = &getContext();
  auto func = getFunction();

  OwningRewritePatternList patterns(ctx);
  patterns.insert<ReplaceFCWithConv2DPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceFCWithConv2D pass.
std::unique_ptr<OperationPass<FuncOp>> createReplaceFCWithConv2DPass() {
  return std::make_unique<ReplaceFCWithConv2D>();
}

static PassRegistration<ReplaceFCWithConv2D>
    pass("xcore-replace-fc-with-conv2d",
         "Replace suitable TFL FullyConnected with TFL Conv2D for XCore.");

} // namespace xcore
} // namespace mlir
