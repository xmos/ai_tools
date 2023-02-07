// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/core/framework/kernel_shape_util.h"

namespace mlir {
namespace xcore {

namespace {
// Apply generated TFL patterns.
struct ApplyTFLPatterns
    : public PassWrapper<ApplyTFLPatterns, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplyTFLPatterns)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  StringRef getArgument() const final { return "xcore-apply-tflpatterns"; }
  StringRef getDescription() const final {
    return "Apply generated TFL optimization patterns.";
  }
  void runOnOperation() override;
};

template <typename T>
SmallVector<Value, 2>
getConvPaddingValues(PatternRewriter &rewriter, T conv2DOp,
                     int64_t dilationHeight, int64_t dilationWidth,
                     int64_t strideHeight, int64_t strideWidth) {
  auto inputType =
      conv2DOp.input().getType().template dyn_cast<RankedTensorType>();
  auto filterType =
      conv2DOp.filter().getType().template dyn_cast<RankedTensorType>();
  auto inputHeight = inputType.getDimSize(1);
  auto inputWidth = inputType.getDimSize(2);
  auto filterHeight = filterType.getDimSize(1);
  auto filterWidth = filterType.getDimSize(2);

  // Find padding values
  int64_t newHeight, newWidth;
  int64_t padTop, padBottom, padLeft, padRight;
  if (tensorflow::GetWindowedOutputSizeVerboseV2(
          inputHeight, filterHeight, dilationHeight, strideHeight,
          tensorflow::Padding::SAME, &newHeight, &padTop,
          &padBottom) != tensorflow::Status::OK()) {
    conv2DOp->emitError("Could not obtain SAME padding values for Conv op!");
  }
  if (tensorflow::GetWindowedOutputSizeVerboseV2(
          inputWidth, filterWidth, dilationWidth, strideWidth,
          tensorflow::Padding::SAME, &newWidth, &padLeft,
          &padRight) != tensorflow::Status::OK()) {
    conv2DOp->emitError("Could not obtain SAME padding values for Conv op!");
  }

  std::vector<int32_t> paddingValues{0,
                                     0,
                                     static_cast<int>(padTop),
                                     static_cast<int>(padBottom),
                                     static_cast<int>(padLeft),
                                     static_cast<int>(padRight),
                                     0,
                                     0};

  RankedTensorType type = RankedTensorType::get({4, 2}, rewriter.getI32Type());
  auto attr = DenseIntElementsAttr::get(type, paddingValues);
  auto paddingOp =
      rewriter.create<arith::ConstantOp>(conv2DOp->getLoc(), type, attr);

  // Obtain the output type so that we can use it to denote the returnType for
  // the PadOp in Tablegen DRR
  auto batch = inputType.getDimSize(0);
  auto depth = inputType.getDimSize(3);
  auto outputHeight = inputHeight + padTop + padBottom;
  auto outputWidth = inputWidth + padLeft + padRight;
  RankedTensorType outputType = RankedTensorType::get(
      {batch, outputHeight, outputWidth, depth}, inputType.getElementType());
  auto outputTypeOp = rewriter.create<arith::ConstantOp>(
      conv2DOp->getLoc(), outputType, rewriter.getUnitAttr());

  return SmallVector<Value, 2>({paddingOp, outputTypeOp});
}

template <typename T>
SmallVector<Value, 2> getConv2DPaddingValues(PatternRewriter &rewriter,
                                             T conv2DOp) {
  return getConvPaddingValues<T>(
      rewriter, conv2DOp, conv2DOp.dilation_h_factor(),
      conv2DOp.dilation_w_factor(), conv2DOp.stride_h(), conv2DOp.stride_w());
}

SmallVector<Value, 2> getBConv2DPaddingValues(PatternRewriter &rewriter,
                                              mlir::lq::Bconv2dOp conv2DOp) {
  return getConvPaddingValues<mlir::lq::Bconv2dOp>(
      rewriter, conv2DOp, conv2DOp.dilation_height_factor(),
      conv2DOp.dilation_width_factor(), conv2DOp.stride_height(),
      conv2DOp.stride_width());
}

struct HoistQuantizeAboveConcatPattern
    : public OpRewritePattern<TFL::QuantizeOp> {
  using OpRewritePattern<TFL::QuantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::QuantizeOp quantizeOp,
                                PatternRewriter &rewriter) const override {
    // Parent op must be concat
    auto concatOp = dyn_cast_or_null<TFL::ConcatenationOp>(
        quantizeOp.input().getDefiningOp());
    if (!concatOp) {
      return failure();
    }

    SmallVector<Value> quantizeOps;
    TFL::QuantizeOp newQuantizeOp;
    for (int i = 0; i < concatOp->getNumOperands(); ++i) {
      auto newQType = quant::CastQuantizedTypeAttrFromExpressedType(
          rewriter, quantizeOp.qtypeAttr(),
          quant::QuantizedType::castToExpressedType(
              concatOp.getOperand(i).getType()),
          -1);
      newQuantizeOp = rewriter.create<TFL::QuantizeOp>(
          quantizeOp.getLoc(), newQType.getValue(), concatOp.getOperand(i),
          newQType);
      quantizeOps.push_back(newQuantizeOp.getResult());
    }

    // We use one of the quantizeops to get the output element type with the new
    // quantized parameters.
    // All of them have the same quantization parameters
    RankedTensorType newOutputType = RankedTensorType::get(
        concatOp.output().getType().cast<RankedTensorType>().getShape(),
        newQuantizeOp.output().getType().cast<ShapedType>().getElementType());

    auto newConcatOp = rewriter.create<TFL::ConcatenationOp>(
        concatOp.getLoc(), newOutputType, quantizeOps, concatOp.axis(),
        concatOp.fused_activation_function());

    rewriter.replaceOp(quantizeOp, newConcatOp.output());

    return success();
  }
};

#include "Transforms/GeneratedTFLPatterns.inc"

void ApplyTFLPatterns::runOnOperation() {
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  func::FuncOp func = getOperation();

  populateWithGenerated(patterns);
  patterns.insert<HoistQuantizeAboveConcatPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ApplyTFLPatterns pass.
std::unique_ptr<OperationPass<func::FuncOp>> createApplyTFLPatternsPass() {
  return std::make_unique<ApplyTFLPatterns>();
}

static PassRegistration<ApplyTFLPatterns> pass;

} // namespace xcore
} // namespace mlir
