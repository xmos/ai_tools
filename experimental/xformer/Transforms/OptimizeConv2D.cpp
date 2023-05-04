// Copyright 2023 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Transforms/Options.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_layout_helper.h"

namespace mlir {
namespace xcore {

namespace {
// Optimize TFL Conv2D.
struct OptimizeConv2D
    : public PassWrapper<OptimizeConv2D, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeConv2D)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-optimize-conv2d"; }
  StringRef getDescription() const final { return "Optimize TFL Conv2D."; }
  void runOnOperation() override;
};

struct ChannelwiseSplitPattern : public OpRewritePattern<TFL::Conv2DOp> {
  using OpRewritePattern<TFL::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::Conv2DOp op,
                                PatternRewriter &rewriter) const override {
    // Lamdba to split filter or bias based on whether it is per channelwise
    // quantization If per channelwise quantization, we have to split the
    // quantization params
    auto getSplitResultType = [](int splitSize, int elems,
                                 std::initializer_list<int64_t> &shape,
                                 TFL::QConstOp op, ShapedType opType) {
      Type splitResultType =
          RankedTensorType::get(shape, opType.getElementType());

      auto opQType = op.getQtype().template cast<RankedTensorType>();

      if (auto qType =
              opQType.getElementType()
                  .dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto scaleVector =
            std::vector<double>{qType.getScales().begin() + elems,
                                qType.getScales().begin() + elems + splitSize};

        auto zeroPointVector = std::vector<int64_t>{
            qType.getZeroPoints().begin() + elems,
            qType.getZeroPoints().begin() + elems + splitSize};

        auto newQType = quant::UniformQuantizedPerAxisType::getChecked(
            op.getLoc(), qType.getFlags(), qType.getStorageType(),
            qType.getExpressedType(), scaleVector, zeroPointVector,
            qType.getQuantizedDimension(), qType.getStorageTypeMin(),
            qType.getStorageTypeMax());

        splitResultType = newQType.castFromExpressedType(
            quant::AnyQuantizedType::castToExpressedType(splitResultType));
      }
      return splitResultType;
    };

    auto filterQConstOp =
        dyn_cast<TFL::QConstOp>(op.getFilter().getDefiningOp());
    auto filterType = op.getFilter().getType().cast<ShapedType>();
    auto filter = filterQConstOp.getValue().cast<DenseElementsAttr>();
    auto filterSize = filter.size();

    // We want to try to keep the split filtersize less than specified size
    int numSplits = ceil(filterSize / convChannelwiseSplitSizeOption);
    // Only try to split if at least two splits are possible
    if (numSplits < 2) {
      return failure();
    }
    // Let's split the filter batch size as that's the same as bias size and
    // output channel size
    auto filterBatchSize = filterType.getShape()[0];
    // We want splits to be multiples of four, so we divide here and multiply
    // after calculating the split sizes
    int tmp = filterBatchSize / 4;
    int d = tmp / numSplits;
    int r = tmp % numSplits;
    llvm::SmallVector<int> splitSizes;
    // If not an even split, we distribute the remainder to the first few splits
    for (int i = 0; i < numSplits; ++i) {
      if (r > 0) {
        splitSizes.push_back((d + 1) * 4);
        r -= 1;
      } else {
        splitSizes.push_back(d * 4);
      }
    }

    auto biasQConstOp = dyn_cast<TFL::QConstOp>(op.getBias().getDefiningOp());
    auto biasType = op.getBias().getType().cast<ShapedType>();
    auto bias = biasQConstOp.getValue().cast<DenseElementsAttr>();

    assert(bias.size() == filterBatchSize);
    assert(op.getOutput().getType().cast<ShapedType>().getShape()[3] ==
           filterBatchSize);

    SmallVector<Value> conv2DOps;
    int elems = 0;
    for (int i = 0; i < splitSizes.size(); ++i) {
      // Create split filter
      auto splitFilterShape = {
          static_cast<int64_t>(splitSizes[i]), filterType.getShape()[1],
          filterType.getShape()[2], filterType.getShape()[3]};
      auto splitFilterResultType = getSplitResultType(
          splitSizes[i], elems, splitFilterShape, filterQConstOp, filterType);
      auto splitFilterValueType =
          RankedTensorType::get(splitFilterShape, rewriter.getIntegerType(8));
      auto filterSizeExcludingBatch = filterType.getShape()[1] *
                                      filterType.getShape()[2] *
                                      filterType.getShape()[3];
      auto filterVector = std::vector<int8_t>{
          filter.getValues<int8_t>().begin() + elems * filterSizeExcludingBatch,
          filter.getValues<int8_t>().begin() +
              ((elems + splitSizes[i]) * filterSizeExcludingBatch)};

      auto splitFilterQConstOp = rewriter.create<TFL::QConstOp>(
          op.getLoc(), mlir::TypeAttr::get(splitFilterResultType),
          mlir::DenseElementsAttr::get(splitFilterValueType,
                                       llvm::ArrayRef(filterVector)));

      // Create split bias
      auto splitBiasShape = {static_cast<int64_t>(splitSizes[i])};
      auto splitBiasResultType = getSplitResultType(
          splitSizes[i], elems, splitBiasShape, biasQConstOp, biasType);
      auto splitBiasValueType =
          RankedTensorType::get(splitBiasShape, rewriter.getIntegerType(32));
      auto biasVector = std::vector<int32_t>{
          bias.getValues<int32_t>().begin() + elems,
          bias.getValues<int32_t>().begin() + elems + splitSizes[i]};

      auto splitBiasQConstOp = rewriter.create<TFL::QConstOp>(
          op.getLoc(), mlir::TypeAttr::get(splitBiasResultType),
          mlir::DenseElementsAttr::get(splitBiasValueType,
                                       llvm::ArrayRef(biasVector)));

      // Add split Conv2Ds
      auto outputType = op.getOutput().getType().cast<ShapedType>();
      auto splitResultType = RankedTensorType::get(
          {outputType.getShape()[0], outputType.getShape()[1],
           outputType.getShape()[2], splitSizes[i]},
          outputType.getElementType());

      auto newConv2DOp = rewriter.create<TFL::Conv2DOp>(
          op.getLoc(), splitResultType, op.getInput(), splitFilterQConstOp,
          splitBiasQConstOp, 1, 1, op.getFusedActivationFunction(), "VALID", 1,
          1);

      conv2DOps.push_back(newConv2DOp.getResult());

      elems += splitSizes[i];
    }

    // Add concatenation op with axis 3, the channel dimension
    auto newConcatOp = rewriter.create<TFL::ConcatenationOp>(
        op.getLoc(), op.getOutput().getType(), conv2DOps, 3, "NONE");

    rewriter.replaceOp(op, newConcatOp.getOutput());
    return success();
  }
};

void OptimizeConv2D::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);

  // Channelwise split conv and add concat
  patterns.insert<ChannelwiseSplitPattern>(ctx);

  // Pad conv so that output depth is 4 and add strided slice

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the OptimizeConv2D pass.
std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeConv2DPass() {
  return std::make_unique<OptimizeConv2D>();
}

static PassRegistration<OptimizeConv2D> pass;

} // namespace xcore
} // namespace mlir
