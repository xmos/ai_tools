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

    // Check for invalid types and return
    // Defining op must be pad
    // auto padOp = dyn_cast_or_null<TFL::PadOp>(op.getInput().getDefiningOp());
    // if (!padOp) {
    //   return failure();
    // }

    // // Get transpose permutation
    // DenseIntElementsAttr perm;
    // if (!matchPattern(op.getPerm(), m_Constant(&perm))) {
    //   return failure();
    // }

    // // Confirm transpose permutation is 0,2,3,1 i.e., NWCH
    // // Remnants of Pytorch to TFlite conversion
    // auto permVal = perm.getValues<int32_t>();
    // if (perm.size() != 4 || permVal[0] != 0 || permVal[1] != 2 ||
    //     permVal[2] != 3 || permVal[3] != 1) {
    //   return failure();
    // }

    // // Get padding val
    // DenseIntElementsAttr pad;
    // if (!matchPattern(padOp.getPadding(), m_Constant(&pad))) {
    //   return failure();
    // }

    // // Confirm padding is only in last two dimensions
    // auto padVal = pad.getValues<int32_t>();
    // if (padVal[{0, 0}] != 0 || padVal[{0, 1}] != 0 || padVal[{1, 0}] != 0 ||
    //     padVal[{1, 1}] != 0 || padVal[{2, 0}] != 1 || padVal[{2, 1}] != 1 ||
    //     padVal[{3, 0}] != 1 || padVal[{3, 1}] != 1) {
    //   return failure();
    // }

    int numSplits = 4;

    // Create new Conv2D ops
    // Expand filter to 4 dims
    auto filterQConstOp =
        dyn_cast<TFL::QConstOp>(op.getFilter().getDefiningOp());
    auto filterType = op.getFilter().getType().cast<ShapedType>();

    auto filter = filterQConstOp.getValue().cast<DenseElementsAttr>();
    auto filterSize = filter.size();

    if (filterSize < 300000) {
      return failure();
    }
    auto numSplitFilterElements = filterSize / numSplits;

    auto biasQConstOp = dyn_cast<TFL::QConstOp>(op.getBias().getDefiningOp());
    auto biasType = op.getBias().getType().cast<ShapedType>();
    auto bias = biasQConstOp.getValue().cast<DenseElementsAttr>();
    auto biasSize = bias.size();
    auto numSplitBiasElements = biasSize / numSplits;


    // TODO, have to split the filter and bias qtype scales


    SmallVector<Value> conv2DOps;
    for (int i = 0; i < numSplits; ++i) {

      auto splitFilterShape = {
          filterType.getShape()[0] / numSplits, filterType.getShape()[1],
          filterType.getShape()[2], filterType.getShape()[3]};
      auto splitFilterResultType =
          RankedTensorType::get(splitFilterShape, filterType.getElementType());
      auto splitFilterValueType =
          RankedTensorType::get(splitFilterShape, rewriter.getIntegerType(8));
      auto filterVector = std::vector<int8_t>{
          filter.getValues<int8_t>().begin() + (i * numSplitFilterElements),
          filter.getValues<int8_t>().begin() +
              (((i + 1) * numSplitFilterElements))};
      auto splitFilterQConstOp = rewriter.create<TFL::QConstOp>(
          op.getLoc(), mlir::TypeAttr::get(splitFilterResultType),
          mlir::DenseElementsAttr::get(splitFilterValueType,
                                       llvm::ArrayRef(filterVector)));

      auto splitBiasShape = {biasType.getShape()[0] / numSplits};
      auto splitBiasResultType =
          RankedTensorType::get(splitBiasShape, biasType.getElementType());
      auto splitBiasValueType =
          RankedTensorType::get(splitBiasShape, rewriter.getIntegerType(32));
      auto biasVector = std::vector<int32_t>{
          bias.getValues<int32_t>().begin() + (i * numSplitBiasElements),
          bias.getValues<int32_t>().begin() +
              (((i + 1) * numSplitBiasElements))};
      auto splitBiasQConstOp = rewriter.create<TFL::QConstOp>(
          op.getLoc(), mlir::TypeAttr::get(splitBiasResultType),
          mlir::DenseElementsAttr::get(splitBiasValueType,
                                       llvm::ArrayRef(biasVector)));

      // Add split Conv2Ds
      auto outputType = op.getOutput().getType().cast<ShapedType>();
      auto splitResultType = RankedTensorType::get(
          {outputType.getShape()[0], outputType.getShape()[1],
           outputType.getShape()[2], outputType.getShape()[3] / numSplits},
          outputType.getElementType());

      auto newConv2DOp = rewriter.create<TFL::Conv2DOp>(
          op.getLoc(), splitResultType, op.getInput(), splitFilterQConstOp,
          splitBiasQConstOp, 1, 1, op.getFusedActivationFunction(), "VALID", 1,
          1);

      conv2DOps.push_back(newConv2DOp.getResult());
    }

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
