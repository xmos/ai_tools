// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

namespace {
// Optimize FullyConnected ops.
struct OptimizeFullyConnected
    : public PassWrapper<OptimizeFullyConnected, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  void runOnFunction() override;
};

#include "Transforms/GeneratedPatterns.inc"

struct LegalizeFullyConnected : public OpRewritePattern<FullyConnectedOp> {
  using OpRewritePattern<FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FullyConnectedOp fcOp,
                                PatternRewriter &rewriter) const override {

    // TODO: Hack to prevent re-matching FullyConnectedOp for now
    if (fcOp.keep_num_dims())
      return failure();

    // Get filter values as int64
    auto filterQConstOp =
        dyn_cast<TFL::QConstOp>(fcOp.filter().getDefiningOp());
    DenseElementsAttr filterAttr =
        filterQConstOp.value().cast<DenseElementsAttr>();
    auto filterVector64 =
        llvm::SmallVector<int64_t, 0>{filterAttr.getValues<int8_t>().begin(),
                                      filterAttr.getValues<int8_t>().end()};

    int p = 0;
    for (auto const &i : filterVector64) {
      llvm::errs() << p++ << " " << (int)i << "\n";
    }

    // Get input zero point
    RankedTensorType inputType =
        fcOp.input().getType().dyn_cast<RankedTensorType>();
    auto inputQType = inputType.getElementType()
                          .dyn_cast<mlir::quant::UniformQuantizedType>();
    int64_t inputZeroPoint = inputQType.getZeroPoint();

    // Multiply filter by input zero point and reduce on axis 1
    // We will be left with "axis 0" number of elements in zeroPointBiasVector64
    // which is the same as the number of outputs
    llvm::SmallVector<int64_t, 0> zeroPointBiasVector64;
    RankedTensorType filterType =
        fcOp.filter().getType().dyn_cast<RankedTensorType>();
    int axis1Size = filterType.getShape()[1];
    int64_t sum = 0;
    for (int i = 0; i < filterVector64.size(); ++i) {
      sum = sum + filterVector64[i] * inputZeroPoint;
      if (i % axis1Size == 0) {
        zeroPointBiasVector64.push_back(sum);
        sum = 0;
      }
    }

    p = 0;
    for (auto const &i : zeroPointBiasVector64) {
      llvm::errs() << p++ << " " << (int)i << "\n";
    }

    // Get bias values as int64
    auto biasQConstOp = dyn_cast<TFL::QConstOp>(fcOp.bias().getDefiningOp());
    DenseElementsAttr biases = biasQConstOp.value().cast<DenseElementsAttr>();
    auto biasVector64 = llvm::SmallVector<int64_t, 0>{
        biases.getValues<int32_t>().begin(), biases.getValues<int32_t>().end()};

    // Subtract to get unified bias
    for (int i = 0; i < biasVector64.size(); ++i) {
      biasVector64[i] = biasVector64[i] - zeroPointBiasVector64[i];
    }

    p = 0;
    for (auto const &i : biasVector64) {
      llvm::errs() << p++ << " " << (int)i << "\n";
    }

    // Find the padded size for the bias result vector
    // This would be a multiple of 16
    // Create biasResultVector with a size of {paddedSize x 2 x 16}
    // This is since we are going to split biasVector64 into upper and lower 16
    // bits of each 32 bit value
    llvm::SmallVector<int16_t, 0> biasResultVector;
    int paddedSize = ceil(biasVector64.size() / 16.0);
    biasResultVector.resize(paddedSize * 2 * 16);

    // Convert biasVector64 values to int32 and split it into upper and lower 16
    // bits of each 32 bit value.
    // This is stored into the biasResultVector in the following format,
    // [Upper 16 bits of first 16 values of biasVector]
    // [Lower 16 bits of first 16 values of biasVector]
    // [Upper 16 bits of next 16 values of biasVector]
    // [Lower 16 bits of next 16 values of biasVector] and so on...
    for (int i = 0; i < biasVector64.size(); ++i) {
      int resultIndex = 16 * floor(i / 16) + i;
      biasResultVector[resultIndex] =
          static_cast<int32_t>(biasVector64[i]) >> 16;
      biasResultVector[resultIndex + 16] =
          static_cast<int32_t>(biasVector64[i]) & 0xFFFF;
    }

    for (int value : biasResultVector) {
      llvm::errs() << value << "\n";
    }

    // Create shape of {paddedSize, 2, 16} for the new bias type and create a
    // new bias op
    ShapedType newBiasType =
        RankedTensorType::get({paddedSize, 2, 16}, rewriter.getIntegerType(16));
    auto newBiasAttr =
        DenseElementsAttr::get<int16_t>(newBiasType, biasResultVector);
    auto newBiasConstantOp =
        rewriter.create<mlir::ConstantOp>(fcOp.getLoc(), newBiasAttr);

    // Create a new FullyConnectedOp with the new bias op and replace the
    // original one
    auto newFcOp = rewriter.create<FullyConnectedOp>(
        fcOp.getLoc(), fcOp.getType(), fcOp.input(), fcOp.filter(),
        newBiasConstantOp,
        rewriter.getStringAttr(fcOp.fused_activation_function()),
        rewriter.getStringAttr(fcOp.weights_format()),
        // TODO: Hack to prevent re-matching FullyConnectedOp for now
        // Setting keep_num_dims = true
        rewriter.getBoolAttr(true));
    rewriter.replaceOp(fcOp, newFcOp.output());

    return success();
  }
};

void OptimizeFullyConnected::runOnFunction() {
  OwningRewritePatternList replacePatterns;
  auto *ctx = &getContext();
  auto func = getFunction();

  populateWithGenerated(ctx, replacePatterns);
  applyPatternsAndFoldGreedily(func, replacePatterns);

  OwningRewritePatternList legalizePatterns;
  legalizePatterns.insert<LegalizeFullyConnected>(ctx);
  applyPatternsAndFoldGreedily(func, legalizePatterns);
}
} // namespace

// Creates an instance of the OptimizeFullyConnected pass.
std::unique_ptr<OperationPass<FuncOp>> createOptimizeFullyConnectedPass() {
  return std::make_unique<OptimizeFullyConnected>();
}

static PassRegistration<OptimizeFullyConnected>
    pass("xcore-optimize-fullyconnected",
         "Optimize FullyConnected operations.");

} // namespace xcore
} // namespace mlir
