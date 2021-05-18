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

// TODO: Is there a better place for these constants?
static constexpr unsigned XCORE_OUTPUT_BITS = 8;
// NOTE: If we would not need to add the offset separately, the intermediate
// could never saturate, and this value would be 8. But decreasing to 7 means
// that we get an extra bit of headroom in the intermediate.
// TODO: Investigate if this could be calculated/estimated from the parameters
static constexpr unsigned XCORE_SHIFT_ADJUSTMENT = 7;
static constexpr unsigned XCORE_MAX_POST_SHIFT =
    22 + XCORE_SHIFT_ADJUSTMENT - XCORE_OUTPUT_BITS;

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

  // TODO: Document
  LogicalResult getTransformedBias(
      FullyConnectedOp fcOp,
      llvm::SmallVectorImpl<int16_t> &transformedBiasResultVector) const {
    // Get filter values as int64
    auto filterQConstOp =
        dyn_cast<TFL::QConstOp>(fcOp.filter().getDefiningOp());
    auto filterAttr = filterQConstOp.value().cast<DenseElementsAttr>();
    auto filterVector64 =
        llvm::SmallVector<int64_t, 0>{filterAttr.getValues<int8_t>().begin(),
                                      filterAttr.getValues<int8_t>().end()};

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
    auto filterType = fcOp.filter().getType().dyn_cast<RankedTensorType>();
    auto axis0Size = filterType.getShape()[0];
    auto axis1Size = filterType.getShape()[1];
    zeroPointBiasVector64.reserve(axis0Size); // for better performance
    for (auto it = filterVector64.begin(); it != filterVector64.end();
         it += axis1Size) {
      auto sum = std::accumulate(it, it + axis1Size, 0);
      zeroPointBiasVector64.push_back(sum * inputZeroPoint);
    }

    // Get bias values as int64
    auto biasQConstOp = dyn_cast<TFL::QConstOp>(fcOp.bias().getDefiningOp());
    auto biases = biasQConstOp.value().cast<DenseElementsAttr>();
    auto biasVector64 = llvm::SmallVector<int64_t, 0>{
        biases.getValues<int32_t>().begin(), biases.getValues<int32_t>().end()};

    // Subtract to get unified bias
    for (int i = 0; i < biasVector64.size(); ++i) {
      biasVector64[i] -= zeroPointBiasVector64[i];
    }

    // Find the pad factor for the bias result vector
    // The padded size would be a multiple of 16
    // Resize bias result vector with a size of {paddedSize x 2 x 16}
    // This is since we are going to split biasVector64 into upper and lower 16
    // bits of each 32 bit value
    int padFactor = ceil(biasVector64.size() / 16.0);
    transformedBiasResultVector.resize(padFactor * 2 * 16);

    // Convert biasVector64 values to int32 and split it into upper and lower 16
    // bits of each 32 bit value
    // This is stored into the bias result vector in the following format,
    // [Upper 16 bits of first 16 values of biasVector]
    // [Lower 16 bits of first 16 values of biasVector]
    // [Upper 16 bits of next 16 values of biasVector]
    // [Lower 16 bits of next 16 values of biasVector] and so on...
    for (int i = 0; i < biasVector64.size(); ++i) {
      int resultIndex = 16 * floor(i / 16) + i;
      int32_t clampedBias =
          std::max(std::min(static_cast<int32_t>(biasVector64[i]),
                            std::numeric_limits<int32_t>::max()),
                   std::numeric_limits<int32_t>::min());
      if (clampedBias != static_cast<int32_t>(biasVector64[i])) {
        llvm::errs() << "Unified bias saturated to 32 bit!";
        return failure();
      }
      transformedBiasResultVector[resultIndex] = clampedBias >> 16;
      transformedBiasResultVector[resultIndex + 16] = clampedBias & 0xFFFF;
    }

    return success();
  }

  // TODO: Document
  LogicalResult
  getTransformedScaleOffset(FullyConnectedOp fcOp,
                            llvm::SmallVectorImpl<int16_t>
                                &transformedScaleOffsetResultVector) const {
    // TODO: Doesn't work for BNNs yet
    // For BNNs, need to adapt these to handle bias with larger rank

    // Get output scale
    RankedTensorType outputType =
        fcOp.output().getType().dyn_cast<RankedTensorType>();
    auto outputQType = outputType.getElementType()
                           .dyn_cast<mlir::quant::UniformQuantizedType>();
    auto outputScale = outputQType.getScale();

    // Get bias scale
    auto biasQConstOp = dyn_cast<TFL::QConstOp>(fcOp.bias().getDefiningOp());
    auto biasType = biasQConstOp.qtype().cast<RankedTensorType>();
    auto biasQType =
        biasType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
    auto biasScale = biasQType.getScale();

    // Do the transformations needed to calculate shiftPre, scale,
    // offsetScale, offset, and shiftPost
    auto multiplier = biasScale / outputScale;
    auto rshift = -(ceil(log2(multiplier))) + 1;
    auto scale = round(multiplier * pow(2, 14 + rshift));
    if (scale == pow(2, 15)) {
      rshift -= 1;
      scale /= 2;
    }
    rshift -= XCORE_SHIFT_ADJUSTMENT;
    auto shiftPre = std::max(rshift, 0.0);
    auto shiftPost = XCORE_MAX_POST_SHIFT + std::min(rshift, 0.0);
    if (shiftPost < 0) {
      llvm::errs() << "Negative shift_post encountered";
      return failure();
    }
    auto outputZeroPoint = outputQType.getZeroPoint();
    auto rawOffset =
        outputZeroPoint * pow(2, shiftPost) * pow(2, (XCORE_OUTPUT_BITS - 8));
    auto offsetScale = round(sqrt(abs(rawOffset)));
    auto offset = round(rawOffset / offsetScale);

    // TODO: The below handling of creating the result vector has to be
    // refactored for adding support for BNNs
    auto biasSize = biasType.getShape()[0];
    int padFactor = ceil(biasSize / 16.0);

    // Create a padded vector of type int16_t
    auto getPaddedVector = [&](int value) {
      llvm::SmallVector<int16_t, 0> vec;
      vec.insert(vec.end(), biasSize, static_cast<int16_t>(value));
      vec.resize(padFactor * 16);
      return vec;
    };

    llvm::SmallVector<int16_t, 0> shiftPreVector = getPaddedVector(shiftPre);
    llvm::SmallVector<int16_t, 0> scaleVector = getPaddedVector(scale);
    llvm::SmallVector<int16_t, 0> offsetScaleVector =
        getPaddedVector(offsetScale);
    llvm::SmallVector<int16_t, 0> offsetVector = getPaddedVector(offset);
    llvm::SmallVector<int16_t, 0> shiftPostVector = getPaddedVector(shiftPost);

    // Create the transformed scale offset vector of size {padFactor * 16 * 5}
    // with 16 values each from shiftPre, scale, offsetScale, offset, and
    // shiftPost
    transformedScaleOffsetResultVector.reserve(padFactor * 16 * 5);
    for (int i = 0; i < padFactor * 16; i += 16) {
      transformedScaleOffsetResultVector.insert(
          transformedScaleOffsetResultVector.end(), shiftPreVector.begin() + i,
          shiftPreVector.begin() + i + 16);
      transformedScaleOffsetResultVector.insert(
          transformedScaleOffsetResultVector.end(), scaleVector.begin() + i,
          scaleVector.begin() + i + 16);
      transformedScaleOffsetResultVector.insert(
          transformedScaleOffsetResultVector.end(),
          offsetScaleVector.begin() + i, offsetScaleVector.begin() + i + 16);
      transformedScaleOffsetResultVector.insert(
          transformedScaleOffsetResultVector.end(), offsetVector.begin() + i,
          offsetVector.begin() + i + 16);
      transformedScaleOffsetResultVector.insert(
          transformedScaleOffsetResultVector.end(), shiftPostVector.begin() + i,
          shiftPostVector.begin() + i + 16);
    }
    return success();
  }

  LogicalResult matchAndRewrite(FullyConnectedOp fcOp,
                                PatternRewriter &rewriter) const override {

    // Return if we have already legalized FullyConnected. The bias QConstOp
    // would have been replaced in the process.
    if (!dyn_cast<TFL::QConstOp>(fcOp.bias().getDefiningOp())) {
      return failure();
    }

    // Obtain the transformed bias and transformed scale offset vectors
    llvm::SmallVector<int16_t, 0> transformedBiasResultVector;
    if (failed(getTransformedBias(fcOp, transformedBiasResultVector))) {
      return failure();
    }
    llvm::SmallVector<int16_t, 0> transformedScaleOffsetResultVector;
    if (failed(getTransformedScaleOffset(fcOp,
                                         transformedScaleOffsetResultVector))) {
      return failure();
    }

    // Find the pad factor
    auto biasQConstOp = dyn_cast<TFL::QConstOp>(fcOp.bias().getDefiningOp());
    auto biasType = biasQConstOp.qtype().cast<RankedTensorType>();
    auto biasSize = biasType.getShape()[0];
    int padFactor = ceil(biasSize / 16.0);

    // Create the new bias vector of size {padFactor * 16 * 7} with 32 values
    // (16 values from bias upper and 16 from bias lower) from transformed bias
    // vector and 80 values (16 values each from shiftPre, scale, offsetScale,
    // offset, and shiftPost) from transformed scale offset vector
    llvm::SmallVector<int16_t, 0> newBiasResultVector;
    newBiasResultVector.reserve(padFactor * 16 * 7);
    for (int i = 0; i < padFactor * 16; i += 16) {
      newBiasResultVector.insert(newBiasResultVector.end(),
                                 transformedBiasResultVector.begin() + i * 2,
                                 transformedBiasResultVector.begin() +
                                     (i + 16) * 2);
      newBiasResultVector.insert(
          newBiasResultVector.end(),
          transformedScaleOffsetResultVector.begin() + i * 5,
          transformedScaleOffsetResultVector.begin() + (i + 16) * 5);
    }

    // Create shape of {padFactor, 7, 16} for the new bias type and create a
    // new bias op from the new bias vector
    ShapedType newBiasType =
        RankedTensorType::get({padFactor, 7, 16}, rewriter.getIntegerType(16));
    auto newBiasAttr =
        DenseElementsAttr::get<int16_t>(newBiasType, newBiasResultVector);
    auto newBiasConstantOp =
        rewriter.create<mlir::ConstantOp>(fcOp.getLoc(), newBiasAttr);

    // Create a new FullyConnectedOp with the new bias op and replace the
    // original one
    auto newFcOp = rewriter.create<FullyConnectedOp>(
        fcOp.getLoc(), fcOp.getType(), fcOp.input(), fcOp.filter(),
        newBiasConstantOp,
        rewriter.getStringAttr(fcOp.fused_activation_function()),
        rewriter.getStringAttr(fcOp.weights_format()),
        rewriter.getBoolAttr(fcOp.keep_num_dims()));
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
