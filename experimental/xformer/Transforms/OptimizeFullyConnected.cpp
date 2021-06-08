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
static constexpr int XCORE_OUTPUT_BITS = 8;
// NOTE: If we would not need to add the offset separately, the intermediate
// could never saturate, and this value would be 8. But decreasing to 7 means
// that we get an extra bit of headroom in the intermediate.
// TODO: Investigate if this could be calculated/estimated from the parameters
static constexpr int XCORE_SHIFT_ADJUSTMENT = 7;
static constexpr int XCORE_MAX_POST_SHIFT =
    22 + XCORE_SHIFT_ADJUSTMENT - XCORE_OUTPUT_BITS;
static constexpr int XCORE_VPU_ACC_PERIOD = 16;
static constexpr int XCORE_WORD_SIZE_BYTES = 4;

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

  // Transform bias via various computations and padding, resulting in biasHi
  // and biasLow which are stored into the result vector in the following
  // format.
  // [Upper 16 bits of first XCORE_VPU_ACC_PERIOD=16 values]
  // [Lower 16 bits of first XCORE_VPU_ACC_PERIOD=16 values]
  // [Upper 16 bits of next XCORE_VPU_ACC_PERIOD=16 values]
  // [Lower 16 bits of next XCORE_VPU_ACC_PERIOD=16 values]
  // and so on...
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
    // The padded size would be a multiple of XCORE_VPU_ACC_PERIOD=16
    // Resize bias result vector with a size of {padFactor x 2 x
    // XCORE_VPU_ACC_PERIOD=16} This is since we are going to split biasVector64
    // into upper and lower 16 bits of each 32 bit value
    int padFactor = ceil(biasVector64.size() / double(XCORE_VPU_ACC_PERIOD));
    transformedBiasResultVector.resize(padFactor * 2 /*biasHi, biasLow*/ *
                                       XCORE_VPU_ACC_PERIOD);

    // Convert biasVector64 values to int32 and split it into upper and lower 16
    // bits of each 32 bit value
    // This is stored into the bias result vector in the following format,
    // [Upper 16 bits of first XCORE_VPU_ACC_PERIOD=16 values of biasVector]
    // [Lower 16 bits of first XCORE_VPU_ACC_PERIOD=16 values of biasVector]
    // [Upper 16 bits of next XCORE_VPU_ACC_PERIOD=16 values of biasVector]
    // [Lower 16 bits of next XCORE_VPU_ACC_PERIOD=16 values of biasVector] and
    // so on...
    for (int i = 0; i < biasVector64.size(); ++i) {
      int resultIndex =
          XCORE_VPU_ACC_PERIOD * floor(i / XCORE_VPU_ACC_PERIOD) + i;
      int32_t clampedBias =
          std::max(std::min(static_cast<int32_t>(biasVector64[i]),
                            std::numeric_limits<int32_t>::max()),
                   std::numeric_limits<int32_t>::min());
      if (clampedBias != static_cast<int32_t>(biasVector64[i])) {
        llvm::errs() << "Unified bias saturated to 32 bit!";
        return failure();
      }
      transformedBiasResultVector[resultIndex] = clampedBias >> 16;
      transformedBiasResultVector[resultIndex + XCORE_VPU_ACC_PERIOD] =
          clampedBias & 0xFFFF;
    }

    return success();
  }

  // Do the transformations needed to calculate shiftPre, scale,
  // offsetScale, offset, and shiftPost
  LogicalResult doScaleOffsetTransformations(
      double biasScale, int biasSize, double outputScale,
      double outputZeroPoint,
      llvm::SmallVectorImpl<int16_t> &shiftPreResultVector,
      llvm::SmallVectorImpl<int16_t> &scaleResultVector,
      llvm::SmallVectorImpl<int16_t> &offsetScaleResultVector,
      llvm::SmallVectorImpl<int16_t> &offsetResultVector,
      llvm::SmallVectorImpl<int16_t> &shiftPostResultVector) const {
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
    auto rawOffset =
        outputZeroPoint * pow(2, shiftPost) * pow(2, (XCORE_OUTPUT_BITS - 8));
    auto offsetScale = round(sqrt(abs(rawOffset)));
    auto offset = offsetScale != 0 ? round(rawOffset / offsetScale) : 0;

    int padFactor = ceil(biasSize / double(XCORE_VPU_ACC_PERIOD));

    // Create a padded vector of type int16_t
    auto getPaddedVector = [&](int value) {
      llvm::SmallVector<int16_t, 0> vec;
      vec.insert(vec.end(), biasSize, static_cast<int16_t>(value));
      vec.resize(padFactor * XCORE_VPU_ACC_PERIOD);
      return vec;
    };

    // Store into the result vectors
    shiftPreResultVector = getPaddedVector(shiftPre);
    scaleResultVector = getPaddedVector(scale);
    offsetScaleResultVector = getPaddedVector(offsetScale);
    offsetResultVector = getPaddedVector(offset);
    shiftPostResultVector = getPaddedVector(shiftPost);
    return success();
  }

  // Do the transformations needed to calculate shiftPre, scale,
  // offsetScale, offset, and shiftPost for the per channel quantized case in
  // which we will have an array of bias scales
  LogicalResult doScaleOffsetTransformationsPerChannelQuantized(
      ArrayRef<double> biasScales, int biasSize, double outputScale,
      double outputZeroPoint,
      llvm::SmallVectorImpl<int16_t> &shiftPreResultVector,
      llvm::SmallVectorImpl<int16_t> &scaleResultVector,
      llvm::SmallVectorImpl<int16_t> &offsetScaleResultVector,
      llvm::SmallVectorImpl<int16_t> &offsetResultVector,
      llvm::SmallVectorImpl<int16_t> &shiftPostResultVector) const {
    llvm::SmallVector<double, 0> multiplier(biasScales.begin(),
                                            biasScales.end());
    std::for_each(multiplier.begin(), multiplier.end(),
                  [&](double &n) { n /= outputScale; });

    llvm::SmallVector<double, 0> rshift;
    rshift.resize(multiplier.size());
    for (int i = 0; i < rshift.size(); ++i) {
      rshift[i] = multiplier[i] != 0 ? -(ceil(log2(multiplier[i]))) + 1 : 16.0;
    }

    llvm::SmallVector<double, 0> scale;
    scale.resize(multiplier.size());
    for (int i = 0; i < scale.size(); ++i) {
      scale[i] = multiplier[i] != 0
                     ? round(multiplier[i] * pow(2, 14 + rshift[i]))
                     : pow(2, 15) - 1;
    }

    for (int i = 0; i < scale.size(); ++i) {
      if (scale[i] == pow(2, 15)) {
        rshift[i] -= 1;
        scale[i] /= 2;
      }
      rshift[i] -= XCORE_SHIFT_ADJUSTMENT;
    }

    int padFactor = ceil(biasSize / double(XCORE_VPU_ACC_PERIOD));

    // Zero pad to a multiple of XCORE_VPU_ACC_PERIOD=16
    rshift.resize(padFactor * XCORE_VPU_ACC_PERIOD);
    scale.resize(padFactor * XCORE_VPU_ACC_PERIOD);

    llvm::SmallVector<int16_t, 0> shiftPre;
    std::transform(
        rshift.begin(), rshift.end(), std::back_inserter(shiftPre),
        [](double n) { return static_cast<int16_t>(std::max(n, 0.0)); });

    llvm::SmallVector<int16_t, 0> shiftPost;
    shiftPost.resize(rshift.size());
    for (int i = 0; i < shiftPost.size(); ++i) {
      shiftPost[i] =
          static_cast<int16_t>(XCORE_MAX_POST_SHIFT + std::min(rshift[i], 0.0));
      if (shiftPost[i] < 0) {
        llvm::errs() << "Negative shift_post encountered";
        return failure();
      }
    }

    llvm::SmallVector<double, 0> rawOffset;
    std::transform(shiftPost.begin(), shiftPost.end(),
                   std::back_inserter(rawOffset), [&](double n) {
                     return outputZeroPoint * pow(2, n) *
                            pow(2, (XCORE_OUTPUT_BITS - 8));
                   });

    llvm::SmallVector<int16_t, 0> offsetScale;
    std::transform(
        rawOffset.begin(), rawOffset.end(), std::back_inserter(offsetScale),
        [](double n) { return static_cast<int16_t>(round(sqrt(abs(n)))); });

    llvm::SmallVector<int16_t, 0> offset;
    offset.resize(rawOffset.size());
    for (int i = 0; i < offset.size(); ++i) {
      offset[i] =
          offsetScale[i] != 0 ? round(rawOffset[i] / offsetScale[i]) : 0;
    }

    // Store into the result vectors
    // The result vectors are in int16_t format
    // The scale vector has not been converted to int16_t yet, so we convert it
    // here
    shiftPreResultVector = shiftPre;
    std::transform(scale.begin(), scale.end(),
                   std::back_inserter(scaleResultVector),
                   [](double n) { return static_cast<int16_t>(n); });
    offsetScaleResultVector = offsetScale;
    offsetResultVector = offset;
    shiftPostResultVector = shiftPost;
    return success();
  }

  // Transform scale and offset via various computations and padding, resulting
  // in transformed scale offset vector of size {padFactor *
  // XCORE_VPU_ACC_PERIOD=16 * 5} with XCORE_VPU_ACC_PERIOD=16 values each from
  // shiftPre, scale, offsetScale, offset, and shiftPost
  LogicalResult
  getTransformedScaleOffset(FullyConnectedOp fcOp,
                            llvm::SmallVectorImpl<int16_t>
                                &transformedScaleOffsetResultVector) const {
    // Get output scale
    RankedTensorType outputType =
        fcOp.output().getType().dyn_cast<RankedTensorType>();
    auto outputQType = outputType.getElementType()
                           .dyn_cast<mlir::quant::UniformQuantizedType>();
    auto outputScale = outputQType.getScale();
    auto outputZeroPoint = outputQType.getZeroPoint();

    // Get bias scale
    auto biasQConstOp = dyn_cast<TFL::QConstOp>(fcOp.bias().getDefiningOp());
    auto biasType = biasQConstOp.qtype().cast<RankedTensorType>();
    int biasSize = biasType.getShape()[0];

    llvm::SmallVector<int16_t, 0> shiftPreVector;
    llvm::SmallVector<int16_t, 0> scaleVector;
    llvm::SmallVector<int16_t, 0> offsetScaleVector;
    llvm::SmallVector<int16_t, 0> offsetVector;
    llvm::SmallVector<int16_t, 0> shiftPostVector;

    // Do the transformations needed to calculate shiftPre, scale,
    // offsetScale, offset, and shiftPost
    // There are two cases here, the bias scale can be per channel quantized in
    // which case we will have multiple bias scales. Otherwise we will have a
    // single bias scale. We have to handle these seperately since the former
    // would be an array of bias scales.
    if (auto biasQType = biasType.getElementType()
                             .dyn_cast<mlir::quant::UniformQuantizedType>()) {
      auto biasScale = biasQType.getScale();
      doScaleOffsetTransformations(
          biasScale, biasSize, outputScale, outputZeroPoint, shiftPreVector,
          scaleVector, offsetScaleVector, offsetVector, shiftPostVector);
    } else if (auto biasQType =
                   biasType.getElementType()
                       .dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
      auto biasScales = biasQType.getScales();
      doScaleOffsetTransformationsPerChannelQuantized(
          biasScales, biasSize, outputScale, outputZeroPoint, shiftPreVector,
          scaleVector, offsetScaleVector, offsetVector, shiftPostVector);
    }

    // Create the transformed scale offset vector of size {padFactor *
    // XCORE_VPU_ACC_PERIOD=16 * 5} with XCORE_VPU_ACC_PERIOD=16 values each
    // from shiftPre, scale, offsetScale, offset, and shiftPost
    int padFactor = ceil(biasSize / double(XCORE_VPU_ACC_PERIOD));
    transformedScaleOffsetResultVector.reserve(
        padFactor * XCORE_VPU_ACC_PERIOD *
        5 /*shiftPre, scale, offsetScale, offset, shiftPost*/);
    for (int i = 0; i < padFactor * XCORE_VPU_ACC_PERIOD;
         i += XCORE_VPU_ACC_PERIOD) {
      transformedScaleOffsetResultVector.insert(
          transformedScaleOffsetResultVector.end(), shiftPreVector.begin() + i,
          shiftPreVector.begin() + i + XCORE_VPU_ACC_PERIOD);
      transformedScaleOffsetResultVector.insert(
          transformedScaleOffsetResultVector.end(), scaleVector.begin() + i,
          scaleVector.begin() + i + XCORE_VPU_ACC_PERIOD);
      transformedScaleOffsetResultVector.insert(
          transformedScaleOffsetResultVector.end(),
          offsetScaleVector.begin() + i,
          offsetScaleVector.begin() + i + XCORE_VPU_ACC_PERIOD);
      transformedScaleOffsetResultVector.insert(
          transformedScaleOffsetResultVector.end(), offsetVector.begin() + i,
          offsetVector.begin() + i + XCORE_VPU_ACC_PERIOD);
      transformedScaleOffsetResultVector.insert(
          transformedScaleOffsetResultVector.end(), shiftPostVector.begin() + i,
          shiftPostVector.begin() + i + XCORE_VPU_ACC_PERIOD);
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

    // Find the bias pad factor
    auto biasQConstOp = dyn_cast<TFL::QConstOp>(fcOp.bias().getDefiningOp());
    auto biasType = biasQConstOp.qtype().cast<RankedTensorType>();
    auto biasSize = biasType.getShape()[0];
    int biasPadFactor = ceil(biasSize / double(XCORE_VPU_ACC_PERIOD));

    // Create the new bias vector of size {biasPadFactor *
    // XCORE_VPU_ACC_PERIOD=16 * 7}. This is with 32 values
    // (XCORE_VPU_ACC_PERIOD=16 values from bias upper and
    // XCORE_VPU_ACC_PERIOD=16 from bias lower) from transformed bias vector and
    // 80 values (XCORE_VPU_ACC_PERIOD=16 values each from shiftPre, scale,
    // offsetScale, offset, and shiftPost) from transformed scale offset vector.
    llvm::SmallVector<int16_t, 0> newBiasResultVector;
    newBiasResultVector.reserve(
        biasPadFactor * XCORE_VPU_ACC_PERIOD *
        7 /*biasHi, biasLow, shiftPre, scale, offsetScale, offset, shiftPost*/);
    for (int i = 0; i < biasPadFactor * XCORE_VPU_ACC_PERIOD;
         i += XCORE_VPU_ACC_PERIOD) {
      newBiasResultVector.insert(
          newBiasResultVector.end(),
          transformedBiasResultVector.begin() + i * 2 /*biasHi, biasLow*/,
          transformedBiasResultVector.begin() + (i + XCORE_VPU_ACC_PERIOD) * 2);
      newBiasResultVector.insert(
          newBiasResultVector.end(),
          transformedScaleOffsetResultVector.begin() +
              i * 5 /*shiftPre, scale, offsetScale, offset, shiftPost*/,
          transformedScaleOffsetResultVector.begin() +
              (i + XCORE_VPU_ACC_PERIOD) * 5);
    }

    // Create shape of {biasPadFactor, 7, XCORE_VPU_ACC_PERIOD=16} for the new
    // bias type and create a new bias op from the new bias vector
    ShapedType newBiasType = RankedTensorType::get(
        {biasPadFactor,
         7 /*biasHi, biasLow, shiftPre, scale, offsetScale, offset, shiftPost*/,
         XCORE_VPU_ACC_PERIOD},
        rewriter.getIntegerType(16));
    auto newBiasAttr =
        DenseElementsAttr::get<int16_t>(newBiasType, newBiasResultVector);
    auto newBiasConstantOp =
        rewriter.create<mlir::ConstantOp>(fcOp.getLoc(), newBiasAttr);

    // Zero pad weights along the column dimension
    // The padded size would be a multiple of XCORE_WORD_SIZE_BYTES=4
    auto weightQConstOp =
        dyn_cast<TFL::QConstOp>(fcOp.filter().getDefiningOp());
    auto weightType = weightQConstOp.qtype().cast<RankedTensorType>();
    auto weightRowSize = weightType.getShape()[0];
    auto weightColumnSize = weightType.getShape()[1];
    int weightZeroPadCount =
        XCORE_WORD_SIZE_BYTES *
            ceil(weightColumnSize / double(XCORE_WORD_SIZE_BYTES)) -
        weightColumnSize;
    auto weights = weightQConstOp.value().cast<DenseElementsAttr>();
    auto weightVector = llvm::SmallVector<int8_t, 0>{
        weights.getValues<int8_t>().begin(), weights.getValues<int8_t>().end()};
    llvm::SmallVector<int8_t, 0> newWeightResultVector;
    newWeightResultVector.reserve(weightRowSize *
                                  (weightColumnSize + weightZeroPadCount));
    for (int i = 0; i < weightVector.size(); i += weightColumnSize) {
      newWeightResultVector.insert(newWeightResultVector.end(),
                                   weightVector.begin() + i,
                                   weightVector.begin() + i + weightColumnSize);
      newWeightResultVector.insert(newWeightResultVector.end(),
                                   weightZeroPadCount, 0);
    }

    // Create shape of {weightRowSize, (weightColumnSize + weightZeroPadCount)}
    // for the new weight type and create a new weight op from the new weight
    // vector
    ShapedType newWeightType = RankedTensorType::get(
        {weightRowSize, (weightColumnSize + weightZeroPadCount)},
        rewriter.getIntegerType(8));
    auto newWeightAttr =
        DenseElementsAttr::get<int8_t>(newWeightType, newWeightResultVector);
    auto newWeightConstantOp =
        rewriter.create<mlir::ConstantOp>(fcOp.getLoc(), newWeightAttr);

    // Create a new FullyConnectedOp with the new bias op and replace the
    // original one
    auto newFcOp = rewriter.create<FullyConnectedOp>(
        fcOp.getLoc(), fcOp.getType(), fcOp.input(), newWeightConstantOp,
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
