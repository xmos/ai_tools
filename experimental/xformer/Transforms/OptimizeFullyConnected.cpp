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

    // Find the padded size for the bias result vector
    // This would be a multiple of 16
    // Create biasResultVector with a size of {paddedSize x 2 x 16}
    // This is since we are going to split biasVector64 into upper and lower 16
    // bits of each 32 bit value
    int paddedSize = ceil(biasVector64.size() / 16.0);
    transformedBiasResultVector.resize(paddedSize * 2 * 16);

    // Convert biasVector64 values to int32 and split it into upper and lower 16
    // bits of each 32 bit value.
    // This is stored into the biasResultVector in the following format,
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

  LogicalResult matchAndRewrite(FullyConnectedOp fcOp,
                                PatternRewriter &rewriter) const override {

    // Return if we have already legalized FullyConnected. The bias QConstOp
    // would have been replaced in the process.
    if (!dyn_cast<TFL::QConstOp>(fcOp.bias().getDefiningOp())) {
      return failure();
    }

    // Get unified bias
    // Get scale offset
    // Combine the two and create op

    llvm::SmallVector<int16_t, 0> transformedBiasResultVector;
    if (failed(getTransformedBias(fcOp, transformedBiasResultVector))) {
      return failure();
    }

    llvm::SmallVector<int16_t, 0> scaleOffsetResultVector;
    {
      /*
        output_scale = self._output.quantization["scale"][0]
          bias_scale = np.array(self._biases.quantization["scale"])
          multiplier = bias_scale / output_scale*/

      // MAYBE THERE CAN BE MULTIPLE BIASES AND SO NEED AN ARRAY
      // LET'S TEST AND CHECK

      // Get output scale
      RankedTensorType outputType =
          fcOp.output().getType().dyn_cast<RankedTensorType>();
      auto outputQType = outputType.getElementType()
                             .dyn_cast<mlir::quant::UniformQuantizedType>();
      auto outputScale = outputQType.getScale();

      // Get bias scale
      auto biasQConstOp = dyn_cast<TFL::QConstOp>(fcOp.bias().getDefiningOp());
      auto biasType = biasQConstOp.qtype().cast<RankedTensorType>();
      auto biasSize = biasType.getShape()[0];

      auto biasQType = biasType.getElementType()
                           .dyn_cast<mlir::quant::UniformQuantizedType>();
      auto biasScale = biasQType.getScale();

      auto multiplier = biasScale / outputScale;
      auto rshift = -(ceil(log2(multiplier))) + 1;

      auto scale = round(multiplier * pow(2, 14 + rshift));

      if (scale == pow(2, 15)) {
        rshift -= 1;
        scale /= 2;
      }
      rshift -= XCORE_SHIFT_ADJUSTMENT;

      auto shiftPre = std::max(rshift, 0.0);
      // MAX_POST_SHIFT = 22 + XCORE_SHIFT_ADJUSTMENT - XCORE_OUTPUT_BITS
      auto shiftPost = (22 + XCORE_SHIFT_ADJUSTMENT - XCORE_OUTPUT_BITS) +
                       std::min(rshift, 0.0);

      if (shiftPost < 0) {
        llvm::errs() << "Negative shift_post encountered";
      }

      auto outputZeroPoint = outputQType.getZeroPoint();
      auto rawOffset =
          outputZeroPoint * pow(2, shiftPost) * pow(2, (XCORE_OUTPUT_BITS - 8));

      auto offsetScale = round(sqrt(abs(rawOffset)));
      auto offset = round(rawOffset / offsetScale);

      // CHANGE TO INT16 AND THEN CREATE biassize ARRAY
      int paddedSize = ceil(biasSize / 16.0) * 16;

      llvm::SmallVector<int16_t, 0> shiftPreVector;
      shiftPreVector.reserve(paddedSize);
      shiftPreVector.insert(shiftPreVector.end(), biasSize,
                            static_cast<int16_t>(shiftPre));
      shiftPreVector.resize(paddedSize);

      llvm::SmallVector<int16_t, 0> scaleVector;
      scaleVector.reserve(paddedSize);
      scaleVector.insert(scaleVector.end(), biasSize,
                         static_cast<int16_t>(scale));
      scaleVector.resize(paddedSize);

      llvm::SmallVector<int16_t, 0> offsetScaleVector;
      offsetScaleVector.reserve(paddedSize);
      offsetScaleVector.insert(offsetScaleVector.end(), biasSize,
                               static_cast<int16_t>(offsetScale));
      offsetScaleVector.resize(paddedSize);

      llvm::SmallVector<int16_t, 0> offsetVector;
      offsetVector.reserve(paddedSize);
      offsetVector.insert(offsetVector.end(), biasSize,
                          static_cast<int16_t>(offset));
      offsetVector.resize(paddedSize);

      llvm::SmallVector<int16_t, 0> shiftPostVector;
      shiftPostVector.reserve(paddedSize);
      shiftPostVector.insert(shiftPostVector.end(), biasSize,
                             static_cast<int16_t>(shiftPost));
      shiftPostVector.resize(paddedSize);

      scaleOffsetResultVector.reserve(paddedSize * 7);
      scaleOffsetResultVector.resize(paddedSize * 7);
      auto resultIndex = 0;
      for (int i = 0; i < paddedSize / 16; i++) {
        for (int j = i * 32; j < (i + 1) * 32; j++) {
          scaleOffsetResultVector[resultIndex++] =
              transformedBiasResultVector[j];
        }
        for (int j = i * 16; j < (i + 1) * 16; j++) {
          scaleOffsetResultVector[resultIndex++] = shiftPreVector[j];
        }
        for (int j = i * 16; j < (i + 1) * 16; j++) {
          scaleOffsetResultVector[resultIndex++] = scaleVector[j];
        }
        for (int j = i * 16; j < (i + 1) * 16; j++) {
          scaleOffsetResultVector[resultIndex++] = offsetScaleVector[j];
        }
        for (int j = i * 16; j < (i + 1) * 16; j++) {
          scaleOffsetResultVector[resultIndex++] = offsetVector[j];
        }
        for (int j = i * 16; j < (i + 1) * 16; j++) {
          scaleOffsetResultVector[resultIndex++] = shiftPostVector[j];
        }
      }

      int kdsfsf = 0;
    }

    // for(auto i : scaleOffsetResultVector){
    //  llvm::errs()<<i<<"\n";
    //}

    // Create shape of {paddedSize, 2, 16} for the new bias type and create a
    // new bias op
    auto biasQConstOp = dyn_cast<TFL::QConstOp>(fcOp.bias().getDefiningOp());
    auto biasType = biasQConstOp.qtype().cast<RankedTensorType>();
    auto biasSize = biasType.getShape()[0];
    int paddedSize = ceil(biasSize / 16.0);

    ShapedType newBiasType =
        RankedTensorType::get({paddedSize, 7, 16}, rewriter.getIntegerType(16));
    auto newBiasAttr =
        DenseElementsAttr::get<int16_t>(newBiasType, scaleOffsetResultVector);
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
