// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Transforms/Options.h"
#include "Utils/Util.h"

#include "lib_nn/api/OutputTransformFn.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

#include <limits>

namespace mlir::xcore {

namespace {

// Estimate the best-case (group or channelwise) xcore output transform
// quantization error for an op with the given per-channel effective
// multipliers/bias/weights, mirroring the Conv2D/DepthwiseConv2D error check
// in ConvPatternsTFL.cpp (see --xcore-conv-err-threshold).
static double computeBestOutputTransformError(
    const std::vector<float> &effectiveMultiplier,
    const std::vector<int32_t> &bias, const std::vector<int8_t> &filter,
    int inputZeroPoint, int outputZeroPoint, int outputChannelCount) {
  nn::MulsAndBias mulAndBias = nn::OutputTransformFnInt8::canonicalise_mul_and_bias(
      effectiveMultiplier, bias, filter, inputZeroPoint, outputZeroPoint,
      outputChannelCount, /*verbose=*/false);

  nn_vlmul_shr_t vlmulShr = targetArchOption == XS3A
                                ? nn_vlmul_shr_t::VLMUL_SHR_XS3A
                                : nn_vlmul_shr_t::VLMUL_SHR_VX4A;

  auto groupQuantizer = nn::OutputTransformFnInt8_Group::Quantizer();
  nn::OutputTransformFnInt8_Group::QuantisationParams groupQp =
      groupQuantizer.quantise_activation(mulAndBias, vlmulShr, /*verbose=*/false);
  double groupError = nn::OutputTransformFnInt8::get_quant_error(
      mulAndBias, groupQp, vlmulShr, convForceErrorCheckOption);

  auto channelwiseQuantizer = nn::OutputTransformFnInt8_Channelwise::Quantizer();
  nn::OutputTransformFnInt8_Channelwise::QuantisationParams channelwiseQp =
      channelwiseQuantizer.quantise_activation(mulAndBias, vlmulShr,
                                               /*verbose=*/false);
  double channelwiseError = nn::OutputTransformFnInt8_Channelwise::get_quant_error(
      mulAndBias, channelwiseQp, vlmulShr, convForceErrorCheckOption);

  return std::min(groupError, channelwiseError);
}

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
    auto fcInputElementType =
        fcOp.getInput().getType().cast<ShapedType>().getElementType();
    auto fcFilterElementType =
        fcOp.getFilter().getType().cast<ShapedType>().getElementType();

    // Check for invalid types and return
    // Input type must be QI8 or QI16
    if (!utils::isNBitSignedQType<8>(fcInputElementType) &&
        !utils::isNBitSignedQType<16>(fcInputElementType))
      return failure();

    if (!utils::isNBitSignedQType<8>(fcFilterElementType))
      return failure();

    // If bias exists, it must be QI32
    if (auto biasType = fcOp.getBias().getType().dyn_cast<ShapedType>()) {
      if (!utils::isNBitSignedQType<32>(biasType.getElementType()))
        return failure();
    }

    if (fcOp.getWeightsFormat() != "DEFAULT") {
      return failure();
    }

    // Optionally leave this FC as a reference TFL FullyConnected op instead
    // of lowering it, if its xcore output transform quantization error
    // exceeds --xcore-fc-err-threshold (higher precision, at the cost of
    // speed for this op). Defaults to no threshold, so every FC is lowered,
    // matching prior behaviour.
    if (fcQuantErrorThresholdOption < std::numeric_limits<double>::max()) {
      auto inputQType =
          fcInputElementType.dyn_cast<quant::UniformQuantizedType>();
      auto outputQType = fcOp.getResult(0)
                             .getType()
                             .cast<ShapedType>()
                             .getElementType()
                             .dyn_cast<quant::UniformQuantizedType>();
      auto filterQConstOp =
          dyn_cast<TFL::QConstOp>(fcOp.getFilter().getDefiningOp());

      if (inputQType && outputQType && filterQConstOp) {
        auto filterQConstType =
            filterQConstOp.getQtype().cast<RankedTensorType>();
        auto filterValues =
            filterQConstOp.getValue().cast<DenseElementsAttr>();
        std::vector<int8_t> filterVector{filterValues.getValues<int8_t>().begin(),
                                        filterValues.getValues<int8_t>().end()};
        auto filterShape =
            fcOp.getFilter().getType().cast<ShapedType>().getShape();
        int outputChannelCount = filterShape[0];

        bool isPerChannelQuantized = false;
        double filterScale = 0;
        ArrayRef<double> filterScales;
        if (auto perTensorQType =
                filterQConstType.getElementType()
                    .dyn_cast<quant::UniformQuantizedType>()) {
          filterScale = perTensorQType.getScale();
        } else if (auto perAxisQType =
                       filterQConstType.getElementType()
                           .dyn_cast<quant::UniformQuantizedPerAxisType>()) {
          isPerChannelQuantized = true;
          filterScales = perAxisQType.getScales();
        }

        std::vector<float> effectiveMultiplier(outputChannelCount);
        for (int i = 0; i < outputChannelCount; ++i) {
          double scale = isPerChannelQuantized ? filterScales[i] : filterScale;
          effectiveMultiplier[i] = static_cast<float>(
              inputQType.getScale() * scale / outputQType.getScale());
        }

        std::vector<int32_t> biasVector;
        if (!fcOp.getBias().getType().isa<NoneType>()) {
          DenseElementsAttr biasAttr;
          if (auto biasQConstOp =
                  dyn_cast<TFL::QConstOp>(fcOp.getBias().getDefiningOp())) {
            biasAttr = biasQConstOp.getValue().cast<DenseElementsAttr>();
          } else {
            matchPattern(fcOp.getBias(), m_Constant(&biasAttr));
          }
          biasVector = {biasAttr.getValues<int32_t>().begin(),
                       biasAttr.getValues<int32_t>().end()};
        } else {
          biasVector = std::vector<int32_t>(outputChannelCount, 0);
        }

        double quantError = computeBestOutputTransformError(
            effectiveMultiplier, biasVector, filterVector,
            inputQType.getZeroPoint(), outputQType.getZeroPoint(),
            outputChannelCount);

        if (quantError > fcQuantErrorThresholdOption) {
          return failure();
        }
      }
    }

    // Add a ReshapeOp before Conv2D for expanding input to 4 dims
    auto inputType = fcOp.getInput().getType().cast<ShapedType>();
    int spatialInDim = inputType.getRank() - 2;
    int channelInDim = inputType.getRank() - 1;

    std::vector<int64_t> expandedInputShapeVector = {
        1LL, 1LL,
        inputType.isDynamicDim(spatialInDim)
            ? 1
            : inputType.getShape()[spatialInDim],
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
        1, 1,
        result0Type.isDynamicDim(spatialOutDim)
            ? 1
            : result0Type.getShape()[spatialOutDim],
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
      squeezedOutputShapeVector = {1, result0Type.getShape()[spatialOutDim],
                                   result0Type.getShape()[channelOutDim]};
      squeezedReshapeConstantVector = {
          1,
          static_cast<int>(newConv2DOutputType.isDynamicDim(2)
                               ? 1
                               : newConv2DOutputType.getDimSize(2)),
          static_cast<int>(newConv2DOutputType.getDimSize(3))};
      squeezedShapeType = RankedTensorType::get({3}, rewriter.getI32Type());
    } else {
      squeezedOutputShapeVector = {result0Type.getShape()[spatialOutDim],
                                   result0Type.getShape()[channelOutDim]};
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

} // namespace mlir::xcore
