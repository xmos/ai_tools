// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Transforms/ConvPatterns.h"

namespace mlir {
namespace xcore {

// Handle Larq BNN Conv2D
LogicalResult
ReplaceBConv2DPattern::checkIfValid(lq::Bconv2dOp conv2DOp) const {
  // For BConv2D, we emit error messages since we don't have a reference op
  // The compilation will fail as we haven't defined a TFL custom op for BConv2D

  // Check for invalid types and return
  // Check filter is I32
  if (!conv2DOp.filter()
           .getType()
           .template cast<ShapedType>()
           .getElementType()
           .isInteger(32)) {
    conv2DOp.emitError("Filter type must be int32(packed binary) for BConv2D!");
    return failure();
  }

  // Check output is I32 or QI8
  if (!(conv2DOp.output()
            .getType()
            .template cast<ShapedType>()
            .getElementType()
            .template isa<quant::QuantizedType>() &&
        conv2DOp.output()
            .getType()
            .template cast<ShapedType>()
            .getElementType()
            .template cast<quant::QuantizedType>()
            .isSigned() &&
        conv2DOp.output()
                .getType()
                .template cast<ShapedType>()
                .getElementType()
                .template cast<quant::QuantizedType>()
                .getStorageTypeIntegralWidth() == 8) &&
      !(conv2DOp.output()
            .getType()
            .template cast<ShapedType>()
            .getElementType()
            .isInteger(32))) {
    conv2DOp.emitError(
        "Output type must be int32(packed binary) or int8 for BConv2D!");
    return failure();
  }

  // If we have QI8 output, check activation function is RELU
  if (conv2DOp.output()
          .getType()
          .template cast<ShapedType>()
          .getElementType()
          .template isa<quant::QuantizedType>() &&
      !(conv2DOp.fused_activation_function() == "RELU")) {
    conv2DOp.emitError("Activation function must be RELU for BConv2D int8!");
    return failure();
  }

  // Check padding is VALID
  if (!(conv2DOp.padding() == "VALID")) {
    conv2DOp.emitError("Only VALID padding is supported for BConv2D!");
    return failure();
  }

  return success();
}

LogicalResult ReplaceBConv2DPattern::getArgs(lq::Bconv2dOp conv2DOp,
                                             BConvArgs &args) const {
  // Retrieve remaining args
  // Find if binary output for the BConv2D
  bool binaryOutput = true;
  if (conv2DOp.output()
          .getType()
          .template cast<ShapedType>()
          .getElementType()
          .template isa<quant::QuantizedType>()) {
    binaryOutput = false;
  }

  // For binary output, convert depth to bits which is the number of input and
  // output channels
  // For int8 output, convert
  // - input depth to bits which is the number of input channels
  // - filter depth which is of type int32 to number of int8s which is the
  // number of output channels
  // - output depth is already of int8 type which should be the correct number
  // of output channels
  args.inputDepth *= sizeof(int32_t) * CHAR_BIT;
  if (binaryOutput) {
    args.outputDepth *= sizeof(int32_t) * CHAR_BIT;
    args.filterDepth *= sizeof(int32_t) * CHAR_BIT;
  } else {
    args.filterDepth *= sizeof(int32_t);
  }

  // Get filter values
  auto filterConstOp =
      dyn_cast<mlir::ConstantOp>(conv2DOp.filter().getDefiningOp());
  auto filter = filterConstOp.value().cast<DenseElementsAttr>();
  auto filterVector = std::vector<int32_t>{filter.getValues<int32_t>().begin(),
                                           filter.getValues<int32_t>().end()};

  std::vector<float> biasVector;
  std::vector<float> multiplierVector;
  std::vector<int32_t> thresholdVector;
  if (binaryOutput) {
    // Get threshold values
    auto thresholdConstOp =
        dyn_cast<mlir::ConstantOp>(conv2DOp.output_threshold().getDefiningOp());
    auto threshold = thresholdConstOp.value().cast<DenseElementsAttr>();
    thresholdVector =
        std::vector<int32_t>{threshold.getValues<int32_t>().begin(),
                             threshold.getValues<int32_t>().end()};
  } else {
    // Get bias values
    auto biasConstOp = dyn_cast<mlir::ConstantOp>(
        conv2DOp.post_activation_bias().getDefiningOp());
    auto biases = biasConstOp.value().cast<DenseElementsAttr>();
    biasVector = std::vector<float>{biases.getValues<float>().begin(),
                                    biases.getValues<float>().end()};

    // Get multiplier values
    auto multiplierConstOp = dyn_cast<mlir::ConstantOp>(
        conv2DOp.post_activation_multiplier().getDefiningOp());
    auto multipliers = multiplierConstOp.value().cast<DenseElementsAttr>();
    multiplierVector =
        std::vector<float>{multipliers.getValues<float>().begin(),
                           multipliers.getValues<float>().end()};

    // Fuse the back-transformation and int8 scale/zero-point into
    // the output transform multiplier/bias
    // Based on OneTimeSetup() in
    // https://github.com/larq/compute-engine/blob/main/larq_compute_engine/tflite/kernels/bconv2d.cc#L353
    auto outputType =
        conv2DOp.output().getType().template dyn_cast<RankedTensorType>();
    auto outputQType =
        outputType.getElementType()
            .template dyn_cast<mlir::quant::UniformQuantizedType>();
    auto outputScale = outputQType.getScale();
    auto outputZeroPoint = outputQType.getZeroPoint();

    // channels_in should be the same as input depth in bits which is the number
    // of input channels
    assert(conv2DOp.channels_in() == args.inputDepth &&
           "BConv2D channels_in and input depth in bits must match!");
    int32_t backtransformAdd =
        args.filterHeight * args.filterWidth * conv2DOp.channels_in();

    for (int i = 0; i < args.outputDepth; ++i) {
      auto postMul = multiplierVector[i];
      auto postBias = biasVector[i];
      multiplierVector[i] = -1 * postMul / outputScale;
      biasVector[i] =
          (postBias + static_cast<float>(backtransformAdd) * postMul) /
              outputScale +
          outputZeroPoint;
    }

    // Initialize min and max clamp values for RELU
    // Based on CalculateActivationRange() in
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/kernel_util.h#L266
    assert(conv2DOp.fused_activation_function() == "RELU" &&
           "Activation function should be RELU for the clamp value "
           "initialization!");
    int32_t nominalClampMin = 0;
    int32_t nominalClampMax = std::numeric_limits<int32_t>::max();

    nominalClampMin = std::max(nominalClampMin, -1 * backtransformAdd);
    nominalClampMax = std::min(nominalClampMax, backtransformAdd);
    args.clampMin = -1 * nominalClampMax + backtransformAdd;
    args.clampMax = -1 * nominalClampMin + backtransformAdd;
  }

  // Init lib_nn structs
  // Output could be int8 or binary
  args.Y = nn::ImageGeometry(args.outputHeight, args.outputWidth,
                             args.outputDepth, binaryOutput ? 1 : 8);
  // Input is in binary
  args.X =
      nn::ImageGeometry(args.inputHeight, args.inputWidth, args.inputDepth, 1);
  args.K = nn::WindowGeometry(
      args.filterHeight, args.filterWidth, args.filterDepth, 0, 0,
      conv2DOp.stride_height(), conv2DOp.stride_width(), 1,
      conv2DOp.dilation_height_factor(), conv2DOp.dilation_width_factor());

  // TODO, we are not padding at the moment, but the pad value might have to be
  // changed for BNNs
  args.padValue = 0;
  args.binaryOutput = binaryOutput;
  args.filter = filterVector;
  args.postActivationBias = biasVector;
  args.postActivationMultiplier = multiplierVector;
  args.threshold = thresholdVector;
  // For emitting errors
  args.loc = conv2DOp.getLoc();

  return success();
}

LogicalResult ReplaceBConv2DPattern::getKernelType(const BConvArgs &args,
                                                   Conv2DType &kt) const {
  // In case of binary output, since the input and output types are int32(packed
  // binary), they will always be a multiple of 32 channels
  // Hence we don't need to check for that case
  if (args.inputDepth % 256 == 0 && args.binaryOutput) {
    kt = Conv2DType::BNNValidDirectBinary;
  } else if (args.inputDepth % 256 == 0 && args.outputDepth % 16 == 0) {
    kt = Conv2DType::BNNValidDirectInt8;
  } else if (args.binaryOutput) {
    kt = Conv2DType::BNNValidIndirectBinary;
  } else if (args.outputDepth % 4 == 0) {
    kt = Conv2DType::BNNValidIndirectInt8;
  } else {
    emitError(args.loc,
              "Output depth must be a multiple of four for BConv2D int8!");
    return failure();
  }

  return success();
}

LogicalResult ReplaceBConv2DPattern::getSerializedParamsAndTensors(
    const BConvArgs &args, const Conv2DType &kt,
    llvm::SmallVector<std::string> &strParams, std::vector<int8_t> &weightsData,
    std::vector<int16_t> &thresholdsData, int &scratchBytes) const {
  switch (kt) {
  case Conv2DType::BNNValidDirectBinary:
    if (failed(getBConv2DValidDirectBinaryParams(
            args, strParams, weightsData, thresholdsData, scratchBytes))) {
      return failure();
    }
    break;
  case Conv2DType::BNNValidIndirectBinary:
    if (failed(getBConv2DValidIndirectBinaryParams(
            args, strParams, weightsData, thresholdsData, scratchBytes))) {
      return failure();
    }
    break;
  case Conv2DType::BNNValidDirectInt8:
    if (failed(getBConv2DValidDirectInt8Params(args, strParams, weightsData,
                                               thresholdsData, scratchBytes))) {
      return failure();
    }
    break;
  case Conv2DType::BNNValidIndirectInt8:
    if (failed(getBConv2DValidIndirectInt8Params(
            args, strParams, weightsData, thresholdsData, scratchBytes))) {
      return failure();
    }
    break;
  default:
    // This shouldn't happen!
    return failure();
  }
  return success();
}

LogicalResult ReplaceBConv2DPattern::getBConv2DValidDirectBinaryParams(
    const BConvArgs &args, llvm::SmallVector<std::string> &strParams,
    std::vector<int8_t> &weightsData, std::vector<int16_t> &thresholdsData,
    int &scratchBytes) const {
  nn::DerefInputFn::Params imToColParams(args.X, args.K);

  std::array<int, 4> filterShape = {args.outputDepth, args.filterHeight,
                                    args.filterWidth, args.inputDepth};
  nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
      (int8_t *)args.filter.data(), filterShape, 1, args.padValue);

  nn::MatMulBinaryDirectFn::Params afParams(args.X, args.K, args.inputDepth);

  // adjust the thresholds from xorpopcount space
  // to xcore space
  auto adjustedThresholds = nn::OT_binary::adjust_thresholds(
      args.threshold, args.inputDepth, args.K, rw);

  nn::OutputTransformFn::pad_final_access(adjustedThresholds, VPU_INT16_EPV,
                                          (int16_t)args.padValue);

  auto ir = nn::ImageRegion(0, 0, 0, args.Y.height, args.Y.width, args.Y.depth);
  nn::Filter2D::Params akParams(args.Y, ir, VPU_INT8_ACC_PERIOD);

  // TODO: Check serialization
  std::string akpStr = akParams.serialise<nn::Filter2D::Params>();
  std::string mfStr = imToColParams.serialise<nn::DerefInputFn::Params>();
  std::string afStr = afParams.serialise<nn::MatMulBinaryDirectFn::Params>();
  std::string otStr = ""; // otParams.serialise<nn::OT_int8::Params>();

  strParams.push_back(akpStr);
  strParams.push_back(mfStr);
  strParams.push_back(afStr);
  strParams.push_back(otStr);
  weightsData = rw.weights;
  thresholdsData = adjustedThresholds;
  scratchBytes = 0;

  return success();
}

LogicalResult ReplaceBConv2DPattern::getBConv2DValidIndirectBinaryParams(
    const BConvArgs &args, llvm::SmallVector<std::string> &strParams,
    std::vector<int8_t> &weightsData, std::vector<int16_t> &thresholdsData,
    int &scratchBytes) const {
  nn::ImToColValid::Params imToColParams(args.X, args.K, args.inputDepth);

  std::array<int, 4> filterShape = {args.outputDepth, args.filterHeight,
                                    args.filterWidth, args.inputDepth};
  nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
      (int8_t *)args.filter.data(), filterShape, 1, args.padValue);

  const int elementsPerByte = 8;
  int inputBytes =
      args.filterHeight * args.filterWidth * args.inputDepth / elementsPerByte;
  nn::MatMulBinary::Params afParams(args.outputDepth, inputBytes);

  // adjust the thresholds from xorpopcount space
  // to xcore space
  auto adjustedThresholds = nn::OT_binary::adjust_thresholds(
      args.threshold, args.inputDepth, args.K, rw);

  nn::OutputTransformFn::pad_final_access(adjustedThresholds, VPU_INT16_EPV,
                                          (int16_t)args.padValue);

  auto ir = nn::ImageRegion(0, 0, 0, args.Y.height, args.Y.width, args.Y.depth);
  nn::Filter2D::Params akParams(args.Y, ir, VPU_INT8_ACC_PERIOD);

  // TODO: Check serialization
  std::string akpStr = akParams.serialise<nn::Filter2D::Params>();
  std::string mfStr = imToColParams.serialise<nn::ImToColValid::Params>();
  std::string afStr = afParams.serialise<nn::MatMulBinary::Params>();
  std::string otStr = ""; // otParams.serialise<nn::OT_int8::Params>();

  strParams.push_back(akpStr);
  strParams.push_back(mfStr);
  strParams.push_back(afStr);
  strParams.push_back(otStr);
  weightsData = rw.weights;
  thresholdsData = adjustedThresholds;
  scratchBytes = nn::MatMulInt8::get_scratch_mem_bytes(inputBytes) + 32;

  return success();
}

LogicalResult ReplaceBConv2DPattern::getBConv2DValidDirectInt8Params(
    const BConvArgs &args, llvm::SmallVector<std::string> &strParams,
    std::vector<int8_t> &weightsData, std::vector<int16_t> &mulsBiasesData,
    int &scratchBytes) const {
  nn::DerefInputFn::Params imToColParams(args.X, args.K);

  std::array<int, 4> filterShape = {args.outputDepth, args.filterHeight,
                                    args.filterWidth, args.inputDepth};
  nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
      (int8_t *)args.filter.data(), filterShape, 1, args.padValue);

  nn::MatMulBinaryDirectFn::Params afParams(args.X, args.K, args.inputDepth);

  int receptiveVolume = args.filterHeight * args.filterWidth * args.inputDepth;
  nn::MulsAndBias mulAndBiases = nn::OT_int8_clamped::canonicalise_mul_and_bias(
      args.postActivationMultiplier, args.postActivationBias, receptiveVolume,
      args.clampMin, args.clampMax, args.outputDepth);
  auto accuOverlaps = nn::OT_int8_clamped::get_accumulator_overlaps(
      receptiveVolume, args.outputDepth, rw);
  nn::QuantisationParams qp =
      nn::OutputTransformFnInt8::quantise_activation(mulAndBiases);

  auto serialisedOffsetsMultipliersAndBiases =
      nn::OutputTransformFn::serialise_memory(accuOverlaps, qp.multipliers,
                                              qp.biases);
  nn::OutputTransformFn::pad_final_access(serialisedOffsetsMultipliersAndBiases,
                                          VPU_INT16_EPV,
                                          (int16_t)args.padValue);

  nn::OT_int8_clamped::Params otParams((int32_t)args.outputDepth,
                                       qp.initial_shr, qp.final_shr);

  auto ir = nn::ImageRegion(0, 0, 0, args.Y.height, args.Y.width, args.Y.depth);
  nn::Filter2D::Params akParams(args.Y, ir, VPU_INT8_ACC_PERIOD);

  // TODO: Check serialization
  std::string akpStr = akParams.serialise<nn::Filter2D::Params>();
  std::string mfStr = imToColParams.serialise<nn::DerefInputFn::Params>();
  std::string afStr = afParams.serialise<nn::MatMulBinaryDirectFn::Params>();
  std::string otStr = otParams.serialise<nn::OT_int8_clamped::Params>();

  strParams.push_back(akpStr);
  strParams.push_back(mfStr);
  strParams.push_back(afStr);
  strParams.push_back(otStr);
  weightsData = rw.weights;
  mulsBiasesData = serialisedOffsetsMultipliersAndBiases;
  scratchBytes = 0;

  return success();
}

LogicalResult ReplaceBConv2DPattern::getBConv2DValidIndirectInt8Params(
    const BConvArgs &args, llvm::SmallVector<std::string> &strParams,
    std::vector<int8_t> &weightsData, std::vector<int16_t> &mulsBiasesData,
    int &scratchBytes) const {
  nn::ImToColValid::Params imToColParams(args.X, args.K, args.inputDepth);

  std::array<int, 4> filterShape = {args.outputDepth, args.filterHeight,
                                    args.filterWidth, args.inputDepth};
  nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
      (int8_t *)args.filter.data(), filterShape, 1, args.padValue);

  const int elementsPerByte = 8;
  int inputBytes =
      args.filterHeight * args.filterWidth * args.inputDepth / elementsPerByte;

  nn::MatMulBinary::Params afParams(args.outputDepth, inputBytes);

  int receptiveVolume = args.filterHeight * args.filterWidth * args.inputDepth;
  nn::MulsAndBias mulAndBiases = nn::OT_int8_clamped::canonicalise_mul_and_bias(
      args.postActivationMultiplier, args.postActivationBias, receptiveVolume,
      args.clampMin, args.clampMax, args.outputDepth);
  auto accuOverlaps = nn::OT_int8_clamped::get_accumulator_overlaps(
      receptiveVolume, args.outputDepth, rw);
  nn::QuantisationParams qp =
      nn::OutputTransformFnInt8::quantise_activation(mulAndBiases);

  auto serialisedOffsetsMultipliersAndBiases =
      nn::OutputTransformFn::serialise_memory(accuOverlaps, qp.multipliers,
                                              qp.biases);
  nn::OutputTransformFn::pad_final_access(serialisedOffsetsMultipliersAndBiases,
                                          VPU_INT16_EPV,
                                          (int16_t)args.padValue);

  nn::OT_int8_clamped::Params otParams((int32_t)args.outputDepth,
                                       qp.initial_shr, qp.final_shr);

  auto ir = nn::ImageRegion(0, 0, 0, args.Y.height, args.Y.width, args.Y.depth);
  nn::Filter2D::Params akParams(args.Y, ir, VPU_INT8_ACC_PERIOD);

  // TODO: Check serialization
  std::string akpStr = akParams.serialise<nn::Filter2D::Params>();
  std::string mfStr = imToColParams.serialise<nn::ImToColValid::Params>();
  std::string afStr = afParams.serialise<nn::MatMulBinary::Params>();
  std::string otStr = otParams.serialise<nn::OT_int8_clamped::Params>();

  strParams.push_back(akpStr);
  strParams.push_back(mfStr);
  strParams.push_back(afStr);
  strParams.push_back(otStr);
  weightsData = rw.weights;
  mulsBiasesData = serialisedOffsetsMultipliersAndBiases;
  scratchBytes = nn::MatMulInt8::get_scratch_mem_bytes(inputBytes) + 32;

  return success();
}

} // namespace xcore
} // namespace mlir
