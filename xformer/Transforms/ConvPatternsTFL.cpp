// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Transforms/ConvPatterns.h"
#include "Transforms/Options.h"
#include "Utils/Diagnostics.h"

#include "tensorflow/core/framework/kernel_shape_util.h"

namespace mlir {
namespace xcore {

// TFL Conv2D Base class implementation
// TFLConvOpType would be XC_FakeConv2D or XC_FakeDepthwiseConv2D
template <typename ConcreteType, typename TFLConvOpType>
LogicalResult ReplaceConv2DBase<ConcreteType, TFLConvOpType>::getArgs(
    TFLConvOpType conv2DOp, TFLConvArgs &args) const {
  // Retrieve remaining args
  // Get output zero point
  auto outputType =
      conv2DOp.getOutput().getType().template dyn_cast<RankedTensorType>();
  auto outputQType =
      outputType.getElementType()
          .template dyn_cast<mlir::quant::UniformQuantizedType>();
  auto outputScale = outputQType.getScale();
  auto outputZeroPoint = outputQType.getZeroPoint();

  // Get input zero point
  auto inputType =
      conv2DOp.getInput().getType().template dyn_cast<RankedTensorType>();
  auto inputQType = inputType.getElementType()
                        .template dyn_cast<mlir::quant::UniformQuantizedType>();
  auto inputScale = inputQType.getScale();
  auto inputZeroPoint = inputQType.getZeroPoint();

  // Get filter values
  auto filterQConstOp =
      dyn_cast<TFL::QConstOp>(conv2DOp.getFilter().getDefiningOp());
  auto filter = filterQConstOp.getValue().template cast<DenseElementsAttr>();
  auto filterVector =
      std::vector<int8_t>{filter.template getValues<int8_t>().begin(),
                          filter.template getValues<int8_t>().end()};

  // Get bias values
  // If no bias exists, create vector with zero values
  std::vector<int32_t> biasVector;
  if (!conv2DOp.getBias().getType().template isa<NoneType>()) {
    DenseElementsAttr biasesAttr;
    if (conv2DOp.getBias()
            .getType()
            .template cast<ShapedType>()
            .getElementType()
            .template isa<quant::QuantizedType>()) {
      auto biasQConstOp =
          dyn_cast<TFL::QConstOp>(conv2DOp.getBias().getDefiningOp());
      biasesAttr = biasQConstOp.getValue().template cast<DenseElementsAttr>();
    } else {
      matchPattern(conv2DOp.getBias(), m_Constant(&biasesAttr));
    }
    biasVector =
        std::vector<int32_t>{biasesAttr.template getValues<int32_t>().begin(),
                             biasesAttr.template getValues<int32_t>().end()};
  } else {
    biasVector = std::vector<int32_t>(args.outputDepth, 0);
  }

  // Calculate effectiveOutputScale
  std::vector<float> effectiveOutputScaleVector;
  auto filterQConstOpType =
      filterQConstOp.getQtype().template cast<RankedTensorType>();
  bool isPerChannelQuantized = false;
  double filterScale;
  ArrayRef<double> filterScales;
  if (auto filterQType =
          filterQConstOpType.getElementType()
              .template dyn_cast<mlir::quant::UniformQuantizedType>()) {
    filterScale = filterQType.getScale();
  } else if (auto filterQType =
                 filterQConstOpType.getElementType()
                     .template dyn_cast<
                         mlir::quant::UniformQuantizedPerAxisType>()) {
    isPerChannelQuantized = true;
    filterScales = filterQType.getScales();
  } else {
    return failure();
  }

  // Conv is quantized along dimension 0
  // DepthwiseConv is quantized along dimension 3
  // https://www.tensorflow.org/lite/performance/quantization_spec
  int dim = static_cast<const ConcreteType *>(this)->getQuantizationIndex();
  auto filterType =
      conv2DOp.getFilter().getType().template dyn_cast<RankedTensorType>();
  auto numOutputChannels = filterType.getDimSize(dim);
  for (int i = 0; i < numOutputChannels; ++i) {
    auto scale = isPerChannelQuantized ? filterScales[i] : filterScale;
    assert(outputScale != 0 && "outputScale should not be zero!");
    effectiveOutputScaleVector.push_back(inputScale * scale / outputScale);
  }

  // Clamp multipliers
  float minVal = *std::min_element(effectiveOutputScaleVector.begin(),
                                   effectiveOutputScaleVector.end());
  // float avgVal = std::accumulate(effectiveOutputScaleVector.begin(),
  // effectiveOutputScaleVector.end(), 0.0) /
  // effectiveOutputScaleVector.size();
  for (int i = 0; i < effectiveOutputScaleVector.size(); ++i) {
    float tmp = std::min(effectiveOutputScaleVector[i],
                         minVal * convMultiplierFactorOption);
    if (tmp != effectiveOutputScaleVector[i]) {
      // Mention which numbers have been clamped
      std::stringstream msg;
      msg << std::endl
          << "CLAMPED conv multiplier index " << i << " from " << std::fixed
          << std::setprecision(18) << effectiveOutputScaleVector[i] << " to "
          << tmp << std::endl;
      conv2DOp.emitRemark(utils::getMsgWithLocPrefix(conv2DOp, msg.str()));
      effectiveOutputScaleVector[i] = tmp;
    }
  }

  // Find padding values
  int64_t newHeight, newWidth;
  int64_t padTop, padBottom, padLeft, padRight;

  if (conv2DOp.getPadding() == "EXPLICIT") {
    DenseElementsAttr paddingAttr;
    matchPattern(conv2DOp.getPaddingValues(), m_Constant(&paddingAttr));
    // The padding values for the PadOp are stored as a 4x2 tensor 0,0 and 0,1
    // is for the batch dimension and 3,0, and 3,1 for the channel/depth 1,0 and
    // 1,1 is top and bottom, and 2,0 and 2,1 is left and right which are the
    // padding values we need
    padTop = paddingAttr.template getValues<int32_t>()[{1, 0}];
    padBottom = paddingAttr.template getValues<int32_t>()[{1, 1}];
    padLeft = paddingAttr.template getValues<int32_t>()[{2, 0}];
    padRight = paddingAttr.template getValues<int32_t>()[{2, 1}];
  } else {
    tensorflow::Padding opPadding = conv2DOp.getPadding() == "VALID"
                                        ? tensorflow::Padding::VALID
                                        : tensorflow::Padding::SAME;
    if (tensorflow::GetWindowedOutputSizeVerboseV2(
            args.inputHeight, args.filterHeight, conv2DOp.getDilationHFactor(),
            conv2DOp.getStrideH(), opPadding, &newHeight, &padTop,
            &padBottom) != tensorflow::OkStatus()) {
      return failure();
    }
    if (tensorflow::GetWindowedOutputSizeVerboseV2(
            args.inputWidth, args.filterWidth, conv2DOp.getDilationWFactor(),
            conv2DOp.getStrideW(), opPadding, &newWidth, &padLeft,
            &padRight) != tensorflow::OkStatus()) {
      return failure();
    }
  }
  args.toBePadded =
      padTop != 0 || padBottom != 0 || padLeft != 0 || padRight != 0;

  args.padding = {static_cast<int16_t>(padTop), static_cast<int16_t>(padLeft),
                  static_cast<int16_t>(padBottom),
                  static_cast<int16_t>(padRight)};
  args.padValue = 0;

  // Init lib_nn structs
  args.Y =
      nn::ImageGeometry(args.outputHeight, args.outputWidth, args.outputDepth);
  args.X =
      nn::ImageGeometry(args.inputHeight, args.inputWidth, args.inputDepth);
  args.K = nn::WindowGeometry(
      args.filterHeight, args.filterWidth, args.filterDepth, -args.padding.top,
      -args.padding.left, conv2DOp.getStrideH(), conv2DOp.getStrideW(), 1,
      conv2DOp.getDilationHFactor(), conv2DOp.getDilationWFactor());

  args.outputZeroPoint = outputZeroPoint;
  args.inputZeroPoint = inputZeroPoint;
  args.filter = filterVector;
  args.bias = biasVector;
  args.effectiveMultiplier = effectiveOutputScaleVector;

  // Obtain quant error threshold from command-line option
  args.quantErrorThreshold = convQuantErrorThresholdOption;
  args.quantErrorFullCheckEnabled = convForceErrorCheckOption;

  return success();
}

// Since we are not defining the template functions in the header, we need
// explicit template class instantiations to avoid linker errors
template class ReplaceConv2DBase<ReplaceConv2DPattern, FakeConv2DOp>;
template class ReplaceConv2DBase<ReplaceDepthwiseConv2DPattern,
                                 FakeDepthwiseConv2DOp>;

//
//
//
// Handle TFL Conv2D specific functions
LogicalResult ReplaceConv2DPattern::getKernelType(const TFLConvArgs &args,
                                                  Conv2DType &kt) const {
  if (args.toBePadded) {
    kt = Conv2DType::PaddedIndirect;
  } else if (args.inputDepth % 32 == 0 && args.outputDepth % 16 == 0) {
    kt = Conv2DType::ValidDirect;
  } else {
    kt = Conv2DType::ValidIndirect;
  }
  return success();
}

LogicalResult ReplaceConv2DPattern::getSerializedParamsAndTensors(
    const TFLConvArgs &args, const Conv2DType &kt, OtType &otType,
    llvm::SmallVector<std::string> &strParams,
    llvm::SmallVector<std::string> &abstractKernelParams,
    std::vector<int8_t> &weightsData, std::vector<int16_t> &mulsBiasesData,
    int &scratchBytes) const {
  switch (kt) {
  case Conv2DType::ValidDirect:
    if (failed(getConv2DValidDirectParams(args, strParams, abstractKernelParams,
                                          weightsData, scratchBytes))) {
      return failure();
    }
    break;
  case Conv2DType::ValidIndirect:
    if (failed(getConv2DValidIndirectParams(args, strParams,
                                            abstractKernelParams, weightsData,
                                            scratchBytes))) {
      return failure();
    }
    break;
  case Conv2DType::PaddedIndirect:
    if (failed(getConv2DPaddedIndirectParams(args, strParams,
                                             abstractKernelParams, weightsData,
                                             scratchBytes))) {
      return failure();
    }
    break;
  default:
    // This shouldn't happen!
    return failure();
  }

  assert(strParams.size() == 2 &&
         "strParams should contain memcpyFn params and aggregateFn params!");
  std::string otStr;
  if (failed(getOutputTransformParams(args, otStr, otType, mulsBiasesData))) {
    return failure();
  }
  strParams.push_back(otStr);

  return success();
}

LogicalResult ReplaceConv2DPattern::getOutputTransformParams(
    const TFLConvArgs &args, std::string &otStr, OtType &otType,
    std::vector<int16_t> &mulsBiasesData) const {

  otType = OtType::Group;

  if (convDebugOption) {
    std::string message;
    llvm::raw_string_ostream os(message);
    std::cout << std::endl;
    args.convOp->print(os);
    std::stringstream msg;
    msg << "Conv2D DEBUG" << std::endl;
    std::cout << message << std::endl
              << utils::getMsgWithLocPrefix(*args.convOp, msg.str())
              << std::endl;
  }

  nn::MulsAndBias mulAndBiases =
      nn::OutputTransformFnInt8::canonicalise_mul_and_bias(
          args.effectiveMultiplier, args.bias, args.filter, args.inputZeroPoint,
          args.outputZeroPoint, args.outputDepth, convDebugOption);
  if (convDebugOption) {
    nn::OutputTransformFn::layerwise_stats(mulAndBiases);
  }

  // Try group OT
  auto quantizer = nn::OutputTransformFnInt8_Group::Quantizer();
  nn::OutputTransformFnInt8_Group::QuantisationParams qp =
      quantizer.quantise_activation(mulAndBiases, convDebugOption);

  double quantError = nn::OutputTransformFnInt8::get_quant_error(
      mulAndBiases, qp, args.quantErrorFullCheckEnabled);
  if (quantError > args.quantErrorThreshold) {
    // Try channelwise OT
    auto quantizer = nn::OutputTransformFnInt8_Channelwise::Quantizer();
    nn::OutputTransformFnInt8_Channelwise::QuantisationParams qp =
        quantizer.quantise_activation(mulAndBiases, convDebugOption);

    quantError = nn::OutputTransformFnInt8_Channelwise::get_quant_error(
        mulAndBiases, qp, true);

    if (quantError > args.quantErrorThreshold) {
      std::stringstream msg;
      msg << "Quantization error of " << quantError
          << " larger than set threshold of " << args.quantErrorThreshold
          << ", therefore reverting to reference Conv2D op!" << std::endl
          << "Inspect the output, and if suitable, set a "
             "higher threshold with --xcore-conv-err-threshold."
          << std::endl;
      args.convOp->emitWarning(
          utils::getMsgWithLocPrefix(*args.convOp, msg.str()));
      return failure();
    } else {
      otType = OtType::Channelwise;

      auto serialisedMultipliersAndBiases =
          nn::OutputTransformFn::serialise_memory(qp.initial_shifts,
                                                  qp.multipliers, qp.biases);
      nn::OutputTransformFn::pad_final_access(serialisedMultipliersAndBiases,
                                              VPU_INT16_EPV,
                                              (int16_t)args.padValue);
      nn::OT_int8_channelwise ot((int32_t)args.outputDepth, qp.final_shr);
      auto otParams = ot.getParams();
      otStr = std::string((char *)&otParams, sizeof(otParams));

      mulsBiasesData = serialisedMultipliersAndBiases;
      return success();
    }
  }

  if (otType == OtType::Group) {
    auto serialisedMultipliersAndBiases =
        nn::OutputTransformFn::serialise_memory(qp.multipliers, qp.biases);
    nn::OutputTransformFn::pad_final_access(
        serialisedMultipliersAndBiases, VPU_INT16_EPV, (int16_t)args.padValue);
    nn::OT_int8 ot((int32_t)args.outputDepth, qp.initial_shr, qp.final_shr);
    auto otParams = ot.getParams();
    otStr = std::string((char *)&otParams, sizeof(otParams));

    mulsBiasesData = serialisedMultipliersAndBiases;
  }

  return success();
}

LogicalResult ReplaceConv2DPattern::getConv2DPaddedIndirectParams(
    const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
    llvm::SmallVector<std::string> &abstractKernelParams,
    std::vector<int8_t> &weightsData, int &scratchBytes) const {

  nn::ImToColPadded imToCol(args.X, args.K, args.padding, args.inputDepth,
                            args.inputZeroPoint);
  auto imToColParams = imToCol.getParams();

  std::array<int, 4> filterShape = {args.outputDepth, args.filterHeight,
                                    args.filterWidth, args.inputDepth};
  nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
      (int8_t *)args.filter.data(), filterShape, 8, args.padValue);
  int inputBytes = args.filterHeight * args.filterWidth * args.inputDepth;
  nn::MatMulInt8 af(args.outputDepth, inputBytes);
  auto afParams = af.getParams();

  std::string mfStr =
      std::string((char *)&imToColParams, sizeof(imToColParams));
  std::string afStr = std::string((char *)&afParams, sizeof(afParams));

  abstractKernelParams =
      getAbstractKernelParamsForMultipleThreads(args.imageRegionSplits, args.Y);
  strParams.push_back(mfStr);
  strParams.push_back(afStr);
  weightsData = rw.weights;
  scratchBytes =
      nn::MatMulInt8::get_scratch_mem_bytes(inputBytes) + 32; //[asj] FIXME

  return success();
}

LogicalResult ReplaceConv2DPattern::getConv2DValidIndirectParams(
    const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
    llvm::SmallVector<std::string> &abstractKernelParams,
    std::vector<int8_t> &weightsData, int &scratchBytes) const {

  nn::ImToColValid imToCol(args.X, args.K, args.inputDepth);
  auto imToColParams = imToCol.getParams();

  std::array<int, 4> filterShape = {args.outputDepth, args.filterHeight,
                                    args.filterWidth, args.inputDepth};
  nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
      (int8_t *)args.filter.data(), filterShape, 8, args.padValue);
  int inputBytes = args.filterHeight * args.filterWidth * args.inputDepth;
  nn::MatMulInt8 af(args.outputDepth, inputBytes);
  auto afParams = af.getParams();

  std::string mfStr =
      std::string((char *)&imToColParams, sizeof(imToColParams));
  std::string afStr = std::string((char *)&afParams, sizeof(afParams));

  abstractKernelParams =
      getAbstractKernelParamsForMultipleThreads(args.imageRegionSplits, args.Y);
  strParams.push_back(mfStr);
  strParams.push_back(afStr);
  weightsData = rw.weights;
  scratchBytes =
      nn::MatMulInt8::get_scratch_mem_bytes(inputBytes) + 32; //[asj] FIXME

  return success();
}

LogicalResult ReplaceConv2DPattern::getConv2DValidDirectParams(
    const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
    llvm::SmallVector<std::string> &abstractKernelParams,
    std::vector<int8_t> &weightsData, int &scratchBytes) const {

  nn::DerefInputFn imToCol(args.X, args.K);
  auto imToColParams = imToCol.getParams();

  std::array<int, 4> filterShape = {args.outputDepth, args.filterHeight,
                                    args.filterWidth, args.inputDepth};
  nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
      (int8_t *)args.filter.data(), filterShape, 8, args.padValue);
  nn::MatMulDirectFn af(args.X, args.K, args.inputDepth);
  auto afParams = af.getParams();

  std::string mfStr =
      std::string((char *)&imToColParams, sizeof(imToColParams));
  std::string afStr = std::string((char *)&afParams, sizeof(afParams));

  abstractKernelParams =
      getAbstractKernelParamsForMultipleThreads(args.imageRegionSplits, args.Y);
  strParams.push_back(mfStr);
  strParams.push_back(afStr);
  weightsData = rw.weights;
  scratchBytes = 0;

  return success();
}

//
//
//
// Handle TFL DepthwiseConv2D
LogicalResult
ReplaceDepthwiseConv2DPattern::getKernelType(const TFLConvArgs &args,
                                             Conv2DType &kt) const {
  if (args.toBePadded) {
    kt = Conv2DType::DepthwisePaddedIndirect;
  } else {
    kt = Conv2DType::DepthwiseValidDirect;
  }
  return success();
}

LogicalResult ReplaceDepthwiseConv2DPattern::getSerializedParamsAndTensors(
    const TFLConvArgs &args, const Conv2DType &kt, OtType &otType,
    llvm::SmallVector<std::string> &strParams,
    llvm::SmallVector<std::string> &abstractKernelParams,
    std::vector<int8_t> &weightsData, std::vector<int16_t> &mulsBiasesData,
    int &scratchBytes) const {
  switch (kt) {
  case Conv2DType::DepthwiseValidDirect:
    if (failed(getDepthwiseConv2DValidDirectParams(
            args, strParams, abstractKernelParams, weightsData,
            scratchBytes))) {
      return failure();
    }
    break;
  case Conv2DType::DepthwisePaddedIndirect:
    if (failed(getDepthwiseConv2DPaddedIndirectParams(
            args, strParams, abstractKernelParams, weightsData,
            scratchBytes))) {
      return failure();
    }
    break;
  default:
    // This shouldn't happen!
    return failure();
  }

  assert(strParams.size() == 2 &&
         "strParams should contain memcpyFn params and aggregateFn params!");
  std::string otStr;

  if (failed(getOutputTransformParams(args, otStr, otType, mulsBiasesData))) {
    return failure();
  }
  strParams.push_back(otStr);

  return success();
}

LogicalResult ReplaceDepthwiseConv2DPattern::getOutputTransformParams(
    const TFLConvArgs &args, std::string &otStr, OtType &otType,
    std::vector<int16_t> &mulsBiasesData) const {
  std::array<int, 4> filterShape = {1, args.filterHeight, args.filterWidth,
                                    args.inputDepth};

  otType = OtType::Group;

  if (convDebugOption) {
    std::string message;
    llvm::raw_string_ostream os(message);
    std::cout << std::endl;
    args.convOp->print(os);
    std::stringstream msg;
    msg << "DepthwiseConv2D DEBUG" << std::endl;
    std::cout << message << std::endl
              << utils::getMsgWithLocPrefix(*args.convOp, msg.str())
              << std::endl;
  }

  nn::MulsAndBias mulAndBiases =
      nn::OutputTransformFnInt8::canonicalise_mul_and_bias_dw(
          args.effectiveMultiplier, args.bias, args.filter, filterShape,
          args.inputZeroPoint, args.outputZeroPoint, args.outputDepth,
          convDebugOption);
  if (convDebugOption) {
    nn::OutputTransformFn::layerwise_stats(mulAndBiases);
  }

  // Try group OT
  auto quantizer = nn::OutputTransformFnInt8_Group::Quantizer();
  nn::OutputTransformFnInt8_Group::QuantisationParams qp =
      quantizer.quantise_activation(mulAndBiases, convDebugOption);

  double quantError = nn::OutputTransformFnInt8::get_quant_error(
      mulAndBiases, qp, args.quantErrorFullCheckEnabled);
  if (quantError > args.quantErrorThreshold) {
    // Try channelwise OT
    auto quantizer = nn::OutputTransformFnInt8_Channelwise::Quantizer();
    nn::OutputTransformFnInt8_Channelwise::QuantisationParams qp =
        quantizer.quantise_activation(mulAndBiases, convDebugOption);

    quantError = nn::OutputTransformFnInt8_Channelwise::get_quant_error(
        mulAndBiases, qp, true);
    if (quantError > args.quantErrorThreshold) {
      std::stringstream msg;
      msg << "Quantization error of " << quantError
          << " larger than set threshold of " << args.quantErrorThreshold
          << ", therefore reverting to reference DepthwiseConv2D op!"
          << std::endl
          << "Inspect the output, and if suitable, set a "
             "higher threshold with --xcore-conv-err-threshold."
          << std::endl;
      args.convOp->emitWarning(
          utils::getMsgWithLocPrefix(*args.convOp, msg.str()));
      return failure();
    } else {
      otType = OtType::Channelwise;

      auto serialisedMultipliersAndBiases =
          nn::OutputTransformFn::serialise_memory(qp.initial_shifts,
                                                  qp.multipliers, qp.biases);
      nn::OutputTransformFn::pad_final_access(serialisedMultipliersAndBiases,
                                              VPU_INT16_EPV,
                                              (int16_t)args.padValue);
      nn::OT_int8_channelwise ot((int32_t)args.outputDepth, qp.final_shr);
      auto otParams = ot.getParams();

      otStr = std::string((char *)&otParams, sizeof(otParams));
      mulsBiasesData = serialisedMultipliersAndBiases;
      return success();
    }
  }
  if (otType == OtType::Group) {
    auto serialisedMultipliersAndBiases =
        nn::OutputTransformFn::serialise_memory(qp.multipliers, qp.biases);
    nn::OutputTransformFn::pad_final_access(
        serialisedMultipliersAndBiases, VPU_INT16_EPV, (int16_t)args.padValue);
    nn::OT_int8 ot((int32_t)args.outputDepth, qp.initial_shr, qp.final_shr);
    auto otParams = ot.getParams();

    otStr = std::string((char *)&otParams, sizeof(otParams));
    mulsBiasesData = serialisedMultipliersAndBiases;
    return success();
  }
}

LogicalResult
ReplaceDepthwiseConv2DPattern::getDepthwiseConv2DValidDirectParams(
    const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
    llvm::SmallVector<std::string> &abstractKernelParams,
    std::vector<int8_t> &weightsData, int &scratchBytes) const {

  nn::DerefInputFn imToCol(args.X, args.K);
  auto imToColParams = imToCol.getParams();

  std::array<int, 4> filterShape = {1, args.filterHeight, args.filterWidth,
                                    args.inputDepth};
  nn::Conv2dReorderedWeights rw = nn::MatMulDirectFn_DW::reorder_kernel_weights(
      (int8_t *)args.filter.data(), filterShape, args.padValue);
  nn::MatMulDirectFn_DW af(args.X, args.K);
  auto afParams = af.getParams();

  std::string mfStr =
      std::string((char *)&imToColParams, sizeof(imToColParams));
  std::string afStr = std::string((char *)&afParams, sizeof(afParams));

  abstractKernelParams =
      getAbstractKernelParamsForMultipleThreads(args.imageRegionSplits, args.Y);
  strParams.push_back(mfStr);
  strParams.push_back(afStr);
  weightsData = rw.weights;
  scratchBytes = 0;

  return success();
}

LogicalResult
ReplaceDepthwiseConv2DPattern::getDepthwiseConv2DPaddedIndirectParams(
    const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
    llvm::SmallVector<std::string> &abstractKernelParams,
    std::vector<int8_t> &weightsData, int &scratchBytes) const {

  nn::ImToColPadded imToCol(args.X, args.K, args.padding, 16,
                            args.inputZeroPoint);
  auto imToColParams = imToCol.getParams();

  std::array<int, 4> filterShape = {1, args.filterHeight, args.filterWidth,
                                    args.inputDepth};
  nn::Conv2dReorderedWeights rw = nn::MatMulDirectFn_DW::reorder_kernel_weights(
      (int8_t *)args.filter.data(), filterShape, args.padValue);
  nn::MatMulDirectFn_DW af(args.K);
  auto afParams = af.getParams();

  std::string mfStr =
      std::string((char *)&imToColParams, sizeof(imToColParams));
  std::string afStr = std::string((char *)&afParams, sizeof(afParams));

  abstractKernelParams =
      getAbstractKernelParamsForMultipleThreads(args.imageRegionSplits, args.Y);
  strParams.push_back(mfStr);
  strParams.push_back(afStr);
  weightsData = rw.weights;
  scratchBytes = nn::MatMulDirectFn_DW::get_scratch_mem_bytes(filterShape);

  return success();
}

} // namespace xcore
} // namespace mlir
