// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "Transforms/ConvPatterns.h"

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
      conv2DOp.output().getType().template dyn_cast<RankedTensorType>();
  auto outputQType =
      outputType.getElementType()
          .template dyn_cast<mlir::quant::UniformQuantizedType>();
  auto outputScale = outputQType.getScale();
  auto outputZeroPoint = outputQType.getZeroPoint();

  // Get input zero point
  auto inputType =
      conv2DOp.input().getType().template dyn_cast<RankedTensorType>();
  auto inputQType = inputType.getElementType()
                        .template dyn_cast<mlir::quant::UniformQuantizedType>();
  auto inputScale = inputQType.getScale();
  auto inputZeroPoint = inputQType.getZeroPoint();

  // Get filter values
  auto filterQConstOp =
      dyn_cast<TFL::QConstOp>(conv2DOp.filter().getDefiningOp());
  auto filter = filterQConstOp.value().template cast<DenseElementsAttr>();
  auto filterVector =
      std::vector<int8_t>{filter.template getValues<int8_t>().begin(),
                          filter.template getValues<int8_t>().end()};

  // Get bias values
  DenseElementsAttr biases;
  if (conv2DOp.bias()
          .getType()
          .template cast<ShapedType>()
          .getElementType()
          .template isa<quant::QuantizedType>()) {
    auto biasQConstOp =
        dyn_cast<TFL::QConstOp>(conv2DOp.bias().getDefiningOp());
    biases = biasQConstOp.value().template cast<DenseElementsAttr>();
  } else {
    auto biasConstOp =
        dyn_cast<mlir::ConstantOp>(conv2DOp.bias().getDefiningOp());
    biases = biasConstOp.value().template cast<DenseElementsAttr>();
  }
  auto biasVector =
      std::vector<int32_t>{biases.template getValues<int32_t>().begin(),
                           biases.template getValues<int32_t>().end()};

  // Calculate effectiveOutputScale
  std::vector<float> effectiveOutputScaleVector;
  auto filterQConstOpType =
      filterQConstOp.qtype().template cast<RankedTensorType>();
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
      conv2DOp.filter().getType().template dyn_cast<RankedTensorType>();
  auto numOutputChannels = filterType.getDimSize(dim);
  for (int i = 0; i < numOutputChannels; ++i) {
    auto scale = isPerChannelQuantized ? filterScales[i] : filterScale;
    assert(outputScale != 0 && "outputScale should not be zero!");
    effectiveOutputScaleVector.push_back(inputScale * scale / outputScale);
  }

  // Find padding values
  int64_t newHeight, newWidth;
  int64_t padTop, padBottom, padLeft, padRight;

  if (conv2DOp.padding() == "EXPLICIT") {
    auto paddingValuesConstOp =
        dyn_cast<mlir::ConstantOp>(conv2DOp.padding_values().getDefiningOp());
    auto paddingValues =
        paddingValuesConstOp.value().template cast<DenseElementsAttr>();
    padTop = paddingValues.template getValue<int32_t>({1, 0});
    padBottom = paddingValues.template getValue<int32_t>({1, 1});
    padLeft = paddingValues.template getValue<int32_t>({2, 0});
    padRight = paddingValues.template getValue<int32_t>({2, 1});
  } else {
    tensorflow::Padding opPadding = conv2DOp.padding() == "VALID"
                                        ? tensorflow::Padding::VALID
                                        : tensorflow::Padding::SAME;
    if (tensorflow::GetWindowedOutputSizeVerboseV2(
            args.inputHeight, args.filterHeight, conv2DOp.dilation_h_factor(),
            conv2DOp.stride_h(), opPadding, &newHeight, &padTop,
            &padBottom) != tensorflow::Status::OK()) {
      return failure();
    }
    if (tensorflow::GetWindowedOutputSizeVerboseV2(
            args.inputWidth, args.filterWidth, conv2DOp.dilation_w_factor(),
            conv2DOp.stride_w(), opPadding, &newWidth, &padLeft,
            &padRight) != tensorflow::Status::OK()) {
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
      -args.padding.left, conv2DOp.stride_h(), conv2DOp.stride_w(), 1,
      conv2DOp.dilation_h_factor(), conv2DOp.dilation_w_factor());

  args.outputZeroPoint = outputZeroPoint;
  args.inputZeroPoint = inputZeroPoint;
  args.filter = filterVector;
  args.bias = biasVector;
  args.effectiveMultiplier = effectiveOutputScaleVector;

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
    const TFLConvArgs &args, const Conv2DType &kt,
    llvm::SmallVector<std::string> &strParams, std::vector<int8_t> &weightsData,
    std::vector<int16_t> &mulsBiasesData, int &scratchBytes) const {
  switch (kt) {
  case Conv2DType::ValidDirect:
    if (failed(getConv2DValidDirectParams(args, strParams, weightsData,
                                          mulsBiasesData, scratchBytes))) {
      return failure();
    }
    break;
  case Conv2DType::ValidIndirect:
    if (failed(getConv2DValidIndirectParams(args, strParams, weightsData,
                                            mulsBiasesData, scratchBytes))) {
      return failure();
    }
    break;
  case Conv2DType::PaddedIndirect:
    if (failed(getConv2DPaddedIndirectParams(args, strParams, weightsData,
                                             mulsBiasesData, scratchBytes))) {
      return failure();
    }
    break;
  default:
    // This shouldn't happen!
    return failure();
  }

  return success();
}

LogicalResult ReplaceConv2DPattern::getConv2DPaddedIndirectParams(
    const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
    std::vector<int8_t> &weightsData, std::vector<int16_t> &mulsBiasesData,
    int &scratchBytes) const {

  nn::ImToColPadded::Params imToColParams(args.X, args.K, args.padding,
                                          args.inputDepth, args.inputZeroPoint);

  std::array<int, 4> filterShape = {args.outputDepth, args.filterHeight,
                                    args.filterWidth, args.inputDepth};
  nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
      (int8_t *)args.filter.data(), filterShape, 8, args.padValue);
  int inputBytes = args.filterHeight * args.filterWidth * args.inputDepth;
  nn::MatMulInt8::Params afParams(args.outputDepth, inputBytes);

  nn::MulsAndBias mulsAndBiases =
      nn::OutputTransformFnInt8::canonicalise_mul_and_bias(
          args.effectiveMultiplier, args.bias, args.filter, args.inputZeroPoint,
          args.outputZeroPoint, args.outputDepth);
  nn::QuantisationParams qp =
      nn::OutputTransformFnInt8::quantise_activation(mulsAndBiases);
  auto serialisedMultipliersAndBiases =
      nn::OutputTransformFn::serialise_memory(qp.multipliers, qp.biases);
  nn::OutputTransformFn::pad_final_access(
      serialisedMultipliersAndBiases, VPU_INT16_EPV, (int16_t)args.padValue);
  nn::OT_int8::Params otParams((int32_t)args.outputDepth, qp.initial_shr,
                               qp.final_shr);

  auto ir = nn::ImageRegion(0, 0, 0, args.Y.height, args.Y.width, args.Y.depth);
  nn::Filter2D::Params akParams(args.Y, ir, VPU_INT8_ACC_PERIOD);

  // TODO: Check serialization
  std::string akpStr = akParams.serialise<nn::Filter2D::Params>();
  std::string mfStr = imToColParams.serialise<nn::ImToColPadded::Params>();
  std::string afStr = afParams.serialise<nn::MatMulInt8::Params>();
  std::string otStr = otParams.serialise<nn::OT_int8::Params>();

  strParams.push_back(akpStr);
  strParams.push_back(mfStr);
  strParams.push_back(afStr);
  strParams.push_back(otStr);
  weightsData = rw.weights;
  mulsBiasesData = serialisedMultipliersAndBiases;
  scratchBytes =
      nn::MatMulInt8::get_scratch_mem_bytes(inputBytes) + 32; //[asj] FIXME

  return success();
}

LogicalResult ReplaceConv2DPattern::getConv2DValidIndirectParams(
    const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
    std::vector<int8_t> &weightsData, std::vector<int16_t> &mulsBiasesData,
    int &scratchBytes) const {

  nn::ImToColValid::Params imToColParams(args.X, args.K, args.inputDepth);

  std::array<int, 4> filterShape = {args.outputDepth, args.filterHeight,
                                    args.filterWidth, args.inputDepth};
  nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
      (int8_t *)args.filter.data(), filterShape, 8, args.padValue);
  int inputBytes = args.filterHeight * args.filterWidth * args.inputDepth;
  nn::MatMulInt8::Params afParams(args.outputDepth, inputBytes);

  nn::MulsAndBias mulsAndBiases =
      nn::OutputTransformFnInt8::canonicalise_mul_and_bias(
          args.effectiveMultiplier, args.bias, args.filter, args.inputZeroPoint,
          args.outputZeroPoint, args.outputDepth);
  nn::QuantisationParams qp =
      nn::OutputTransformFnInt8::quantise_activation(mulsAndBiases);
  auto serialisedMultipliersAndBiases =
      nn::OutputTransformFn::serialise_memory(qp.multipliers, qp.biases);
  nn::OutputTransformFn::pad_final_access(
      serialisedMultipliersAndBiases, VPU_INT16_EPV, (int16_t)args.padValue);
  nn::OT_int8::Params otParams((int32_t)args.outputDepth, qp.initial_shr,
                               qp.final_shr);

  auto ir = nn::ImageRegion(0, 0, 0, args.Y.height, args.Y.width, args.Y.depth);
  nn::Filter2D::Params akParams(args.Y, ir, VPU_INT8_ACC_PERIOD);

  // TODO: Check serialization
  std::string akpStr = akParams.serialise<nn::Filter2D::Params>();
  std::string mfStr = imToColParams.serialise<nn::ImToColValid::Params>();
  std::string afStr = afParams.serialise<nn::MatMulInt8::Params>();
  std::string otStr = otParams.serialise<nn::OT_int8::Params>();

  strParams.push_back(akpStr);
  strParams.push_back(mfStr);
  strParams.push_back(afStr);
  strParams.push_back(otStr);
  weightsData = rw.weights;
  mulsBiasesData = serialisedMultipliersAndBiases;
  scratchBytes =
      nn::MatMulInt8::get_scratch_mem_bytes(inputBytes) + 32; //[asj] FIXME

  return success();
}

LogicalResult ReplaceConv2DPattern::getConv2DValidDirectParams(
    const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
    std::vector<int8_t> &weightsData, std::vector<int16_t> &mulsBiasesData,
    int &scratchBytes) const {

  nn::DerefInputFn::Params imToColParams(args.X, args.K);

  std::array<int, 4> filterShape = {args.outputDepth, args.filterHeight,
                                    args.filterWidth, args.inputDepth};
  nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
      (int8_t *)args.filter.data(), filterShape, 8, args.padValue);
  nn::MatMulDirectFn::Params afParams(args.X, args.K, args.inputDepth);

  nn::MulsAndBias mulsAndBiases =
      nn::OutputTransformFnInt8::canonicalise_mul_and_bias(
          args.effectiveMultiplier, args.bias, args.filter, args.inputZeroPoint,
          args.outputZeroPoint, args.outputDepth);
  nn::QuantisationParams qp =
      nn::OutputTransformFnInt8::quantise_activation(mulsAndBiases);
  auto serialisedMultipliersAndBiases =
      nn::OutputTransformFn::serialise_memory(qp.multipliers, qp.biases);
  nn::OutputTransformFn::pad_final_access(
      serialisedMultipliersAndBiases, VPU_INT16_EPV, (int16_t)args.padValue);
  nn::OT_int8::Params otParams((int32_t)args.outputDepth, qp.initial_shr,
                               qp.final_shr);

  auto ir = nn::ImageRegion(0, 0, 0, args.Y.height, args.Y.width, args.Y.depth);
  nn::Filter2D::Params akParams(args.Y, ir, VPU_INT8_ACC_PERIOD);

  // TODO: Check serialization
  std::string akpStr = akParams.serialise<nn::Filter2D::Params>();
  std::string mfStr = imToColParams.serialise<nn::DerefInputFn::Params>();
  std::string afStr = afParams.serialise<nn::MatMulDirectFn::Params>();
  std::string otStr = otParams.serialise<nn::OT_int8::Params>();

  strParams.push_back(akpStr);
  strParams.push_back(mfStr);
  strParams.push_back(afStr);
  strParams.push_back(otStr);
  weightsData = rw.weights;
  mulsBiasesData = serialisedMultipliersAndBiases;
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
    const TFLConvArgs &args, const Conv2DType &kt,
    llvm::SmallVector<std::string> &strParams, std::vector<int8_t> &weightsData,
    std::vector<int16_t> &mulsBiasesData, int &scratchBytes) const {
  switch (kt) {
  case Conv2DType::DepthwiseValidDirect:
    if (failed(getDepthwiseConv2DValidDirectParams(
            args, strParams, weightsData, mulsBiasesData, scratchBytes))) {
      return failure();
    }
    break;
  case Conv2DType::DepthwisePaddedIndirect:
    if (failed(getDepthwiseConv2DPaddedIndirectParams(
            args, strParams, weightsData, mulsBiasesData, scratchBytes))) {
      return failure();
    }
    break;
  default:
    // This shouldn't happen!
    return failure();
  }

  return success();
}

LogicalResult
ReplaceDepthwiseConv2DPattern::getDepthwiseConv2DValidDirectParams(
    const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
    std::vector<int8_t> &weightsData, std::vector<int16_t> &mulsBiasesData,
    int &scratchBytes) const {

  nn::DerefInputFn::Params imToColParams(args.X, args.K);

  std::array<int, 4> filterShape = {1, args.filterHeight, args.filterWidth,
                                    args.inputDepth};
  nn::Conv2dReorderedWeights rw = nn::MatMulDirectFn_DW::reorder_kernel_weights(
      (int8_t *)args.filter.data(), filterShape, args.padValue);
  nn::MatMulDirectFn_DW::Params afParams(args.X, args.K);

  nn::MulsAndBias mulsAndBiases =
      nn::OutputTransformFnInt8::canonicalise_mul_and_bias_dw(
          args.effectiveMultiplier, args.bias, args.filter, filterShape,
          args.inputZeroPoint, args.outputZeroPoint, args.outputDepth);
  nn::QuantisationParams qp =
      nn::OutputTransformFnInt8::quantise_activation(mulsAndBiases);
  auto serialisedMultipliersAndBiases =
      nn::OutputTransformFn::serialise_memory(qp.multipliers, qp.biases);
  nn::OutputTransformFn::pad_final_access(
      serialisedMultipliersAndBiases, VPU_INT16_EPV, (int16_t)args.padValue);
  nn::OT_int8::Params otParams((int32_t)args.outputDepth, qp.initial_shr,
                               qp.final_shr);

  auto ir = nn::ImageRegion(0, 0, 0, args.Y.height, args.Y.width, args.Y.depth);
  nn::Filter2D_DW::Params akParams(args.Y, ir, VPU_INT8_ACC_PERIOD);

  // TODO: Check serialization
  std::string akpStr = akParams.serialise<nn::Filter2D_DW::Params>();
  std::string mfStr = imToColParams.serialise<nn::DerefInputFn::Params>();
  std::string afStr = afParams.serialise<nn::MatMulDirectFn_DW::Params>();
  std::string otStr = otParams.serialise<nn::OT_int8::Params>();

  strParams.push_back(akpStr);
  strParams.push_back(mfStr);
  strParams.push_back(afStr);
  strParams.push_back(otStr);
  weightsData = rw.weights;
  mulsBiasesData = serialisedMultipliersAndBiases;
  scratchBytes = 0;

  return success();
}

LogicalResult
ReplaceDepthwiseConv2DPattern::getDepthwiseConv2DPaddedIndirectParams(
    const TFLConvArgs &args, llvm::SmallVector<std::string> &strParams,
    std::vector<int8_t> &weightsData, std::vector<int16_t> &mulsBiasesData,
    int &scratchBytes) const {

  nn::ImToColPadded::Params imToColParams(args.X, args.K, args.padding, 16,
                                          args.inputZeroPoint);

  std::array<int, 4> filterShape = {1, args.filterHeight, args.filterWidth,
                                    args.inputDepth};
  nn::Conv2dReorderedWeights rw = nn::MatMulDirectFn_DW::reorder_kernel_weights(
      (int8_t *)args.filter.data(), filterShape, args.padValue);
  nn::MatMulDirectFn_DW::Params afParams(args.K);

  nn::MulsAndBias mulsAndBiases =
      nn::OutputTransformFnInt8::canonicalise_mul_and_bias_dw(
          args.effectiveMultiplier, args.bias, args.filter, filterShape,
          args.inputZeroPoint, args.outputZeroPoint, args.outputDepth);
  nn::QuantisationParams qp =
      nn::OutputTransformFnInt8::quantise_activation(mulsAndBiases);
  auto serialisedMultipliersAndBiases =
      nn::OutputTransformFn::serialise_memory(qp.multipliers, qp.biases);
  nn::OutputTransformFn::pad_final_access(
      serialisedMultipliersAndBiases, VPU_INT16_EPV, (int16_t)args.padValue);
  nn::OT_int8::Params otParams((int32_t)args.outputDepth, qp.initial_shr,
                               qp.final_shr);

  auto ir = nn::ImageRegion(0, 0, 0, args.Y.height, args.Y.width, args.Y.depth);
  nn::Filter2D_DW::Params akParams(args.Y, ir, VPU_INT8_ACC_PERIOD);

  // TODO: Check serialization
  std::string akpStr = akParams.serialise<nn::Filter2D_DW::Params>();
  std::string mfStr = imToColParams.serialise<nn::ImToColPadded::Params>();
  std::string afStr = afParams.serialise<nn::MatMulDirectFn_DW::Params>();
  std::string otStr = otParams.serialise<nn::OT_int8::Params>();

  strParams.push_back(akpStr);
  strParams.push_back(mfStr);
  strParams.push_back(afStr);
  strParams.push_back(otStr);
  weightsData = rw.weights;
  mulsBiasesData = serialisedMultipliersAndBiases;
  scratchBytes = nn::MatMulDirectFn_DW::get_scratch_mem_bytes(filterShape);

  return success();
}

} // namespace xcore
} // namespace mlir
