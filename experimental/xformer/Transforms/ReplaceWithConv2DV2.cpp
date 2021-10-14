// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "lib_nn/api/Conv2d.hpp"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include <numeric>

namespace mlir {
namespace xcore {

namespace {
// Replace TFL Conv2D and DepthwiseConv2D with XC Conv2DV2 ops.
struct ReplaceWithConv2DV2
    : public PassWrapper<ReplaceWithConv2DV2, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
    registry.insert<XCoreDialect>();
  }
  void runOnFunction() override;
};

struct Conv2DArgs {
  int outputHeight, outputWidth, outputDepth, outputZeroPoint;
  int inputHeight, inputWidth, inputDepth, inputZeroPoint;
  int filterHeight, filterWidth, filterDepth;
  int strideH, strideW;
  int dilationH, dilationW;
  std::vector<int8_t> filter;
  std::vector<int32_t> bias;
  std::vector<float> effectiveMultiplier;
  int16_t topPad, leftPad, bottomPad, rightPad;
  int8_t padValue;
};

template <typename TFLOp>
struct ReplaceWithConv2DV2Pattern : public OpRewritePattern<TFLOp> {
  using OpRewritePattern<TFLOp>::OpRewritePattern;

  //
  llvm::SmallVector<std::string>
  getDepthwiseConv2DValidDirectParams(const Conv2DArgs &args) const {
    llvm::SmallVector<std::string> conv2DParams;

    nn::padding_t padding = {args.topPad, args.leftPad, args.bottomPad,
                             args.rightPad};
    nn::ImageGeometry Y(args.outputHeight, args.outputWidth, args.outputDepth);
    nn::ImageGeometry X(args.inputHeight, args.inputWidth, args.inputDepth);
    nn::WindowGeometry K(args.filterHeight, args.filterWidth, args.filterDepth,
                         -padding.top, -padding.left, args.strideH,
                         args.strideW, 1, args.dilationH, args.dilationW);
    nn::Filter2dGeometry geom(X, Y, K);

    nn::DerefInputFn::Params imToColParams(X, K);

    std::array<int, 4> weightsShape = {1, args.filterHeight, args.filterWidth,
                                       args.inputDepth};
    nn::Conv2dReorderedWeights rw =
        nn::MatMulDirectFn_DW::reorder_kernel_weights(
            (int8_t *)args.filter.data(), weightsShape, args.padValue);
    nn::MatMulDirectFn_DW::Params afParams(X, K, rw.weights.data(),
                                           rw.weights.size());

    nn::OutputTransformFnInt8::CanonicalMulAndBias canonical_values =
        nn::OutputTransformFnInt8::canonicalise_mul_and_bias_dw(
            args.effectiveMultiplier, args.bias, args.filter, weightsShape,
            args.inputZeroPoint, args.outputZeroPoint, args.outputDepth);
    nn::QuantisationParams qp = nn::OutputTransformFnInt8::quantise_activation(
        canonical_values.f_multipliers, canonical_values.f_biases,
        canonical_values.accu_min, canonical_values.accu_max);
    nn::OutputTransformFn::pad(qp.biases, VPU_INT16_EPV,
                               (int16_t)args.padValue);
    nn::OutputTransformFn::pad(qp.multipliers, VPU_INT16_EPV,
                               (int16_t)args.padValue);
    nn::OT_int8::Params otParams((int32_t)args.outputDepth, &qp.otv, qp.biases,
                                 qp.multipliers);

    auto ir = nn::ImageRegion(0, 0, 0, Y.height, Y.width, Y.depth);
    nn::Filter2D_DW::Params akParams(Y, ir, VPU_INT8_ACC_PERIOD);

    // TODO: Check serialization
    std::string akpStr = akParams.serialise<nn::Filter2D_DW::Params>();
    std::string mfStr = imToColParams.serialise<nn::DerefInputFn::Params>();
    std::string afStr = afParams.serialise<nn::MatMulDirectFn_DW::Params>();
    std::string otStr = otParams.serialise<nn::OT_int8::Params>();

    conv2DParams.push_back(akpStr);
    conv2DParams.push_back(mfStr);
    conv2DParams.push_back(afStr);
    conv2DParams.push_back(otStr);

    return conv2DParams;
  }

  //
  llvm::SmallVector<std::string>
  getDepthwiseConv2DPaddedIndirectParams(const Conv2DArgs &args) const {
    llvm::SmallVector<std::string> conv2DParams;

    nn::padding_t padding = {args.topPad, args.leftPad, args.bottomPad,
                             args.rightPad};
    nn::ImageGeometry Y(args.outputHeight, args.outputWidth, args.outputDepth);
    nn::ImageGeometry X(args.inputHeight, args.inputWidth, args.inputDepth);
    nn::WindowGeometry K(args.filterHeight, args.filterWidth, args.filterDepth,
                         -padding.top, -padding.left, args.strideH,
                         args.strideW, 1, args.dilationH, args.dilationW);
    nn::Filter2dGeometry geom(X, Y, K);

    nn::ImToColPadded::Params imToColParams(X, K, padding, 16,
                                            args.inputZeroPoint);

    std::array<int, 4> weightsShape = {1, args.filterHeight, args.filterWidth,
                                       args.inputDepth};
    nn::Conv2dReorderedWeights rw =
        nn::MatMulDirectFn_DW::reorder_kernel_weights(
            (int8_t *)args.filter.data(), weightsShape, args.padValue);
    nn::MatMulDirectFn_DW::Params afParams(K, rw.weights.data(),
                                           rw.weights.size());

    nn::OutputTransformFnInt8::CanonicalMulAndBias canonical_values =
        nn::OutputTransformFnInt8::canonicalise_mul_and_bias_dw(
            args.effectiveMultiplier, args.bias, args.filter, weightsShape,
            args.inputZeroPoint, args.outputZeroPoint, args.outputDepth);
    nn::QuantisationParams qp = nn::OutputTransformFnInt8::quantise_activation(
        canonical_values.f_multipliers, canonical_values.f_biases,
        canonical_values.accu_min, canonical_values.accu_max);
    // change
    nn::OutputTransformFn::pad(qp.biases, VPU_INT16_EPV,
                               (int16_t)args.padValue);
    nn::OutputTransformFn::pad(qp.multipliers, VPU_INT16_EPV,
                               (int16_t)args.padValue);
    nn::OT_int8::Params otParams((int32_t)args.outputDepth, &qp.otv, qp.biases,
                                 qp.multipliers);

    auto ir = nn::ImageRegion(0, 0, 0, Y.height, Y.width, Y.depth);
    nn::Filter2D_DW::Params akParams(Y, ir, VPU_INT8_ACC_PERIOD);

    // TODO: Check serialization
    std::string akpStr = akParams.serialise<nn::Filter2D_DW::Params>();
    std::string mfStr = imToColParams.serialise<nn::ImToColPadded::Params>();
    std::string afStr = afParams.serialise<nn::MatMulDirectFn_DW::Params>();
    std::string otStr = otParams.serialise<nn::OT_int8::Params>();

    conv2DParams.push_back(akpStr);
    conv2DParams.push_back(mfStr);
    conv2DParams.push_back(afStr);
    conv2DParams.push_back(otStr);

    return conv2DParams;
  }

  //
  llvm::SmallVector<std::string>
  getConv2DPaddedIndirectParams(const Conv2DArgs &args) const {
    llvm::SmallVector<std::string> conv2DParams;

    nn::padding_t padding = {args.topPad, args.leftPad, args.bottomPad,
                             args.rightPad};
    nn::ImageGeometry Y(args.outputHeight, args.outputWidth, args.outputDepth);
    nn::ImageGeometry X(args.inputHeight, args.inputWidth, args.inputDepth);
    nn::WindowGeometry K(args.filterHeight, args.filterWidth, args.filterDepth,
                         -padding.top, -padding.left, args.strideH,
                         args.strideW, 1, args.dilationH, args.dilationW);
    nn::Filter2dGeometry geom(X, Y, K);

    nn::ImToColPadded::Params imToColParams(X, K, padding, args.inputDepth,
                                            args.inputZeroPoint);

    int inputBytes = args.filterHeight * args.filterWidth * args.inputDepth;
    std::array<int, 4> shape = {args.outputDepth, args.filterHeight,
                                args.filterWidth, args.inputDepth};
    nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
        (int8_t *)args.filter.data(), shape, 8, args.padValue);
    nn::MatMulInt8::Params afParams(args.outputDepth, inputBytes,
                                    rw.weights.data());

    nn::OutputTransformFnInt8::CanonicalMulAndBias canonical_values =
        nn::OutputTransformFnInt8::canonicalise_mul_and_bias(
            args.effectiveMultiplier, args.bias, args.filter,
            args.inputZeroPoint, args.outputZeroPoint, args.outputDepth);
    nn::QuantisationParams qp = nn::OutputTransformFnInt8::quantise_activation(
        canonical_values.f_multipliers, canonical_values.f_biases,
        canonical_values.accu_min, canonical_values.accu_max);
    nn::OT_int8::Params otParams((int32_t)args.outputDepth, &qp.otv, qp.biases,
                                 qp.multipliers);

    auto ir = nn::ImageRegion(0, 0, 0, Y.height, Y.width, Y.depth);
    nn::Filter2D::Params akParams(Y, ir, VPU_INT8_ACC_PERIOD);

    // TODO: Check serialization
    std::string akpStr = akParams.serialise<nn::Filter2D::Params>();
    std::string mfStr = imToColParams.serialise<nn::ImToColPadded::Params>();
    std::string afStr = afParams.serialise<nn::MatMulInt8::Params>();
    std::string otStr = otParams.serialise<nn::OT_int8::Params>();

    conv2DParams.push_back(akpStr);
    conv2DParams.push_back(mfStr);
    conv2DParams.push_back(afStr);
    conv2DParams.push_back(otStr);

    return conv2DParams;
  }

  //
  llvm::SmallVector<std::string>
  getConv2DValidIndirectParams(const Conv2DArgs &args) const {
    llvm::SmallVector<std::string> conv2DParams;

    nn::padding_t padding = {args.topPad, args.leftPad, args.bottomPad,
                             args.rightPad};
    nn::ImageGeometry Y(args.outputHeight, args.outputWidth, args.outputDepth);
    nn::ImageGeometry X(args.inputHeight, args.inputWidth, args.inputDepth);
    nn::WindowGeometry K(args.filterHeight, args.filterWidth, args.filterDepth,
                         -padding.top, -padding.left, args.strideH,
                         args.strideW, 1, args.dilationH, args.dilationW);
    nn::Filter2dGeometry geom(X, Y, K);

    nn::ImToColValid::Params imToColParams(X, K, args.inputDepth);

    // TODO: What is overread bytes for?
    // int overread_bytes = memcpy.get_overread_bytes();
    // input.resize(input.size() + overread_bytes / sizeof(int8_t), 0);

    std::array<int, 4> shape = {args.outputDepth, args.filterHeight,
                                args.filterWidth, args.inputDepth};
    nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
        (int8_t *)args.filter.data(), shape, 8, args.padValue);
    int inputBytes = args.filterHeight * args.filterWidth * args.inputDepth;
    nn::MatMulInt8::Params afParams(args.outputDepth, inputBytes,
                                    rw.weights.data());

    nn::OutputTransformFnInt8::CanonicalMulAndBias canonicalValues =
        nn::OutputTransformFnInt8::canonicalise_mul_and_bias(
            args.effectiveMultiplier, args.bias, args.filter,
            args.inputZeroPoint, args.outputZeroPoint, args.outputDepth);
    nn::QuantisationParams qp = nn::OutputTransformFnInt8::quantise_activation(
        canonicalValues.f_multipliers, canonicalValues.f_biases,
        canonicalValues.accu_min, canonicalValues.accu_max);
    nn::OT_int8::Params otParams((int32_t)args.outputDepth, &qp.otv, qp.biases,
                                 qp.multipliers);

    auto ir = nn::ImageRegion(0, 0, 0, Y.height, Y.width, Y.depth);
    nn::Filter2D::Params akParams(Y, ir, VPU_INT8_ACC_PERIOD);

    // TODO: Check serialization
    std::string akpStr = akParams.serialise<nn::Filter2D::Params>();
    std::string mfStr = imToColParams.serialise<nn::ImToColValid::Params>();
    std::string afStr = afParams.serialise<nn::MatMulInt8::Params>();
    std::string otStr = otParams.serialise<nn::OT_int8::Params>();

    conv2DParams.push_back(akpStr);
    conv2DParams.push_back(mfStr);
    conv2DParams.push_back(afStr);
    conv2DParams.push_back(otStr);

    return conv2DParams;
  }

  //
  llvm::SmallVector<std::string>
  getConv2DValidDirectParams(const Conv2DArgs &args) const {
    llvm::SmallVector<std::string> conv2DParams;

    nn::padding_t padding = {args.topPad, args.leftPad, args.bottomPad,
                             args.rightPad};
    nn::ImageGeometry X(args.inputHeight, args.inputWidth, args.inputDepth);
    nn::ImageGeometry Y(args.outputHeight, args.outputWidth, args.outputDepth);
    nn::WindowGeometry K(args.filterHeight, args.filterWidth, args.filterDepth,
                         -padding.top, -padding.left, args.strideH,
                         args.strideW, 1, args.dilationH, args.dilationW);

    nn::DerefInputFn::Params imToColParams(X, K);

    std::array<int, 4> shape = {args.outputDepth, args.filterHeight,
                                args.filterWidth, args.inputDepth};
    nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
        (int8_t *)args.filter.data(), shape, 8, args.padValue);
    nn::MatMulDirectFn::Params afParams(
        X, K, args.inputDepth, rw.weights.data(), (int)rw.weights.size());

    nn::OutputTransformFnInt8::CanonicalMulAndBias canonicalValues =
        nn::OutputTransformFnInt8::canonicalise_mul_and_bias(
            args.effectiveMultiplier, args.bias, args.filter,
            args.inputZeroPoint, args.outputZeroPoint, args.outputDepth);
    nn::QuantisationParams qp = nn::OutputTransformFnInt8::quantise_activation(
        canonicalValues.f_multipliers, canonicalValues.f_biases,
        canonicalValues.accu_min, canonicalValues.accu_max);
    nn::OT_int8::Params otParams((int32_t)args.outputDepth, &qp.otv, qp.biases,
                                 qp.multipliers);

    auto ir = nn::ImageRegion(0, 0, 0, Y.height, Y.width, Y.depth);
    nn::Filter2D::Params akParams(Y, ir, VPU_INT8_ACC_PERIOD);

    // TODO: Check serialization
    std::string akpStr = akParams.serialise<nn::Filter2D::Params>();
    std::string mfStr = imToColParams.serialise<nn::DerefInputFn::Params>();
    std::string afStr = afParams.serialise<nn::MatMulDirectFn::Params>();
    std::string otStr = otParams.serialise<nn::OT_int8::Params>();

    conv2DParams.push_back(akpStr);
    conv2DParams.push_back(mfStr);
    conv2DParams.push_back(afStr);
    conv2DParams.push_back(otStr);

    return conv2DParams;
  }

  LogicalResult matchAndRewrite(TFLOp conv2DOp,
                                PatternRewriter &rewriter) const override {
    // Check for invalid types and return
    // Input type must be QI8
    if (!(conv2DOp.input()
              .getType()
              .template cast<ShapedType>()
              .getElementType()
              .template isa<quant::QuantizedType>() &&
          conv2DOp.input()
              .getType()
              .template cast<ShapedType>()
              .getElementType()
              .template cast<quant::QuantizedType>()
              .isSigned() &&
          conv2DOp.input()
                  .getType()
                  .template cast<ShapedType>()
                  .getElementType()
                  .template cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    // Filter type must be QI8
    if (!(conv2DOp.filter()
              .getType()
              .template cast<ShapedType>()
              .getElementType()
              .template isa<quant::QuantizedType>() &&
          conv2DOp.filter()
              .getType()
              .template cast<ShapedType>()
              .getElementType()
              .template cast<quant::QuantizedType>()
              .isSigned() &&
          conv2DOp.filter()
                  .getType()
                  .template cast<ShapedType>()
                  .getElementType()
                  .template cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    // TODO: What to do if no bias?
    // Bias type must be QI32
    if (!(conv2DOp.bias()
              .getType()
              .template cast<ShapedType>()
              .getElementType()
              .template isa<quant::QuantizedType>() &&
          conv2DOp.bias()
              .getType()
              .template cast<ShapedType>()
              .getElementType()
              .template cast<quant::QuantizedType>()
              .isSigned() &&
          conv2DOp.bias()
                  .getType()
                  .template cast<ShapedType>()
                  .getElementType()
                  .template cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 32)) {
      return failure();
    }

    // Output depth and input depth must be a multiple of four
    // If this is not the case, we return to the reference
    // implementation
    auto outputType =
        conv2DOp.output().getType().template dyn_cast<RankedTensorType>();
    auto inputType =
        conv2DOp.input().getType().template dyn_cast<RankedTensorType>();
    auto outputDepth = outputType.getDimSize(3);
    auto inputDepth = inputType.getDimSize(3);
    if (outputDepth % 4 != 0 || inputDepth % 4 != 0) {
      return failure();
    }

    // Is it a DepthwiseConv2D?
    bool isDepthwise = std::is_same<TFLOp, TFL::DepthwiseConv2DOp>::value;

    auto filterType =
        conv2DOp.filter().getType().template dyn_cast<RankedTensorType>();
    auto filterHeight = filterType.getDimSize(1);
    auto filterWidth = filterType.getDimSize(2);
    auto inputHeight = inputType.getDimSize(1);
    auto inputWidth = inputType.getDimSize(2);

    // Find padding values
    tensorflow::int64 newHeight, newWidth;
    tensorflow::int64 padTop, padBottom, padLeft, padRight;
    tensorflow::Padding opPadding =
        symbolizePadding(conv2DOp.padding()) == Padding::VALID
            ? tensorflow::Padding::VALID
            : tensorflow::Padding::SAME;
    if (tensorflow::GetWindowedOutputSizeVerboseV2(
            inputHeight, filterHeight, conv2DOp.dilation_h_factor(),
            conv2DOp.stride_h(), opPadding, &newHeight, &padTop,
            &padBottom) != tensorflow::Status::OK()) {
      return failure();
    }
    if (tensorflow::GetWindowedOutputSizeVerboseV2(
            inputWidth, filterWidth, conv2DOp.dilation_w_factor(),
            conv2DOp.stride_w(), opPadding, &newWidth, &padLeft,
            &padRight) != tensorflow::Status::OK()) {
      return failure();
    }
    bool toBePadded =
        padTop != 0 || padBottom != 0 || padLeft != 0 || padRight != 0;

    // TODO: With multithreading support, we could have a different kernel type
    // for each thread
    Conv2DType kernelType;
    if (isDepthwise) {
      if (toBePadded) {
        kernelType = Conv2DType::DepthwisePaddedIndirect;
      } else {
        kernelType = Conv2DType::DepthwiseValidDirect;
      }
    } else {
      if (toBePadded) {
        kernelType = Conv2DType::PaddedIndirect;
      } else if (inputDepth % 32 == 0 && outputDepth % 16 == 0) {
        kernelType = Conv2DType::ValidDirect;
      } else {
        kernelType = Conv2DType::ValidIndirect;
      }
    }

    // Retrieve the remaining args
    auto outputHeight = outputType.getDimSize(1);
    auto outputWidth = outputType.getDimSize(2);
    auto filterDepth = filterType.getDimSize(3);

    // Get output zero point
    auto outputQType =
        outputType.getElementType()
            .template dyn_cast<mlir::quant::UniformQuantizedType>();
    auto outputScale = outputQType.getScale();
    auto outputZeroPoint = outputQType.getZeroPoint();

    // Get input zero point
    auto inputQType =
        inputType.getElementType()
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
    auto biasQConstOp =
        dyn_cast<TFL::QConstOp>(conv2DOp.bias().getDefiningOp());
    auto biases = biasQConstOp.value().template cast<DenseElementsAttr>();
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
    auto numOutputChannels =
        isDepthwise ? filterType.getDimSize(3) : filterType.getDimSize(0);
    for (int i = 0; i < numOutputChannels; ++i) {
      auto scale = isPerChannelQuantized ? filterScales[i] : filterScale;
      assert(outputScale != 0 && "outputScale should not be zero!");
      effectiveOutputScaleVector.push_back(inputScale * scale / outputScale);
    }

    // Create a struct of Conv2DArgs to pass in parameters
    Conv2DArgs args = {
        .outputHeight = static_cast<int>(outputHeight),
        .outputWidth = static_cast<int>(outputWidth),
        .outputDepth = static_cast<int>(outputDepth),
        .outputZeroPoint = static_cast<int>(outputZeroPoint),
        .inputHeight = static_cast<int>(inputHeight),
        .inputWidth = static_cast<int>(inputWidth),
        .inputDepth = static_cast<int>(inputDepth),
        .inputZeroPoint = static_cast<int>(inputZeroPoint),
        .filterHeight = static_cast<int>(filterHeight),
        .filterWidth = static_cast<int>(filterWidth),
        .filterDepth = static_cast<int>(filterDepth),
        .strideH = static_cast<int>(conv2DOp.stride_h()),
        .strideW = static_cast<int>(conv2DOp.stride_w()),
        .dilationH = static_cast<int>(conv2DOp.dilation_h_factor()),
        .dilationW = static_cast<int>(conv2DOp.dilation_w_factor()),
        .filter = filterVector,
        .bias = biasVector,
        .effectiveMultiplier = effectiveOutputScaleVector,
        .topPad = static_cast<int16_t>(padTop),
        .leftPad = static_cast<int16_t>(padLeft),
        .bottomPad = static_cast<int16_t>(padBottom),
        .rightPad = static_cast<int16_t>(padRight),
        // TODO: For BNNs, pad value cannot be zero
        // We should be ideally using a different Conv2D operator for BNNs
        .padValue = 0};

    llvm::SmallVector<std::string> abstractKernelParams, memcpyFnParams,
        aggregateFnParams, outputTransformFnParams, kernelTypeEnumParams;
    llvm::SmallVector<int32_t> scratchByteParams;

    // TODO: Get thread count as command-line option
    // Currently thread count is one
    const int threadCount = 1;

    // TODO: Multithread analysis to determine how to split up the data
    // between threads. Might be better to do this as an analysis pass and
    // access the analysis results here
    for (int i = 0; i < threadCount; ++i) {
      llvm::SmallVector<std::string> conv2DParams;
      // TODO: Determine which kernel type for each thread.
      kernelTypeEnumParams.push_back(stringifyConv2DType(kernelType).str());

      // The scratch size needed is zero for ValidDirect kernel types
      // For other types, we calculate it below
      int scratchBytes = 0;

      // TODO: Call the right kernel type function
      // Call the kernel type function which returns a vector of four strings
      // for the four Conv2D params
      switch (kernelType) {
      case Conv2DType::ValidDirect:
        conv2DParams = getConv2DValidDirectParams(args);
        break;
      case Conv2DType::ValidIndirect:
        conv2DParams = getConv2DValidIndirectParams(args);
        scratchBytes =
            nn::MatMulInt8::get_scratch_mem_bytes(
                args.filterHeight * args.filterWidth * args.inputDepth) +
            32; //[asj] FIXME
        break;
      case Conv2DType::PaddedIndirect:
        conv2DParams = getConv2DPaddedIndirectParams(args);
        scratchBytes =
            nn::MatMulInt8::get_scratch_mem_bytes(
                args.filterHeight * args.filterWidth * args.inputDepth) +
            32; //[asj] FIXME
        break;
      case Conv2DType::DepthwiseValidDirect:
        conv2DParams = getDepthwiseConv2DValidDirectParams(args);
        break;
      case Conv2DType::DepthwisePaddedIndirect:
        conv2DParams = getDepthwiseConv2DPaddedIndirectParams(args);
        auto weightsShape = std::array<int, 4>(
            {1, args.filterHeight, args.filterWidth, args.inputDepth});
        scratchBytes =
            nn::MatMulDirectFn_DW::get_scratch_mem_bytes(weightsShape);
        break;
      }

      abstractKernelParams.push_back(conv2DParams[0]);
      memcpyFnParams.push_back(conv2DParams[1]);
      aggregateFnParams.push_back(conv2DParams[2]);
      outputTransformFnParams.push_back(conv2DParams[3]);
      scratchByteParams.push_back(scratchBytes);
    }

    // Create a string array attr from a vector of strings
    auto getStringArrayAttr = [&](llvm::SmallVector<std::string> value) {
      auto attrs = llvm::to_vector<8>(
          llvm::map_range(value, [&](std::string v) -> Attribute {
            return rewriter.getStringAttr(v);
          }));
      return rewriter.getArrayAttr(attrs);
    };

    // Create the Conv2DV2 Op with the params and kernel type
    auto newConv2DV2Op = rewriter.create<Conv2DV2Op>(
        conv2DOp.getLoc(), conv2DOp.getType(), conv2DOp.input(),
        rewriter.getI32IntegerAttr(threadCount),
        rewriter.getI32ArrayAttr(scratchByteParams),
        getStringArrayAttr(abstractKernelParams),
        getStringArrayAttr(memcpyFnParams),
        getStringArrayAttr(aggregateFnParams),
        getStringArrayAttr(outputTransformFnParams),
        getStringArrayAttr(kernelTypeEnumParams));
    rewriter.replaceOp(conv2DOp, newConv2DV2Op.output());

    return success();
  }
};

void ReplaceWithConv2DV2::runOnFunction() {
  auto *ctx = &getContext();
  auto func = getFunction();

  OwningRewritePatternList patterns(ctx);
  patterns.insert<ReplaceWithConv2DV2Pattern<TFL::Conv2DOp>>(ctx);
  patterns.insert<ReplaceWithConv2DV2Pattern<TFL::DepthwiseConv2DOp>>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceWithConv2DV2 pass.
std::unique_ptr<OperationPass<FuncOp>> createReplaceWithConv2DV2Pass() {
  return std::make_unique<ReplaceWithConv2DV2>();
}

static PassRegistration<ReplaceWithConv2DV2>
    pass("xcore-replace-with-conv2dv2",
         "Replace TFL Conv2D and DepthwiseConv2D with XC Conv2DV2 operations.");

} // namespace xcore
} // namespace mlir
