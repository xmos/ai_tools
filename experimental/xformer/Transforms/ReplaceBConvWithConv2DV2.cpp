// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "lib_nn/api/Conv2d.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/core/framework/kernel_shape_util.h"

namespace mlir {
namespace xcore {

namespace {
// Replace BConv with XC Conv2DV2 ops.
struct ReplaceBConvWithConv2DV2
    : public PassWrapper<ReplaceBConvWithConv2DV2, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
    registry.insert<XCoreDialect>();
    registry.insert<lq::LarqDialect>();
  }
  void runOnFunction() override;
};

struct Conv2DArgs {
  int outputHeight, outputWidth, outputDepth;
  int inputHeight, inputWidth, inputDepth;
  int filterHeight, filterWidth, filterDepth;
  // std::vector<int8_t> filter;
  std::vector<int32_t> filter;
  // std::vector<int32_t> bias;
  std::vector<float> bias;
  std::vector<float> effectiveMultiplier;
  std::vector<int32_t> threshold;

  int8_t padValue;
  // nn::padding_t padding;
  nn::ImageGeometry Y;
  nn::ImageGeometry X;
  nn::WindowGeometry K;
};

struct ReplaceBConvWithConv2DV2Pattern
    : public OpRewritePattern<lq::Bconv2dOp> {
  using OpRewritePattern<lq::Bconv2dOp>::OpRewritePattern;

  //
  llvm::SmallVector<std::string> getBNNConv2DValidIndirectBinaryParams(
      Conv2DArgs &args, std::vector<int8_t> &weightsTensorData,
      std::vector<int16_t> &multipliersAndBiasesTensorData) const {
    llvm::SmallVector<std::string> conv2DParams;

    nn::ImToColValid::Params imToColParams(args.X, args.K, args.inputDepth);

    std::array<int, 4> filterShape = {args.outputDepth, args.filterHeight,
                                      args.filterWidth, args.inputDepth};
    nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
        (int8_t *)args.filter.data(), filterShape, 1, args.padValue);

    const int elementsPerByte = 8;
    int inputBytes = args.filterHeight * args.filterWidth * args.inputDepth /
                     elementsPerByte;
    nn::MatMulBinary::Params afParams(args.outputDepth, inputBytes);

    // adjust the thresholds from xorpopcount space
    // to xcore space
    auto adjustedThresholds = nn::OT_binary::adjust_thresholds(
        args.threshold, args.inputDepth, args.K, rw);

    nn::OutputTransformFn::pad_final_access(adjustedThresholds, VPU_INT16_EPV,
                                            (int16_t)args.padValue);

    auto ir =
        nn::ImageRegion(0, 0, 0, args.Y.height, args.Y.width, args.Y.depth);
    nn::Filter2D::Params akParams(args.Y, ir, VPU_INT8_ACC_PERIOD);

    // TODO: Check serialization
    std::string akpStr = akParams.serialise<nn::Filter2D::Params>();
    std::string mfStr = imToColParams.serialise<nn::ImToColValid::Params>();
    std::string afStr = afParams.serialise<nn::MatMulBinary::Params>();
    std::string otStr = ""; // otParams.serialise<nn::OT_int8::Params>();

    conv2DParams.push_back(akpStr);
    conv2DParams.push_back(mfStr);
    conv2DParams.push_back(afStr);
    conv2DParams.push_back(otStr);
    weightsTensorData = rw.weights;
    multipliersAndBiasesTensorData = adjustedThresholds;

    return conv2DParams;
  }

  LogicalResult matchAndRewrite(lq::Bconv2dOp conv2DOp,
                                PatternRewriter &rewriter) const override {
    // Check for invalid types and return
    // check activation only relu
    // check filter not float
    // check output not float
    // check padding is VALID

    bool bconv2DBinaryOutput = true;
    if (conv2DOp.output()
            .getType()
            .template cast<ShapedType>()
            .getElementType()
            .template isa<quant::QuantizedType>()) {
      bconv2DBinaryOutput = false;
    }

    // If this is not the case, we return to the reference
    // implementation
    auto outputType = conv2DOp.output().getType().dyn_cast<RankedTensorType>();
    auto inputType = conv2DOp.input().getType().dyn_cast<RankedTensorType>();
    auto outputDepth = outputType.getDimSize(3);
    auto inputDepth = inputType.getDimSize(3);

    auto filterType = conv2DOp.filter().getType().dyn_cast<RankedTensorType>();
    auto filterHeight = filterType.getDimSize(1);
    auto filterWidth = filterType.getDimSize(2);
    auto inputHeight = inputType.getDimSize(1);
    auto inputWidth = inputType.getDimSize(2);

    // Find padding values
    // int64_t newHeight, newWidth;
    // int64_t padTop, padBottom, padLeft, padRight;
    // tensorflow::Padding opPadding = conv2DOp.padding() == "VALID"
    //                                     ? tensorflow::Padding::VALID
    //                                     : tensorflow::Padding::SAME;
    // if (tensorflow::GetWindowedOutputSizeVerboseV2(
    //         inputHeight, filterHeight, conv2DOp.dilation_height_factor(),
    //         conv2DOp.stride_height(), opPadding, &newHeight, &padTop,
    //         &padBottom) != tensorflow::Status::OK()) {
    //   return failure();
    // }
    // if (tensorflow::GetWindowedOutputSizeVerboseV2(
    //         inputWidth, filterWidth, conv2DOp.dilation_width_factor(),
    //         conv2DOp.stride_width(), opPadding, &newWidth, &padLeft,
    //         &padRight) != tensorflow::Status::OK()) {
    //   return failure();
    // }
    // bool toBePadded =
    //     padTop != 0 || padBottom != 0 || padLeft != 0 || padRight != 0;

    // TODO: With multithreading support, we could have a different kernel type
    // for each thread
    if (!bconv2DBinaryOutput) {
      return failure();
    }
    Conv2DType kernelType = Conv2DType::BNNValidIndirectBinary;
    // if (isDepthwise) {
    //   if (toBePadded) {
    //     kernelType = Conv2DType::DepthwisePaddedIndirect;
    //   } else {
    //     kernelType = Conv2DType::DepthwiseValidDirect;
    //   }
    // } else {
    //   if (toBePadded) {
    //     kernelType = Conv2DType::PaddedIndirect;
    //   } else if (inputDepth % 32 == 0 && outputDepth % 16 == 0) {
    //     kernelType = Conv2DType::ValidDirect;
    //   } else {
    //     kernelType = Conv2DType::ValidIndirect;
    //   }
    // }

    // Retrieve the remaining args
    auto outputHeight = outputType.getDimSize(1);
    auto outputWidth = outputType.getDimSize(2);
    auto filterDepth = filterType.getDimSize(3);

    // Get filter values
    auto filterConstOp =
        dyn_cast<mlir::ConstantOp>(conv2DOp.filter().getDefiningOp());
    auto filter = filterConstOp.value().cast<DenseElementsAttr>();
    auto filterVector = std::vector<int32_t>{
        filter.getValues<int32_t>().begin(), filter.getValues<int32_t>().end()};

    // Get bias values
    std::vector<float> biasVector;
    std::vector<float> effectiveOutputScaleVector;
    std::vector<int32_t> thresholdVector;
    if (!bconv2DBinaryOutput) {
      auto biasConstOp = dyn_cast<mlir::ConstantOp>(
          conv2DOp.post_activation_bias().getDefiningOp());
      auto biases = biasConstOp.value().cast<DenseElementsAttr>();
      biasVector = std::vector<float>{biases.getValues<float>().begin(),
                                      biases.getValues<float>().end()};

      // Get effectiveOutputMultiplier values
      auto multiplierConstOp = dyn_cast<mlir::ConstantOp>(
          conv2DOp.post_activation_multiplier().getDefiningOp());
      auto multipliers = multiplierConstOp.value().cast<DenseElementsAttr>();
      effectiveOutputScaleVector =
          std::vector<float>{multipliers.getValues<float>().begin(),
                             multipliers.getValues<float>().end()};
    } else {
      // Get threshold values
      auto thresholdConstOp = dyn_cast<mlir::ConstantOp>(
          conv2DOp.output_threshold().getDefiningOp());
      auto threshold = thresholdConstOp.value().cast<DenseElementsAttr>();
      thresholdVector =
          std::vector<int32_t>{threshold.getValues<int32_t>().begin(),
                               threshold.getValues<int32_t>().end()};
    }
    // Calculate effectiveOutputScale
    // std::vector<float> effectiveOutputScaleVector;
    // auto filterQConstOpType =
    //     filterQConstOp.qtype().cast<RankedTensorType>();
    // bool isPerChannelQuantized = false;
    // double filterScale;
    // ArrayRef<double> filterScales;
    // if (auto filterQType =
    //         filterQConstOpType.getElementType()
    //             .dyn_cast<mlir::quant::UniformQuantizedType>()) {
    //   filterScale = filterQType.getScale();
    // } else if (auto filterQType =
    //                filterQConstOpType.getElementType()
    //                    .dyn_cast<
    //                        mlir::quant::UniformQuantizedPerAxisType>()) {
    //   isPerChannelQuantized = true;
    //   filterScales = filterQType.getScales();
    // } else {
    //   return failure();
    // }

    // Conv is quantized along dimension 0
    // DepthwiseConv is quantized along dimension 3
    // https://www.tensorflow.org/lite/performance/quantization_spec
    // auto numOutputChannels =
    //     isDepthwise ? filterType.getDimSize(3) : filterType.getDimSize(0);
    // for (int i = 0; i < numOutputChannels; ++i) {
    //   auto scale = isPerChannelQuantized ? filterScales[i] : filterScale;
    //   assert(outputScale != 0 && "outputScale should not be zero!");
    //   effectiveOutputScaleVector.push_back(inputScale * scale / outputScale);
    // }

    // nn::padding_t padding = {
    //     static_cast<int16_t>(padTop), static_cast<int16_t>(padLeft),
    //     static_cast<int16_t>(padBottom), static_cast<int16_t>(padRight)};

    // For BNNs, convert depth to bits
    inputDepth = inputDepth * 32;
    filterDepth = filterDepth * 32;
    if (!bconv2DBinaryOutput) {
      outputDepth = outputDepth * 8;
    } else {
      outputDepth = outputDepth * 32;
    }

    nn::ImageGeometry Y(outputHeight, outputWidth, outputDepth, 1);
    nn::ImageGeometry X(inputHeight, inputWidth, inputDepth, 1);
    nn::WindowGeometry K(filterHeight, filterWidth, filterDepth, 0, 0,
                         conv2DOp.stride_height(), conv2DOp.stride_width(), 1,
                         conv2DOp.dilation_height_factor(),
                         conv2DOp.dilation_width_factor());

    // Create a struct of Conv2DArgs to pass in parameters
    Conv2DArgs args = {
        .outputHeight = static_cast<int>(outputHeight),
        .outputWidth = static_cast<int>(outputWidth),
        .outputDepth = static_cast<int>(outputDepth),
        .inputHeight = static_cast<int>(inputHeight),
        .inputWidth = static_cast<int>(inputWidth),
        .inputDepth = static_cast<int>(inputDepth),
        .filterHeight = static_cast<int>(filterHeight),
        .filterWidth = static_cast<int>(filterWidth),
        .filterDepth = static_cast<int>(filterDepth),
        .filter = filterVector,
        .bias = biasVector,
        .effectiveMultiplier = effectiveOutputScaleVector,
        .threshold = thresholdVector,
        // TODO: For BNNs, pad value cannot be zero
        // We should be ideally using a different Conv2D operator for BNNs
        .padValue = 0,
        //.padding = padding,
        .Y = Y,
        .X = X,
        .K = K};

    llvm::SmallVector<std::string> abstractKernelParams, memcpyFnParams,
        aggregateFnParams, outputTransformFnParams, kernelTypeEnumParams;
    llvm::SmallVector<int32_t> scratchByteParams;

    // TODO: We only have one thread now
    // If we have more threads, we'll need to combine the tensor data
    // and save the sizes for each thread
    std::vector<int8_t> weightsTensorData;
    std::vector<int16_t> multipliersAndBiasesTensorData;

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
      case Conv2DType::BNNValidIndirectBinary:
        conv2DParams = getBNNConv2DValidIndirectBinaryParams(
            args, weightsTensorData, multipliersAndBiasesTensorData);
        const int elementsPerByte = 8;
        int inputBytes = args.filterHeight * args.filterWidth *
                         args.inputDepth / elementsPerByte;
        scratchBytes = nn::MatMulInt8::get_scratch_mem_bytes(inputBytes) + 32;
        break;
        //   case Conv2DType::ValidIndirect:
        //     conv2DParams = getConv2DValidIndirectParams(
        //         args, weightsTensorData, multipliersAndBiasesTensorData);
        //     scratchBytes =
        //         nn::MatMulInt8::get_scratch_mem_bytes(
        //             args.filterHeight * args.filterWidth * args.inputDepth) +
        //         32; //[asj] FIXME
        //     break;
        //   case Conv2DType::PaddedIndirect:
        //     conv2DParams = getConv2DPaddedIndirectParams(
        //         args, weightsTensorData, multipliersAndBiasesTensorData);
        //     scratchBytes =
        //         nn::MatMulInt8::get_scratch_mem_bytes(
        //             args.filterHeight * args.filterWidth * args.inputDepth) +
        //         32; //[asj] FIXME
        //     break;
        //   case Conv2DType::DepthwiseValidDirect:
        //     conv2DParams = getDepthwiseConv2DValidDirectParams(
        //         args, weightsTensorData, multipliersAndBiasesTensorData);
        //     break;
        //   case Conv2DType::DepthwisePaddedIndirect:
        //     conv2DParams = getDepthwiseConv2DPaddedIndirectParams(
        //         args, weightsTensorData, multipliersAndBiasesTensorData);
        //     auto filterShape = std::array<int, 4>(
        //         {1, args.filterHeight, args.filterWidth, args.inputDepth});
        //     scratchBytes =
        //         nn::MatMulDirectFn_DW::get_scratch_mem_bytes(filterShape);
        //     break;
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

    // Create the tensors for weights and multipliers_and_biases
    assert(threadCount == 1 &&
           "Tensor data has to be combined for more than one thread!");
    ShapedType weightsType = RankedTensorType::get(
        {static_cast<long long>(weightsTensorData.size())},
        rewriter.getIntegerType(8));
    auto weightsAttr =
        DenseElementsAttr::get<int8_t>(weightsType, weightsTensorData);
    auto weightsConstantOp =
        rewriter.create<mlir::ConstantOp>(conv2DOp.getLoc(), weightsAttr);

    ShapedType multipliersAndBiasesType = RankedTensorType::get(
        {static_cast<long long>(multipliersAndBiasesTensorData.size())},
        rewriter.getIntegerType(16));
    auto multipliersAndBiasesAttr = DenseElementsAttr::get<int16_t>(
        multipliersAndBiasesType, multipliersAndBiasesTensorData);
    auto multipliersAndBiasesConstantOp = rewriter.create<mlir::ConstantOp>(
        conv2DOp.getLoc(), multipliersAndBiasesAttr);

    // Create the Conv2DV2 Op with the params and kernel type
    auto newConv2DV2Op = rewriter.create<Conv2DV2Op>(
        conv2DOp.getLoc(), conv2DOp.getType(), conv2DOp.input(),
        rewriter.getI32IntegerAttr(threadCount),
        rewriter.getI32ArrayAttr(scratchByteParams), weightsConstantOp,
        multipliersAndBiasesConstantOp,
        getStringArrayAttr(abstractKernelParams),
        getStringArrayAttr(memcpyFnParams),
        getStringArrayAttr(aggregateFnParams),
        getStringArrayAttr(outputTransformFnParams),
        getStringArrayAttr(kernelTypeEnumParams));
    rewriter.replaceOp(conv2DOp, newConv2DV2Op.output());

    return success();
  }
};

void ReplaceBConvWithConv2DV2::runOnFunction() {
  auto *ctx = &getContext();
  auto func = getFunction();

  OwningRewritePatternList patterns(ctx);
  patterns.insert<ReplaceBConvWithConv2DV2Pattern>(ctx);
  ;
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceBConvWithConv2DV2 pass.
std::unique_ptr<OperationPass<FuncOp>> createReplaceBConvWithConv2DV2Pass() {
  return std::make_unique<ReplaceBConvWithConv2DV2>();
}

static PassRegistration<ReplaceBConvWithConv2DV2>
    pass("xcore-replace-bconv-with-conv2dv2",
         "Replace BConv with XC Conv2DV2 operations.");

} // namespace xcore
} // namespace mlir
