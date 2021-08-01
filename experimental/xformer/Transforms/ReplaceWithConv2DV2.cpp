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
#include <numeric>

namespace mlir {
namespace xcore {

namespace {
// Replace TFL Conv2D with XC Conv2DV2 ops.
struct ReplaceWithConv2DV2
    : public PassWrapper<ReplaceWithConv2DV2, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
    registry.insert<XCoreDialect>();
  }
  void runOnFunction() override;
};

// TODO
struct Conv2DArgs {
  int outputHeight, outputWidth, outputZeroPoint;
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

struct ReplaceWithConv2DV2Pattern : public OpRewritePattern<TFL::Conv2DOp> {
  using OpRewritePattern<TFL::Conv2DOp>::OpRewritePattern;

  // TODO
  llvm::SmallVector<std::string>
  getConv2DValidDirectParams(const Conv2DArgs &args) const {
    llvm::SmallVector<std::string> conv2DParams;

    nn::padding_t padding = {args.topPad, args.leftPad, args.bottomPad,
                             args.rightPad};
    nn::ImageGeometry X(args.inputHeight, args.inputWidth, args.inputDepth);
    nn::ImageGeometry Y(args.outputHeight, args.outputWidth, args.filterDepth);
    nn::WindowGeometry K(args.filterHeight, args.filterWidth, args.filterDepth,
                         -padding.top, -padding.left, args.strideH,
                         args.strideW, 1, args.dilationH, args.dilationW);

    nn::DerefInputFn::Params imToColParams(X, K);
    nn::DerefInputFn memcpyFn(&imToColParams);

    std::array<int, 4> shape = {args.filterDepth, args.filterHeight,
                                args.filterWidth, args.inputDepth};
    nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
        (int8_t *)args.filter.data(), shape, 8, args.padValue);
    nn::MatMulDirectFn::Params afParams(X, K, args.inputDepth,
                                        rw.weights.data());
    nn::MatMulDirectFn aggregateFn(&afParams);

    nn::OutputTransformFnInt8::CanonicalMulAndBias canonicalValues =
        nn::OutputTransformFnInt8::canonicalise_mul_and_bias(
            args.effectiveMultiplier, args.bias, args.filter,
            args.inputZeroPoint, args.outputZeroPoint, args.filterDepth);
    nn::QuantisationParams qp = nn::OutputTransformFnInt8::quantise_activation(
        canonicalValues.f_multipliers, canonicalValues.f_biases,
        canonicalValues.accu_min, canonicalValues.accu_max);
    nn::OT_int8::Params otParams((int32_t)args.filterDepth, &qp.otv,
                                 qp.biases.data(), qp.multipliers.data());
    nn::OT_int8 otFn(&otParams);

    auto ir = nn::ImageRegion(0, 0, 0, Y.height, Y.width, Y.depth);
    nn::Filter2D::Params akParams(Y, ir, VPU_INT8_ACC_PERIOD);

    // TODO: This method of serializing by casting can only work when the
    // param structs don't have pointer members.
    // Otherwise, we would need to use a proper serialize method which returns a
    // serialized string.
    std::string akpStr(reinterpret_cast<const char *>(&akParams),
                       sizeof(akParams));
    std::string otStr(reinterpret_cast<const char *>(&otParams),
                      sizeof(otParams));

    nn::Conv2dValidDirect conv2d(&akParams, &memcpyFn, &aggregateFn, &otFn);

    conv2DParams.push_back(akpStr);
    conv2DParams.push_back(akpStr);
    conv2DParams.push_back(akpStr);
    conv2DParams.push_back(otStr);

    return conv2DParams;
  }

  LogicalResult matchAndRewrite(TFL::Conv2DOp conv2DOp,
                                PatternRewriter &rewriter) const override {
    // Check for invalid types and return
    // Input type must be QI8
    if (!(conv2DOp.input()
              .getType()
              .cast<ShapedType>()
              .getElementType()
              .isa<quant::QuantizedType>() &&
          conv2DOp.input()
              .getType()
              .cast<ShapedType>()
              .getElementType()
              .cast<quant::QuantizedType>()
              .isSigned() &&
          conv2DOp.input()
                  .getType()
                  .cast<ShapedType>()
                  .getElementType()
                  .cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    // Check if not fitting the three kernel types, then return
    // Output channel must be a multiple of four
    if (conv2DOp.output().getType().cast<ShapedType>().getDimSize(3) % 4 != 0) {
      return failure();
    }

    // TODO: For Conv2D for padding type = "SAME", get the correct padding
    // values
    int16_t top_pad, left_pad, bottom_pad, right_pad;
    top_pad = left_pad = bottom_pad = right_pad = 0;

    // TODO: For BNNs, pad value cannot be zero
    // We should be ideally using a different Conv2D operator for BNNs
    int8_t kernel_pad_val = 0;

    // Retrieve these
    int output_height = 1;
    int output_width = 1;

    int k_height = 1;
    int k_width = 1;
    int k_depth = 16;

    int x_height = 1;
    int x_width = 1;
    int x_channels = 32;

    int k_v_stride = 1;
    int k_h_stride = 1;
    int k_v_dilation = 1;
    int k_h_dilation = 1;

    std::vector<int8_t> weights(k_height * k_width * k_depth * x_channels, 1);
    std::vector<int32_t> bias(k_depth, 1);
    std::vector<float> eff_mult(k_depth, 1);
    int input_zero_point = 1;
    int output_zero_point = 1;

    Conv2DArgs args = {.outputHeight = output_height,
                       .outputWidth = output_width,
                       .outputZeroPoint = output_zero_point,
                       .inputHeight = x_height,
                       .inputWidth = x_width,
                       .inputDepth = x_channels,
                       .inputZeroPoint = input_zero_point,
                       .filterHeight = k_height,
                       .filterWidth = k_width,
                       .filterDepth = k_depth,
                       .strideH = k_v_stride,
                       .strideW = k_h_stride,
                       .dilationH = k_v_dilation,
                       .dilationW = k_h_dilation,
                       .filter = weights,
                       .bias = bias,
                       .effectiveMultiplier = eff_mult,
                       .topPad = top_pad,
                       .leftPad = left_pad,
                       .bottomPad = bottom_pad,
                       .rightPad = right_pad,
                       .padValue = kernel_pad_val};

    llvm::SmallVector<std::string> abstractKernelParams, memcpyFnParams,
        aggregateFnParams, outputTransformFnParams;
    llvm::SmallVector<int32_t> scratchByteParams;

    // TODO: Get thread count as command-line option
    // Currently thread count is one
    int threadCount = 2;

    // TODO: Multithread analysis to determine how to split up the data between
    // threads.
    // Also to determine which kernel type for each thread.
    // Might be better to do this as an analysis pass and access the analysis
    // results here
    for (int i = 0; i < threadCount; ++i) {
      llvm::SmallVector<std::string> conv2DParams;

      // Call the kernel type function which returns a vector of four strings
      // for the four Conv2D params
      conv2DParams = getConv2DValidDirectParams(args);

      abstractKernelParams.push_back(conv2DParams[0]);
      memcpyFnParams.push_back(conv2DParams[1]);
      aggregateFnParams.push_back(conv2DParams[2]);
      outputTransformFnParams.push_back(conv2DParams[3]);

      // Find out the scratch size needed
      int inputBytes = args.filterHeight * args.filterWidth * args.filterDepth;
      int scratchBytes =
          nn::MatMulInt8::get_scratch_mem_bytes(inputBytes) + 32; //[asj] FIXME
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
        rewriter.getStringAttr(stringifyConv2DType(Conv2DType::ValidInDirect)));
    rewriter.replaceOp(conv2DOp, newConv2DV2Op.output());

    return success();
  }
};

void ReplaceWithConv2DV2::runOnFunction() {
  auto *ctx = &getContext();
  auto func = getFunction();

  OwningRewritePatternList patterns(ctx);
  patterns.insert<ReplaceWithConv2DV2Pattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceWithConv2DV2 pass.
std::unique_ptr<OperationPass<FuncOp>> createReplaceWithConv2DV2Pass() {
  return std::make_unique<ReplaceWithConv2DV2>();
}

static PassRegistration<ReplaceWithConv2DV2>
    pass("xcore-replace-with-conv2dv2",
         "Replace TFL Conv2D with XC Conv2DV2 operations.");

} // namespace xcore
} // namespace mlir
