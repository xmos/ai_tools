// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Utils/Util.h"

#include "lib_nn/api/TransposeConv.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::xcore {

namespace {
// Replace TFL TransposeConv with Conv for XCore.
struct ReplaceTransposeConv
    : public PassWrapper<ReplaceTransposeConv, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceTransposeConv)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-transposeconv"; }
  StringRef getDescription() const final {
    return "Replace TFL TransposeConv with Conv for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceTransposeConvPattern
    : public OpRewritePattern<TFL::TransposeConvOp> {
  using OpRewritePattern<TFL::TransposeConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeConvOp tConvOp,
                                PatternRewriter &rewriter) const override {
    // Check for invalid types and return
    // Input type must be QI8 or QI16
    auto inputElementType =
        tConvOp.getInput().getType().cast<ShapedType>().getElementType();
    auto weightsElementType =
        tConvOp.getWeights().getType().cast<ShapedType>().getElementType();
    auto outputElementType =
        tConvOp.getOutput().getType().cast<ShapedType>().getElementType();

    if (!utils::hasNBitSignedQType(inputElementType) &&
        !utils::hasNBitSignedQType<16>(inputElementType))
      return failure();

    if (!utils::hasNBitSignedQType(weightsElementType))
      return failure();

    if (!utils::hasNBitSignedQType(outputElementType) &&
        !utils::hasNBitSignedQType<16>(outputElementType))
      return failure();

    if (tConvOp.getPadding() != "VALID") {
      return failure();
    }

    bool i16TransposeConv = false;
    if (inputElementType.cast<quant::QuantizedType>()
                .getStorageTypeIntegralWidth() == 16 &&
        outputElementType.cast<quant::QuantizedType>()
                .getStorageTypeIntegralWidth() == 16) {
      i16TransposeConv = true;
    }

    auto inputType =
        tConvOp.getInput().getType().template dyn_cast<RankedTensorType>();
    auto outputType =
        tConvOp.getOutput().getType().template dyn_cast<RankedTensorType>();
    auto weightsType =
        tConvOp.getWeights().getType().template dyn_cast<RankedTensorType>();
    auto inputWidth = inputType.getDimSize(2);
    auto inputDepth = inputType.getDimSize(3);
    auto outputDepth = outputType.getDimSize(3);
    auto weightsHeight = weightsType.getDimSize(1);
    auto weightsWidth = weightsType.getDimSize(2);

    // Input and output depth must be multiple of four
    if (inputDepth % 4 != 0 || outputDepth % 4 != 0) {
      return failure();
    }

    // If int16, then input and output depth must be multiple of sixteen
    if (i16TransposeConv && (inputDepth % 16 != 0 || outputDepth % 16 != 0)) {
      return failure();
    }

    // Get weights values
    auto weightsQConstOp =
        dyn_cast<TFL::QConstOp>(tConvOp.getWeights().getDefiningOp());
    auto weightsQConstOpType =
        weightsQConstOp.getQtype().template cast<RankedTensorType>();
    auto weights =
        weightsQConstOp.getValue().template cast<DenseElementsAttr>();
    auto weightsVector =
        std::vector<int8_t>{weights.template getValues<int8_t>().begin(),
                            weights.template getValues<int8_t>().end()};

    //
    //
    //
    //

    // This is the shape of the conv2d transpose kernel
    std::array<int, 4> original_kernel_shape = {
        {static_cast<int>(outputDepth), static_cast<int>(weightsHeight),
         static_cast<int>(weightsWidth), static_cast<int>(inputDepth)}};

    std::vector<ConvParams> convParams = transpose_conv_reorder_kernel_weights(
        (int8_t *)weightsVector.data(), original_kernel_shape,
        tConvOp.getStrideH(), tConvOp.getStrideW());

    int vertical_padding = 0;
    int horizontal_padding = 0;
    for (ConvParams c : convParams) {
      std::array<int, 4> sub_kernel_shape = c.kernelShape;
      vertical_padding = std::max(vertical_padding, sub_kernel_shape[1] - 1);
      horizontal_padding =
          std::max(horizontal_padding, sub_kernel_shape[2] - 1);
    }

    //
    //
    // Create pad op if necessary
    //
    //
    RankedTensorType paddingsType =
        RankedTensorType::get({4, 2}, rewriter.getI32Type());

    // Pad the input depth
    if (vertical_padding > 0 || horizontal_padding > 0) {
      std::vector<int32_t> paddingsValues = {0,
                                             0,
                                             vertical_padding,
                                             vertical_padding,
                                             horizontal_padding,
                                             horizontal_padding,
                                             0,
                                             0};
      Value paddings = rewriter.create<TFL::ConstOp>(
          tConvOp.getLoc(),
          DenseIntElementsAttr::get(paddingsType, paddingsValues));
      auto inputShape = tConvOp.getInput()
                            .getType()
                            .template cast<RankedTensorType>()
                            .getShape();

      auto paddedInputResultType = RankedTensorType::get(
          {inputShape[0], inputShape[1] + 2 * vertical_padding,
           inputShape[2] + 2 * horizontal_padding, inputShape[3]},
          tConvOp.getInput()
              .getType()
              .template cast<ShapedType>()
              .getElementType());

      // Set the PadOp output as the Conv2D input
      Value padOpOutput =
          rewriter.create<TFL::PadOp>(tConvOp.getLoc(), paddedInputResultType,
                                      tConvOp.getInput(), paddings);
      // Strangely, operand 2 is the input for TFL Transpose Conv
      tConvOp.setOperand(2, padOpOutput);
    }

    //
    //
    //
    //

    FakeConv2DOp prevOp = nullptr;
    FakeConv2DOp currentOp = nullptr;
    for (ConvParams c : convParams) {
      // Calculate input offset
      auto this_kernels_vertical_padding = c.kernelShape[1] - 1;
      auto this_kernels_horizontal_padding = c.kernelShape[2] - 1;
      auto inputOffset =
          inputDepth * (horizontal_padding - this_kernels_horizontal_padding) +
          (inputDepth * (vertical_padding - this_kernels_vertical_padding) *
           (inputWidth + 2 * horizontal_padding));

      // inputOffset is in bytes.
      // For int16, we have to multiply by two as the offset is double that of
      // int8
      inputOffset = i16TransposeConv ? inputOffset * 2 : inputOffset;

      int64_t subWeightsShape[] = {c.kernelShape[0], c.kernelShape[1],
                                   c.kernelShape[2], c.kernelShape[3]};
      auto subWeightsResultType = RankedTensorType::get(
          subWeightsShape, weightsQConstOpType.getElementType());
      auto subWeightsValueType =
          RankedTensorType::get(subWeightsShape, rewriter.getIntegerType(8));

      auto subWeightsQConstOp = rewriter.create<TFL::QConstOp>(
          tConvOp.getLoc(), mlir::TypeAttr::get(subWeightsResultType),
          mlir::DenseElementsAttr::get(subWeightsValueType,
                                       llvm::ArrayRef(c.weights)));

      auto noneValue = rewriter.create<TFL::NoValueOp>(rewriter.getUnknownLoc(),
                                                       rewriter.getNoneType(),
                                                       rewriter.getUnitAttr());
      // We want the input strides to be 1,1 and the output strides to be the
      // transpose conv strides
      if (!prevOp) {
        currentOp = rewriter.create<FakeConv2DOp>(
            tConvOp.getLoc(), tConvOp.getType(), tConvOp.getInput(),
            subWeightsQConstOp, tConvOp.getBias(), noneValue,
            /*dilation_h_factor=*/1,
            /*dilation_w_factor=*/1,
            /*fused_activation_function=*/tConvOp.getFusedActivationFunction(),
            /*padding=*/tConvOp.getPadding(), noneValue,
            /*stride_h=*/1,
            /*stride_w=*/1, c.subH, c.subW, tConvOp.getStrideH(),
            tConvOp.getStrideW(), inputOffset);
      } else {
        currentOp = rewriter.create<FakeConv2DOp>(
            tConvOp.getLoc(), tConvOp.getType(), tConvOp.getInput(),
            subWeightsQConstOp, tConvOp.getBias(), prevOp.getOutput(),
            /*dilation_h_factor=*/1,
            /*dilation_w_factor=*/1,
            /*fused_activation_function=*/tConvOp.getFusedActivationFunction(),
            /*padding=*/tConvOp.getPadding(), noneValue,
            /*stride_h=*/1,
            /*stride_w=*/1, c.subH, c.subW, tConvOp.getStrideH(),
            tConvOp.getStrideW(), inputOffset);
      }
      prevOp = currentOp;
    }

    rewriter.replaceOp(tConvOp, currentOp.getOutput());

    return success();
  }
};

void ReplaceTransposeConv::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceTransposeConvPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceTransposeConv pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceTransposeConvPass() {
  return std::make_unique<ReplaceTransposeConv>();
}

static PassRegistration<ReplaceTransposeConv> pass;

} // namespace mlir::xcore
