// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "lib_nn/api/AbstractKernel.hpp"
#include "lib_nn/api/AggregateFn.hpp"
#include "lib_nn/api/MemCpyFn.hpp"
#include "lib_nn/api/OutputTransformFn.hpp"
#include "lib_nn/api/TransposeConv.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/core/framework/kernel_shape_util.h"

namespace mlir {
namespace xcore {

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
    // We don't currently handle the unusual case where both input shapes have
    // to be broadcasted. Either both input shapes must match the output or one
    // of the inputs has to be broadcasted.
    // if (failed(utils::hasSameShape(
    //         tConvOp.getRhs().getType().cast<ShapedType>(),
    //         tConvOp.getOutput().getType().cast<ShapedType>())) &&
    //     failed(utils::hasSameShape(
    //         tConvOp.getLhs().getType().cast<ShapedType>(),
    //         tConvOp.getOutput().getType().cast<ShapedType>()))) {
    //   return failure();
    // }

    // auto lhsType =
    // tConvOp.getLhs().getType().cast<ShapedType>().getElementType();
    // // Lhs type must be QI8
    // if (!(lhsType.isa<quant::QuantizedType>() &&
    //       lhsType.cast<quant::QuantizedType>().isSigned() &&
    //       lhsType.cast<quant::QuantizedType>().getStorageTypeIntegralWidth()
    //       ==
    //           8)) {
    //   return failure();
    // }

    // auto rhsType =
    // tConvOp.getRhs().getType().cast<ShapedType>().getElementType();
    // // Rhs type must be QI8
    // if (!(rhsType.isa<quant::QuantizedType>() &&
    //       rhsType.cast<quant::QuantizedType>().isSigned() &&
    //       rhsType.cast<quant::QuantizedType>().getStorageTypeIntegralWidth()
    //       ==
    //           8)) {
    //   return failure();
    // }

    // auto outputType =
    //     tConvOp.getOutput().getType().cast<ShapedType>().getElementType();
    // // Output type must be QI8
    // if (!(outputType.isa<quant::QuantizedType>() &&
    //       outputType.cast<quant::QuantizedType>().isSigned() &&
    //       outputType.cast<quant::QuantizedType>()
    //               .getStorageTypeIntegralWidth() == 8)) {
    //   return failure();
    // }

    auto inputType =
        tConvOp.getInput().getType().template dyn_cast<RankedTensorType>();
    auto outputType =
        tConvOp.getOutput().getType().template dyn_cast<RankedTensorType>();
    auto weightsType =
        tConvOp.getWeights().getType().template dyn_cast<RankedTensorType>();
    auto inputDepth = inputType.getDimSize(3);
    auto outputDepth = outputType.getDimSize(3);
    auto weightsHeight = weightsType.getDimSize(1);
    auto weightsWidth = weightsType.getDimSize(2);

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
    for (ConvParams c : convParams) {
      // apply each convolution, writing the result to the correct location in
      // the output
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
      // tConvOp.getStrideH(), tConvOp.getStrideW()
      auto conv2DOp = rewriter.create<FakeConv2DOp>(
          tConvOp.getLoc(), tConvOp.getType(), tConvOp.getInput(),
          subWeightsQConstOp, tConvOp.getBias(),
          /*dilation_h_factor=*/1,
          /*dilation_w_factor=*/1,
          /*fused_activation_function=*/tConvOp.getFusedActivationFunction(),
          /*padding=*/tConvOp.getPadding(), noneValue,
          /*stride_h=*/tConvOp.getStrideH(),
          /*stride_w=*/tConvOp.getStrideW(), c.subH, c.subW,
          tConvOp.getStrideH(), tConvOp.getStrideW());

      rewriter.replaceOp(tConvOp, conv2DOp.getOutput());
    }

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

} // namespace xcore
} // namespace mlir
