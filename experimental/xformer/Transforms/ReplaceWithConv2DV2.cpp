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

struct ReplaceWithConv2DV2Pattern : public OpRewritePattern<TFL::Conv2DOp> {
  using OpRewritePattern<TFL::Conv2DOp>::OpRewritePattern;

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

    llvm::SmallVector<llvm::StringRef> abstractKernelParams, memcpyFnParams,
        aggregateFnParams, outputTransformFnParams;
    llvm::SmallVector<int32_t> scratchBytes;
    // TODO: Get thread count as command-line option
    // Currently thread count is one
    int threadCount = 2;

    // TODO: Multithread analysis to determine how to split up the data between
    // threads.
    // Also to determine which kernel type for each thread.
    // Might be better to do this as an analysis pass and access the analysis
    // results here

    for (int i = 0; i < threadCount; ++i) {
      // Retrieve some of the parameters from the op
      // This is to find the correct kernel type

      // Retrieve the rest of the parameters from the op
      // Call the function for that which returns a vector of four strings for
      // the four params

      {
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

        int VPU_INT8_ACC_PERIOD = 16;

        std::vector<int8_t> weights(k_height * k_width * k_depth * x_channels,
                                    1);
        std::vector<int32_t> bias(k_depth, 1);
        std::vector<float> eff_mult(k_depth, 1);
        int input_zero_point = 1;
        int output_zero_point = 1;

        int input_bytes = 1; // geom.window.shape.height *
                             // geom.window.shape.width * geom.input.depth;
        int scratch_bytes = nn::MatMulInt8::get_scratch_mem_bytes(input_bytes) +
                            32; //[asj] FIXME

        nn::padding_t padding = {(int16_t)top_pad, (int16_t)left_pad,
                                 (int16_t)bottom_pad, (int16_t)right_pad};

        // here output_height + width muct match the
        // allocated memory for y
        nn::ImageGeometry Y(output_height, output_width, k_depth);

        nn::ImageGeometry X(x_height, x_width, x_channels);

        nn::WindowGeometry K(k_height, k_width, k_depth, -padding.top,
                             -padding.left, k_v_stride, k_h_stride, 1,
                             k_v_dilation, k_h_dilation);

        nn::DerefInputFn::Params im_to_col_params(X, K);
        nn::DerefInputFn memcpy(&im_to_col_params);

        std::array<int, 4> shape = {k_depth, k_height, k_width, x_channels};
        nn::Conv2dReorderedWeights rw = nn::MatMulInt8::reorder_kernel_weights(
            (int8_t *)weights.data(), shape, 8, kernel_pad_val);

        nn::MatMulDirectFn::Params p(X, K, x_channels, rw.weights.data());
        nn::MatMulDirectFn aggregator(&p);

        nn::OutputTransformFnInt8::CanonicalMulAndBias canonical_values =
            nn::OutputTransformFnInt8::canonicalise_mul_and_bias(
                eff_mult, bias, weights, input_zero_point, output_zero_point,
                k_depth);

        nn::QuantisationParams qp =
            nn::OutputTransformFnInt8::quantise_activation(
                canonical_values.f_multipliers, canonical_values.f_biases,
                canonical_values.accu_min, canonical_values.accu_max);
        nn::OT_int8::Params ot_params((int32_t)k_depth, &qp.otv,
                                      qp.biases.data(), qp.multipliers.data());
        nn::OT_int8 ot(&ot_params);
        auto ir = nn::ImageRegion(0, 0, 0, Y.height, Y.width, Y.depth);

        nn::Filter2D::Params akp(Y, ir, VPU_INT8_ACC_PERIOD);

        // TODO: This method of serializing by casting can only work when the
        // param structs don't have pointer members. Otherwise, we would need to
        // use a proper serialize method which returns a serialized string.
        llvm::StringRef akpStr(reinterpret_cast<const char *>(&akp),
                               sizeof(akp));
        llvm::StringRef otStr(reinterpret_cast<const char *>(&ot_params),
                              sizeof(ot_params));

        // Store the param vector for each thread
        scratchBytes.push_back(scratch_bytes);
        abstractKernelParams.push_back(akpStr);
        memcpyFnParams.push_back(akpStr);
        aggregateFnParams.push_back(akpStr);
        outputTransformFnParams.push_back(otStr);

        nn::Conv2dValidDirect conv2d(&akp, &memcpy, &aggregator, &ot);
      }

      // Find out the scratch size needed calling the lib_nn functions

      // have to accumulate the vector of strings to be written out later
    }

    // Create the Conv2DV2 Op with the params and kernel type
    auto newConv2DV2Op = rewriter.create<Conv2DV2Op>(
        conv2DOp.getLoc(), conv2DOp.getType(), conv2DOp.input(),
        rewriter.getI32IntegerAttr(threadCount),
        rewriter.getI32ArrayAttr(scratchBytes),
        rewriter.getStrArrayAttr(abstractKernelParams),
        rewriter.getStrArrayAttr(memcpyFnParams),
        rewriter.getStrArrayAttr(aggregateFnParams),
        rewriter.getStrArrayAttr(outputTransformFnParams),
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
