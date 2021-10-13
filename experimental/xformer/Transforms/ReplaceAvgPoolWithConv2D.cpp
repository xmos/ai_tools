// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include <numeric>

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

namespace mlir {
namespace xcore {

namespace {
// Replace TFL AveragePool2D with TFL DepthwiseConv2D.
struct ReplaceAvgPoolWithConv2D
    : public PassWrapper<ReplaceAvgPoolWithConv2D, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  void runOnFunction() override;
};

struct ReplaceAvgPoolWithConv2DPattern
    : public OpRewritePattern<TFL::AveragePool2DOp> {
  using OpRewritePattern<TFL::AveragePool2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::AveragePool2DOp avgPoolOp,
                                PatternRewriter &rewriter) const override {

    auto input_elemental_type = avgPoolOp.input()
                  .getType()
                  .template cast<ShapedType>()
                  .getElementType();

    // Check for invalid types and return
    // Input type must be QI8
    if (!(input_elemental_type.template isa<quant::QuantizedType>() &&
          input_elemental_type.template cast<quant::QuantizedType>().isSigned() &&
          input_elemental_type.template cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      avgPoolOp.emitOpError() << "incorrect input elemental type";     
      return failure();
    }

    auto output_elemental_type = avgPoolOp.output()
                  .getType()
                  .template cast<ShapedType>()
                  .getElementType();

    // Output type must be QI8
    if (!(output_elemental_type.template isa<quant::QuantizedType>() &&
          output_elemental_type.template cast<quant::QuantizedType>().isSigned() &&
          output_elemental_type.template cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      avgPoolOp.emitOpError() << "incorrect output elemental type";
      return failure();
    }

    auto inputType =
        avgPoolOp.input().getType().template dyn_cast<RankedTensorType>();
    auto inputDepth = inputType.getDimSize(3);

    auto filter_height = avgPoolOp.filter_height();
    auto filter_width = avgPoolOp.filter_width();

    float scale_factor = 1./(filter_height* filter_width);

    int64_t storage_type_min =
        quant::QuantizedType::getDefaultMinimumForInteger(/*isSigned=*/true, 8);
    int64_t storage_type_max =
        quant::QuantizedType::getDefaultMaximumForInteger(/*isSigned=*/true, 8);

    UniformQuantizedType int8_element_qtype =
        mlir::quant::UniformQuantizedType::get(
            true, rewriter.getIntegerType(8), rewriter.getF32Type(), scale_factor, 0,
            storage_type_min, storage_type_max);

    auto filterResultType = RankedTensorType::get(
        {1, filter_height, filter_width, inputDepth},
        int8_element_qtype);

    RankedTensorType filterValueType = RankedTensorType::get(
        {1, filter_height, filter_width, inputDepth},
        rewriter.getIntegerType(8));

    std::vector<int8_t>filterVector(filter_height*filter_width*inputDepth, 1);

    Value filter = rewriter.create<TFL::QConstOp>(
        avgPoolOp.getLoc(), 
        mlir::TypeAttr::get(filterResultType),
        DenseElementsAttr::get<int8_t>(filterValueType,
                                       filterVector));

    //[asj] This may need to be QI32 but I32 seems to work 
    RankedTensorType biasType =
        RankedTensorType::get({inputDepth}, rewriter.getI32Type());
    std::vector<int32_t> biasValues(inputDepth, 0);
    auto bias = rewriter.create<TFL::ConstOp>(avgPoolOp->getLoc(), 
      DenseIntElementsAttr::get(biasType, biasValues));

   auto conv2dOp = rewriter.create<TFL::DepthwiseConv2DOp>(
        avgPoolOp.getLoc(), 
        avgPoolOp.getType(), 
        avgPoolOp.input(),
        filter, 
        bias, //TODO [asj]how do we drop the bias?
        /*dilation_h_factor=*/rewriter.getI32IntegerAttr(1),
        /*dilation_w_factor=*/rewriter.getI32IntegerAttr(1),
        /*fused_activation_function=*/rewriter.getStringAttr(avgPoolOp.fused_activation_function()),
        /*padding=*/rewriter.getStringAttr(avgPoolOp.padding()),
        /*stride_h=*/rewriter.getI32IntegerAttr(avgPoolOp.stride_h()),
        /*stride_w=*/rewriter.getI32IntegerAttr(avgPoolOp.stride_w()),
        /*depth_multiplier=*/rewriter.getI32IntegerAttr(1));

    rewriter.replaceOp(avgPoolOp, conv2dOp.output());

    return success();
  }
};

void ReplaceAvgPoolWithConv2D::runOnFunction() {
  auto *ctx = &getContext();
  auto func = getFunction();

  OwningRewritePatternList patterns(ctx);
  patterns.insert<ReplaceAvgPoolWithConv2DPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
}  // namespace

// Creates an instance of the ReplaceAvgPoolWithConv2D pass.
std::unique_ptr<OperationPass<FuncOp>> createReplaceAvgPoolWithConv2DPass() {
  return std::make_unique<ReplaceAvgPoolWithConv2D>();
}

static PassRegistration<ReplaceAvgPoolWithConv2D> pass(
    "xcore-replace-avgpool-with-conv2d",
    "Replace TFL Avgpool with Conv2D operations.");

}  // namespace xcore
}  // namespace mlir
