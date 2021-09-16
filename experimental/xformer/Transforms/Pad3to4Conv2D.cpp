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
// Add padding before TFL Conv2D to align input depth from three to four.
struct Pad3to4Conv2D : public PassWrapper<Pad3to4Conv2D, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
    registry.insert<XCoreDialect>();
  }
  void runOnFunction() override;
};

struct Pad3to4Conv2DPattern : public OpRewritePattern<TFL::Conv2DOp> {
  using OpRewritePattern<TFL::Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::Conv2DOp conv2DOp,
                                PatternRewriter &rewriter) const override {
    // Check for invalid types and return
    // Input depth must be three
    auto inputDepth =
        conv2DOp.input().getType().cast<ShapedType>().getDimSize(3);
    if (inputDepth != 3) {
      return failure();
    }

    // pad op test
    RankedTensorType paddings_ty =
        RankedTensorType::get({4, 2}, rewriter.getI32Type());
    std::vector<int32_t> paddings_values = {0, 0, 0, 0, 0, 0, 0, 1};
    Value paddings = rewriter.create<TFL::ConstOp>(
        conv2DOp.getLoc(),
        DenseIntElementsAttr::get(paddings_ty, paddings_values));

    auto input_shape =
        conv2DOp.input().getType().cast<RankedTensorType>().getShape();
    std::vector<int64_t> pad_shape = {input_shape[0], input_shape[1],
                                      input_shape[2], input_shape[3] + 1};
    SmallVector<int64_t, 4> expand_shape(pad_shape.begin(), pad_shape.end());
    auto expand_result_type = RankedTensorType::get(
        expand_shape,
        conv2DOp.input().getType().cast<ShapedType>().getElementType());

    Value padOutput = rewriter.create<TFL::PadOp>(
        conv2DOp.getLoc(), expand_result_type, conv2DOp.input(), paddings);

    conv2DOp.setOperand(0, padOutput);

    // Pad the filter accordingly
    // We need to do this at compile time instead of using a PadOp as we use the
    // padded filter values for the boggling calculations Get filter values
    auto filterQConstOp =
        dyn_cast<TFL::QConstOp>(conv2DOp.filter().getDefiningOp());
    auto filter = filterQConstOp.value().cast<DenseElementsAttr>();
    auto filterVector = std::vector<int8_t>{filter.getValues<int8_t>().begin(),
                                            filter.getValues<int8_t>().end()};

    auto filter_shape =
        conv2DOp.filter().getType().cast<RankedTensorType>().getShape();
    std::vector<int64_t> pad_filter_shape = {
        filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3] + 1};

    llvm::SmallVector<int8_t, 0> paddedFilterVector;
    paddedFilterVector.reserve(pad_filter_shape[0] * pad_filter_shape[1] *
                               pad_filter_shape[2] * pad_filter_shape[3]);

    for (int i = 0; i < filterVector.size(); i += filter_shape[3]) {
      paddedFilterVector.insert(paddedFilterVector.end(),
                                filterVector.begin() + i,
                                filterVector.begin() + i + filter_shape[3]);
      paddedFilterVector.insert(paddedFilterVector.end(), 0);
    }

    SmallVector<int64_t, 4> expand_filter_shape(pad_filter_shape.begin(),
                                                pad_filter_shape.end());
    auto expand_filter_result_type = RankedTensorType::get(
        expand_filter_shape,
        conv2DOp.filter().getType().cast<ShapedType>().getElementType());

    RankedTensorType newFilterType =
        RankedTensorType::get(expand_filter_shape, rewriter.getIntegerType(8));

    Value padFilterOutput = rewriter.create<TFL::QConstOp>(
        conv2DOp.getLoc(), mlir::TypeAttr::get(expand_filter_result_type),
        DenseElementsAttr::get<int8_t>(newFilterType, paddedFilterVector));

    // Value padFilterOutput = rewriter.create<TFL::PadOp>(
    //     conv2DOp.getLoc(), expand_filter_result_type, conv2DOp.filter(),
    //     paddings);

    conv2DOp.setOperand(1, padFilterOutput);

    // // Create the Conv2DV2 Op with the params and kernel type
    // auto newConv2DV2Op = rewriter.create<Conv2DV2Op>(
    //     conv2DOp.getLoc(), conv2DOp.getType(), output,
    //     rewriter.getI32IntegerAttr(threadCount),
    //     rewriter.getI32ArrayAttr(scratchByteParams),
    //     getStringArrayAttr(abstractKernelParams),
    //     getStringArrayAttr(memcpyFnParams),
    //     getStringArrayAttr(aggregateFnParams),
    //     getStringArrayAttr(outputTransformFnParams),
    //     getStringArrayAttr(kernelTypeEnumParams));
    // rewriter.replaceOp(conv2DOp, newConv2DV2Op.output());

    return success();
  }
};

void Pad3to4Conv2D::runOnFunction() {
  auto *ctx = &getContext();
  auto func = getFunction();

  OwningRewritePatternList patterns(ctx);
  patterns.insert<Pad3to4Conv2DPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the Pad3to4Conv2D pass.
std::unique_ptr<OperationPass<FuncOp>> createPad3to4Conv2DPass() {
  return std::make_unique<Pad3to4Conv2D>();
}

static PassRegistration<Pad3to4Conv2D> pass(
    "xcore-pad-3to4-conv2d",
    "Add padding before TFL Conv2D to align input depth from three to four.");

} // namespace xcore
} // namespace mlir
