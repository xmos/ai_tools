// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

namespace {
// Add padding before TFL Conv2D to align input depth from three to four.
struct ApplyPaging : public PassWrapper<ApplyPaging, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
    registry.insert<XCoreDialect>();
  }
  void runOnFunction() override;
};

struct ApplyPagingPattern : public OpRewritePattern<TFL::DepthwiseConv2DOp> {
  using OpRewritePattern<TFL::DepthwiseConv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::DepthwiseConv2DOp dConvOp,
                                PatternRewriter &rewriter) const override {
    // Check for invalid types and return
    // These are the types applicable for lowering to an XC Conv2D
    // Input type must be QI8
    // if (!(conv2DOp.input()
    //           .getType()
    //           .template cast<ShapedType>()
    //           .getElementType()
    //           .template isa<quant::QuantizedType>() &&
    //       conv2DOp.input()
    //           .getType()
    //           .template cast<ShapedType>()
    //           .getElementType()
    //           .template cast<quant::QuantizedType>()
    //           .isSigned() &&
    //       conv2DOp.input()
    //               .getType()
    //               .template cast<ShapedType>()
    //               .getElementType()
    //               .template cast<quant::QuantizedType>()
    //               .getStorageTypeIntegralWidth() == 8)) {
    //   return failure();
    // }

    // // Filter type must be QI8
    // if (!(conv2DOp.filter()
    //           .getType()
    //           .template cast<ShapedType>()
    //           .getElementType()
    //           .template isa<quant::QuantizedType>() &&
    //       conv2DOp.filter()
    //           .getType()
    //           .template cast<ShapedType>()
    //           .getElementType()
    //           .template cast<quant::QuantizedType>()
    //           .isSigned() &&
    //       conv2DOp.filter()
    //               .getType()
    //               .template cast<ShapedType>()
    //               .getElementType()
    //               .template cast<quant::QuantizedType>()
    //               .getStorageTypeIntegralWidth() == 8)) {
    //   return failure();
    // }

    // // We don't bother padding if output depth is not a multiple of four as
    // we
    // // cannot optimize it with an XC Conv2D
    // auto outputDepth =
    //     conv2DOp.output().getType().cast<ShapedType>().getDimSize(3);
    // if (outputDepth % 4 != 0) {
    //   return failure();
    // }

    // // Input depth must be three
    // auto inputDepth =
    //     conv2DOp.input().getType().cast<ShapedType>().getDimSize(3);
    // if (inputDepth != 3) {
    //   return failure();
    // }

    if (isa<PageInOp>(dConvOp.input().getDefiningOp())) {
      return failure();
    }

    //    // Pad the Conv2D input
    // RankedTensorType paddingsType =
    //     RankedTensorType::get({4, 2}, rewriter.getI32Type());

    // // Pad the input depth by one, i.e, from 3 to 4
    // std::vector<int32_t> paddingsValues = {0, 0, 0, 0, 0, 0, 0, 0};
    // Value paddings = rewriter.create<TFL::ConstOp>(
    //     dConvOp.getLoc(),
    //     DenseIntElementsAttr::get(paddingsType, paddingsValues));
    // auto inputShape =
    //     dConvOp.input().getType().cast<RankedTensorType>().getShape();

    // // The pad output depth would increse from 3 to 4
    // auto paddedInputResultType = RankedTensorType::get(
    //     {inputShape[0], inputShape[1] - 1, inputShape[2], inputShape[3]},
    //     dConvOp.input().getType().cast<ShapedType>().getElementType());

    // // Set the PadOp output as the Conv2D input
    // auto newPadOp = rewriter.create<TFL::PadOp>(
    //     dConvOp.getLoc(), paddedInputResultType, dyn_cast<TFL::PadOp>(dConvOp.input().getDefiningOp()).input(), paddings);


    RankedTensorType pageOutOutputType =
        RankedTensorType::get({0}, rewriter.getI32Type());

    // Set the PadOp output as the Conv2D input
    auto pageOutOp = rewriter.create<PageOutOp>(
        dConvOp.getLoc(), pageOutOutputType, dConvOp.input(), 100);

    //rewriter.replaceOp(dConvOp.input().getDefiningOp(), pageOutOp.output());

    auto pageInOp = rewriter.create<PageInOp>(
        dConvOp.getLoc(), dConvOp.input().getType(), pageOutOp, 10, 10);

    dConvOp.setOperand(0, pageInOp);

    // // Pad the Conv2D filter
    // // We need to do this at compile time instead of using a PadOp
    // // as we use the padded filter values for the boggling calculations
    // // for creating the XC Conv2D ops
    // auto filterQConstOp =
    //     dyn_cast<TFL::QConstOp>(conv2DOp.filter().getDefiningOp());
    // auto filter = filterQConstOp.value().cast<DenseElementsAttr>();
    // auto filterVector =
    // std::vector<int8_t>{filter.getValues<int8_t>().begin(),
    //                                         filter.getValues<int8_t>().end()};
    // auto filterShape =
    //     conv2DOp.filter().getType().cast<RankedTensorType>().getShape();
    // llvm::SmallVector<int8_t, 0> paddedFilterVector;
    // paddedFilterVector.reserve(filterShape[0] * filterShape[1] *
    //                            filterShape[2] * (filterShape[3] + 1));

    // // Pad the filter depth by one, i.e, from 3 to 4
    // for (int i = 0; i < filterVector.size(); i += filterShape[3]) {
    //   paddedFilterVector.insert(paddedFilterVector.end(),
    //                             filterVector.begin() + i,
    //                             filterVector.begin() + i + filterShape[3]);
    //   paddedFilterVector.insert(paddedFilterVector.end(), 0);
    // }

    // auto paddedFilterResultType = RankedTensorType::get(
    //     {filterShape[0], filterShape[1], filterShape[2], filterShape[3] + 1},
    //     conv2DOp.filter().getType().cast<ShapedType>().getElementType());
    // RankedTensorType paddedFilterValueType = RankedTensorType::get(
    //     {filterShape[0], filterShape[1], filterShape[2], filterShape[3] + 1},
    //     rewriter.getIntegerType(8));

    // // Create a new QConstOp with the padded data
    // Value paddedQConstOpOutput = rewriter.create<TFL::QConstOp>(
    //     conv2DOp.getLoc(), mlir::TypeAttr::get(paddedFilterResultType),
    //     DenseElementsAttr::get<int8_t>(paddedFilterValueType,
    //                                    paddedFilterVector));

    // // Set the QConstOp output as the Conv2D filter
    // conv2DOp.setOperand(1, paddedQConstOpOutput);

    return success();
  }
};

void ApplyPaging::runOnFunction() {
  auto *ctx = &getContext();
  auto func = getFunction();

  OwningRewritePatternList patterns(ctx);
  patterns.insert<ApplyPagingPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ApplyPaging pass.
std::unique_ptr<OperationPass<FuncOp>> createApplyPagingPass() {
  return std::make_unique<ApplyPaging>();
}

static PassRegistration<ApplyPaging> pass(
    "xcore-apply-paging",
    "Add padding before TFL Conv2D to align input depth from three to four.");

} // namespace xcore
} // namespace mlir
