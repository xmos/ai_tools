// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/core/framework/kernel_shape_util.h"

namespace mlir {
namespace xcore {

namespace {
// Apply generated TFL patterns.
struct ApplyTFLPatterns : public PassWrapper<ApplyTFLPatterns, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<XCoreDialect>();
  }
  void runOnFunction() override;
};

SmallVector<Value, 2> getBConv2DPaddingValues(PatternRewriter &rewriter,
                                              mlir::lq::Bconv2dOp conv2DOp) {
  auto inputType = conv2DOp.input().getType().dyn_cast<RankedTensorType>();
  auto filterType = conv2DOp.filter().getType().dyn_cast<RankedTensorType>();
  auto inputHeight = inputType.getDimSize(1);
  auto inputWidth = inputType.getDimSize(2);
  auto filterHeight = filterType.getDimSize(1);
  auto filterWidth = filterType.getDimSize(2);

  // Find padding values
  int64_t newHeight, newWidth;
  int64_t padTop, padBottom, padLeft, padRight;
  if (tensorflow::GetWindowedOutputSizeVerboseV2(
          inputHeight, filterHeight, conv2DOp.dilation_height_factor(),
          conv2DOp.stride_height(), tensorflow::Padding::SAME, &newHeight,
          &padTop, &padBottom) != tensorflow::Status::OK()) {
    conv2DOp->emitError("Could not obtain SAME padding values for BConv2D!");
  }
  if (tensorflow::GetWindowedOutputSizeVerboseV2(
          inputWidth, filterWidth, conv2DOp.dilation_width_factor(),
          conv2DOp.stride_width(), tensorflow::Padding::SAME, &newWidth,
          &padLeft, &padRight) != tensorflow::Status::OK()) {
    conv2DOp->emitError("Could not obtain SAME padding values for BConv2D!");
  }

  std::vector<int32_t> paddingValues{0,
                                     0,
                                     static_cast<int>(padTop),
                                     static_cast<int>(padBottom),
                                     static_cast<int>(padLeft),
                                     static_cast<int>(padRight),
                                     0,
                                     0};

  RankedTensorType type = RankedTensorType::get({4, 2}, rewriter.getI32Type());
  auto attr = DenseIntElementsAttr::get(type, paddingValues);
  auto paddingOp = rewriter.create<ConstantOp>(conv2DOp->getLoc(), type, attr);

  // Obtain the output type so that we can use it to denote the returnType for
  // the PadOp in Tablegen DRR
  auto batch = 1;
  auto depth = inputType.getDimSize(3);
  auto outputHeight = inputHeight + padTop + padBottom;
  auto outputWidth = inputWidth + padLeft + padRight;
  std::vector<int32_t> dummy(batch * outputHeight * outputWidth * depth, 0);
  RankedTensorType outputType = RankedTensorType::get(
      {batch, outputHeight, outputWidth, depth}, rewriter.getI32Type());
  auto outputTypeOp =
      rewriter.create<ConstantOp>(conv2DOp->getLoc(), outputType,
                                  DenseIntElementsAttr::get(outputType, dummy));

  return SmallVector<Value, 2>({paddingOp, outputTypeOp});
}

#include "Transforms/GeneratedTFLPatterns.inc"

void ApplyTFLPatterns::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  auto func = getFunction();

  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ApplyTFLPatterns pass.
std::unique_ptr<OperationPass<FuncOp>> createApplyTFLPatternsPass() {
  return std::make_unique<ApplyTFLPatterns>();
}

static PassRegistration<ApplyTFLPatterns>
    pass("xcore-apply-tflpatterns",
         "Apply generated TFL optimization patterns.");

} // namespace xcore
} // namespace mlir
