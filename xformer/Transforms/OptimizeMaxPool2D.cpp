#include "IR/XCoreOps.h"
#include "Transforms/Options.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/padding.h"

namespace mlir::xcore {

namespace {

struct OptimizeMaxPool2D
    : public PassWrapper<OptimizeMaxPool2D, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeMaxPool2D)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-optimize-maxpool2d"; }
  StringRef getDescription() const final {
    return "Convert MaxPool2D with SAME padding to use explicit padding.";
  }
  void runOnOperation() override;
};

struct ConvertMaxPool2DSamePaddingPattern : public OpRewritePattern<TFL::MaxPool2DOp> {
  using OpRewritePattern<TFL::MaxPool2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MaxPool2DOp mPoolOp,
                               PatternRewriter &rewriter) const override {
    // Only handle SAME padding case
    if (mPoolOp.getPadding() != "SAME") {
      return failure();
    }

    auto inputType = mPoolOp.getInput().getType().dyn_cast<RankedTensorType>();
    auto inputHeight = inputType.getDimSize(1);
    auto inputWidth = inputType.getDimSize(2);
    
    // Calculate padding values using TFLite's padding calculation
    int outHeight, outWidth;
    auto paddingValues = tflite::ComputePaddingHeightWidth(
        mPoolOp.getStrideH(), mPoolOp.getStrideW(),
        /*dilation_rate_height=*/1, /*dilation_rate_width=*/1,
        inputHeight, inputWidth,
        mPoolOp.getFilterHeight(), mPoolOp.getFilterWidth(),
        kTfLitePaddingSame,
        &outHeight, &outWidth);

    // Create padding values tensor
    std::vector<int32_t> paddingValuesVec{0, 0,                    // batch
                                      paddingValues.height, paddingValues.height + paddingValues.height_offset,  // height
                                      paddingValues.width, paddingValues.width + paddingValues.width_offset,    // width
                                      0, 0};                     // channels
    RankedTensorType paddingsType = RankedTensorType::get({4, 2}, rewriter.getI32Type());
    Value paddings = rewriter.create<TFL::ConstOp>(
        mPoolOp.getLoc(),
        DenseIntElementsAttr::get(paddingsType, paddingValuesVec));

    // Create padded input type
    auto paddedInputType = RankedTensorType::get(
        {inputType.getDimSize(0),
         inputType.getDimSize(1) + paddingValues.height * 2 + paddingValues.height_offset,
         inputType.getDimSize(2) + paddingValues.width * 2 + paddingValues.width_offset,
         inputType.getDimSize(3)},
        inputType.getElementType());

    // Create pad op
    auto padOp = rewriter.create<TFL::PadOp>(
        mPoolOp.getLoc(), paddedInputType,
        mPoolOp.getInput(), paddings);

    // Create new maxpool with VALID padding
    auto newMaxPool = rewriter.create<TFL::MaxPool2DOp>(
        mPoolOp.getLoc(),
        mPoolOp.getType(),
        padOp.getOutput(),
        rewriter.getStringAttr("VALID"),  // padding
        mPoolOp.getStrideWAttr(),         // stride_w
        mPoolOp.getStrideHAttr(),         // stride_h
        mPoolOp.getFilterWidthAttr(),     // filter_width
        mPoolOp.getFilterHeightAttr(),    // filter_height
        mPoolOp.getFusedActivationFunctionAttr()); // fused_activation_function

    rewriter.replaceOp(mPoolOp, newMaxPool.getOutput());
    return success();
  }
};

void OptimizeMaxPool2D::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ConvertMaxPool2DSamePaddingPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeMaxPool2DPass() {
  return std::make_unique<OptimizeMaxPool2D>();
}

static PassRegistration<OptimizeMaxPool2D> pass;

} // namespace mlir::xcore