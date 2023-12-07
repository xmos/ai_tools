#include "IR/XCoreOps.h"
#include "Transforms/Options.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {
struct ReplaceMaxPool2D
    : public PassWrapper<ReplaceMaxPool2D, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceMaxPool2D)
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-maxpool2d"; }
  StringRef getDescription() const final {
    return "Replace TFL MaxPool2D with MaxPool2D for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceMaxPool2DPattern : public OpRewritePattern<TFL::MaxPool2DOp> {
  using OpRewritePattern<TFL::MaxPool2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MaxPool2DOp mPoolOp,
                                PatternRewriter &rewriter) const override {
    auto inputType =
        mPoolOp.getInput().getType().template dyn_cast<RankedTensorType>();
    auto outputType =
        mPoolOp.getOutput().getType().template dyn_cast<RankedTensorType>();
  }
};
} // namespace xcore
} // namespace mlir
