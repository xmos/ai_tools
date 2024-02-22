// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Utils/Util.h"

extern "C" {
#include "lib_nn/api/nn_layers.h"
}
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"

namespace mlir::xcore {

namespace {
// Replace TFL Mul with Mul for XCore.
struct ReplaceMul
    : public PassWrapper<ReplaceMul, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceMul)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-mul"; }
  StringRef getDescription() const final {
    return "Replace TFL Mul with Mul for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceMulPattern : public OpRewritePattern<TFL::MulOp> {
  using OpRewritePattern<TFL::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MulOp mulOp,
                                PatternRewriter &rewriter) const override {

    if (!utils::checkBinaryCompatibility(mulOp))
      return failure();

    auto lhsQType = utils::getQType(mulOp.getLhs());
    auto lhsScale = lhsQType.getScale();
    auto lhsZeroPoint = lhsQType.getZeroPoint();

    auto rhsQType = utils::getQType(mulOp.getRhs());
    auto rhsScale = rhsQType.getScale();
    auto rhsZeroPoint = rhsQType.getZeroPoint();

    auto outputQType = utils::getQType(mulOp.getOutput());
    auto outputScale = outputQType.getScale();
    auto outputZeroPoint = outputQType.getZeroPoint();

    nn_mul_params_t mp;
    mul_boggle(&mp, lhsScale, rhsScale, outputScale, lhsZeroPoint, rhsZeroPoint,
               outputZeroPoint);
    auto mpStr = std::string((char *)&mp, sizeof(nn_mul_params_t));

    auto xcMulOp =
        rewriter.create<MulOp>(mulOp.getLoc(), mulOp.getType(), mulOp.getLhs(),
                               mulOp.getRhs(), rewriter.getStringAttr(mpStr));
    rewriter.replaceOp(mulOp, xcMulOp.getOutput());

    return success();
  }
};

void ReplaceMul::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceMulPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceMul pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceMulPass() {
  return std::make_unique<ReplaceMul>();
}

static PassRegistration<ReplaceMul> pass;

} // namespace mlir::xcore
