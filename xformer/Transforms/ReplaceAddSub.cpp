// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"
#include "Utils/Util.h"

#include "lib_nn/api/MemCpyFn.hpp"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::xcore {

namespace {
// Replace TFL Add with Add for XCore.
struct ReplaceAddSub
    : public PassWrapper<ReplaceAddSub, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceAddSub)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-addsub"; }
  StringRef getDescription() const final {
    return "Replace TFL Add/Sub with Add for XCore.";
  }
  void runOnOperation() override;
};

template <typename T>
LogicalResult replaceAddorSub(T addOp, PatternRewriter &rewriter,
                              bool negateForSub) {
  if (!utils::checkBinaryCompatibility(addOp))
    return failure();

  auto lhsQType = utils::getQType(addOp.getLhs());
  auto lhsScale = lhsQType.getScale();
  auto lhsZeroPoint = lhsQType.getZeroPoint();

  auto rhsQType = utils::getQType(addOp.getRhs());
  auto rhsScale = rhsQType.getScale();
  auto rhsZeroPoint = rhsQType.getZeroPoint();

  auto outputQType = utils::getQType(addOp.getOutput());
  auto outputScale = outputQType.getScale();
  auto outputZeroPoint = outputQType.getZeroPoint();

  double lhsRatio = lhsScale / outputScale;
  double rhsRatio = rhsScale / outputScale;

  // We find the max in case there is a large difference
  // between lhs and rhs scales.
  double maxR = std::max(lhsRatio, rhsRatio);
  // We want the max shift to be 14 bits
  int shift = int(floor(log2(pow(2, 14) / maxR)));

  // For doing subtraction with add op
  rhsRatio = negateForSub ? -rhsRatio : rhsRatio;

  // Multipliers are converted to fixed-point
  int m1 = round(lhsRatio * pow(2, shift));
  int m2 = round(rhsRatio * pow(2, shift));
  int bias = round((outputZeroPoint - (lhsZeroPoint * lhsRatio) -
                    (rhsZeroPoint * rhsRatio)) *
                   pow(2, shift));

  auto xcAddOp = rewriter.create<AddOp>(
      addOp.getLoc(), addOp.getType(), addOp.getLhs(), addOp.getRhs(),
      rewriter.getStringAttr(addOp.getFusedActivationFunction()),
      rewriter.getI32IntegerAttr(m1), rewriter.getI32IntegerAttr(m2),
      rewriter.getI32IntegerAttr(bias), rewriter.getI32IntegerAttr(shift));
  rewriter.replaceOp(addOp, xcAddOp.getOutput());

  return success();
}

struct ReplaceAddPattern : public OpRewritePattern<TFL::AddOp> {
  using OpRewritePattern<TFL::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    return replaceAddorSub(addOp, rewriter, /*negateForSub=*/false);
  }
};

struct ReplaceSubPattern : public OpRewritePattern<TFL::SubOp> {
  using OpRewritePattern<TFL::SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SubOp subOp,
                                PatternRewriter &rewriter) const override {
    return replaceAddorSub(subOp, rewriter, /*negateForSub=*/true);
  }
};

void ReplaceAddSub::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceAddPattern>(ctx);
  patterns.insert<ReplaceSubPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceAddSub pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceAddSubPass() {
  return std::make_unique<ReplaceAddSub>();
}

static PassRegistration<ReplaceAddSub> pass;

} // namespace mlir::xcore
