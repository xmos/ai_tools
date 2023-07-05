// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "lib_nn/api/MemCpyFn.hpp"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

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

    // Check for invalid types and return
    // We only currently handle muls with a single element or 1x..xN as RHS and
    // a tensor as LHS
    auto shapeRHS = mulOp.getRhs().getType().cast<ShapedType>();
    if (shapeRHS.getNumElements() !=
        shapeRHS.getDimSize(shapeRHS.getRank() - 1)) {
      return failure();
    }

    auto lhsType = mulOp.getLhs().getType().cast<ShapedType>().getElementType();
    // Lhs type must be QI8
    if (!(lhsType.isa<quant::QuantizedType>() &&
          lhsType.cast<quant::QuantizedType>().isSigned() &&
          lhsType.cast<quant::QuantizedType>().getStorageTypeIntegralWidth() ==
              8)) {
      return failure();
    }

    auto rhsType = mulOp.getRhs().getType().cast<ShapedType>().getElementType();
    // Rhs type must be QI8
    if (!(rhsType.isa<quant::QuantizedType>() &&
          rhsType.cast<quant::QuantizedType>().isSigned() &&
          rhsType.cast<quant::QuantizedType>().getStorageTypeIntegralWidth() ==
              8)) {
      return failure();
    }

    auto outputType =
        mulOp.getOutput().getType().cast<ShapedType>().getElementType();
    // Output type must be QI8
    if (!(outputType.isa<quant::QuantizedType>() &&
          outputType.cast<quant::QuantizedType>().isSigned() &&
          outputType.cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    auto lhsQType = lhsType.dyn_cast<mlir::quant::UniformQuantizedType>();
    auto lhsScale = lhsQType.getScale();
    auto lhsZeroPoint = lhsQType.getZeroPoint();

    auto rhsQType = rhsType.dyn_cast<mlir::quant::UniformQuantizedType>();
    auto rhsScale = rhsQType.getScale();
    auto rhsZeroPoint = rhsQType.getZeroPoint();

    auto outputQType = outputType.dyn_cast<mlir::quant::UniformQuantizedType>();
    auto outputScale = outputQType.getScale();
    auto outputZeroPoint = outputQType.getZeroPoint();

    // x2 = ((S * (-b1 * x0 + -b0 * x1 +  x0 * x1) + (1<<13) >> 14) + B ) +
    // (1<<5) >> 6
    // B =  (b0 * b1 * S + b2)

    double scaleRatio = lhsScale * rhsScale / outputScale;
    int S = round(scaleRatio * pow(2, 14 + 6));

    double biasTerm =
        lhsZeroPoint * rhsZeroPoint * scaleRatio + outputZeroPoint;
    int B = round(biasTerm * pow(2, 6));

    auto xcMulOp = rewriter.create<MulOp>(
        mulOp.getLoc(), mulOp.getType(), mulOp.getLhs(), mulOp.getRhs(),
        rewriter.getI32IntegerAttr(B), rewriter.getI32IntegerAttr(S));
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

} // namespace xcore
} // namespace mlir
