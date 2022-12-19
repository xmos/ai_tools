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
// Replace TFL Add with Add for XCore.
struct ReplaceAdd
    : public PassWrapper<ReplaceAdd, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceAdd)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-add"; }
  StringRef getDescription() const final {
    return "Replace TFL Add with Add for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceAddPattern : public OpRewritePattern<TFL::AddOp> {
  using OpRewritePattern<TFL::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::AddOp addOp,
                                PatternRewriter &rewriter) const override {

    // Check for invalid types and return
    // Both input shapes must match
    if (failed(
            verifyCompatibleShapes(addOp.lhs().getType().cast<ShapedType>(),
                                   addOp.rhs().getType().cast<ShapedType>()))) {
      return failure();
    }

    auto lhsType = addOp.lhs().getType().cast<ShapedType>().getElementType();
    // Lhs type must be QI8
    if (!(lhsType.isa<quant::QuantizedType>() &&
          lhsType.cast<quant::QuantizedType>().isSigned() &&
          lhsType.cast<quant::QuantizedType>().getStorageTypeIntegralWidth() ==
              8)) {
      return failure();
    }

    auto rhsType = addOp.rhs().getType().cast<ShapedType>().getElementType();
    // Rhs type must be QI8
    if (!(rhsType.isa<quant::QuantizedType>() &&
          rhsType.cast<quant::QuantizedType>().isSigned() &&
          rhsType.cast<quant::QuantizedType>().getStorageTypeIntegralWidth() ==
              8)) {
      return failure();
    }

    auto outputType =
        addOp.output().getType().cast<ShapedType>().getElementType();
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

    double lhsRatio = lhsScale / outputScale;
    double rhsRatio = rhsScale / outputScale;

    // We find the max in case there is a large difference
    // between lhs and rhs scales.
    double maxR = std::max(lhsRatio, rhsRatio);
    // We want the max shift to be 14 bits
    int shift = int(floor(log2(pow(2, 14) / maxR)));

    // Multipliers are converted to fixed-point
    int m1 = round(lhsRatio * pow(2, shift));
    int m2 = round(rhsRatio * pow(2, shift));
    int bias = round((outputZeroPoint - (lhsZeroPoint * lhsRatio) -
                      (rhsZeroPoint * rhsRatio)) *
                     pow(2, shift));

    auto xcAddOp = rewriter.create<AddOp>(
        addOp.getLoc(), addOp.getType(), addOp.lhs(), addOp.rhs(),
        rewriter.getStringAttr(addOp.fused_activation_function()),
        rewriter.getI32IntegerAttr(m1), rewriter.getI32IntegerAttr(m2),
        rewriter.getI32IntegerAttr(bias), rewriter.getI32IntegerAttr(shift));
    rewriter.replaceOp(addOp, xcAddOp.output());

    return success();
  }
};

void ReplaceAdd::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceAddPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceAdd pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceAddPass() {
  return std::make_unique<ReplaceAdd>();
}

static PassRegistration<ReplaceAdd> pass;

} // namespace xcore
} // namespace mlir
