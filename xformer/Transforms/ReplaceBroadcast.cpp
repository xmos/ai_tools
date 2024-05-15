// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "Utils/Util.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

extern "C" {
#include "lib_nn/api/nn_layers.h"
}

namespace mlir::xcore {

namespace {

// Replace TFL Broadcast with Broadcast for XCore.
struct ReplaceBroadcast
    : public PassWrapper<ReplaceBroadcast, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceBroadcast)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-broadcast"; }
  StringRef getDescription() const final {
    return "Replace TFL Broadcast with Broadcast for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceBroadcastPattern : public OpRewritePattern<TFL::BroadcastToOp> {
  using OpRewritePattern<TFL::BroadcastToOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::BroadcastToOp broadcastOp,
                                PatternRewriter &rewriter) const override {

    auto inputType = broadcastOp.getInput().getType().cast<RankedTensorType>();
    auto outputType =
        broadcastOp.getOutput().getType().cast<RankedTensorType>();

    if (!inputType.hasStaticShape())
      return failure();

    if (utils::checkSliceNoOp(inputType, outputType)) {
      rewriter.replaceOp(broadcastOp, broadcastOp.getInput());
      return success();
    }

    // If the input is a constant, LLVM's Canonicalizer will
    // fold the broadcast into a constant later.
    if (matchPattern(broadcastOp.getInput(), m_Constant()) ||
        matchPattern(broadcastOp.getInput(), m_Op<TFL::ShapeOp>())) {
      return failure();
    }

    Type inputElementType = inputType.getElementType();

    auto inShape = inputType.getShape();
    auto outShape = outputType.getShape();

    std::vector<int32_t> inShapeVec(inShape.begin(), inShape.end());
    std::vector<int32_t> outShapeVec(outShape.begin(), outShape.end());
    // check only broadcasting along 1 dimension
    bool canBroadcast = true;
    int size = utils::getTypeSize(inputElementType);
    int num_copies = 1, num_broadcasts = 1;
    for (int i = 0; i < inShapeVec.size(); i++) {
      if (canBroadcast) {
        if (inShapeVec[i] != outShapeVec[i])
          num_copies *= outShapeVec[i];
        else {
          if (num_copies != 1) {
            canBroadcast = false;
            size *= inShapeVec[i];
          } else
            num_broadcasts *= outShapeVec[i];
        }
      } else {
        if (inShapeVec[i] != outShapeVec[i])
          return failure();
        size *= inShapeVec[i];
      }
    }

    bool isVpu = size % 4 == 0;
    auto binaryObjectBroadcastOp = rewriter.create<BroadcastOp>(
        broadcastOp.getLoc(), broadcastOp.getType(), broadcastOp.getInput(),
        rewriter.getI32IntegerAttr(size),
        rewriter.getI32IntegerAttr(num_copies),
        rewriter.getI32IntegerAttr(num_broadcasts),
        rewriter.getBoolAttr(isVpu));

    rewriter.replaceOp(broadcastOp, binaryObjectBroadcastOp.getOutput());

    return success();
  }
};

void ReplaceBroadcast::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceBroadcastPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceBroadcast pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceBroadcastPass() {
  return std::make_unique<ReplaceBroadcast>();
}

static PassRegistration<ReplaceBroadcast> pass;

} // namespace mlir::xcore
