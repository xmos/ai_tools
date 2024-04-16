// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "Utils/Util.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

extern "C" {
#include "lib_nn/api/nn_layers.h"
#include "lib_nn/api/vpu_memset_256.h"
}

namespace mlir::xcore {

namespace {

// Replace TFL Pad with Pad for XCore.
struct ReplacePad
    : public PassWrapper<ReplacePad, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplacePad)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-pad"; }
  StringRef getDescription() const final {
    return "Replace TFL Pad with Pad for XCore.";
  }
  void runOnOperation() override;
};

struct ReplacePadPattern : public OpRewritePattern<TFL::PadOp> {
  using OpRewritePattern<TFL::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::PadOp padOp,
                                PatternRewriter &rewriter) const override {

    // If the input is a constant, LLVM's Canonicalizer will
    // fold the pad into a constant later.
    if (matchPattern(padOp.getInput(), m_Constant()) ||
        matchPattern(padOp.getInput(), m_Op<TFL::ShapeOp>())) {
      return failure();
    }

    auto inputType = padOp.getInput().getType().cast<RankedTensorType>();
    auto outputType = padOp.getOutput().getType().cast<RankedTensorType>();

    if (!inputType.hasStaticShape()) {
      return failure();
    }

    Type elementType = inputType.getElementType();
    const size_t dtype_size = utils::getTypeSize(elementType);
    const int rank = inputType.getRank();

    if (rank != 4)
      return failure();

    int32_t zero_point = 0;
    if (elementType.isa<quant::QuantizedType>()) {
      auto inputQType = elementType.dyn_cast<quant::UniformQuantizedType>();
      zero_point = BROADCAST_8_TO_32(inputQType.getZeroPoint());
    }

    DenseElementsAttr paddingAttr;
    matchPattern(padOp.getPadding(), m_Constant(&paddingAttr));
    auto paddingValues = paddingAttr.getValues<int32_t>();

    if (paddingValues[0] != 0 || paddingValues[1] != 0)
      return failure();

    bool paddingHW = false;
    for (int i = 2; i < 6; i++)
      if (paddingValues[i] != 0)
        paddingHW = true;
    if (paddingHW && (paddingValues[6] != 0 || paddingValues[7] != 0))
      return failure();

    bool isNoOp = true;
    for (int i = 0; i < 8; i++)
      if (paddingValues[i] != 0)
        isNoOp = false;

    if (isNoOp) {
      rewriter.replaceOp(padOp, padOp.getInput());
      return success();
    }

    auto inShape = inputType.getShape();
    auto outShape = outputType.getShape();

    int32_t start, pad_size, size, num_copies, end;
    const int mulW = inShape[3] * dtype_size;
    if (paddingHW) {
      if (outShape[0] != 1)
        return failure();
      size = inShape[2] * mulW;
      pad_size = (outShape[2] - inShape[2]) * mulW;
      start = (paddingValues[2] * outShape[2] + paddingValues[4]) * mulW;
      end = (paddingValues[3] * outShape[2] + paddingValues[5]) * mulW;
      // -1 because the last memcpy is done separately, since we also need to
      // memset
      num_copies = inShape[1] - 1;
    } else {
      // Pad3to4 is a special case, handled by it's own op
      if (inShape[3] == 3 && outShape[3] == 4)
        return failure();
      size = mulW;
      pad_size = (outShape[3] - inShape[3]) * dtype_size;
      start = paddingValues[6] * dtype_size;
      end = paddingValues[7] * dtype_size;
      // -1 because the last memcpy is done separately, since we also need to
      // memset
      num_copies = inShape[2] * inputType.getShape()[1] * inShape[0] - 1;
    }

    auto binaryObjectPadOp = rewriter.create<PadOp>(
        padOp.getLoc(), padOp.getType(), padOp.getInput(),
        rewriter.getI32IntegerAttr(start), rewriter.getI32IntegerAttr(pad_size),
        rewriter.getI32IntegerAttr(size),
        rewriter.getI32IntegerAttr(num_copies),
        rewriter.getI32IntegerAttr(zero_point),
        rewriter.getI32IntegerAttr(end));

    rewriter.replaceOp(padOp, binaryObjectPadOp.getOutput());

    return success();
  }
};

void ReplacePad::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplacePadPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplacePad pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplacePadPass() {
  return std::make_unique<ReplacePad>();
}

static PassRegistration<ReplacePad> pass;

} // namespace mlir::xcore
