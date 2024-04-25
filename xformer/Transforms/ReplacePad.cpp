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

    auto inputType = padOp.getInput().getType().cast<RankedTensorType>();
    auto outputType = padOp.getOutput().getType().cast<RankedTensorType>();

    if (!inputType.hasStaticShape()) {
      return failure();
    }

    if (utils::checkSliceNoOp(inputType, outputType)) {
      rewriter.replaceOp(padOp, padOp.getInput());
      return success();
    }

    // If the input is a constant, LLVM's Canonicalizer will
    // fold the pad into a constant later.
    if (matchPattern(padOp.getInput(), m_Constant()) ||
        matchPattern(padOp.getInput(), m_Op<TFL::ShapeOp>())) {
      return failure();
    }

    DenseElementsAttr paddingAttr;
    matchPattern(padOp.getPadding(), m_Constant(&paddingAttr));
    auto paddingValues = paddingAttr.getValues<int32_t>();

    std::vector<int32_t> paddingValuesStart;
    std::vector<int32_t> paddingValuesEnd;
    for (int i = 0; i < paddingValues.size(); i += 2)
      paddingValuesStart.push_back(paddingValues[i]);
    for (int i = 1; i < paddingValues.size(); i += 2)
      paddingValuesEnd.push_back(paddingValues[i]);
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();
    std::vector<int32_t> inShapeVec(inputShape.begin(), inputShape.end());
    std::vector<int32_t> outShapeVec(outputShape.begin(), outputShape.end());

    // Assuming we can translate the pad op to reshape -> pad -> reshape
    // such that the number of dimensions of the pad in the middle is as small
    // as possible, rank would represent that number of dimensions.
    int rank = utils::mergeAxes(paddingValuesStart, paddingValuesEnd,
                                inShapeVec, outShapeVec, inputType.getRank());

    if (rank > 2)
      return failure();

    Type elementType = inputType.getElementType();
    const size_t dtype_size = utils::getTypeSize(elementType);
    paddingValuesStart[rank - 1] *= dtype_size;
    paddingValuesEnd[rank - 1] *= dtype_size;
    outShapeVec[rank - 1] *= dtype_size;
    inShapeVec[rank - 1] *= dtype_size;

    int32_t zero_point = 0;
    if (elementType.isa<quant::QuantizedType>()) {
      auto inputQType = elementType.dyn_cast<quant::UniformQuantizedType>();
      zero_point = BROADCAST_8_TO_32(inputQType.getZeroPoint());
    }

    int32_t start, pad_size, size, num_copies, end;
    if (rank == 1) {
      start = paddingValuesStart[0];
      pad_size = 0;
      size = inShapeVec[0];
      num_copies = 0;
      end = paddingValuesEnd[0];
    } else {
      // We have a special optimised Pad3to4 op for this case
      if ((inShapeVec[1] == 3) && (outShapeVec[1] == 4))
        return failure();
      start = paddingValuesStart[0] * outShapeVec[1] + paddingValuesStart[1];
      pad_size = paddingValuesStart[1] + paddingValuesEnd[1];
      size = inShapeVec[1];
      num_copies = inShapeVec[0] - 1;
      end = paddingValuesEnd[0] * outShapeVec[1] + paddingValuesEnd[1];
    }
    bool isVpu =
        start % 4 == 0 && pad_size % 4 == 0 && size % 4 == 0 && end % 4 == 0;

    auto binaryObjectPadOp = rewriter.create<PadOp>(
        padOp.getLoc(), padOp.getType(), padOp.getInput(),
        rewriter.getI32IntegerAttr(start), rewriter.getI32IntegerAttr(pad_size),
        rewriter.getI32IntegerAttr(size),
        rewriter.getI32IntegerAttr(num_copies),
        rewriter.getI32IntegerAttr(zero_point), rewriter.getI32IntegerAttr(end),
        rewriter.getBoolAttr(isVpu));

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
