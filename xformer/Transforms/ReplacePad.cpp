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
    if (!inputType.hasStaticShape()) {
      return failure();
    }

    Type elementType =
        padOp.getInput().getType().cast<ShapedType>().getElementType();

    // TODO: Remove this
    // Currently QINT8 is handled by the original XC_PadOp
    // because this doesn't support non zero zero points or
    // the buffer re-use optimisation.
    if (utils::hasNBitSignedQType(elementType) && (inputType.getRank() == 4) &&
        (inputType.getShape()[3] % 4 == 0)) {
      return failure();
    }

    // TODO: Remove this as well
    if (elementType.isa<quant::QuantizedType>()) {
      auto inputQType = elementType.dyn_cast<quant::UniformQuantizedType>();
      int padValue = inputQType.getZeroPoint();
      if (padValue != 0) {
        return failure();
      }
    }

    Type inputElementType = inputType.getElementType();
    auto outputType = padOp.getOutput().getType().cast<RankedTensorType>();

    DenseElementsAttr paddingAttr;
    matchPattern(padOp.getPadding(), m_Constant(&paddingAttr));
    auto paddingValues = paddingAttr.getValues<int32_t>();

    const int rank = inputType.getRank();

    int beginValues[5], sizeValues[5];
    for (int i = 0; i < rank; i++) {
      beginValues[i] = paddingValues[i * 2];
      sizeValues[i] = inputType.getShape()[i];
    }

    if (utils::checkSliceNoOp(beginValues, sizeValues, outputType)) {
      rewriter.replaceOp(padOp, padOp.getInput());
      return success();
    }

    int begin_dst[5], end_dst[5], in_offsets[4], out_offsets[4], shape_dst[5];

    const size_t dtype_size = utils::getTypeSize(inputElementType);

    // Cast beginValues and sizeValues to int* for slice_memcpy_get_params
    size_t totalElements = dtype_size;
    int begin[5], size[5], shape[5];
    for (int i = 0; i < rank; i++) {
      begin[i] = beginValues[i];
      size[i] = sizeValues[i];
      shape[i] = outputType.getShape()[i];
      totalElements *= shape[i];
    }

    // TODO: Fix this.
    // For now we just use vpu_memset_32 to set everything to zero
    // this means some memory would be left over if the output size
    // is not a multiple of 4.
    if (totalElements % 4) {
      return failure();
    }

    slice_memcpy_get_params(begin_dst, end_dst, in_offsets, out_offsets,
                            shape_dst, begin, size, shape, dtype_size, rank);
    auto binaryObjectPadOp = rewriter.create<PadOpV2>(
        padOp.getLoc(), padOp.getType(), padOp.getInput(),
        rewriter.getI32ArrayAttr(begin_dst), rewriter.getI32ArrayAttr(end_dst),
        rewriter.getI32ArrayAttr(in_offsets),
        rewriter.getI32ArrayAttr(out_offsets));

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
