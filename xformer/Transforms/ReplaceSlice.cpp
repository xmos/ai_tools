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

// Replace TFL Slice with Slice for XCore.
struct ReplaceSlice
    : public PassWrapper<ReplaceSlice, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceSlice)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-slice"; }
  StringRef getDescription() const final {
    return "Replace TFL Slice with Slice for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceSlicePattern : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp sliceOp,
                                PatternRewriter &rewriter) const override {

    // If the input is a constant, LLVM's Canonicalizer will
    // fold the slice into a constant later.
    if (matchPattern(sliceOp.getInput(), m_Constant()) ||
        matchPattern(sliceOp.getInput(), m_Op<TFL::ShapeOp>())) {
      return failure();
    }

    auto inputType = sliceOp.getInput().getType().cast<RankedTensorType>();
    if (!inputType.hasStaticShape())
      return failure();

    Type inputElementType = inputType.getElementType();
    auto outputType = sliceOp.getOutput().getType().cast<RankedTensorType>();

    DenseElementsAttr beginAttr;
    matchPattern(sliceOp.getBegin(), m_Constant(&beginAttr));
    auto beginValues = beginAttr.getValues<int32_t>();

    DenseElementsAttr sizeAttr;
    matchPattern(sliceOp.getSize(), m_Constant(&sizeAttr));
    auto sizeValues = sizeAttr.getValues<int32_t>();

    const int rank = inputType.getRank();

    if (utils::checkSliceNoOp(beginValues, sizeValues, inputType)) {
      rewriter.replaceOp(sliceOp, sliceOp.getInput());
      return success();
    }

    int begin_dst[5], end_dst[5], in_offsets[4], out_offsets[4], shape_dst[5];

    // TFLite supports up to 5 dimensions, if the input is less we pad
    const size_t dtype_size = utils::getTypeSize(inputElementType);

    // Cast beginValues and sizeValues to int* for slice_memcpy_get_params
    int begin[5], size[5], shape[5];
    for (int i = 0; i < rank; i++) {
      begin[i] = beginValues[i];
      size[i] = sizeValues[i];
      shape[i] = inputType.getShape()[i];
    }

    slice_memcpy_get_params(begin_dst, end_dst, in_offsets, out_offsets,
                            shape_dst, begin, size, shape, dtype_size, rank);
    auto binaryObjectSliceOp = rewriter.create<SliceOp>(
        sliceOp.getLoc(), sliceOp.getType(), sliceOp.getInput(),
        rewriter.getI32ArrayAttr(begin_dst), rewriter.getI32ArrayAttr(end_dst),
        rewriter.getI32ArrayAttr(in_offsets),
        rewriter.getI32ArrayAttr(out_offsets));

    rewriter.replaceOp(sliceOp, binaryObjectSliceOp.getOutput());

    return success();
  }
};

void ReplaceSlice::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceSlicePattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceSlice pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceSlicePass() {
  return std::make_unique<ReplaceSlice>();
}

static PassRegistration<ReplaceSlice> pass;

} // namespace mlir::xcore
