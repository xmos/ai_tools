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

int mergeAxes(std::vector<int32_t> &begin, std::vector<int32_t> &size,
              std::vector<int32_t> &inShape, std::vector<int32_t> &outShape,
              int rank) {

  for (int i = rank - 1; i > 0; i--) {
    while ((inShape[i] == outShape[i]) && (i > 0)) {
      const int mul = inShape[i];
      inShape[i - 1] *= mul;
      outShape[i - 1] *= mul;
      begin[i - 1] *= mul;
      size[i - 1] *= mul;
      inShape.erase(inShape.begin() + i);
      outShape.erase(outShape.begin() + i);
      begin.erase(begin.begin() + i);
      size.erase(size.begin() + i);
      rank -= 1;
      i -= 1;
    }
  }
  if ((inShape[0] == 1) && (outShape[0] == 1)) {
    inShape.erase(inShape.begin());
    outShape.erase(outShape.begin());
    begin.erase(begin.begin());
    size.erase(size.begin());
    rank -= 1;
  }
  return rank;
}

struct ReplaceSlicePattern : public OpRewritePattern<TFL::SliceOp> {
  using OpRewritePattern<TFL::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::SliceOp sliceOp,
                                PatternRewriter &rewriter) const override {

    auto inputType = sliceOp.getInput().getType().cast<RankedTensorType>();
    auto outputType = sliceOp.getOutput().getType().cast<RankedTensorType>();

    if (!inputType.hasStaticShape())
      return failure();

    if (utils::checkSliceNoOp(inputType, outputType)) {
      rewriter.replaceOp(sliceOp, sliceOp.getInput());
      return success();
    }

    // If the input is a constant, LLVM's Canonicalizer will
    // fold the slice into a constant later.
    if (matchPattern(sliceOp.getInput(), m_Constant()) ||
        matchPattern(sliceOp.getInput(), m_Op<TFL::ShapeOp>())) {
      return failure();
    }

    Type inputElementType = inputType.getElementType();

    DenseElementsAttr beginAttr;
    matchPattern(sliceOp.getBegin(), m_Constant(&beginAttr));
    auto beginValues = beginAttr.getValues<int32_t>();

    DenseElementsAttr sizeAttr;
    matchPattern(sliceOp.getSize(), m_Constant(&sizeAttr));
    auto sizeValues = sizeAttr.getValues<int32_t>();

    auto inShape = inputType.getShape();
    auto outShape = outputType.getShape();

    std::vector<int32_t> begin(beginValues.begin(), beginValues.end());
    std::vector<int32_t> sizes(sizeValues.begin(), sizeValues.end());
    std::vector<int32_t> inShapeVec(inShape.begin(), inShape.end());
    std::vector<int32_t> outShapeVec(outShape.begin(), outShape.end());

    int rank =
        mergeAxes(begin, sizes, inShapeVec, outShapeVec, inputType.getRank());

    if (rank > 2)
      return failure();

    const size_t dtype_size = utils::getTypeSize(inputElementType);
    begin[rank - 1] *= dtype_size;
    sizes[rank - 1] *= dtype_size;
    inShapeVec[rank - 1] *= dtype_size;
    outShapeVec[rank - 1] *= dtype_size;

    int32_t start, offset, size, num_copies;
    if (rank == 1) {
      start = begin[0];
      offset = inShapeVec[0];
      size = outShapeVec[0];
      num_copies = 1;
    } else {
      start = begin[0] * inShapeVec[1] + begin[1];
      offset = inShapeVec[1];
      size = outShapeVec[1];
      num_copies = outShapeVec[0];
    }

    bool isVpu = start % 4 == 0 && size % 4 == 0 && offset % 4 == 0;
    auto binaryObjectSliceOp = rewriter.create<SliceOp>(
        sliceOp.getLoc(), sliceOp.getType(), sliceOp.getInput(),
        rewriter.getI32IntegerAttr(start), rewriter.getI32IntegerAttr(offset),
        rewriter.getI32IntegerAttr(size),
        rewriter.getI32IntegerAttr(num_copies), rewriter.getBoolAttr(isVpu));

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
