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

// Replace TFL Concatenate with Concat for XCore.
struct ReplaceConcat
    : public PassWrapper<ReplaceConcat, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceConcat)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-concat"; }
  StringRef getDescription() const final {
    return "Replace TFL Concatenate with Concat for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceConcatPattern : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern<TFL::ConcatenationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concatOp,
                                PatternRewriter &rewriter) const override {

    auto values = concatOp.getValues();
    if (values.size() != 2) {
      return failure();
    }

    auto inputType1 = values[0].getType().cast<RankedTensorType>();
    auto inputType2 = values[1].getType().cast<RankedTensorType>();
    auto outputType = concatOp.getOutput().getType().cast<RankedTensorType>();
    if (!inputType1.hasStaticShape() || !inputType2.hasStaticShape()) {
      return failure();
    }

    Type elementType = inputType1.getElementType();

    int axis = concatOp.getAxis();
    const int rank = inputType1.getRank();
    if (axis < 0)
      axis = rank + axis;
    auto in_shp1 = inputType1.getShape();
    auto in_shp2 = inputType2.getShape();

    int beginValues1[5], sizeValues1[5];
    int beginValues2[5], sizeValues2[5];
    for (int i = 0; i < rank; i++) {
      beginValues1[i] = 0;
      sizeValues1[i] = in_shp1[i];
      beginValues2[i] = i == axis ? in_shp1[i] : 0;
      sizeValues2[i] = in_shp2[i];
    }
    const size_t dtype_size = utils::getTypeSize(elementType);

    // Cast beginValues and sizeValues to int* for slice_memcpy_get_params
    int begin1[5], size1[5], shape[5], begin2[5], size2[5];
    int begin_dst1[5], end_dst1[5], in_offsets1[4], out_offsets1[4],
        shape_dst1[5];
    int begin_dst2[5], end_dst2[5], in_offsets2[4], out_offsets2[4],
        shape_dst2[5];
    for (int i = 0; i < rank; i++) {
      begin1[i] = beginValues1[i];
      size1[i] = sizeValues1[i];
      begin2[i] = beginValues2[i];
      size2[i] = sizeValues2[i];
      shape[i] = outputType.getShape()[i];
    }

    slice_memcpy_get_params(begin_dst1, end_dst1, in_offsets1, out_offsets1,
                            shape_dst1, begin1, size1, shape, dtype_size, rank);
    slice_memcpy_get_params(begin_dst2, end_dst2, in_offsets2, out_offsets2,
                            shape_dst2, begin2, size2, shape, dtype_size, rank);

    auto binaryObjectConcatOp = rewriter.create<ConcatOp>(
        concatOp.getLoc(), concatOp.getType(), values[0], values[1],
        // we don't need begin1: always 0
        rewriter.getI32ArrayAttr(end_dst1),
        rewriter.getI32ArrayAttr(in_offsets1),
        rewriter.getI32ArrayAttr(out_offsets1),
        // begin2 is always 0 apart from last element:
        rewriter.getI32IntegerAttr(begin_dst2[4]),
        rewriter.getI32ArrayAttr(end_dst2),
        // in_offsets2 == in_offsets1
        rewriter.getI32ArrayAttr(out_offsets2));

    rewriter.replaceOp(concatOp, binaryObjectConcatOp.getOutput());

    return success();
  }
};

void ReplaceConcat::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceConcatPattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceConcat pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceConcatPass() {
  return std::make_unique<ReplaceConcat>();
}

static PassRegistration<ReplaceConcat> pass;

} // namespace mlir::xcore
