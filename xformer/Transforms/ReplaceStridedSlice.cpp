// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1

#include "IR/XCoreOps.h"

#include "lib_nn/api/MemCpyFn.hpp"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace xcore {

namespace {
// Replace TFL StridedSlice with StridedSlice for XCore.
struct ReplaceStridedSlice
    : public PassWrapper<ReplaceStridedSlice, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceStridedSlice)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-stridedslice"; }
  StringRef getDescription() const final {
    return "Replace TFL StridedSlice with StridedSlice for XCore.";
  }
  void runOnOperation() override;
};

// Utility to check if a StridedSliceOp can be replaced with a SliceOp
bool canReplaceWithSlice(TFL::StridedSliceOp stridedSliceOp) {
  // Check input shape is static
  auto inputType = stridedSliceOp.getInput().getType().dyn_cast<ShapedType>();
  if (!inputType || !inputType.hasStaticShape()) {
    return false;
  }

  // Check all strides are 1
  DenseIntElementsAttr stridesAttr;
  matchPattern(stridedSliceOp.getStrides(), m_Constant(&stridesAttr));
  if (!stridesAttr)
    return false;
  for (auto stride : stridesAttr)
    if (!stride.isOne())
      return false;

  if (stridedSliceOp.getEllipsisMask() != 0 ||
      stridedSliceOp.getShrinkAxisMask() != 0) {
    return false;
  }
  return true;
}

struct ReplaceStridedSlicePattern
    : public OpRewritePattern<TFL::StridedSliceOp> {
  using OpRewritePattern<TFL::StridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::StridedSliceOp stridedSliceOp,
                                PatternRewriter &rewriter) const override {

    if (!canReplaceWithSlice(stridedSliceOp))
      return failure();
    auto inputType = stridedSliceOp.getInput().getType().dyn_cast<ShapedType>();
    auto rank =
        stridedSliceOp.getInput().getType().cast<ShapedType>().getRank();
    DenseIntElementsAttr beginAttr;
    matchPattern(stridedSliceOp.getBegin(), m_Constant(&beginAttr));
    if (!beginAttr)
      return failure();
    auto begin = beginAttr.getValues<int32_t>();

    DenseIntElementsAttr endAttr;
    matchPattern(stridedSliceOp.getEnd(), m_Constant(&endAttr));
    if (!beginAttr)
      return failure();
    auto end = endAttr.getValues<int32_t>();
    std::vector<int32_t> newBegin(rank), newSize(rank);
    for (int i = 0; i < rank; i++) {
      newBegin[i] = (stridedSliceOp.getBeginMask() & (1 << i)) ? 0 : begin[i];
      newSize[i] = (stridedSliceOp.getEndMask() & (1 << i))
                       ? inputType.getShape()[i]
                       : end[i] - newBegin[i];
    }
    int64_t shrinkMask = stridedSliceOp.getShrinkAxisMask();
    std::vector<int64_t> newOutputShape;
    for (int i = 0; i < rank; ++i) {
      if (!(shrinkMask & (1 << i))) {         // Check if we should NOT shrink
        newOutputShape.push_back(newSize[i]); // Retain size
      }
    }

    auto shapeAttrType =
        RankedTensorType::get({rank}, rewriter.getIntegerType(32));

    // create constant ops for begin and size
    auto beginConstantOp = rewriter.create<arith::ConstantOp>(
        stridedSliceOp.getLoc(), shapeAttrType,
        DenseIntElementsAttr::get(shapeAttrType, newBegin));
    auto sizeConstantOp = rewriter.create<arith::ConstantOp>(
        stridedSliceOp.getLoc(), shapeAttrType,
        DenseIntElementsAttr::get(shapeAttrType, newSize));

    // create sliceOp
    std::vector<int64_t> newSize64(newSize.begin(), newSize.end());
    auto sliceOp = rewriter.create<TFL::SliceOp>(
        stridedSliceOp.getLoc(),
        RankedTensorType::get(ArrayRef<int64_t>(newSize64),
                              stridedSliceOp.getType().getElementType()),
        stridedSliceOp.getInput(), beginConstantOp, sizeConstantOp);

    // add reshape if shrinkMask is not 0
    if (shrinkMask != 0) {
      auto shapeConstantOp = rewriter.create<arith::ConstantOp>(
          stridedSliceOp.getLoc(), shapeAttrType,
          DenseIntElementsAttr::get(shapeAttrType, newOutputShape));
      auto newOutputType = RankedTensorType::get(
          newOutputShape, sliceOp.getType().getElementType());
      auto reshape = rewriter.create<TFL::ReshapeOp>(
          stridedSliceOp.getLoc(), newOutputType, sliceOp, shapeConstantOp);
      rewriter.replaceOp(stridedSliceOp, reshape.getOutput());
    } else {
      rewriter.replaceOp(stridedSliceOp, sliceOp.getOutput());
    }
    return success();
  }
};

void ReplaceStridedSlice::runOnOperation() {
  auto *ctx = &getContext();
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.insert<ReplaceStridedSlicePattern>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace

// Creates an instance of the ReplaceStridedSlice pass.
std::unique_ptr<OperationPass<func::FuncOp>> createReplaceStridedSlicePass() {
  return std::make_unique<ReplaceStridedSlice>();
}

static PassRegistration<ReplaceStridedSlice> pass;

} // namespace xcore
} // namespace mlir
