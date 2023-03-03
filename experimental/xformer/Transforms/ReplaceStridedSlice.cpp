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

struct ReplaceStridedSlicePattern
    : public OpRewritePattern<TFL::StridedSliceOp> {
  using OpRewritePattern<TFL::StridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::StridedSliceOp stridedSliceOp,
                                PatternRewriter &rewriter) const override {

    auto inputType = stridedSliceOp.input().getType().cast<RankedTensorType>();
    auto inputElementType = inputType.getElementType();

    // Check for invalid types and return
    // Input type must be QI8
    if (!(inputElementType.isa<quant::QuantizedType>() &&
          inputElementType.cast<quant::QuantizedType>().isSigned() &&
          inputElementType.cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    auto outputType =
        stridedSliceOp.output().getType().cast<RankedTensorType>();
    auto outputElementType = outputType.getElementType();

    // Output type must be QI8
    if (!(outputElementType.isa<quant::QuantizedType>() &&
          outputElementType.cast<quant::QuantizedType>().isSigned() &&
          outputElementType.cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    // Depth must be a multiple of four
    if (inputType.getDimSize(3) % 4 != 0 || outputType.getDimSize(3) % 4 != 0) {
      return failure();
    }

    // Extract args from the op
    DenseElementsAttr beginAttr;
    matchPattern(stridedSliceOp.begin(), m_Constant(&beginAttr));

    DenseElementsAttr endAttr;
    matchPattern(stridedSliceOp.end(), m_Constant(&endAttr));

    DenseElementsAttr stridesAttr;
    matchPattern(stridedSliceOp.strides(), m_Constant(&stridesAttr));

    auto inputHeight = inputType.getDimSize(1);
    auto inputWidth = inputType.getDimSize(2);
    auto inputDepth = inputType.getDimSize(3);
    auto beginX = beginAttr.getValues<int32_t>()[2];
    auto beginY = beginAttr.getValues<int32_t>()[1];
    auto endX = endAttr.getValues<int32_t>()[2];
    auto endY = endAttr.getValues<int32_t>()[1];
    auto strideX = stridesAttr.getValues<int32_t>()[2];
    auto strideY = stridesAttr.getValues<int32_t>()[1];

    auto image_geom = nn::ImageGeometry(inputHeight, inputWidth,
                                        static_cast<int>(inputDepth));

    int xDiff = endX - beginX;
    int yDiff = endY - beginY;
    auto window_geom =
        nn::WindowGeometry({yDiff, xDiff, static_cast<int>(inputDepth)},
                           {beginY, beginX}, {1, 1, 1}, {strideY, strideX});

    nn::ImToColValid::Params imToColParams(image_geom, window_geom,
                                           static_cast<int>(inputDepth),
                                           /*dont_zero_pad_at_the_end=*/true);

    std::string mfStr = imToColParams.serialise<nn::ImToColValid::Params>();

    auto binaryObjectStridedSliceOp = rewriter.create<StridedSliceOp>(
        stridedSliceOp.getLoc(), stridedSliceOp.getType(),
        stridedSliceOp.input(), rewriter.getI32IntegerAttr(beginX),
        rewriter.getI32IntegerAttr(beginY), rewriter.getStringAttr(mfStr));
    rewriter.replaceOp(stridedSliceOp, binaryObjectStridedSliceOp.output());

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
