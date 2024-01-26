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
// Replace TFL StridedSlice with Slice for XCore.
struct ReplaceStridedSlice
    : public PassWrapper<ReplaceStridedSlice, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceStridedSlice)

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
  StringRef getArgument() const final { return "xcore-replace-stridedslice"; }
  StringRef getDescription() const final {
    return "Replace TFL StridedSlice with Slice for XCore.";
  }
  void runOnOperation() override;
};

struct ReplaceStridedSlicePattern
    : public OpRewritePattern<TFL::StridedSliceOp> {
  using OpRewritePattern<TFL::StridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::StridedSliceOp stridedSliceOp,
                                PatternRewriter &rewriter) const override {

    auto inputType =
        stridedSliceOp.getInput().getType().cast<RankedTensorType>();
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
        stridedSliceOp.getOutput().getType().cast<RankedTensorType>();
    auto outputElementType = outputType.getElementType();

    // Output type must be QI8
    if (!(outputElementType.isa<quant::QuantizedType>() &&
          outputElementType.cast<quant::QuantizedType>().isSigned() &&
          outputElementType.cast<quant::QuantizedType>()
                  .getStorageTypeIntegralWidth() == 8)) {
      return failure();
    }

    // Check if both input and output tensors have a rank of 4
    if (inputType.getRank() != 4 || outputType.getRank() != 4) {
      return failure();
    }

    DenseElementsAttr beginAttr;
    matchPattern(stridedSliceOp.getBegin(), m_Constant(&beginAttr));
    auto beginValues = beginAttr.getValues<int32_t>();

    DenseElementsAttr endAttr;
    matchPattern(stridedSliceOp.getEnd(), m_Constant(&endAttr));
    auto endValues = endAttr.getValues<int32_t>();

    DenseElementsAttr stridesAttr;
    matchPattern(stridedSliceOp.getStrides(), m_Constant(&stridesAttr));

    // TODO: We don't support masks yet
    if (stridedSliceOp.getBeginMask() != 0 ||
        stridedSliceOp.getEndMask() != 0 ||
        stridedSliceOp.getEllipsisMask() != 0 ||
        stridedSliceOp.getNewAxisMask() != 0 ||
        stridedSliceOp.getShrinkAxisMask() != 0) {
      return failure();
    }

    // Check if strides are all 1
    for (auto stride : stridesAttr.getValues<int32_t>()) {
      if (stride != 1) {
        return failure();
      }
    }

    if (beginValues[3] != 0) {
      return failure();
    }
    for (int i = 0; i < 3; i++) {
      if (beginValues[i] != 0 || endValues[i] != inputType.getDimSize(i)) {
        return failure();
      }
    }

    SliceMemcpyType memcpyType;
    if (inputType.getDimSize(2) == outputType.getDimSize(2) &&
        inputType.getDimSize(3) == outputType.getDimSize(3)) {
      // Single CPU memcpy
      memcpyType = SliceMemcpyType::SliceCpy;
      // } else if (inputType.getDimSize(2) % 4 == 0 &&
      //            outputType.getDimSize(2) == inputType.getDimSize(2)) {
      //   // If depth * output width is a multiple of four and the x,y location
      //   of
      //   // the starting pixel is word-aligned, we can do a slice copy
      //   instead.
      //   // That is ((y * input depth + x) * depth) is a multiple of four.
      //   // We use a simple memcpy to do the copy in the runtime.
      //   // Input and output tensors must have the same width.
      //   memcpyType = SliceMemcpyType::VpuCpy;
    } else if (inputType.getDimSize(3) % 4 == 0 &&
               outputType.getDimSize(3) % 4 == 0) {
      // If not a slice copy, then if both depths are multiples of four, we can
      // do pixel by pixel VPU copy.
      memcpyType = SliceMemcpyType::VpuCpy;
    } else {
      // Pixel by pixel CPU memcpy, when depth not a multiple of four
      memcpyType = SliceMemcpyType::MemCpy;
    }
    int32_t inputHeight = inputType.getDimSize(1);
    int32_t inputWidth = inputType.getDimSize(2);
    int32_t inputDepth = inputType.getDimSize(3);
    int32_t beginX = beginValues[2];
    int32_t beginY = beginValues[1];
    int32_t endX = endValues[2];
    int32_t endY = endValues[1];

    auto image_geom = nn::ImageGeometry(inputHeight, inputWidth, inputDepth);

    int xDiff = endX - beginX;
    int yDiff = endY - beginY;
    auto window_geom = nn::WindowGeometry({yDiff, xDiff, inputDepth},
                                          {beginY, beginX}, {1, 1, 1}, {1, 1});

    nn::ImToColValid imToCol(image_geom, window_geom,
                             static_cast<int>(inputDepth),
                             /*dont_zero_pad_at_the_end=*/true);
    auto imToColParams = imToCol.getParams();
    auto mfStr = std::string((char *)&imToColParams, sizeof(imToColParams));

    auto binaryObjectSliceOp = rewriter.create<SliceOp>(
        stridedSliceOp.getLoc(), stridedSliceOp.getType(),
        stridedSliceOp.getInput(), rewriter.getI32IntegerAttr(beginX),
        rewriter.getI32IntegerAttr(beginY), rewriter.getStringAttr(mfStr),
        rewriter.getI32IntegerAttr((int32_t)memcpyType));
    rewriter.replaceOp(stridedSliceOp, binaryObjectSliceOp.getOutput());

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
